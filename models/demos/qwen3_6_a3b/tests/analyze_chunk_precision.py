# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only analysis: does bf16 matmul-input precision (Tensix behavior) materially degrade the
single-chunk Gated-DeltaNet output? Decides whether the fused kernel needs an fp32/stabilized inverse.

Runs torch_chunk_gated_delta_rule's per-chunk math twice: (a) fp32 everywhere, (b) every matmul with
operands rounded to bf16 (accumulate fp32) — exactly what a tt-lang/Tensix kernel does. Reports PCC of
the chunk OUTPUT and of the inverse alone, at realistic post-l2norm magnitudes.
"""
import torch
import torch.nn.functional as F


def bf16mm(a, b):
    # Tensix matmul: operands rounded to bf16, fp32 accumulate.
    return a.to(torch.bfloat16).float() @ b.to(torch.bfloat16).float()


def chunk_output(q, k, v, g, beta, mm, C=64):
    """One chunk of torch_chunk_gated_delta_rule (initial_state=0), using matmul fn `mm`."""
    q = F.normalize(q, dim=-1, eps=1e-6) * (q.shape[-1] ** -0.5)
    k = F.normalize(k, dim=-1, eps=1e-6)
    g = g.cumsum(-1)
    decay = (g[:, None] - g[None, :]).tril().exp().tril()
    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]
    eye = torch.eye(C)
    tri_excl = torch.triu(torch.ones(C, C, dtype=torch.bool), 0)
    L = -(mm(k_beta, k.T) * decay).masked_fill(tri_excl, 0)  # strictly-lower
    # inverse T = (I - L)^-1 via doubling product, using mm (matches the device kernel)
    p = L.clone()
    T = eye + L
    for _ in range(int(torch.log2(torch.tensor(float(C)))) - 1):
        p = mm(p, p)
        T = mm(T, eye + p)
    w = mm(T, v_beta)
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool), 1)
    attn = (mm(q, k.T) * decay).masked_fill(mask, 0)
    out = mm(attn, w)  # intra-chunk (initial_state=0 so no inter term)
    return out, T, L


def main():
    torch.manual_seed(0)
    C, D = 64, 128
    # realistic pre-conv projections; magnitudes that produce typical beta/decay
    q = torch.randn(C, D)
    k = torch.randn(C, D)
    v = torch.randn(C, D) * 0.5
    beta = torch.sigmoid(torch.randn(C))  # (0,1)
    A_log = torch.rand(8).uniform_(0, 16).log_()  # like the real A_log range
    a = torch.randn(C)
    g = -A_log.exp().mean() * F.softplus(a + 1.0)  # negative decays

    out32, T32, L = chunk_output(q, k, v, g, beta, lambda a, b: a @ b)
    outbf, Tbf, _ = chunk_output(q, k, v, g, beta, bf16mm)

    def pcc(x, y):
        return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()

    print(f"L max|.|={L.abs().max():.3f}   inverse T max|.|={T32.abs().max():.2f}")
    print(f"[analyze] inverse (I-L)^-1   bf16-mm vs fp32 PCC: {pcc(Tbf, T32):.5f}")
    print(f"[analyze] CHUNK OUTPUT       bf16-mm vs fp32 PCC: {pcc(outbf, out32):.5f}")
    print(
        "[analyze] -> if chunk-output PCC > 0.99, the bf16 inverse is fine end-to-end; "
        "the 0.95 inverse PCC does NOT need fixing."
    )


if __name__ == "__main__":
    main()
