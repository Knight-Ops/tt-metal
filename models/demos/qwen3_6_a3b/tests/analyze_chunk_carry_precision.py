# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only precision study of the CROSS-CHUNK STATE CARRY in the fused chunk_state kernel.

The single-chunk bf16 output is already fine (analyze_chunk_precision.py: PCC 0.99999). The fused
bf16 prefill kernel nonetheless drifts to incoherence over 30 layers. This script isolates *why*, by
faithfully emulating the kernel's per-chunk recurrence (reference/qwen3_5_moe.py::
torch_chunk_gated_delta_rule inner loop, lines ~301-312) under three device precision regimes and
measuring how chunk-output accuracy degrades as the number of chunks Nc grows (Nc = seq/64):

  mode "bf16dst_bf16carry"  : every op output rounds to bf16 (fp32_dest_acc_en=False), state S stored
                              bf16 between chunks.  == the CURRENT kernel.
  mode "fp32dst_bf16carry"  : op outputs kept fp32 (fp32 DST), but S rounded to bf16 at the carry
                              write.  Isolates whether fp32 DST alone (no fp32 storage) suffices.
  mode "fp32dst_fp32carry"  : op outputs fp32 AND S carried fp32.  == the proposed fp32 kernel.

In ALL modes matmul INPUTS are rounded to bf16 (Tensix always feeds bf16 to the FPU) — so this study
does NOT assume an unreachable fp32-input matmul; it only varies the accumulator/storage precision,
which is exactly what fp32_dest_acc_en + the S buffer dtype control.

Reports chunk-output PCC and relative L2 error vs a full-fp32 reference, plus a coarse 30-layer
compounding bound (rel*30 coherent / rel*sqrt(30) independent) so we can predict whether the fp32
kernel will hold across the real 30-layer stack. Realistic head-dim 128, strong-decay gated-delta data.
"""
import math

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
C = 64  # chunk size (matches _CHUNK / reference default)


def rbf(x):  # round to bf16, keep float32 container
    return x.to(BF16).float()


def _mm_inputs(a, b, fid):
    """Emulate Tensix matmul-input fidelity. LoFi/HiFi2: a single bf16 rounding of each operand (what a
    plain tt-lang matmul_block does). HiFi4: split each operand into hi + lo bf16 components and sum the
    two partial products (a ≈ bf16(a) + bf16(a-bf16(a))) — recovers ~fp32 input precision, which is what
    ttnn's HiFi4 MathFidelity does via multi-pass on the bf16 FPU."""
    if fid == "hifi4":
        ah, bh = rbf(a), rbf(b)
        al, bl = rbf(a - ah), rbf(b - bh)
        return ah @ bh + ah @ bl + al @ bh  # drop al@bl (2nd-order, negligible) — 3-pass HiFi4-ish
    return rbf(a) @ rbf(b)  # single-pass bf16 inputs (LoFi/HiFi2)


def mm(a, b, dst, fid="lofi"):
    """Tensix matmul: input fidelity `fid`, fp32 accumulate, output at DST precision (bf16|fp32)."""
    o = _mm_inputs(a, b, fid)
    return rbf(o) if dst == "bf16" else o


def elt(x, dst):
    """Elementwise result lands in DST too, so it rounds with the DST precision."""
    return rbf(x) if dst == "bf16" else x


def chunk_prep_fp32(q, k, v, g, beta):
    """fp32/HiFi4 prep (matches the model's eager prep) -> per-chunk terms, one chunk [C,D].
    Returns terms the kernel consumes (bf16-rounded once when fed to the kernel, like hm())."""
    scale = q.shape[-1] ** -0.5
    q = l2n(q) * scale
    k = l2n(k)
    gc = g.cumsum(0)
    decay = (gc[:, None] - gc[None, :]).tril().exp().tril()
    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]
    L = torch.tril(-(k_beta @ k.T * decay), diagonal=-1)
    T = torch.inverse(torch.eye(C) - L)
    w = T @ v_beta
    kcd = T @ (k_beta * gc.exp()[:, None])
    qg = q * gc.exp()[:, None]
    kgt = (k * (gc[-1] - gc).exp()[:, None]).T
    glast = gc[-1].exp().item()
    aintra = torch.tril(q @ k.T * decay)
    return dict(q=q, k=k, decay=decay, w=w, kcd=kcd, qg=qg, kgt=kgt, glast=glast, aintra=aintra)


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def run_chunked(terms_per_chunk, Dk, Dv, dst, carry, fid="lofi"):
    """Emulate the fused kernel recurrence across chunks in the given precision regime.
    v_new = w - kcd@S ; out = qg@S + aintra@v_new ; Snew = S*glast + kgt@v_new."""
    S = torch.zeros(Dk, Dv)
    outs = []
    for t in terms_per_chunk:
        kcdS = mm(t["kcd"], S, dst, fid)
        v_new = elt(t["w"] - kcdS, dst)
        qgS = mm(t["qg"], S, dst, fid)
        av = mm(t["aintra"], v_new, dst, fid)
        out = elt(qgS + av, dst)
        sg = elt(S * t["glast"], dst)
        skv = mm(t["kgt"], v_new, dst, fid)
        Snew = elt(sg + skv, dst)
        S = rbf(Snew) if carry == "bf16" else Snew
        outs.append(out)
    return torch.cat(outs, 0)


def run_reference(terms_per_chunk, Dk, Dv):
    """Full-fp32 gold recurrence (no bf16 anywhere)."""
    S = torch.zeros(Dk, Dv)
    outs = []
    for t in terms_per_chunk:
        v_new = t["w"] - t["kcd"] @ S
        out = t["qg"] @ S + t["aintra"] @ v_new
        S = S * t["glast"] + t["kgt"] @ v_new
        outs.append(out)
    return torch.cat(outs, 0)


def pcc(x, y):
    return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()


def relerr(x, ref):
    return (x - ref).norm().item() / (ref.norm().item() + 1e-30)


def main():
    torch.manual_seed(0)
    Dk = Dv = 128
    # (name, dst, carry, matmul-input-fidelity). The first three vary DST/carry with LoFi (single-pass
    # bf16) inputs = what the tt-lang kernel does today. The last is HiFi4 (multi-pass ~fp32 inputs) =
    # what ttnn's _chunk_state_ttnn HiFi4 path does. Isolates whether the fix is DST or input fidelity.
    modes = [
        ("bf16dst_bf16carry_lofi", "bf16", "bf16", "lofi"),
        ("fp32dst_bf16carry_lofi", "fp32", "bf16", "lofi"),
        ("fp32dst_fp32carry_lofi", "fp32", "fp32", "lofi"),
        ("fp32dst_fp32carry_HIFI4", "fp32", "fp32", "hifi4"),
    ]
    print(
        f"head-dim {Dk}, chunk {C}. LoFi=single-pass bf16 inputs (tt-lang kernel); HiFi4=multi-pass ~fp32 inputs (ttnn path). PCC/relerr vs full-fp32.\n"
    )
    header = f"{'Nc':>3} {'seq':>5} | " + " | ".join(f"{m[0]:>26}" for m in modes)
    print(header)
    print("-" * len(header))
    for Nc in (1, 2, 4, 8, 16, 32):
        # fresh realistic data per chunk (strong-decay gated-delta regime)
        terms = []
        for _ in range(Nc):
            q = torch.randn(C, Dk)
            k = torch.randn(C, Dk)
            v = torch.randn(C, Dv) * 0.5
            beta = torch.sigmoid(torch.randn(C))
            A_log = torch.rand(1).uniform_(0, 16).log_()
            g = -A_log.exp() * F.softplus(torch.randn(C) + 1.0)
            terms.append(chunk_prep_fp32(q, k, v, g, beta))
        ref = run_reference(terms, Dk, Dv)
        cells = []
        for _, dst, carry, fid in modes:
            out = run_chunked(terms, Dk, Dv, dst, carry, fid)
            p, re = pcc(out, ref), relerr(out, ref)
            cells.append(f"{p:.5f}/{re*100:5.2f}%")
        print(f"{Nc:>3} {Nc*C:>5} | " + " | ".join(f"{c:>26}" for c in cells))

    # 30-layer compounding bound from the Nc=32 (seq 2048) per-layer relative error
    print("\n30-layer compounding bound (from Nc=32 per-layer relerr; the real prefill stacks 30 GDN layers):")
    terms = []
    for _ in range(32):
        q = torch.randn(C, Dk)
        k = torch.randn(C, Dk)
        v = torch.randn(C, Dv) * 0.5
        beta = torch.sigmoid(torch.randn(C))
        g = -torch.rand(1).uniform_(0, 16).log_().exp() * F.softplus(torch.randn(C) + 1.0)
        terms.append(chunk_prep_fp32(q, k, v, g, beta))
    ref = run_reference(terms, Dk, Dv)
    for name, dst, carry, fid in modes:
        re = relerr(run_chunked(terms, Dk, Dv, dst, carry, fid), ref)
        print(
            f"  {name:>26}: per-layer {re*100:5.2f}%  -> ~{re*30*100:6.1f}% coherent / ~{re*math.sqrt(30)*100:5.1f}% indep over 30L"
        )


if __name__ == "__main__":
    main()
