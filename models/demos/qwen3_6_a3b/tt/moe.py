# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn MoE block for Qwen3.6-35B-A3B.

Matches HF ``Qwen3_5MoeSparseMoeBlock``:
  router: softmax over all experts -> top-k -> renormalize;
  experts: SwiGLU per expert (packed gate_up_proj + down_proj);
  shared expert: an always-on MLP gated by ``sigmoid(shared_expert_gate)``;
  output = routed + shared.

PREFILL (default) computes ALL experts for ALL tokens via a single batched matmul and masks by the
(scattered) routing weights. This is ~num_experts/top_k x the needed FLOPs but, at this expert size,
the MoE prefill is dispatch-bound not FLOP-bound, so dense is measured faster than the sparse per-tile
path and is the default. DECODE (T=1) uses the sparse/gather path (only the top_k active experts).

An optional SPARSE per-tile prefill path (``forward_sparse_prefill``, opt-in via
``QWEN36_SPARSE_PREFILL=1``) processes tokens in 32-row tiles and computes, per tile, only the
experts that *some* token in the tile selected (the per-tile union mask), via ``ttnn.sparse_matmul``
with ``nnz=None`` — the same mechanism ``forward_sparse_decode`` uses, generalized from M=1 to M=32.
It is PCC-identical to the dense path (experts not in the union were selected by no token; selected
(token, expert) products are still computed). (The single ``both-sparse`` down projection forces an
E-length sparsity per call, which is why tiles are processed one at a time rather than as one batched
cross-product matmul.)

PERF NOTE — it is OFF by default because it is currently SLOWER, not faster, at this model's expert
size. Measured on a single P150 (E=256, inter=512, bf4 experts): dense prefill ~30-35ms and nearly
flat in T (one batched matmul over all experts), while the per-tile sparse path costs ~82ms at T=128
and ~164ms at T=256 (0.37x / 0.22x) — its per-tile host-dispatch overhead grows linearly with T.
MoE prefill here is dispatch/overhead-bound, not FLOP-bound, so cutting the ~32x expert-FLOP waste
buys nothing and the loop overhead dominates. Kept as a correct reference / for larger-expert configs
where prefill becomes FLOP-bound. Dense is the default and recommended path.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.common import moe_gather
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.common import as_weight, build_dram_shard, to_tt

# DRAM-shard the DECODE shared-expert down projection se_down (K=se_inter, N=hidden). Microbench showed
# 1.46x per op, but at 40L it is a WASH: se_down's ~6us matmul saving is eaten by the ~8us reshard in/out
# (K=512 is small, so the fixed reshard cost dominates). Default OFF; opt in with QWEN36_MOE_SE_DRAM_SHARD=1.
_SE_DRAM_SHARD = os.environ.get("QWEN36_MOE_SE_DRAM_SHARD", "0") == "1"

# Max tokens per dense-expert matmul chunk. The tuned batched-matmul program config holds each
# expert's full [tc, N] output in L1 (per_core_M = tc/32); tc=256 (per_core_M=8) is proven to fit,
# tc~384 overflows. Prefill chunks T into <=256 blocks so any sequence length stays L1-bounded.
_DENSE_TMAX = 256


class TtMoE(LightweightModule):
    def __init__(
        self,
        mesh_device,
        weights,
        num_experts,
        top_k,
        expert_dtype=ttnn.bfloat4_b,
        down_dtype=None,
        dtype=ttnn.bfloat16,
        sparse_decode=True,
        compute_kernel_config=None,
        cache_path=None,
        hidden=None,
        inter=None,
        se_inter=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        cn = (lambda role: f"{cache_path}/{role}") if cache_path else (lambda role: None)

        # weights[*] may be lazy thunks (production loader) or plain tensors (module tests). W()
        # materializes either; it is only invoked on a cache miss for the big expert weights.
        def W(k):
            v = weights[k]
            return v() if callable(v) else v

        self.num_experts = num_experts
        self.top_k = top_k
        # sparse_decode=True uses the gather-top-k path (less compute, but a host readback per call,
        # which forces a device sync -> bad for latency/trace). False uses the dense path (no sync).
        self.sparse_decode = sparse_decode
        # nnz hint for the decode sparse_matmuls: top_k makes the kernel iterate only the active
        # experts instead of scanning all E slots (see forward_sparse_decode). QWEN36_MOE_NNZ=0 -> None.
        self._decode_nnz = None if os.environ.get("QWEN36_MOE_NNZ") == "0" else top_k
        # Indexed/gather decode: pass the active expert ids to ttnn.sparse_matmul so the kernels iterate
        # only the top_k selected experts (compact output) instead of scanning all E sparsity slots.
        # Orthogonal to _decode_nnz; default ON. QWEN36_MOE_GATHER=0 falls back to the 256-slot scan.
        self._decode_gather = os.environ.get("QWEN36_MOE_GATHER", "1") != "0"
        # Compute-kernel config for the dense PREFILL expert matmuls (LoFi is correct for BFP4 experts;
        # PCC-identical to the default here). None falls back to the ttnn default.
        self.compute_kernel_config = compute_kernel_config
        g = mesh_device.compute_with_storage_grid_size()
        self._grid = (g.x, g.y)  # full compute grid (Blackhole P150: 11x10)

        # router: logits = x @ gate_w^T  ; reference gate weight is [E, hidden]
        self.gate_w = as_weight(weights["gate"], mesh_device, dtype=dtype, cache_file_name=cn("gate"))  # -> [hidden, E]

        # Expert weights in sparse_matmul layout [1, E, in, out], used for BOTH dense prefill
        # (batched matmul, squeeze leading dim) and sparse_matmul decode. gate+up are kept FUSED
        # (HF layout: first inter rows = gate, next inter = up) as [1, E, H, 2*inter] so the gate+up
        # stage issues ONE (sparse_)matmul instead of two -> halves that stage's dispatch overhead
        # (the dominant decode cost is per-launch sparse_matmul overhead, not weight bandwidth).
        # down transposed to [1, E, inter, hidden]. Total weight bytes unchanged (1x).
        E = num_experts
        # Prefer config dims (so a cache hit needn't read the weight just to learn its shape); fall
        # back to a shape read for direct callers (tests) that don't pass them.
        H = hidden if hidden is not None else W("gate_up_proj").shape[2]
        I = inter if inter is not None else W("gate_up_proj").shape[1] // 2
        # Routed-expert precision. gate_up stays at expert_dtype (bf4 by default); down_proj can be
        # pinned higher via down_dtype (the AesSedai/Ubergarm mixed-precision recipe: down_proj is the
        # more sensitive projection, so lifting only it recovers accuracy at ~1/3 the memory cost of
        # lifting the whole expert). down_dtype=None -> same as expert_dtype (uniform).
        down_dtype = down_dtype or expert_dtype
        self.hidden, self.inter, self.expert_dtype, self.down_dtype = H, I, expert_dtype, down_dtype
        self.gate_up_sp = to_tt(
            lambda: W("gate_up_proj").transpose(1, 2).reshape(1, E, H, 2 * I).contiguous(),
            mesh_device,
            dtype=expert_dtype,
            cache_file_name=cn("gate_up_proj"),
        )
        self.down_sp = to_tt(
            lambda: W("down_proj").transpose(1, 2).reshape(1, E, I, H).contiguous(),
            mesh_device,
            dtype=down_dtype,
            cache_file_name=cn("down_proj"),
        )

        # shared expert (MLP) + its gate. se_gate_proj/se_up_proj both map hidden -> se_inter; fuse
        # into one [hidden, 2*se_inter] matmul (one launch), split the output for the SwiGLU.
        self.se_inter = se_inter if se_inter is not None else W("se_gate_proj").shape[0]
        self.se_gate_up = as_weight(
            lambda: torch.cat([W("se_gate_proj"), W("se_up_proj")], dim=0),
            mesh_device,
            dtype=dtype,
            cache_file_name=cn("se_gate_up"),
        )
        self.se_down = as_weight(weights["se_down_proj"], mesh_device, dtype=dtype, cache_file_name=cn("se_down"))
        self.se_router = as_weight(weights["se_router"], mesh_device, dtype=dtype, cache_file_name=cn("se_router"))
        # DRAM-sharded decode se_down (built from the interleaved weight; prefill keeps interleaved).
        self._se_down_dram = None
        if _SE_DRAM_SHARD:
            self._se_down_dram, self._se_amc, self._se_omc, self._se_pc = build_dram_shard(
                self.se_down, self.se_down.shape[-2], self.se_down.shape[-1]
            )

    @staticmethod
    def _grid_for(Nt):
        # pick a rectangular grid with cx*cy == Nt (per_core_N=1 -> every core has work)
        for d in range(min(10, Nt), 0, -1):
            if Nt % d == 0 and Nt // d <= 10:
                return (Nt // d, d)
        return (1, 1)

    @staticmethod
    def _largest_divisor_leq(n, cap):
        for d in range(min(n, cap), 0, -1):
            if n % d == 0:
                return d
        return 1

    def _dense_expert_pc(self, m, n):
        """Batched-matmul program config for the dense expert matmuls [E,M,K]@[E,K,N]. The default
        ttnn heuristic serializes the E=256 batch onto few cores (~15 ms); spreading each expert's
        full [M,N] output across the whole compute grid is ~7x faster (PCC-identical). m,n are the
        (padded-to-tile) output element dims. out_subblock_h*out_subblock_w must fit DST: 8 tiles for
        bf16 dest-acc (LoFi/the model path), 4 for fp32 dest-acc (the ttnn default); prefer wide N."""
        m_tiles = (m + 31) // 32
        n_tiles = (n + 31) // 32
        ck = self.compute_kernel_config
        fp32_dst = getattr(ck, "fp32_dest_acc_en", True) if ck is not None else True
        dst_cap = 4 if fp32_dst else 8
        sw = self._largest_divisor_leq(n_tiles, 4)
        sh = self._largest_divisor_leq(m_tiles, max(1, dst_cap // sw))
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self._grid),
            in0_block_w=1,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=m_tiles,
            per_core_N=n_tiles,
        )

    @staticmethod
    def _split_last(t, half):
        """Split a tensor into [..., :half] and [..., half:] along the last dim, rank-agnostic
        (sparse_matmul output keeps its native batched rank, so build begins/ends from t.shape)."""
        sh = list(t.shape)
        r = len(sh)
        lo = ttnn.slice(t, [0] * r, sh[:-1] + [half])
        hi = ttnn.slice(t, [0] * (r - 1) + [half], list(sh))
        return lo, hi

    @classmethod
    def _sparse_pc(cls, m, n):
        Nt = (n + 31) // 32
        cx, cy = cls._grid_for(Nt)
        # in0_block_w = K-tiles processed per K-block. Larger -> fewer K-block iterations -> fewer
        # per-block multicast-semaphore handshakes (the measured decode MoE bottleneck). Must divide
        # Kt for both expert matmuls (gate_up Kt=64, down Kt=16 -> common divisors 1,2,4,8,16). Measured
        # 16 > 8 at 40L: MoE 0.348 -> 0.321 ms/op (~-1.1 ms/token), PCC-identical (program config only).
        in0bw = int(os.environ.get("QWEN36_SPARSE_IN0BW", "16"))
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
            in0_block_w=in0bw,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=max(32, m) // 32,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def forward_sparse_decode(self, x2, sparsity, topv=None, indices=None):
        """x2: [1, hidden] (single token). sparsity: [1,1,1,E] bf16 routing weights. Computes only
        the experts with nonzero routing weight via ttnn.sparse_matmul (no host sync, traceable).
        gate+up are one fused sparse_matmul, then split for the SwiGLU.

        If ``indices`` (the top_k active expert ids, [1,1,1,top_k] uint16) and ``topv`` ([1,top_k]
        routing weights) are given, runs the INDEXED/GATHER path: both sparse_matmuls iterate only the
        top_k selected experts and return a COMPACT [.., top_k, ..] output (no 256-slot scan); the
        combine is sum_i topv[i]*down[i] over top_k. Otherwise the legacy 256-slot sparsity-scan path
        (weight by sparsity, sum over E) is used.

        nnz: with nnz=None the kernel still iterates ALL E sparsity slots (per-slot multicast
        semaphores) even though it skips the DRAM reads of zero experts — that 256-slot scan is the
        dominant decode cost (~1.4 ms/layer). Passing nnz=top_k makes the compute kernel process only
        the active experts (num_batch_compute=nnz in the 1D-optimized factory), iterating ~8 not 256.
        This is SAFE here because decode routing is softmax -> top-k -> renormalize, so EXACTLY top_k
        experts are active every token with positive (sum-to-1) weights — they never flush to zero
        (the gpt-oss deadlock was a routing where the active count could drop below a static nnz; that
        cannot happen for a fixed top-k). Disable with QWEN36_MOE_NNZ=0 if a routing ever violates this."""
        E, H, I = self.num_experts, self.hidden, self.inter
        x4 = ttnn.reshape(x2, [1, 1, 1, H])

        if indices is not None:
            # --- indexed/gather path: iterate only the top_k active experts, compact output ---
            gu = ttnn.sparse_matmul(
                x4,
                self.gate_up_sp,
                sparsity=sparsity,
                indices=indices,
                program_config=self._sparse_pc(1, 2 * I),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )  # [.., top_k, 1, 2I]
            gate, up = self._split_last(gu, I)
            h = ttnn.multiply(ttnn.silu(gate), up)  # [.., top_k, 1, I]
            down = ttnn.sparse_matmul(
                h,
                self.down_sp,
                sparsity=sparsity,
                indices=indices,
                is_input_a_sparse=True,
                program_config=self._sparse_pc(1, H),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )  # [.., top_k, 1, H]
            return moe_gather.gather_combine(down, topv, H, self.top_k)  # [hidden]

        # --- legacy 256-slot sparsity-scan path ---
        nnz = self._decode_nnz
        gu = ttnn.sparse_matmul(
            x4,
            self.gate_up_sp,
            sparsity=sparsity,
            nnz=nnz,
            program_config=self._sparse_pc(1, 2 * I),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate, up = self._split_last(gu, I)
        h = ttnn.multiply(ttnn.silu(gate), up)  # [1, E, 1, I]
        down = ttnn.sparse_matmul(
            h,
            self.down_sp,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=True,
            program_config=self._sparse_pc(1, H),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # weight each expert output by its routing weight, sum over experts
        down = ttnn.reshape(down, [self.num_experts, H])  # [E, hidden]
        w = ttnn.reshape(sparsity, [self.num_experts, 1])
        return ttnn.sum(ttnn.multiply(down, w), dim=0)  # [hidden] -> [1, hidden] via caller

    def _experts_for_tile(self, xt, sparsity, wt):
        """One 32-row tile of the sparse expert path. xt: [32, hidden]. sparsity: [1,1,1,E] (the
        tile's union mask, ROW_MAJOR bf16). wt: [E, 32, 1] per-(expert,token) routing weights.
        Returns [32, hidden]: sum_e wt[e] * down(silu(gate_e(xt)) * up_e(xt)) over the tile."""
        E, H, I = self.num_experts, self.hidden, self.inter
        x4 = ttnn.reshape(xt, [1, 1, 32, H])
        gu = ttnn.sparse_matmul(
            x4,
            self.gate_up_sp,
            sparsity=sparsity,
            nnz=None,
            program_config=self._sparse_pc(32, 2 * I),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate, up = self._split_last(gu, I)
        h = ttnn.multiply(ttnn.silu(gate), up)  # [1, E, 32, I]
        h = ttnn.reshape(h, [1, E, 32, I])
        down = ttnn.sparse_matmul(
            h,
            self.down_sp,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            program_config=self._sparse_pc(32, H),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        down = ttnn.reshape(down, [E, 32, H])  # [E, 32, hidden]
        return ttnn.sum(ttnn.multiply(down, wt), dim=0)  # [32, hidden]

    def forward_sparse_prefill(self, x2, routing):
        """x2: [T, hidden] (T a multiple of 32). routing: [T, E] scattered top-k weights.
        Per-32-token-tile sparse expert compute (union mask). Returns routed output [T, hidden]."""
        E, H = self.num_experts, self.hidden
        T = x2.shape[0]
        num_tiles = T // 32
        outs = []
        for t in range(num_tiles):
            xt = ttnn.slice(x2, [t * 32, 0], [(t + 1) * 32, H])  # [32, hidden]
            rt = ttnn.slice(routing, [t * 32, 0], [(t + 1) * 32, E])  # [32, E]
            # union mask: nonzero where any of the 32 tokens picked the expert (weights are >= 0)
            sparsity = ttnn.to_layout(ttnn.reshape(ttnn.sum(rt, dim=0), [1, 1, 1, E]), ttnn.ROW_MAJOR_LAYOUT)
            wt = ttnn.reshape(ttnn.transpose(rt, 0, 1), [E, 32, 1])  # [E, 32, 1]
            outs.append(self._experts_for_tile(xt, sparsity, wt))
        return outs[0] if num_tiles == 1 else ttnn.concat(outs, dim=0)  # [T, hidden]

    def _shared_and_router(self, x2):
        with sp.region("moe.shared"):
            se_gu = ttnn.linear(x2, self.se_gate_up)  # [T, 2*se_inter]
            se_gate, se_up = self._split_last(se_gu, self.se_inter)
            shared = ttnn.multiply(ttnn.silu(se_gate), se_up)
            if self._se_down_dram is not None and x2.shape[0] == 1:
                # Decode: DRAM-sharded se_down. Reshard in, matmul, reshard back to interleaved DRAM.
                sh_in = ttnn.to_memory_config(shared, self._se_amc)
                shared = ttnn.linear(sh_in, self._se_down_dram, program_config=self._se_pc, memory_config=self._se_omc)
                shared = ttnn.to_memory_config(shared, ttnn.DRAM_MEMORY_CONFIG)
            else:
                shared = ttnn.linear(shared, self.se_down)
            shared = ttnn.multiply(shared, ttnn.sigmoid(ttnn.linear(x2, self.se_router)))  # [T, hidden]
        with sp.region("moe.router"):
            logits = ttnn.linear(x2, self.gate_w)  # [T, E]
            probs = ttnn.softmax(logits, dim=-1)
            topv, topi = ttnn.topk(probs, self.top_k, dim=-1, largest=True, sorted=True)  # [T, k]
            topv = ttnn.divide(topv, ttnn.sum(topv, dim=-1, keepdim=True))  # renormalize
        return shared, topv, topi, probs

    def forward(self, x):
        """x: [1, 1, T, hidden] -> [1, 1, T, hidden]. sparse_matmul decode (T==1); dense prefill by
        default (T>1); optional sparse per-tile prefill if QWEN36_SPARSE_PREFILL=1 and T%32==0."""
        T = x.shape[2]
        hidden = x.shape[3]
        x2 = ttnn.reshape(x, [T, hidden])
        # NOTE: no prof.phase here — this runs in the T==1 decode path too, and prof.phase syncs the
        # device (would break trace capture if profiling were enabled during decode). The prefill-only
        # sub-timers below (T>1 dense branch) are safe.
        shared, topv, topi, probs = self._shared_and_router(x2)
        # zeros via a compute op (probs*0), not ttnn.zeros_like: a fill is illegal inside a captured
        # trace, whereas an eltwise multiply is a normal traced kernel (PCC-identical, also fine eager).
        with sp.region("moe.scatter"):
            routing = ttnn.scatter(ttnn.multiply(probs, 0.0), 1, topi, topv)  # [T, E]

        if T == 1:
            # --- sparse_matmul decode: compute only selected experts, NO host sync (traceable) ---
            with sp.region("moe.experts"):
                sparsity = ttnn.to_layout(ttnn.reshape(routing, [1, 1, 1, self.num_experts]), ttnn.ROW_MAJOR_LAYOUT)
                if self._decode_gather:
                    # gather path: pass the active expert ids -> kernels iterate top_k, not all E
                    indices = moe_gather.topk_to_indices(topi, self.top_k)
                    routed = ttnn.reshape(self.forward_sparse_decode(x2, sparsity, topv, indices), [1, hidden])
                else:
                    routed = ttnn.reshape(self.forward_sparse_decode(x2, sparsity), [1, hidden])
            with sp.region("moe.combine"):
                return ttnn.reshape(ttnn.add(routed, shared), [1, 1, T, hidden])

        if T % 32 == 0 and os.environ.get("QWEN36_SPARSE_PREFILL"):
            # --- opt-in sparse per-tile prefill: each 32-token tile computes only experts it touches.
            # Off by default: PCC-identical to dense but slower at this expert size (dispatch-bound). ---
            routed = self.forward_sparse_prefill(x2, routing)  # [T, hidden]
            return ttnn.reshape(ttnn.add(routed, shared), [1, 1, T, hidden])

        # --- dense prefill (DEFAULT): all experts via batched matmul on the sparse-layout weights ---
        # Chunk over T (<= _DENSE_TMAX): the tuned program config gives each core an expert's full
        # [tc, N] output, which sits in L1 (per_core_M = tc/32). The down-proj (N=hidden) overflows L1
        # past tc~320, so cap tc at 256 (per_core_M<=8, proven to fit) and concat — same fast config
        # per chunk, L1-bounded for any T. One chunk (no concat) when T <= _DENSE_TMAX.
        E, H, I = self.num_experts, self.hidden, self.inter
        gate_up_w = ttnn.reshape(self.gate_up_sp, [E, H, 2 * I])
        down_w = ttnn.reshape(self.down_sp, [E, I, H])
        outs = []
        for t0 in range(0, T, _DENSE_TMAX):
            t1 = min(t0 + _DENSE_TMAX, T)
            tc = t1 - t0
            with prof.phase(self.mesh_device, "moe.repeat"), sp.region("moe.repeat"):
                xe = ttnn.repeat(
                    ttnn.reshape(ttnn.slice(x2, [t0, 0], [t1, hidden]), [1, tc, hidden]), ttnn.Shape([E, 1, 1])
                )  # [E, tc, hidden]
            with prof.phase(self.mesh_device, "moe.gate_up_mm"), sp.region("moe.gate_up_mm"):
                hgu = ttnn.matmul(
                    xe,
                    gate_up_w,
                    program_config=self._dense_expert_pc(tc, 2 * I),
                    compute_kernel_config=self.compute_kernel_config,
                )  # [E, tc, 2*inter]
            with prof.phase(self.mesh_device, "moe.swiglu"), sp.region("moe.swiglu"):
                gate = ttnn.slice(hgu, [0, 0, 0], [E, tc, I])
                up = ttnn.slice(hgu, [0, 0, I], [E, tc, 2 * I])
                h = ttnn.multiply(ttnn.silu(gate), up)  # [E, tc, inter]
                # Fold the routing weight into the down-projection INPUT. down is a linear matmul (no
                # bias), so sum_e w_e·down(h_e) == sum_e down(w_e·h_e); scaling the [E,tc,I] input is
                # ~4x less data than scaling the [E,tc,H] output and lets the expert-reduce be a sum.
                wc = ttnn.reshape(ttnn.transpose(ttnn.slice(routing, [t0, 0], [t1, E]), 0, 1), [E, tc, 1])
                h = ttnn.multiply(h, wc)
            with prof.phase(self.mesh_device, "moe.down_mm"), sp.region("moe.down_mm"):
                ye = ttnn.matmul(
                    h,
                    down_w,
                    program_config=self._dense_expert_pc(tc, H),
                    compute_kernel_config=self.compute_kernel_config,
                )  # [E, tc, hidden]
            with prof.phase(self.mesh_device, "moe.reduce"), sp.region("moe.reduce"):
                outs.append(ttnn.sum(ye, dim=0))  # [tc, hidden]
        routed = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)  # [T, hidden]
        with sp.region("moe.combine"):
            return ttnn.reshape(ttnn.add(routed, shared), [1, 1, T, hidden])
