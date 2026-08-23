# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Model + device configuration for Qwen3.6-35B-A3B on Blackhole.

Wraps the architecture config (parsed from the HF ``config.json`` via the standalone reference
``Qwen35MoeConfig``) and adds tt-nn / device-specific settings. Default target is a single
Blackhole P150 (32 GB) which forces BFP4 expert/MLP weights (the 35B model is ~17.5 GB in BFP4
vs ~35 GB in BFP8).
"""
from __future__ import annotations

import os

import ttnn
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig


class ModelArgs:
    """Holds architecture dims (from HF config) plus tt-nn device/precision config."""

    # Default context. Only the 10 full_attention layers hold a KV cache (the 30 linear_attention
    # layers carry a fixed-size recurrent state), so KV costs 10 * 2(K,V) * n_kv_heads(2) *
    # head_dim(256) * 2 B = 20 KiB/token -- a quarter of the 80 KiB/token a dense 40-layer model of
    # these dims would need. 32K context is therefore ~0.64 GB of KV against ~17.5 GB of BFP4
    # weights; even the checkpoint's native 262144 is only ~5.0 GB. See README ("Memory").
    DEFAULT_MAX_SEQ_LEN = 32768

    def __init__(
        self,
        mesh_device,
        ckpt_dir: str | None = None,
        max_batch_size: int = 1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.ckpt_dir = ckpt_dir or os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        self.config = Qwen35MoeConfig.from_hf_config(self.ckpt_dir)

        # On-disk cache of converted (quantized + tilized) weights. The first build writes one
        # .tensorbin per weight; later runs load them directly and skip the block-float quantization
        # (the bulk of cold-load time). Root defaults to $TT_CACHE_PATH or <ckpt>/tt_weight_cache.
        # QWEN36_WEIGHT_CACHE=0 disables it (always convert from the HF checkpoint).
        if os.environ.get("QWEN36_WEIGHT_CACHE") == "0":
            self.weight_cache_path = None
        else:
            root = os.environ.get("TT_CACHE_PATH") or os.path.join(self.ckpt_dir, "tt_weight_cache")
            self.weight_cache_path = root

        # --- convenience aliases (architecture) ---
        c = self.config
        self.dim = c.hidden_size
        self.n_layers = c.num_hidden_layers
        self.n_heads = c.num_attention_heads
        self.n_kv_heads = c.num_key_value_heads
        self.head_dim = c.head_dim
        self.vocab_size = c.vocab_size
        self.norm_eps = c.rms_norm_eps
        self.rope_theta = c.rope_theta
        self.rotary_dim = int(c.head_dim * c.partial_rotary_factor)  # 64
        self.layer_types = c.layer_types
        # Fidelity guards for the two architecture facts the tt-nn modules hardcode rather than
        # read: the per-head attention output gate (tt/attention.py splits q_proj into query and
        # gate halves and always applies sigmoid(gate)), and the absence of attention bias. Both
        # hold for this checkpoint; a variant that flipped either would produce silently wrong
        # output, so fail loudly at build time instead of generating garbage.
        assert c.attn_output_gate, "attn_output_gate=False is not implemented (tt/attention.py always gates)"
        assert not c.attention_bias, "attention_bias=True is not implemented (q/k/v/o are bias-free)"

        # linear-attention (Gated DeltaNet) dims
        self.lin_num_k_heads = c.linear_num_key_heads
        self.lin_num_v_heads = c.linear_num_value_heads
        self.lin_head_k_dim = c.linear_key_head_dim
        self.lin_head_v_dim = c.linear_value_head_dim
        self.lin_key_dim = c.linear_key_head_dim * c.linear_num_key_heads
        self.lin_value_dim = c.linear_value_head_dim * c.linear_num_value_heads
        self.lin_conv_dim = self.lin_key_dim * 2 + self.lin_value_dim
        self.conv_kernel_size = c.linear_conv_kernel_dim

        # MoE dims
        self.num_experts = c.num_experts
        self.num_experts_per_tok = c.num_experts_per_tok
        self.moe_intermediate_size = c.moe_intermediate_size
        self.shared_expert_intermediate_size = c.shared_expert_intermediate_size

        # MTP (multi-token prediction) head — one full-attention+MoE decoder layer shipped under the
        # checkpoint's `mtp.*` prefix, sharing embed_tokens and lm_head with the backbone. Only used
        # when speculative decode is enabled; the backbone ignores it. See MTP.md.
        self.mtp_num_hidden_layers = c.mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = c.mtp_use_dedicated_embeddings

        # --- device config ---
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        self.is_blackhole = mesh_device is not None and ttnn.device.is_blackhole(mesh_device)

        # --- precision (single-card memory budget) ---
        # Routed experts dominate the parameter count -> BFP4. Attention/linear-attn projections
        # and norms stay higher precision for accuracy.
        # QWEN36_EXPERT_DTYPE={bf4,bf8} overrides the routed-expert + shared-MLP precision. Default
        # bf4 (the 35B model is ~17.5 GB in BFP4 vs ~35 GB in BFP8, and only BFP4 fits a 32 GB P150).
        # bf8 is used by the accuracy sweep in evaluation/ to measure the BFP4 accuracy cost (it needs
        # more DRAM, so pair it with fewer QWEN36_LAYERS if it OOMs on a single card).
        _expert_dtype = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}.get(
            os.environ.get("QWEN36_EXPERT_DTYPE", "bf4").lower(), ttnn.bfloat4_b
        )
        self.expert_weight_dtype = _expert_dtype
        self.mlp_weight_dtype = _expert_dtype
        # Mixed-precision recipe (AesSedai/Ubergarm): pin the more-sensitive routed-expert down_proj
        # to bf8 while gate_up stays bf4. Costs ~+5GB (still fits a 32GB P150), unlike uniform bf8
        # experts (~+16GB -> OOM). QWEN36_EXPERT_DOWN_BF8=1 enables; default = same as expert dtype.
        self.expert_down_weight_dtype = (
            ttnn.bfloat8_b if os.environ.get("QWEN36_EXPERT_DOWN_BF8") == "1" else self.expert_weight_dtype
        )
        # --- per-block weight precision, all env-overridable ---
        # `ROOFLINE.md` §2.1 is what makes these worth touching: the gated-delta projections run at
        # 74-87% of DRAM peak, i.e. they are NEAR-ROOFLINE, so their time is directly proportional to
        # their byte count and halving the dtype nearly halves them. That is the opposite of the
        # "decode is overhead-bound so precision is a small lever" reading in FUTURE_OPTIMIZATIONS
        # Lever 4, which averaged over a step whose worst components sit at 3-6% of peak.
        # Per-token weight bytes at the defaults (of a 2.60 GB total):
        #   gated-delta projections  1075 MB   bf8 -> bf4 saves ~504 MB
        #   attention projections     290 MB   bf8 -> bf4 saves ~136 MB
        #   MoE router + shared       294 MB   bf16 -> bf8 saves ~138 MB, -> bf4 saves ~211 MB
        # Gate any change on BOTH instruments, because MMLU alone does not cover the T==1 paths:
        #   evaluation/run_mmlu_bench.py            (paired, prefill-scored)
        #   tests/probe_teacher_forced_drift.py     (decode logit PCC, the T==1 half)
        _DT = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}

        def _dt(var, default):
            v = os.environ.get(var, default).lower()
            assert v in _DT, f"{var}={v!r} must be one of {sorted(_DT)}"
            return _DT[v]

        # MEASURED 2026-08-22 (bench_decode 40L + paired MMLU-Redux n=200):
        #   gdn  bf8 -> bf4   -0.45 ms/token   (0.316 -> 0.301 ms/layer x 30)
        #   attn bf8 -> bf4   -0.09 ms/token   (0.254 -> 0.245 ms/layer x 10)
        #   together: 25.49 -> 24.88 ms/token, 39.2 -> 40.2 tok/s/user (+2.6%), and ~0.68 GB less
        #   device DRAM for the weights.
        # Accuracy: 79.0% -> 80.5% on the same 200 questions (nominally UP; paired McNemar p=0.508, so
        # no detectable change either way). That run was the MORE aggressive all-bf4 config -- it also
        # had the shared expert at bf4 -- so it bounds the shipped config from above. Caveat worth
        # keeping: 15 of 200 predictions changed, 5x the churn of the program-config work, so this is a
        # materially bigger numerical perturbation even though the score is indistinguishable.
        # NOT SHIPPED -- default stays bf8. bf4 breaks a correctness gate for +2.6%:
        # `test_long_prefill.py::test_long_prefill_chunk_invariant` (chunked prefill == single-shot)
        # FAILS at bf4 and PASSES at bf8, isolated by flipping only these two vars. MMLU-Redux was
        # fine (79.0% -> 80.5%, paired McNemar p=0.508) and so was the decode side, which is exactly
        # why the module gates matter: a 200-question eval cannot see a chunk-boundary invariant.
        # Opt in with QWEN36_GDN_DTYPE=bf4 QWEN36_ATTN_DTYPE=bf4 if you want the 0.54 ms and the
        # ~0.68 GB of device DRAM and can live with that gate red.
        self.attn_weight_dtype = _dt("QWEN36_ATTN_DTYPE", "bf8")
        self.linear_attn_weight_dtype = _dt("QWEN36_GDN_DTYPE", "bf8")
        # The MoE router + shared expert. These are bf16 today NOT by design but because
        # `mlp_weight_dtype` above is assigned and never read -- decoder.py passes
        # `dtype=args.activation_dtype` to TtMoE, which is what sets the router/shared/gate precision.
        # STAYS bf16, on measurement: bf16 -> bf8 here removes 138 MB/token of weight traffic and buys
        # -0.04 ms, i.e. nothing. `moe.shared` runs at 21% of DRAM peak and `moe.router` at 3%, so bytes
        # are not their binding constraint -- exactly the test that "it is 12% of the weight budget"
        # fails to apply. Same reasoning as dropping lm_head from the wide-1D default set: no gain, so
        # do not spend the numerical perturbation. Selectable for anyone chasing DRAM footprint rather
        # than latency (bf8 saves ~138 MB/token of reads and ~0.3 GB of resident weights).
        self.moe_shared_weight_dtype = _dt("QWEN36_SHARED_DTYPE", "bf16")
        # KV cache precision. bf16 today; `tt_transformers` parameterises this and its sglang path
        # DEFAULTS to bfloat8_b (generator_sglang.py:34), so bf16 makes us the outlier.
        # MEASURED speed (tests/probe_paged_kv_perf.py, sdpa-decode per attention layer):
        #     pos      bf16      bf8    speedup
        #     8192    83.6us   62.5us    1.34x
        #     32768  228.3    137.3      1.66x
        #     131072 796.7    438.0      1.82x
        # i.e. -0.21 / -0.91 / -3.6 ms per TOKEN across the 10 attention layers, tracking the byte
        # ratio -- so it is bandwidth, and it grows with context. It also halves the cache (5.23 GB ->
        # 2.78 GB at max_seq=262144).
        # NOTE ON GATING IT: MMLU-Redux cannot see this. Single-shot prefill runs SDPA on the live q/k/v
        # and only FILLS the cache, so a prefill-scored eval never READS it. The gate is
        # tests/probe_kv_dtype_accuracy.py (teacher-forced decode, which does read it).
        self.kv_cache_dtype = _dt("QWEN36_KV_DTYPE", "bf16")
        self.activation_dtype = ttnn.bfloat16
        # Decode MoE: dense (no host sync, traceable, measured faster at batch=1) is the default.
        # gather-top-k sparse path (a host readback per call) is opt-in via QWEN36_SPARSE_DECODE=1.
        self.sparse_moe_decode = os.environ.get("QWEN36_SPARSE_DECODE", "0") == "1"

        self.compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def is_linear_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    def reference_config(self) -> Qwen35MoeConfig:
        return self.config

    @property
    def model_name(self) -> str:
        return "Qwen3.6-35B-A3B"
