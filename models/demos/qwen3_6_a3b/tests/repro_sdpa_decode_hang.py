#!/usr/bin/env python
"""Minimal scaled_dot_product_attention_decode repro for the Qwen3.6 decode hang.

ROOT CAUSE: stale Blackhole firmware. On FW bundle 18.8.0 (eth-fw 1.4.2) the flash
scaled_dot_product_attention_decode op DEADLOCKS for this attention shape -- two Tensix cores stuck
in a circular-buffer wait (watcher-confirmed). Upgrading to >=19.5.0 (we used 19.6.0, eth-fw >=1.8.1)
fixes it; the stock op then decodes normally. Keep this as a regression check: it hangs on old FW and
passes on new FW, with no model dependency.

Usage: repro_sdpa_decode_hang.py <head_dim> <config> <cur_pos>
Faithful to the model: KV cache bfloat16, [1, n_kv=2, max_seq=512, head_dim], q [1,1,nh=16,hd].
Prints REPRO_PASS or raises/HANGs. Run under `timeout` since it hangs on affected firmware.
"""
import sys

import torch

import ttnn

head_dim = int(sys.argv[1])
cfg_name = sys.argv[2]
pos = int(sys.argv[3]) if len(sys.argv) > 3 else 5
nh, nkv, max_seq = 16, 2, 512

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    q_t = torch.randn(1, 1, nh, head_dim) * 0.1
    k_t = torch.randn(1, nkv, max_seq, head_dim) * 0.1
    v_t = torch.randn(1, nkv, max_seq, head_dim) * 0.1
    q = ttnn.from_torch(q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    kc = ttnn.from_torch(k_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # bf16 like model
    vc = ttnn.from_torch(v_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cur = ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    pcs = {
        "default": None,
        "model": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=128
        ),
        "kchunk64": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=64
        ),
    }
    pc = pcs[cfg_name]
    kwargs = dict(cur_pos_tensor=cur, scale=head_dim**-0.5)
    if pc is not None:
        kwargs["program_config"] = pc
    print(f"[repro] hd={head_dim} cfg={cfg_name} pos={pos}: calling sdpa_decode...", flush=True)
    out = ttnn.transformer.scaled_dot_product_attention_decode(q, kc, vc, **kwargs)
    o = ttnn.to_torch(out)
    print(
        f"REPRO_PASS hd={head_dim} cfg={cfg_name} pos={pos} out_shape={tuple(o.shape)} mean={o.float().mean():.4f}",
        flush=True,
    )
finally:
    ttnn.close_device(dev)
