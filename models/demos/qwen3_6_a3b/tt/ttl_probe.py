# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Minimal tt-lang (ttl) fused-kernel probe: y = silu(a) * b, fused in one custom op.

Confirms tt-lang compiles and runs on this Blackhole, and demonstrates the authoring pattern
(@ttl.operation / @ttl.datamovement / @ttl.compute / dataflow buffers) used by a future fused
chunked Gated-DeltaNet kernel.
"""
import torch
import ttl

import ttnn

TILE = 32


@ttl.operation(grid=(1, 1))
def _fused_swiglu(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
    m_tiles = a.shape[0] // TILE
    n_tiles = a.shape[1] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for m in range(m_tiles):
            for n in range(n_tiles):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    ta = ttl.copy(a[m, n], a_blk)
                    tb = ttl.copy(b[m, n], b_blk)
                    ta.wait()
                    tb.wait()

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(n_tiles):
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(ttl.math.silu(a_blk) * b_blk)

    @ttl.datamovement()
    def write():
        for m in range(m_tiles):
            for n in range(n_tiles):
                with y_dfb.wait() as y_blk:
                    ttl.copy(y_blk, y[m, n]).wait()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    try:
        M, N = 64, 128
        a = torch.randn(M, N, dtype=torch.bfloat16)
        b = torch.randn(M, N, dtype=torch.bfloat16)
        expected = (torch.nn.functional.silu(a.float()) * b.float()).to(torch.bfloat16)

        def up(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        at, bt = up(a), up(b)
        yt = up(torch.zeros(M, N, dtype=torch.bfloat16))
        _fused_swiglu(at, bt, yt)
        y = ttnn.to_torch(yt)
        pcc = torch.corrcoef(torch.stack([y.flatten().float(), expected.flatten().float()]))[0, 1].item()
        print(f"[ttl_probe] fused silu(a)*b PCC {pcc:.5f}")
        assert pcc > 0.99, pcc
        print("[ttl_probe] tt-lang fused kernel compiles + runs on Blackhole ✓")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
