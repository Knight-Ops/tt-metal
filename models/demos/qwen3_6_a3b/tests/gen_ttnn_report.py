# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate a ttnn op/memory report (db.sqlite) for ttnn-visualizer's --profiler-path.

Runs EAGER decode (the op-graph + per-op buffer allocations are captured at dispatch, so this must be
eager, not traced). enable_logging must be ON at load (it initializes the report path/db), but the
graph + detailed-buffer reports are toggled on ONLY around the decode step — leaving them on during
the model build snapshots the weight tensors and fails (ndarray_import). Run:

    TTNN_CONFIG_OVERRIDES='{"enable_logging": true, "report_name": "qwen36_decode",
                            "enable_fast_runtime_mode": false}' \
    QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/gen_ttnn_report.py

Output: generated/ttnn/reports/<id>/db.sqlite (+ config.json). Point ttnn-visualizer at that folder.
"""
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


@torch.no_grad()
def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=128)
        loader = CheckpointLoader(ckpt)
        model = TtModel(mesh, args, loader, num_layers=n_layers)

        torch.manual_seed(0)
        prompt = torch.randint(0, args.vocab_size, (1, 8))
        logits = model.forward(prompt)  # prefill (logging off)
        nxt = int(logits[0, -1].argmax())
        model.start_decode(nxt)
        model.decode_step_eager()  # warmup (reports off)
        ttnn.synchronize_device(mesh)
        # enable the graph + detailed-buffer reports ONLY for this decode step (on during build -> crash)
        with ttnn.manage_config("enable_graph_report", True), ttnn.manage_config("enable_detailed_buffer_report", True):
            model.decode_step_eager()  # ONE decode step: op graph + buffers -> db.sqlite
            ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
