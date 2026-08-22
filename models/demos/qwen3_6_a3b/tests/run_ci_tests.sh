#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# CI entrypoint for the qwen3_6_a3b module PCC gates. Runs on a SINGLE Blackhole and needs NO
# checkpoint: these are the checkpoint-free module tests (random / per-module weights). The
# checkpoint-required tests (test_model/test_trace/test_long_prefill/test_ragged_prefill) and the
# reference-venv cross-validation (test_reference_smoke) are intentionally excluded — see
# pcc_thresholds.json. test_cache_topology reads only the checkpoint's config.json (no weights) and
# skips cleanly if it is absent.
#
# Wire this into tt-metal CI by adding one entry that invokes this script to the Blackhole
# single-card unit-test matrix (tests/pipeline_reorg/*.yaml, consumed by
# .github/scripts/utils/prepare_test_matrix.py) — the same mechanism TT uses for
# models/demos/blackhole/qwen36. Locally: `bash models/demos/qwen3_6_a3b/tests/run_ci_tests.sh`.
set -euo pipefail

PYTHON="${PYTHON:-./python_env/bin/python}"
TESTDIR="models/demos/qwen3_6_a3b/tests"

exec "$PYTHON" -m pytest -p no:cacheprovider -q \
  "$TESTDIR/test_cache_topology.py" \
  "$TESTDIR/test_norms.py" \
  "$TESTDIR/test_attention.py" \
  "$TESTDIR/test_attention_decode.py" \
  "$TESTDIR/test_gated_delta.py" \
  "$TESTDIR/test_gated_delta_decode.py" \
  "$TESTDIR/test_moe.py" \
  "$TESTDIR/test_moe_sparse.py" \
  "$TESTDIR/test_tool_parsing.py" \
  "$@"
