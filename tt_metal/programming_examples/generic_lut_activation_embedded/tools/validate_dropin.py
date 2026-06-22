#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Drop-in Replacement Validator — TTNN integration smoke test.

Validates that a modified ckernel_sfpu_<activation>.h works correctly
through the full TTNN pipeline. Uses exhaustive BF16 + dense FP32 inputs.

For authoritative device-profiler timing and ULP metrics, use run_csv.sh
which runs the LUT kernel directly with Tracy profiler instrumentation.

This script complements run_csv.sh by verifying the TTNN end-to-end path:
  - Kernel compilation through TTNN dispatch
  - Parameter handling (beta, threshold, etc.)
  - Dtype selection (#ifdef INP_FLOAT32)
  - Tile padding and output correctness

Usage:
  python3 validate_dropin.py softplus
  python3 validate_dropin.py softplus --precision bf16
  python3 validate_dropin.py gelu sigmoid softplus
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def exhaustive_bf16_inputs(lo=-10.0, hi=10.0):
    all_bits = np.arange(0, 65536, dtype=np.uint16)
    f32_bits = all_bits.astype(np.uint32) << 16
    all_vals = np.frombuffer(f32_bits.tobytes(), dtype=np.float32)
    mask = np.isfinite(all_vals) & (all_vals >= lo) & (all_vals <= hi)
    return torch.from_numpy(np.sort(all_vals[mask])).bfloat16()


def compute_ulps(hw, ref, is_bf16, inputs=None):
    """Per-point canonical Goldberg ULP error (ttpoly.spec.units).

    Delegates to the SINGLE ULP owner so this validator reports the SAME robust
    metric as the device sweep / fitter (FTZ before spacing, subnormal golden &
    inputs masked to NaN). Replaces the old bit-distance reimplementation, which
    blew up near roots and is the forbidden "raw" alternative. Returns a float
    array (NaN where masked); aggregate with nanmax / nanmean / nanpercentile.
    """
    import os as _os

    _fit_dir = _os.environ.get("TT_POLY_FIT_DIR", "/localdev/nkapre/tt-polynomial-fitter")
    if _fit_dir not in sys.path:
        sys.path.insert(0, _fit_dir)
    from ttpoly.spec import units as _units

    precision = "bf16" if is_bf16 else "fp32"
    return _units.ulp_error(
        np.asarray(ref, dtype=np.float64),
        np.asarray(hw, dtype=np.float64),
        precision=precision,
        inputs=None if inputs is None else np.asarray(inputs, dtype=np.float64),
        flush_to_zero=True,
    )


def load_activation_configs():
    """Load activation ranges from sweep_config.dat."""
    configs = {}
    for search_dir in [
        Path(__file__).parent.parent,
        Path(__file__).parent.parent.parent / "generic_lut_activation_embedded",
    ]:
        config_path = search_dir / "sweep_config.dat"
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if line.startswith("ACTIVATION_RANGES="):
                        for pair in line.strip().split("=", 1)[1].split(","):
                            parts = pair.split(":")
                            if len(parts) == 3:
                                configs[parts[0]] = {"lo": float(parts[1]), "hi": float(parts[2])}
                        break
            if configs:
                break
    # Fallback for common activations
    if not configs:
        configs = {
            "softplus": {"lo": -10, "hi": 10},
            "gelu": {"lo": -10, "hi": 10},
            "sigmoid": {"lo": -10, "hi": 10},
            "hardmish": {"lo": -10, "hi": 10},
        }
    return configs


ACTIVATIONS = load_activation_configs()


def get_ttnn_fn(name):
    import ttnn

    fn = getattr(ttnn, name, None)
    if fn is None:
        raise ValueError(f"ttnn.{name} not found")
    return fn


# Ground truth from tt-polynomial-fitter
_gt_module = None


def get_torch_fn(name):
    global _gt_module
    # Try to use ground_truth module from tt-polynomial-fitter (canonical reference)
    if _gt_module is None:
        try:
            import importlib.util, os

            gt_dir = os.environ.get("TT_POLY_FIT_DIR", "/localdev/nkapre/tt-polynomial-fitter")
            gt_path = os.path.join(gt_dir, "ground_truth.py")
            if os.path.exists(gt_path):
                spec = importlib.util.spec_from_file_location("ground_truth", gt_path)
                _gt_module_local = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_gt_module_local)
                _gt_module = _gt_module_local
        except Exception:
            pass

    if _gt_module and hasattr(_gt_module, "compute_ground_truth"):

        def fn(x):
            result = _gt_module.compute_ground_truth(name, x.numpy())
            return torch.from_numpy(np.asarray(result, dtype=np.float64))

        return fn

    # Fallback to common torch functions
    fns = {
        "softplus": lambda x: torch.nn.functional.softplus(x),
        "gelu": lambda x: torch.nn.functional.gelu(x),
        "sigmoid": lambda x: torch.sigmoid(x),
        "tanh": lambda x: torch.tanh(x),
        "exp": lambda x: torch.exp(x),
        "silu": lambda x: torch.nn.functional.silu(x),
        "elu": lambda x: torch.nn.functional.elu(x),
        "erf": lambda x: torch.erf(x),
        "sin": lambda x: torch.sin(x),
        "cos": lambda x: torch.cos(x),
        "hardmish": lambda x: x * torch.clamp(x + 2.8, 0.0, 5.0) / 5.0,
    }
    if name in fns:
        return fns[name]
    raise ValueError(f"No torch reference for {name}. Add to ground_truth.py or validate_dropin.py")


def validate(act_name, device, precisions, timing_iters=20):
    import ttnn

    cfg = ACTIVATIONS[act_name]
    lo, hi = cfg["lo"], cfg["hi"]
    ttnn_fn = get_ttnn_fn(act_name)
    torch_fn = get_torch_fn(act_name)
    results = {}

    for prec in precisions:
        is_bf16 = prec == "bf16"
        dtype_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32

        # Inputs
        if is_bf16:
            x_cpu = exhaustive_bf16_inputs(lo, hi)
        else:
            x_cpu = torch.linspace(lo, hi, 65536, dtype=torch.float32)
        n = len(x_cpu)

        # Pad to tile boundary
        pad_to = ((n + 1023) // 1024) * 1024
        x_padded = torch.zeros(pad_to, dtype=x_cpu.dtype)
        x_padded[:n] = x_cpu

        # Reference
        y_ref = torch_fn(x_cpu.double()).numpy().astype(np.float32)

        # Hardware
        x_tt = ttnn.from_torch(x_padded.reshape(1, 1, 1, -1), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype_tt)
        y_tt = ttnn_fn(x_tt)
        y_hw = ttnn.to_torch(y_tt).squeeze().float().numpy()[:n]

        # Metrics — canonical Goldberg ULP (NaN where masked); aggregate with nan*
        x_in = x_cpu.float().numpy()
        ulps = compute_ulps(y_hw, y_ref, is_bf16, inputs=x_in)
        abs_err = np.abs(y_hw - y_ref)

        # Timing (256 tiles)
        x_t = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16 if is_bf16 else torch.float32)
        x_tt_t = ttnn.from_torch(x_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype_tt)
        for _ in range(5):
            ttnn_fn(x_tt_t)
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(timing_iters):
            ttnn_fn(x_tt_t)
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - start) / timing_iters * 1e6

        results[prec] = {
            "n": n,
            "mae": float(np.mean(abs_err)),
            "max_err": float(np.max(abs_err)),
            "max_ulp": float(np.nanmax(ulps)) if np.any(np.isfinite(ulps)) else float("nan"),
            "mean_ulp": float(np.nanmean(ulps)) if np.any(np.isfinite(ulps)) else float("nan"),
            "p99_ulp": float(np.nanpercentile(ulps, 99)) if np.any(np.isfinite(ulps)) else float("nan"),
            "host_us": us,
        }

    return results


def print_results(name, results):
    print(f"\n{'='*72}")
    print(f"  {name.upper()} — TTNN Drop-in Validation (exhaustive BF16 / dense FP32)")
    print(f"{'='*72}\n")

    hdr = f"  {'Prec':<6} {'Pts':>6} {'MAE':>10} {'MaxErr':>10} {'MaxULP':>8} {'MeanULP':>8} {'P99':>6} {'Host us':>8}"
    print(hdr)
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for prec, r in results.items():
        print(
            f"  {prec.upper():<6} {r['n']:>6} {r['mae']:>10.2e} {r['max_err']:>10.2e} "
            f"{r['max_ulp']:>8.2f} {r['mean_ulp']:>8.2f} {r['p99_ulp']:>6.2f} {r['host_us']:>7.1f}"
        )

    print(f"\n  PR Markdown:")
    print(f"  | Precision | Points | MAE | MaxErr | MaxULP | MeanULP | P99 ULP | Host (us) |")
    print(f"  |-----------|--------|-----|--------|--------|---------|---------|-----------|")
    for prec, r in results.items():
        print(
            f"  | {prec.upper()} | {r['n']} | {r['mae']:.2e} | {r['max_err']:.2e} | "
            f"{r['max_ulp']:.2f} | {r['mean_ulp']:.2f} | {r['p99_ulp']:.2f} | {r['host_us']:.1f} |"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="TTNN drop-in replacement smoke test")
    parser.add_argument("activations", nargs="+")
    parser.add_argument("--precision", choices=["bf16", "fp32", "both"], default="both")
    args = parser.parse_args()

    import ttnn

    precs = ("bf16", "fp32") if args.precision == "both" else (args.precision,)
    device = ttnn.open_device(device_id=0)

    for act in args.activations:
        results = validate(act, device, precs)
        print_results(act, results)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
