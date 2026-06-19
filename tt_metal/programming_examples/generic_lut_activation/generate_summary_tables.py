#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate summary statistics tables from polynomial sweep CSV results.

Used by sweep scripts to show aggregated performance metrics.
"""

import csv
import sys
import argparse
import shutil
from collections import defaultdict


def _pivot_rows_to_config(rows):
    """Pivot row-per-shape CSV rows into one record per config with per-shape timing.

    Input: list of dicts with 'shape', 'time_profiler_us', 'mae', 'max_error', 'max_ulp', ...
    Output: list of dicts with 'single', '8tiles', '256tiles', 'height', 'yolov4' timing keys
            and 'mae', 'max_error', 'max_ulp' from single_tile shape.
    """
    SHAPE_TO_KEY = {
        'single_tile': 'single',
        '8_tiles': '8tiles',
        '256_tiles': '256tiles',
        'height_sharded': 'height',
        'yolov4': 'yolov4',
    }

    # Group by config key (everything except shape and metrics)
    configs = defaultdict(dict)
    for row in rows:
        # Build config key from non-shape, non-metric columns
        shape = row.get('shape', '')
        key_parts = []
        for col in ('activation', 'precision', 'depth', 'degree', 'segmentation',
                     'num_degree', 'den_degree', 'segments'):
            if col in row:
                key_parts.append(row[col])
        config_key = tuple(key_parts)
        configs[config_key][shape] = row

    result = []
    for config_key, shapes in configs.items():
        # Use single_tile for error metrics (representative)
        base = shapes.get('single_tile', next(iter(shapes.values())))
        entry = {k: v for k, v in base.items() if k not in ('shape', 'time_profiler_us', 'time_host_ms')}

        # Populate per-shape timing
        for shape_name, timing_key in SHAPE_TO_KEY.items():
            if shape_name in shapes:
                entry[timing_key] = float(shapes[shape_name].get('time_profiler_us', 0)) / 1000.0
            else:
                entry[timing_key] = 0.0

        result.append(entry)
    return result


def generate_polynomial_tables(results_file):
    """Generate summary tables for polynomial approximations."""
    # Group results by degree
    by_degree = defaultdict(list)
    by_segmentation = defaultdict(list)
    by_depth = defaultdict(list)

    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        raw_rows = [row for row in reader if row["status"] == "pass"]

    # Detect format: row-per-shape has 'shape' column
    if raw_rows and 'shape' in raw_rows[0]:
        pivoted = _pivot_rows_to_config(raw_rows)
    else:
        # Legacy wide format — convert to same structure
        pivoted = []
        for row in raw_rows:
            entry = dict(row)
            entry['single'] = float(row.get("time_profiler_single_tile_us", 0)) / 1000.0
            entry['8tiles'] = float(row.get("time_profiler_8_tiles_us", 0)) / 1000.0
            entry['256tiles'] = float(row.get("time_profiler_256_tiles_us", 0)) / 1000.0
            entry['height'] = float(row.get("time_profiler_height_sharded_us", 0)) / 1000.0
            entry['yolov4'] = float(row.get("time_profiler_yolov4_us", 0)) / 1000.0
            pivoted.append(entry)

    for entry in pivoted:
        # Handle both polynomial degrees (int) and rational degrees (e.g., "14/13")
        degree_str = entry["degree"]
        if "/" in degree_str:
            degree = f"r{degree_str}"
        else:
            degree = int(degree_str)

        segmentation = entry["segmentation"]
        depth = int(entry["depth"])
        mae = float(entry.get("mae", entry.get("mae_single_tile", 0)))
        max_error = float(entry.get("max_error", entry.get("max_error_single_tile", 0)))
        max_ulp_raw = entry.get("max_ulp", entry.get("max_ulp_error_single_tile", "0"))
        max_ulp = float(max_ulp_raw) if max_ulp_raw else 0.0

        timing = {
            "single": float(entry.get("single", 0)),
            "8tiles": float(entry.get("8tiles", 0)),
            "256tiles": float(entry.get("256tiles", 0)),
            "height": float(entry.get("height", 0)),
            "yolov4": float(entry.get("yolov4", 0)),
        }

        by_degree[degree].append({"mae": mae, "max_error": max_error, "max_ulp": max_ulp, **timing})
        by_segmentation[segmentation].append({"mae": mae, "max_error": max_error, "max_ulp": max_ulp, **timing})
        by_depth[depth].append({"mae": mae, "max_error": max_error, "max_ulp": max_ulp, **timing})

    # Print summary by degree
    print("By Degree:")
    print("┌────────────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Degree     │ Count │ Best MAE     │ Best MaxErr  │ Best ULP     │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
    print("├────────────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    degree_names = {1: "Linear", 2: "Quad", 3: "Cubic", 4: "Quartic", 6: "Hexic", 8: "Octic"}

    # Sort keys: integers first (numerically), then strings (lexicographically)
    int_keys = sorted([k for k in by_degree.keys() if isinstance(k, int)])
    str_keys = sorted([k for k in by_degree.keys() if isinstance(k, str)])

    for degree in int_keys + str_keys:
        data = by_degree[degree]
        count = len(data)
        best_mae = min(d["mae"] for d in data)
        best_max = min(d["max_error"] for d in data)
        best_ulp = min(d["max_ulp"] for d in data)
        best_t1 = min(d["single"] for d in data)
        best_t8 = min(d["8tiles"] for d in data)
        best_t256 = min(d["256tiles"] for d in data)
        best_ths = min(d["height"] for d in data)
        best_tyolo = min(d["yolov4"] for d in data)

        mae_str = f"{best_mae:.2e}" if best_mae < 1e-3 else f"{best_mae:.6f}"
        max_str = f"{best_max:.2e}" if best_max < 1e-3 else f"{best_max:.6f}"
        ulp_str = f"{best_ulp:.1f}"
        degree_display = degree_names.get(degree, str(degree))
        print(f"│ {degree_display:>10} │ {count:>5} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")
    print("└────────────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # Print summary by segmentation
    print("By Segmentation:")
    print("┌────────────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Type       │ Count │ Best MAE     │ Best MaxErr  │ Best ULP     │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
    print("├────────────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    for seg in sorted(by_segmentation.keys()):
        data = by_segmentation[seg]
        count = len(data)
        best_mae = min(d["mae"] for d in data)
        best_max = min(d["max_error"] for d in data)
        best_ulp = min(d["max_ulp"] for d in data)
        best_t1 = min(d["single"] for d in data)
        best_t8 = min(d["8tiles"] for d in data)
        best_t256 = min(d["256tiles"] for d in data)
        best_ths = min(d["height"] for d in data)
        best_tyolo = min(d["yolov4"] for d in data)

        mae_str = f"{best_mae:.2e}" if best_mae < 1e-3 else f"{best_mae:.6f}"
        max_str = f"{best_max:.2e}" if best_max < 1e-3 else f"{best_max:.6f}"
        ulp_str = f"{best_ulp:.1f}"
        print(f"│ {seg:>10} │ {count:>5} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")
    print("└────────────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # Print summary by depth
    print("By Depth:")
    print("┌───────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Depth │ Count │ Best MAE     │ Best MaxErr  │ Best ULP     │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
    print("├───────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    for depth in sorted(by_depth.keys()):
        data = by_depth[depth]
        count = len(data)
        best_mae = min(d["mae"] for d in data)
        best_max = min(d["max_error"] for d in data)
        best_ulp = min(d["max_ulp"] for d in data)
        best_t1 = min(d["single"] for d in data)
        best_t8 = min(d["8tiles"] for d in data)
        best_t256 = min(d["256tiles"] for d in data)
        best_ths = min(d["height"] for d in data)
        best_tyolo = min(d["yolov4"] for d in data)

        mae_str = f"{best_mae:.2e}" if best_mae < 1e-3 else f"{best_mae:.6f}"
        max_str = f"{best_max:.2e}" if best_max < 1e-3 else f"{best_max:.6f}"
        ulp_str = f"{best_ulp:.1f}"
        print(f"│ {depth:>5} │ {count:>5} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")
    print("└───────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()


def generate_rational_tables(results_file):
    """Generate summary tables for rational approximations."""
    # Group results by config type and activation
    by_config = defaultdict(list)
    by_num_degree = defaultdict(list)
    by_activation = defaultdict(list)

    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        raw_rows = [row for row in reader if row["status"] == "pass"]

    # Detect format: row-per-shape has 'shape' column
    if raw_rows and 'shape' in raw_rows[0]:
        pivoted = _pivot_rows_to_config(raw_rows)
    else:
        pivoted = []
        for row in raw_rows:
            entry = dict(row)
            entry['single'] = float(row.get("time_profiler_single_tile_us", 0)) / 1000.0
            entry['8tiles'] = float(row.get("time_profiler_8_tiles_us", 0)) / 1000.0
            entry['256tiles'] = float(row.get("time_profiler_256_tiles_us", 0)) / 1000.0
            entry['height'] = float(row.get("time_profiler_height_sharded_us", 0)) / 1000.0
            entry['yolov4'] = float(row.get("time_profiler_yolov4_us", 0)) / 1000.0
            pivoted.append(entry)

    for entry in pivoted:
        num_deg = int(entry["num_degree"])
        den_deg = int(entry["den_degree"])
        segs = int(entry["segments"])
        activation = entry["activation"]

        mae_raw = entry.get("mae", entry.get("mae_single_tile", "0"))
        mae = float(mae_raw) if mae_raw not in ("N/A", "nan", "", None) else float("inf")
        max_raw = entry.get("max_error", entry.get("max_error_single_tile", "0"))
        max_err = float(max_raw) if max_raw not in ("N/A", "nan", "", None) else float("inf")
        ulp_raw = entry.get("max_ulp", entry.get("max_ulp_error_single_tile", "0"))
        max_ulp = float(ulp_raw) if ulp_raw and ulp_raw not in ("N/A", "nan", "", None) else 0.0

        timing = {
            "single": float(entry.get("single", 0)),
            "8tiles": float(entry.get("8tiles", 0)),
            "256tiles": float(entry.get("256tiles", 0)),
            "height": float(entry.get("height", 0)),
            "yolov4": float(entry.get("yolov4", 0)),
        }

        config = f"r{num_deg}d{den_deg}_s{segs}"
        by_config[config].append({"mae": mae, "max_error": max_err, "max_ulp": max_ulp, **timing, "activation": activation})
        by_num_degree[num_deg].append({"mae": mae, "max_error": max_err, "max_ulp": max_ulp, **timing})
        by_activation[activation].append({"mae": mae, "max_error": max_err, "max_ulp": max_ulp, **timing, "config": config})

    # Print summary by config
    print("By Rational Config (across all activations):")
    print("┌──────────────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Config       │ Count │ Best MAE     │ Best MaxErr  │ Best ULP     │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
    print("├──────────────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    for config in sorted(
        by_config.keys(),
        key=lambda x: (int(x.split("d")[0][1:]), int(x.split("_")[0].split("d")[1]), int(x.split("s")[1])),
    ):
        data = by_config[config]
        count = len(data)
        best_mae = min(d["mae"] for d in data)
        best_max = min(d["max_error"] for d in data)
        best_ulp = min(d["max_ulp"] for d in data)
        best_t1 = min(d["single"] for d in data)
        best_t8 = min(d["8tiles"] for d in data)
        best_t256 = min(d["256tiles"] for d in data)
        best_ths = min(d["height"] for d in data)
        best_tyolo = min(d["yolov4"] for d in data)

        if best_mae == float("inf"):
            mae_str = "N/A"
        elif best_mae < 1e-3:
            mae_str = f"{best_mae:.2e}"
        else:
            mae_str = f"{best_mae:.6f}"

        if best_max == float("inf"):
            max_str = "N/A"
        elif best_max < 1e-3:
            max_str = f"{best_max:.2e}"
        else:
            max_str = f"{best_max:.6f}"

        ulp_str = f"{best_ulp:.1f}"

        print(f"│ {config:<12} │ {count:>5} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")
    print("└──────────────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # Print summary by numerator degree
    print("By Numerator Degree:")
    print("┌────────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Degree │ Count │ Best MAE     │ Best MaxErr  │ Best ULP     │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
    print("├────────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    for deg in sorted(by_num_degree.keys()):
        data = by_num_degree[deg]
        count = len(data)
        best_mae = min(d["mae"] for d in data)
        best_max = min(d["max_error"] for d in data)
        best_ulp = min(d["max_ulp"] for d in data)
        best_t1 = min(d["single"] for d in data)
        best_t8 = min(d["8tiles"] for d in data)
        best_t256 = min(d["256tiles"] for d in data)
        best_ths = min(d["height"] for d in data)
        best_tyolo = min(d["yolov4"] for d in data)

        if best_mae == float("inf"):
            mae_str = "N/A"
        elif best_mae < 1e-3:
            mae_str = f"{best_mae:.2e}"
        else:
            mae_str = f"{best_mae:.6f}"

        if best_max == float("inf"):
            max_str = "N/A"
        elif best_max < 1e-3:
            max_str = f"{best_max:.2e}"
        else:
            max_str = f"{best_max:.6f}"

        ulp_str = f"{best_ulp:.1f}"

        print(f"│ {deg:>6} │ {count:>5} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")
    print("└────────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # Print summary by activation (best config for each)
    if len(by_activation) > 1:
        print("Best Config per Activation (by optimization metric):")
        print("┌─────────────┬───────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
        print("│ Activation  │ Metric    │ Config       │ MAE          │ MaxErr       │ ULP          │ T(1tile)ms   │ T(8tiles)ms  │ T(256t)ms    │ T(HS)ms      │ T(YOLOv4)ms  │")
        print("├─────────────┼───────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
        for activation in sorted(by_activation.keys()):
            data = by_activation[activation]

            # Show all three metric optimizations
            metrics = [
                ("best_mae", lambda d: d["mae"]),
                ("best_max", lambda d: d["max_error"]),
                ("best_ulp", lambda d: d["max_ulp"])
            ]

            for i, (metric_name, key_func) in enumerate(metrics):
                best = min(data, key=key_func)
                best_mae = best["mae"]
                best_max = best["max_error"]
                best_ulp = best["max_ulp"]
                best_t1 = best["single"]
                best_t8 = best["8tiles"]
                best_t256 = best["256tiles"]
                best_ths = best["height"]
                best_tyolo = best["yolov4"]
                best_config = best["config"]

                mae_str = "N/A" if best_mae == float("inf") else (f"{best_mae:.2e}" if best_mae < 1e-3 else f"{best_mae:.6f}")
                max_str = "N/A" if best_max == float("inf") else (f"{best_max:.2e}" if best_max < 1e-3 else f"{best_max:.6f}")
                ulp_str = f"{best_ulp:.1f}"

                # Show activation name only on first row
                act_display = activation if i == 0 else ""

                print(f"│ {act_display:<11} │ {metric_name:<9} │ {best_config:<12} │ {mae_str:>12} │ {max_str:>12} │ {ulp_str:>12} │ {best_t1:>12.3f} │ {best_t8:>12.3f} │ {best_t256:>12.3f} │ {best_ths:>12.3f} │ {best_tyolo:>12.3f} │")

                # Add separator between activations (except after last activation)
                if i == len(metrics) - 1 and activation != sorted(by_activation.keys())[-1]:
                    print("├─────────────┼───────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

        print("└─────────────┴───────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")


def generate_best_comparison_table(results_file):
    """Generate comparison table for best configurations (theory vs hardware vs SFPU)."""
    import os

    comparisons = []

    # Detect platform prefix from results file name
    results_basename = os.path.basename(results_file)
    if results_basename.startswith("wormhole_"):
        platform_prefix = "wormhole_"
    elif results_basename.startswith("blackhole_"):
        platform_prefix = "blackhole_"
    else:
        platform_prefix = ""

    # Load SFPU results if available
    results_dir = os.path.dirname(results_file)
    sfpu_file = os.path.join(results_dir, f"{platform_prefix}native_sfpu_results.csv")
    sfpu_data = {}
    if os.path.exists(sfpu_file):
        with open(sfpu_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] != "pass":
                    continue
                # Use fast mode as baseline for comparison
                if row.get("sfpu_mode") != "fast":
                    continue
                # Row-per-shape format: use 256_tiles shape as representative
                shape = row.get("shape", "")
                if shape != "256_tiles":
                    continue
                key = (row["activation"], row["precision"])
                runtime_ms = float(row.get("time_profiler_us", 0)) / 1000.0
                sfpu_data[key] = {
                    "mae": float(row.get("mae", 0)),
                    "max_error": float(row.get("max_error", 0)),
                    "runtime": runtime_ms,
                    "mode": row.get("sfpu_mode", "fast"),
                }

    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        raw_rows = [row for row in reader if row["status"] == "pass"]

    # Detect format and pivot if row-per-shape
    if raw_rows and 'shape' in raw_rows[0]:
        pivoted = _pivot_rows_to_config(raw_rows)
    else:
        # Legacy wide format
        pivoted = []
        for row in raw_rows:
            entry = dict(row)
            entry['single'] = float(row.get("time_profiler_single_tile_us", 0)) / 1000.0
            entry['256tiles'] = float(row.get("time_profiler_256_tiles_us", 0)) / 1000.0
            pivoted.append(entry)

    for entry in pivoted:
        # Skip rows with missing theoretical values
        if not entry.get("theoretical_mae") or not entry.get("theoretical_max_error"):
            continue

        activation = entry["activation"]
        precision = entry["precision"]
        depth = entry["depth"]
        degree = entry["degree"]
        segmentation = entry["segmentation"]

        mae_hw = float(entry.get("mae", entry.get("mae_single_tile", 0)))
        max_hw = float(entry.get("max_error", entry.get("max_error_single_tile", 0)))
        mae_theory = float(entry["theoretical_mae"])
        max_theory = float(entry["theoretical_max_error"])

        # Use 256 tiles as representative timing (most common workload)
        runtime = float(entry.get("256tiles", 0))

        mae_ratio = mae_hw / mae_theory if mae_theory > 1e-15 else 1.0
        max_ratio = max_hw / max_theory if max_theory > 1e-15 else 1.0

        # ULP metrics
        ulp_raw = entry.get("max_ulp", entry.get("max_ulp_error_single_tile", "0"))
        ulp_hw = float(ulp_raw) if ulp_raw else 0.0
        ulp_theory_raw = entry.get("theoretical_ulp", "")
        ulp_theory = float(ulp_theory_raw) if ulp_theory_raw and ulp_theory_raw not in ("", "N/A") else None
        ulp_ratio = ulp_hw / ulp_theory if ulp_theory and ulp_theory > 0.1 else None

        # Get SFPU data if available
        sfpu_key = (activation, precision)
        sfpu_mae = sfpu_data.get(sfpu_key, {}).get("mae", None)
        sfpu_max = sfpu_data.get(sfpu_key, {}).get("max_error", None)
        sfpu_time = sfpu_data.get(sfpu_key, {}).get("runtime", None)

        # Compute SFPU ratios
        max_ratio_sfpu = max_hw / sfpu_max if sfpu_max and sfpu_max > 1e-15 else None
        time_ratio_sfpu = runtime / sfpu_time if sfpu_time and sfpu_time > 1e-15 else None

        comparisons.append(
            {
                "activation": activation,
                "precision": precision,
                "depth": depth,
                "degree": degree,
                "segmentation": segmentation,
                "mae_theory": mae_theory,
                "mae_hw": mae_hw,
                "mae_ratio": mae_ratio,
                "max_theory": max_theory,
                "max_hw": max_hw,
                "max_ratio": max_ratio,
                "ulp_hw": ulp_hw,
                "ulp_theory": ulp_theory,
                "ulp_ratio": ulp_ratio,
                "runtime": runtime,
                "sfpu_mae": sfpu_mae,
                "sfpu_max": sfpu_max,
                "sfpu_time": sfpu_time,
                "max_ratio_sfpu": max_ratio_sfpu,
                "time_ratio_sfpu": time_ratio_sfpu,
            }
        )

    if not comparisons:
        print("No passing configurations to compare")
        return

    # ANSI color codes (basic)
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def ratio_to_rgb_gradient(ratio):
        """
        Map ratio to RGB color gradient: green → yellow → orange → red.
        - ratio < 0.5: dark green
        - ratio = 1.0: green/yellow
        - ratio = 2.0: orange
        - ratio > 3.0: red
        """
        if ratio < 0.5:
            # Dark green to green
            t = ratio / 0.5  # 0.0 to 1.0
            r = int(0 + t * 100)
            g = int(150 + t * 105)  # 150 to 255
            b = int(0)
        elif ratio < 1.0:
            # Green to bright green
            t = (ratio - 0.5) / 0.5  # 0.0 to 1.0
            r = int(100 + t * 155)  # 100 to 255
            g = int(255)
            b = int(0)
        elif ratio < 2.0:
            # Bright green/yellow to orange
            t = (ratio - 1.0) / 1.0  # 0.0 to 1.0
            r = int(255)
            g = int(255 - t * 90)  # 255 to 165
            b = int(0)
        elif ratio < 3.0:
            # Orange to red-orange
            t = (ratio - 2.0) / 1.0  # 0.0 to 1.0
            r = int(255)
            g = int(165 - t * 65)  # 165 to 100
            b = int(0)
        else:
            # Red-orange to deep red
            t = min((ratio - 3.0) / 2.0, 1.0)  # 0.0 to 1.0 (cap at ratio=5.0)
            r = int(255)
            g = int(100 - t * 100)  # 100 to 0
            b = int(0)
        return f"\033[38;2;{r};{g};{b}m"

    # Check if we have SFPU data
    has_sfpu = any(c["sfpu_max"] is not None for c in comparisons)
    # Check if we have theoretical ULP data (new format CSV)
    has_ulp_theory = any(c["ulp_theory"] is not None and c["ulp_theory"] > 0.0 for c in comparisons)

    # Calculate column widths dynamically based on actual content
    def format_number(val):
        """Format a number as it would appear in the table (2 sig figs)"""
        if val is None:
            return "N/A"
        return f"{val:.1e}"  # 2 significant digits in scientific notation

    def format_ratio(val, use_scientific=False):
        """Format a ratio compactly for MAE/Max ratio columns"""
        if val is None:
            return "N/A"
        # Simple compact format: "1.2x" or "0.95x ✓"
        if val < 0.98:
            # Hardware is better
            return f"{val:.2f}x ✓"
        elif val > 1.02:
            # Hardware is worse
            return f"{val:.2f}x  "
        else:
            # About the same
            return f"{val:.2f}x  "

    # Auto-size columns based on actual content width (header + data)
    col_widths = {
        "activation": max(len("Activation"), max(len(c["activation"]) for c in comparisons)),
        "prec": max(len("Prec"), max(len(c["precision"]) for c in comparisons)),
        "depth": max(len("Dep"), max(len(str(c["depth"])) for c in comparisons)),
        "degree": max(len("Deg"), max(len(str(c["degree"])) for c in comparisons)),
        "seg": max(len("Seg"), max(len(c["segmentation"][:3]) for c in comparisons)),
        "mae_theory": max(len("Theory MAE"), max(len(format_number(c["mae_theory"])) for c in comparisons)),
        "mae_hw": max(len("Hardware MAE"), max(len(format_number(c["mae_hw"])) for c in comparisons)),
        "mae_ratio": max(len("MAE Ratio"), max(len(format_ratio(c["mae_ratio"])) for c in comparisons)),
        "max_theory": max(len("Theory Max"), max(len(format_number(c["max_theory"])) for c in comparisons)),
        "max_hw": max(len("Hardware Max"), max(len(format_number(c["max_hw"])) for c in comparisons)),
        "max_ratio": max(len("Max Ratio"), max(len(format_ratio(c["max_ratio"])) for c in comparisons)),
        "time": max(len("Time"), max(len(f"{c['runtime']:.1f}ms") for c in comparisons)),
    }

    if has_ulp_theory:
        col_widths.update(
            {
                "ulp_theory": max(
                    len("Theory ULP"),
                    max(len(f"{c['ulp_theory']:.2f}") if c["ulp_theory"] is not None else 3 for c in comparisons),
                ),
                "ulp_hw": max(
                    len("HW ULP"),
                    max(len(f"{c['ulp_hw']:.2f}") for c in comparisons),
                ),
                "ulp_ratio": max(
                    len("ULP Ratio"),
                    max(len(format_ratio(c["ulp_ratio"])) if c["ulp_ratio"] is not None else 3 for c in comparisons),
                ),
            }
        )

    if has_sfpu:
        col_widths.update(
            {
                "sfpu_max": max(
                    len("SFPU Max"),
                    max(len(format_number(c["sfpu_max"])) if c["sfpu_max"] is not None else 3 for c in comparisons),
                ),
                "sfpu_mae": max(
                    len("SFPU MAE"),
                    max(len(format_number(c["sfpu_mae"])) if c["sfpu_mae"] is not None else 3 for c in comparisons),
                ),
                "sfpu_time": max(
                    len("SFPU Time"),
                    max(len(f"{c['sfpu_time']:.1f}ms") if c["sfpu_time"] is not None else 3 for c in comparisons),
                ),
                "max_ratio_sfpu": max(
                    len("Max vs SFPU"),
                    15,  # Width for "7212x better ✓" format
                ),
                "time_ratio_sfpu": max(
                    len("Time vs SFPU"),
                    15,  # Width for "1.23x slower ✗" format
                ),
            }
        )

    # Print comparison table header
    if has_sfpu:
        print("THEORETICAL vs HARDWARE vs SFPU ERROR COMPARISON")
    else:
        print("THEORETICAL vs HARDWARE ERROR COMPARISON")

    # Build table borders dynamically based on available columns
    top_border_parts = [
        f"┌─{'─'*col_widths['activation']}─┬─{'─'*col_widths['prec']}─┬─{'─'*col_widths['depth']}─┬─",
        f"{'─'*col_widths['degree']}─┬─{'─'*col_widths['seg']}─┬─{'─'*col_widths['mae_theory']}─┬─",
        f"{'─'*col_widths['mae_hw']}─┬─{'─'*col_widths['mae_ratio']}─┬─{'─'*col_widths['max_theory']}─┬─",
        f"{'─'*col_widths['max_hw']}─┬─{'─'*col_widths['max_ratio']}─",
    ]
    if has_ulp_theory:
        top_border_parts.append(
            f"┬─{'─'*col_widths['ulp_theory']}─┬─{'─'*col_widths['ulp_hw']}─┬─{'─'*col_widths['ulp_ratio']}─"
        )
    top_border_parts.append(f"┬─{'─'*col_widths['time']}─")
    if has_sfpu:
        top_border_parts.append(
            f"┬─{'─'*col_widths['sfpu_max']}─┬─{'─'*col_widths['sfpu_mae']}─┬─{'─'*col_widths['sfpu_time']}─┬─"
            f"{'─'*col_widths['max_ratio_sfpu']}─┬─{'─'*col_widths['time_ratio_sfpu']}─┐"
        )
    else:
        top_border_parts.append("┐")
    print("".join(top_border_parts))

    # Header row
    header_parts = [
        f"│ {'Activation':<{col_widths['activation']}} │ {'Prec':<{col_widths['prec']}} │ ",
        f"{'Dep':>{col_widths['depth']}} │ {'Deg':>{col_widths['degree']}} │ {'Seg':>{col_widths['seg']}} │ ",
        f"{'Theory MAE':>{col_widths['mae_theory']}} │ {'Hardware MAE':>{col_widths['mae_hw']}} │ ",
        f"{'MAE Ratio':>{col_widths['mae_ratio']}} │ {'Theory Max':>{col_widths['max_theory']}} │ ",
        f"{'Hardware Max':>{col_widths['max_hw']}} │ {'Max Ratio':>{col_widths['max_ratio']}} │ ",
    ]
    if has_ulp_theory:
        header_parts.append(
            f"{'Theory ULP':>{col_widths['ulp_theory']}} │ {'HW ULP':>{col_widths['ulp_hw']}} │ "
            f"{'ULP Ratio':>{col_widths['ulp_ratio']}} │ "
        )
    header_parts.append(f"{'Time':>{col_widths['time']}} │")
    if has_sfpu:
        header_parts.append(
            f" {'SFPU Max':>{col_widths['sfpu_max']}} │ {'SFPU MAE':>{col_widths['sfpu_mae']}} │ "
            f"{'SFPU Time':>{col_widths['sfpu_time']}} │ {'Max vs SFPU':>{col_widths['max_ratio_sfpu']}} │ "
            f"{'Time vs SFPU':>{col_widths['time_ratio_sfpu']}} │"
        )
    print("".join(header_parts))

    # Separator
    separator_parts = [
        f"├─{'─'*col_widths['activation']}─┼─{'─'*col_widths['prec']}─┼─{'─'*col_widths['depth']}─┼─",
        f"{'─'*col_widths['degree']}─┼─{'─'*col_widths['seg']}─┼─{'─'*col_widths['mae_theory']}─┼─",
        f"{'─'*col_widths['mae_hw']}─┼─{'─'*col_widths['mae_ratio']}─┼─{'─'*col_widths['max_theory']}─┼─",
        f"{'─'*col_widths['max_hw']}─┼─{'─'*col_widths['max_ratio']}─",
    ]
    if has_ulp_theory:
        separator_parts.append(
            f"┼─{'─'*col_widths['ulp_theory']}─┼─{'─'*col_widths['ulp_hw']}─┼─{'─'*col_widths['ulp_ratio']}─"
        )
    separator_parts.append(f"┼─{'─'*col_widths['time']}─")
    if has_sfpu:
        separator_parts.append(
            f"┼─{'─'*col_widths['sfpu_max']}─┼─{'─'*col_widths['sfpu_mae']}─┼─{'─'*col_widths['sfpu_time']}─┼─"
            f"{'─'*col_widths['max_ratio_sfpu']}─┼─{'─'*col_widths['time_ratio_sfpu']}─┤"
        )
    else:
        separator_parts.append("┤")
    print("".join(separator_parts))

    for c in comparisons:
        # Format numbers (2 significant digits)
        mae_theory_str = f"{c['mae_theory']:.1e}"
        mae_hw_str = f"{c['mae_hw']:.1e}"
        max_theory_str = f"{c['max_theory']:.1e}"
        max_hw_str = f"{c['max_hw']:.1e}"

        # Format ratios with color and indicators
        if c["mae_ratio"] < 1.0:
            mae_ratio_str = f"{GREEN}{c['mae_ratio']:>4.2f}x ✓{RESET}"
            mae_indicator = "✓"
            mae_ratio_width = 7  # "0.98x ✓" = 7 chars
        elif c["mae_ratio"] <= 2.0:
            mae_ratio_str = f"{c['mae_ratio']:>4.2f}x  "
            mae_indicator = " "
            mae_ratio_width = 7
        elif c["mae_ratio"] <= 5.0:
            mae_ratio_str = f"{c['mae_ratio']:>4.2f}x ⚠"
            mae_indicator = "⚠"
            mae_ratio_width = 7
        else:
            mae_ratio_str = f"{RED}{c['mae_ratio']:>4.2f}x ✗{RESET}"
            mae_indicator = "✗"
            mae_ratio_width = 7

        # Use gradient coloring for max ratio (red → orange → yellow → green)
        if c["max_ratio"] < 1.0:
            max_indicator = "✓"
        elif c["max_ratio"] <= 2.0:
            max_indicator = " "
        elif c["max_ratio"] <= 5.0:
            max_indicator = "⚠"
        else:
            max_indicator = "✗"

        max_color = ratio_to_rgb_gradient(c["max_ratio"])
        max_ratio_str = f"{max_color}{c['max_ratio']:>4.2f}x {max_indicator}{RESET}"
        max_ratio_width = 7  # "1.00x ✓" or "1.00x  " = 7 chars

        seg_abbrev = c["segmentation"][:3] if len(c["segmentation"]) > 3 else c["segmentation"]
        time_str = f"{c['runtime']:.1f}ms"

        # Calculate padding needed for color codes (they don't take visual space but count as characters)
        mae_ratio_padding = col_widths["mae_ratio"] + (len(mae_ratio_str) - mae_ratio_width)
        max_ratio_padding = col_widths["max_ratio"] + (len(max_ratio_str) - max_ratio_width)

        # Format ULP values if available
        if has_ulp_theory:
            ulp_theory_str = f"{c['ulp_theory']:.2f}" if c["ulp_theory"] is not None else "N/A"
            ulp_hw_str = f"{c['ulp_hw']:.2f}"
            if c["ulp_ratio"] is None:
                ulp_ratio_str = "N/A"
                ulp_ratio_padding = col_widths["ulp_ratio"]
            elif c["ulp_ratio"] < 1.0:
                ulp_ratio_str = f"{GREEN}{c['ulp_ratio']:>4.2f}x ✓{RESET}"
                ulp_ratio_padding = col_widths["ulp_ratio"] + (len(ulp_ratio_str) - 7)
            elif c["ulp_ratio"] <= 2.0:
                ulp_ratio_str = f"{c['ulp_ratio']:>4.2f}x  "
                ulp_ratio_padding = col_widths["ulp_ratio"]
            elif c["ulp_ratio"] <= 5.0:
                ulp_ratio_str = f"{c['ulp_ratio']:>4.2f}x ⚠"
                ulp_ratio_padding = col_widths["ulp_ratio"]
            else:
                ulp_ratio_str = f"{RED}{c['ulp_ratio']:>4.2f}x ✗{RESET}"
                ulp_ratio_padding = col_widths["ulp_ratio"] + (len(ulp_ratio_str) - 7)

        # Build base row
        row_parts = [
            f"│ {c['activation']:<{col_widths['activation']}} │ {c['precision']:<{col_widths['prec']}} │ ",
            f"{c['depth']:>{col_widths['depth']}} │ {str(c['degree']):>{col_widths['degree']}} │ ",
            f"{seg_abbrev:>{col_widths['seg']}} │ {mae_theory_str:>{col_widths['mae_theory']}} │ ",
            f"{mae_hw_str:>{col_widths['mae_hw']}} │ {mae_ratio_str:>{mae_ratio_padding}} │ ",
            f"{max_theory_str:>{col_widths['max_theory']}} │ {max_hw_str:>{col_widths['max_hw']}} │ ",
            f"{max_ratio_str:>{max_ratio_padding}} │ ",
        ]
        if has_ulp_theory:
            row_parts.append(
                f"{ulp_theory_str:>{col_widths['ulp_theory']}} │ {ulp_hw_str:>{col_widths['ulp_hw']}} │ "
                f"{ulp_ratio_str:>{ulp_ratio_padding}} │ "
            )
        row_parts.append(f"{time_str:>{col_widths['time']}} │")

        # Add SFPU columns if available
        if has_sfpu:
            if c["sfpu_max"] is not None:
                # Format SFPU values (2 sig figs)
                sfpu_max_str = f"{c['sfpu_max']:.1e}"
                sfpu_mae_str = f"{c['sfpu_mae']:.1e}"
                sfpu_time_str = f"{c['sfpu_time']:.1f}ms"

                # Format SFPU ratios with gradient coloring (better/worse format)
                if c["max_ratio_sfpu"] is not None:
                    ratio = c["max_ratio_sfpu"]
                    sfpu_max_color = ratio_to_rgb_gradient(ratio)

                    # Format as "Nx better" or "Nx worse" - consistent formatting
                    if ratio == 0.0:
                        # Perfect hardware match (zero error)
                        max_ratio_sfpu_str = f"{sfpu_max_color}{'perfect ✓':>15}{RESET}"
                        plain_len = 15
                    elif ratio < 1.0:
                        # Hardware is better (less error) - show reciprocal
                        inverse = 1.0 / ratio
                        if inverse >= 1000:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{inverse:>5.0f}x better ✓{RESET}"
                            plain_len = 15  # " 7212x better ✓"
                        elif inverse >= 10:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{inverse:>5.1f}x better ✓{RESET}"
                            plain_len = 15  # " 71.4x better ✓"
                        else:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{inverse:>5.2f}x better ✓{RESET}"
                            plain_len = 15  # "  7.14x better ✓"
                    elif ratio > 1.01:
                        # Hardware is worse (more error)
                        if ratio >= 1000:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{ratio:>5.0f}x worse  ✗{RESET}"
                            plain_len = 15  # " 1600x worse  ✗"
                        elif ratio >= 10:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{ratio:>5.1f}x worse  ✗{RESET}"
                            plain_len = 15  # " 16.0x worse  ✗"
                        else:
                            max_ratio_sfpu_str = f"{sfpu_max_color}{ratio:>5.2f}x worse  ✗{RESET}"
                            plain_len = 15  # "  1.60x worse  ✗"
                    else:
                        # About the same
                        max_ratio_sfpu_str = f"{sfpu_max_color}{'~same':>15}{RESET}"
                        plain_len = 15

                    max_ratio_sfpu_padding = col_widths["max_ratio_sfpu"] + (len(max_ratio_sfpu_str) - plain_len)
                else:
                    max_ratio_sfpu_str = "N/A"
                    max_ratio_sfpu_padding = col_widths["max_ratio_sfpu"]

                if c["time_ratio_sfpu"] is not None:
                    # Time ratio: lower is better (LUT faster than SFPU)
                    ratio = c["time_ratio_sfpu"]
                    # Invert coloring: small ratio = green (LUT faster), large ratio = red (LUT slower)
                    time_color = ratio_to_rgb_gradient(ratio)

                    # Format as "Nx faster" or "Nx slower" - consistent formatting
                    if ratio == 0.0:
                        # Instant execution (zero time)
                        time_ratio_sfpu_str = f"{GREEN}{'instant ✓':>15}{RESET}"
                        plain_len = 15
                    elif ratio < 0.99:
                        # LUT is faster - show reciprocal
                        inverse = 1.0 / ratio
                        if inverse >= 10:
                            time_ratio_sfpu_str = f"{GREEN}{inverse:>5.2f}x faster ✓{RESET}"
                            plain_len = 15  # " 12.34x faster ✓"
                        else:
                            time_ratio_sfpu_str = f"{GREEN}{inverse:>5.2f}x faster ✓{RESET}"
                            plain_len = 15  # "  1.23x faster ✓"
                    elif ratio > 1.01:
                        # LUT is slower
                        if ratio >= 10:
                            time_ratio_sfpu_str = f"{time_color}{ratio:>5.2f}x slower ✗{RESET}"
                            plain_len = 15  # " 12.34x slower ✗"
                        else:
                            time_ratio_sfpu_str = f"{time_color}{ratio:>5.2f}x slower ✗{RESET}"
                            plain_len = 15  # "  1.23x slower ✗"
                    else:
                        # About the same
                        time_ratio_sfpu_str = f"{'~same':>15}"
                        plain_len = 15

                    time_ratio_sfpu_padding = col_widths["time_ratio_sfpu"] + (len(time_ratio_sfpu_str) - plain_len)
                else:
                    time_ratio_sfpu_str = "N/A"
                    time_ratio_sfpu_padding = col_widths["time_ratio_sfpu"]

                row_parts.append(
                    f" {sfpu_max_str:>{col_widths['sfpu_max']}} │ {sfpu_mae_str:>{col_widths['sfpu_mae']}} │ "
                    f"{sfpu_time_str:>{col_widths['sfpu_time']}} │ {max_ratio_sfpu_str:>{max_ratio_sfpu_padding}} │ "
                    f"{time_ratio_sfpu_str:>{time_ratio_sfpu_padding}} │"
                )
            else:
                # No SFPU data for this activation
                row_parts.append(
                    f" {'N/A':>{col_widths['sfpu_max']}} │ {'N/A':>{col_widths['sfpu_mae']}} │ "
                    f"{'N/A':>{col_widths['sfpu_time']}} │ {'N/A':>{col_widths['max_ratio_sfpu']}} │ "
                    f"{'N/A':>{col_widths['time_ratio_sfpu']}} │"
                )

        print("".join(row_parts))

    # Footer
    footer_parts = [
        f"└─{'─'*col_widths['activation']}─┴─{'─'*col_widths['prec']}─┴─{'─'*col_widths['depth']}─┴─",
        f"{'─'*col_widths['degree']}─┴─{'─'*col_widths['seg']}─┴─{'─'*col_widths['mae_theory']}─┴─",
        f"{'─'*col_widths['mae_hw']}─┴─{'─'*col_widths['mae_ratio']}─┴─{'─'*col_widths['max_theory']}─┴─",
        f"{'─'*col_widths['max_hw']}─┴─{'─'*col_widths['max_ratio']}─",
    ]
    if has_ulp_theory:
        footer_parts.append(
            f"┴─{'─'*col_widths['ulp_theory']}─┴─{'─'*col_widths['ulp_hw']}─┴─{'─'*col_widths['ulp_ratio']}─"
        )
    footer_parts.append(f"┴─{'─'*col_widths['time']}─")
    if has_sfpu:
        footer_parts.append(
            f"┴─{'─'*col_widths['sfpu_max']}─┴─{'─'*col_widths['sfpu_mae']}─┴─{'─'*col_widths['sfpu_time']}─┴─"
            f"{'─'*col_widths['max_ratio_sfpu']}─┴─{'─'*col_widths['time_ratio_sfpu']}─┘"
        )
    else:
        footer_parts.append("┘")
    print("".join(footer_parts))
    print()

    # Statistics
    mae_ratios = [c["mae_ratio"] for c in comparisons]
    max_ratios = [c["max_ratio"] for c in comparisons]

    best_mae_ratio = min(mae_ratios)
    worst_mae_ratio = max(mae_ratios)
    avg_mae_ratio = sum(mae_ratios) / len(mae_ratios)

    within_1x = sum(1 for r in mae_ratios if r <= 1.0)
    within_2x = sum(1 for r in mae_ratios if r <= 2.0)
    within_5x = sum(1 for r in mae_ratios if r <= 5.0)

    best_max_ratio = min(max_ratios)
    worst_max_ratio = max(max_ratios)
    avg_max_ratio = sum(max_ratios) / len(max_ratios)

    within_1x_max = sum(1 for r in max_ratios if r <= 1.0)
    within_2x_max = sum(1 for r in max_ratios if r <= 2.0)
    within_5x_max = sum(1 for r in max_ratios if r <= 5.0)

    print(f"Total configurations: {len(comparisons)}")
    print()
    print("Hardware vs Theory (MAE):")
    print(f"  Best ratio: {best_mae_ratio:.2f}x")
    print(f"  Worst ratio: {worst_mae_ratio:.2f}x")
    print(f"  Average ratio: {avg_mae_ratio:.2f}x")
    print()
    print(f"  Better than theory (<1x):     {within_1x}/{len(comparisons)} ({100*within_1x/len(comparisons):.0f}%)")
    print(f"  Within 2x of theory:           {within_2x}/{len(comparisons)} ({100*within_2x/len(comparisons):.0f}%)")
    print(f"  Within 5x of theory:           {within_5x}/{len(comparisons)} ({100*within_5x/len(comparisons):.0f}%)")
    print()
    print("Hardware vs Theory (Max Error):")
    print(f"  Best ratio: {best_max_ratio:.2f}x")
    print(f"  Worst ratio: {worst_max_ratio:.2f}x")
    print(f"  Average ratio: {avg_max_ratio:.2f}x")
    print()
    print(f"  Better than theory (<1x):     {within_1x_max}/{len(comparisons)} ({100*within_1x_max/len(comparisons):.0f}%)")
    print(f"  Within 2x of theory:           {within_2x_max}/{len(comparisons)} ({100*within_2x_max/len(comparisons):.0f}%)")
    print(f"  Within 5x of theory:           {within_5x_max}/{len(comparisons)} ({100*within_5x_max/len(comparisons):.0f}%)")

    if has_ulp_theory:
        ulp_ratios = [c["ulp_ratio"] for c in comparisons if c["ulp_ratio"] is not None]
        if ulp_ratios:
            best_ulp_ratio = min(ulp_ratios)
            worst_ulp_ratio = max(ulp_ratios)
            avg_ulp_ratio = sum(ulp_ratios) / len(ulp_ratios)

            within_1x_ulp = sum(1 for r in ulp_ratios if r <= 1.0)
            within_2x_ulp = sum(1 for r in ulp_ratios if r <= 2.0)
            within_5x_ulp = sum(1 for r in ulp_ratios if r <= 5.0)

            print()
            print("Hardware vs Theory (ULP):")
            print(f"  Best ratio: {best_ulp_ratio:.2f}x")
            print(f"  Worst ratio: {worst_ulp_ratio:.2f}x")
            print(f"  Average ratio: {avg_ulp_ratio:.2f}x")
            print()
            print(f"  Better than theory (<1x):     {within_1x_ulp}/{len(ulp_ratios)} ({100*within_1x_ulp/len(ulp_ratios):.0f}%)")
            print(f"  Within 2x of theory:           {within_2x_ulp}/{len(ulp_ratios)} ({100*within_2x_ulp/len(ulp_ratios):.0f}%)")
            print(f"  Within 5x of theory:           {within_5x_ulp}/{len(ulp_ratios)} ({100*within_5x_ulp/len(ulp_ratios):.0f}%)")

    # SFPU statistics
    if has_sfpu:
        sfpu_max_ratios = [c["max_ratio_sfpu"] for c in comparisons if c["max_ratio_sfpu"] is not None]
        sfpu_time_ratios = [c["time_ratio_sfpu"] for c in comparisons if c["time_ratio_sfpu"] is not None]

        if sfpu_max_ratios:
            best_max_sfpu = min(sfpu_max_ratios)
            worst_max_sfpu = max(sfpu_max_ratios)
            avg_max_sfpu = sum(sfpu_max_ratios) / len(sfpu_max_ratios)

            within_1x_sfpu = sum(1 for r in sfpu_max_ratios if r <= 1.0)
            within_2x_sfpu = sum(1 for r in sfpu_max_ratios if r <= 2.0)
            within_5x_sfpu = sum(1 for r in sfpu_max_ratios if r <= 5.0)

            print()
            print("Hardware LUT vs SFPU (Max Error):")
            print(f"  Best ratio: {best_max_sfpu:.2f}x")
            print(f"  Worst ratio: {worst_max_sfpu:.2f}x")
            print(f"  Average ratio: {avg_max_sfpu:.2f}x")
            print()
            print(
                f"  Better than SFPU (<1x):     {within_1x_sfpu}/{len(sfpu_max_ratios)} ({100*within_1x_sfpu/len(sfpu_max_ratios):.0f}%)"
            )
            print(
                f"  Within 2x of SFPU:          {within_2x_sfpu}/{len(sfpu_max_ratios)} ({100*within_2x_sfpu/len(sfpu_max_ratios):.0f}%)"
            )
            print(
                f"  Within 5x of SFPU:          {within_5x_sfpu}/{len(sfpu_max_ratios)} ({100*within_5x_sfpu/len(sfpu_max_ratios):.0f}%)"
            )

        if sfpu_time_ratios:
            best_time_sfpu = min(sfpu_time_ratios)
            worst_time_sfpu = max(sfpu_time_ratios)
            avg_time_sfpu = sum(sfpu_time_ratios) / len(sfpu_time_ratios)

            faster_than_sfpu = sum(1 for r in sfpu_time_ratios if r < 1.0)
            within_2x_time = sum(1 for r in sfpu_time_ratios if r <= 2.0)

            print()
            print("Hardware LUT vs SFPU (Runtime):")
            print(f"  Best ratio: {best_time_sfpu:.2f}x")
            print(f"  Worst ratio: {worst_time_sfpu:.2f}x")
            print(f"  Average ratio: {avg_time_sfpu:.2f}x")
            print()
            print(
                f"  Faster than SFPU (<1x):     {faster_than_sfpu}/{len(sfpu_time_ratios)} ({100*faster_than_sfpu/len(sfpu_time_ratios):.0f}%)"
            )
            print(
                f"  Within 2x of SFPU:          {within_2x_time}/{len(sfpu_time_ratios)} ({100*within_2x_time/len(sfpu_time_ratios):.0f}%)"
            )

    print()
    print("Legend: ✓ = better than reference (<1x), ⚠ = moderate deviation (2-5x), ✗ = large deviation (>5x)")


def detect_format(results_file):
    """Auto-detect whether the CSV contains polynomial, rational, or best comparison data."""
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        # Check column names
        fieldnames = reader.fieldnames
        if "theoretical_mae" in fieldnames and "theoretical_max_error" in fieldnames:
            return "best"
        elif "num_degree" in fieldnames and "den_degree" in fieldnames:
            return "rational"
        else:
            return "polynomial"


def main():
    parser = argparse.ArgumentParser(description="Generate summary statistics tables from sweep results")
    parser.add_argument("results_csv", help="Path to results CSV file")
    parser.add_argument(
        "--type",
        choices=["polynomial", "rational", "best"],
        required=False,
        help="Type of approximation (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Auto-detect format if not specified
    sweep_type = args.type if args.type else detect_format(args.results_csv)

    # Show grouped summary tables for polynomial/rational
    if sweep_type == "polynomial":
        generate_polynomial_tables(args.results_csv)
        print()
    elif sweep_type == "rational":
        generate_rational_tables(args.results_csv)
        print()

    # Check if theoretical comparison is available
    with open(args.results_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        has_theoretical = "theoretical_mae" in fieldnames and "theoretical_max_error" in fieldnames

    # Show theory vs hardware comparison if available
    if has_theoretical:
        generate_best_comparison_table(args.results_csv)


if __name__ == "__main__":
    main()
