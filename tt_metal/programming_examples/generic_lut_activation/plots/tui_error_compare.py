#!/usr/bin/env python3
"""
Hardware Error Comparison TUI - Ratatui-style Terminal Plot

Compare hardware output CSVs (input,output pairs) against a reference function.
Shows ULP, absolute, and relative error curves in the terminal.
Plots automatically adapt to terminal width for more detail on wider terminals.

Two modes:
  1. Auto-discovery mode (like plot_hardware_error_analysis.py):
     python3 tui_error_compare.py --arch blackhole --activation gelu --precision bf16

  2. Manual CSV mode:
     python3 tui_error_compare.py csv1.csv csv2.csv --activation gelu

CSV format expected:
    input,output
    -3.5,0.00012
    -3.4,0.00015
    ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

# Import shared utilities from hwoutcsv2errcsv
from hwoutcsv2errcsv import (
    load_hardware_outputs,
    load_or_compute_errors,
    compute_pytorch_baseline,
    is_normal_number,
    ACTIVATION_FUNCTIONS as BASE_ACTIVATION_FUNCTIONS,
)


# ── Activation functions ───────────────────────────────────────────────────────

# Extend base activation functions with TUI-specific ones
ACTIVATION_FUNCTIONS = {
    **BASE_ACTIVATION_FUNCTIONS,
    'exp2': lambda x: torch.pow(2.0, x),
    'atan': torch.atan,
    'abs': torch.abs,
    'identity': lambda x: x,
}

# Activations with native SFPU support
NATIVE_SFPU_ACTIVATIONS = [
    'cos', 'elu', 'erf', 'exp', 'gelu', 'hardsigmoid',
    'leaky_relu', 'relu', 'selu', 'sin', 'softplus', 'tanh',
    'cosh', 'sinh', 'atanh', 'celu', 'sigmoid'
]


# Use compute_pytorch_baseline from hwoutcsv2errcsv (aliased for compatibility)
compute_reference = compute_pytorch_baseline

# Aliases for compatibility
load_hardware_csv = load_hardware_outputs


# ── Auto-discovery of CSV files ────────────────────────────────────────────────

def discover_csvs(data_dir: Path, arch: str, activation: str, precision: str) -> list[tuple[Path, str]]:
    """
    Auto-discover CSV files for the given arch/activation/precision.
    Returns list of (path, label) tuples.
    """
    activation_dir = data_dir / arch / activation
    found = []

    if not activation_dir.exists():
        return found

    # 1. Look for SFPU fast/precise
    for variant in ['fast', 'precise']:
        pattern = str(activation_dir / f'native_sfpu_{activation}_{precision}_{variant}_tiles*.csv')
        matching = glob.glob(pattern)
        if matching:
            # Sort by tile count (descending) and pick the largest
            matching.sort(key=lambda f: int(f.split('tiles')[-1].replace('.csv', '')), reverse=True)
            found.append((Path(matching[0]), f"SFPU {variant}"))

    # 2. Look for theoretical/LUT approximation
    theo_pattern = str(activation_dir / f'theoretical_{activation}_*_{precision}_tiles*.csv')
    theo_files = glob.glob(theo_pattern)
    if theo_files:
        theo_files.sort(key=lambda f: int(f.split('tiles')[-1].replace('.csv', '')), reverse=True)
        approx_csv = Path(theo_files[0])

        # Extract label from filename: theoretical_gelu_32_4_curvature_any_bf16_tiles3200.csv
        fname_parts = approx_csv.stem.replace('theoretical_', '').split('_')
        if len(fname_parts) >= 4:
            try:
                segments_val = fname_parts[1]
                degree_val = fname_parts[2]
                seg_type_val = fname_parts[3]
                label = f"LUT p{degree_val}s{segments_val}"
            except:
                label = "LUT"
        else:
            label = "LUT"

        found.append((approx_csv, label))

    # 3. Look for other hardware approximation files (fallback)
    if not any('LUT' in label for _, label in found):
        # Look for files like: gelu_bf16_32_4_curvature_any_tiles*.csv
        hw_pattern = str(activation_dir / f'{activation}_{precision}_*_tiles*.csv')
        hw_files = glob.glob(hw_pattern)
        # Exclude native_sfpu files
        hw_files = [f for f in hw_files if 'native_sfpu' not in f and 'theoretical' not in f]
        if hw_files:
            hw_files.sort(key=lambda f: int(f.split('tiles')[-1].replace('.csv', '')), reverse=True)
            found.append((Path(hw_files[0]), "HW Approx"))

    return found


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_stats(errors: dict) -> dict:
    """Compute summary statistics."""
    nm = errors['normal_mask']
    ulp = errors['ulp_err'][nm]
    abs_e = errors['abs_err'][nm]

    valid_ulp = ulp[np.isfinite(ulp)]
    valid_abs = abs_e[np.isfinite(abs_e)]

    return {
        'max_ulp': float(valid_ulp.max()) if len(valid_ulp) > 0 else 0.0,
        'mean_ulp': float(valid_ulp.mean()) if len(valid_ulp) > 0 else 0.0,
        'p95_ulp': float(np.percentile(valid_ulp, 95)) if len(valid_ulp) > 0 else 0.0,
        'p99_ulp': float(np.percentile(valid_ulp, 99)) if len(valid_ulp) > 0 else 0.0,
        'mae': float(valid_abs.mean()) if len(valid_abs) > 0 else 0.0,
        'max_abs': float(valid_abs.max()) if len(valid_abs) > 0 else 0.0,
        'n_points': int(nm.sum()),
    }


# ── Colors for up to 3 datasets ────────────────────────────────────────────────

COLORS = ["orange", "blue", "green"]
COLOR_NAMES = ["orange1", "blue", "green"]  # For textual markup


# ── Plot Widgets ───────────────────────────────────────────────────────────────

class ULPComparisonPlot(PlotextPlot):
    """ULP error scatter plot comparing datasets."""

    def __init__(self, datasets: list[dict], labels: list[str], **kw):
        super().__init__(**kw)
        self._datasets = datasets
        self._labels = labels

    def on_mount(self) -> None:
        plt = self.plt
        plt.clear_figure()

        all_ulp = []

        for i, (data, label) in enumerate(zip(self._datasets, self._labels)):
            nm = data['normal_mask']
            ulp = data['ulp_err']
            x = data['inputs']
            mask = nm & (ulp > 0) & np.isfinite(ulp)
            if mask.any():
                plt.scatter(x[mask].tolist(), ulp[mask].tolist(),
                           color=COLORS[i % len(COLORS)], label=label, marker="dot")
                all_ulp.extend(ulp[mask].tolist())

        # ULP reference lines
        plt.horizontal_line(1.0, color="cyan")
        plt.horizontal_line(3.0, color="yellow")
        plt.horizontal_line(10.0, color="red")

        plt.yscale("log")
        plt.title("ULP Error Comparison (log scale)")
        plt.xlabel("Input x")
        plt.ylabel("ULP Error")

        # Set y-axis ticks
        if all_ulp:
            lo = max(0, int(math.floor(math.log10(min(all_ulp)))))
            hi = int(math.ceil(math.log10(max(all_ulp))))
            plt.yticks(
                [10.0 ** e for e in range(lo, hi + 1)],
                [f"1e{e}" for e in range(lo, hi + 1)],
            )

        self.refresh()

    def on_resize(self, event) -> None:
        """Redraw on resize to adapt to terminal width."""
        self.on_mount()


class AbsErrorComparisonPlot(PlotextPlot):
    """Absolute error scatter plot comparing datasets."""

    def __init__(self, datasets: list[dict], labels: list[str], **kw):
        super().__init__(**kw)
        self._datasets = datasets
        self._labels = labels

    def on_mount(self) -> None:
        plt = self.plt
        plt.clear_figure()

        all_abs = []

        for i, (data, label) in enumerate(zip(self._datasets, self._labels)):
            nm = data['normal_mask']
            abs_e = data['abs_err']
            x = data['inputs']
            mask = nm & (abs_e > 0) & np.isfinite(abs_e)
            if mask.any():
                plt.scatter(x[mask].tolist(), abs_e[mask].tolist(),
                           color=COLORS[i % len(COLORS)], label=label, marker="dot")
                all_abs.extend(abs_e[mask].tolist())

        plt.yscale("log")
        plt.title("Absolute Error |f(x) - approx(x)| (log scale)")
        plt.xlabel("Input x")
        plt.ylabel("|Error|")

        if all_abs:
            lo = int(math.floor(math.log10(min(all_abs))))
            hi = int(math.ceil(math.log10(max(all_abs))))
            plt.yticks(
                [10.0 ** e for e in range(lo, hi + 1)],
                [f"1e{e}" for e in range(lo, hi + 1)],
            )

        self.refresh()

    def on_resize(self, event) -> None:
        self.on_mount()


class OutputComparisonPlot(PlotextPlot):
    """Output values comparison: reference vs all approximations."""

    def __init__(self, datasets: list[dict], labels: list[str], **kw):
        super().__init__(**kw)
        self._datasets = datasets
        self._labels = labels

    def on_mount(self) -> None:
        plt = self.plt
        plt.clear_figure()

        # Use first dataset's inputs for reference
        x = self._datasets[0]['inputs']
        ref = self._datasets[0]['reference']

        # Sort for line plot
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        ref_sorted = ref[sort_idx]

        # Plot reference
        plt.plot(x_sorted.tolist(), ref_sorted.tolist(),
                color="white", label="Reference")

        # Plot each dataset's outputs
        for i, (data, label) in enumerate(zip(self._datasets, self._labels)):
            out = data['outputs']
            xi = data['inputs']
            plt.scatter(xi.tolist(), out.tolist(),
                       color=COLORS[i % len(COLORS)], label=label, marker="dot")

        plt.title("Output Comparison: Reference vs Approximations")
        plt.xlabel("Input x")
        plt.ylabel("Output f(x)")

        self.refresh()

    def on_resize(self, event) -> None:
        self.on_mount()


class RelErrorComparisonPlot(PlotextPlot):
    """Relative error scatter plot comparing datasets."""

    def __init__(self, datasets: list[dict], labels: list[str], **kw):
        super().__init__(**kw)
        self._datasets = datasets
        self._labels = labels

    def on_mount(self) -> None:
        plt = self.plt
        plt.clear_figure()

        all_rel = []

        for i, (data, label) in enumerate(zip(self._datasets, self._labels)):
            nm = data['normal_mask']
            rel = data['rel_err']
            x = data['inputs']
            mask = nm & (rel > 0) & np.isfinite(rel)
            if mask.any():
                plt.scatter(x[mask].tolist(), rel[mask].tolist(),
                           color=COLORS[i % len(COLORS)], label=label, marker="dot")
                all_rel.extend(rel[mask].tolist())

        plt.yscale("log")
        plt.title("Relative Error |f(x) - approx(x)| / |f(x)| (log scale)")
        plt.xlabel("Input x")
        plt.ylabel("Relative Error")

        if all_rel:
            lo = int(math.floor(math.log10(min(all_rel))))
            hi = int(math.ceil(math.log10(max(all_rel))))
            plt.yticks(
                [10.0 ** e for e in range(lo, hi + 1)],
                [f"1e{e}" for e in range(lo, hi + 1)],
            )

        self.refresh()

    def on_resize(self, event) -> None:
        self.on_mount()


# ── Sidebar ────────────────────────────────────────────────────────────────────

class StatsSidebar(Vertical):
    """Statistics panel for all datasets."""

    def __init__(self, stats_list: list[dict], labels: list[str],
                 activation: str, precision: str, **kw):
        super().__init__(**kw)
        self._stats = stats_list
        self._labels = labels
        self._act = activation
        self._prec = precision

    def compose(self) -> ComposeResult:
        yield Label(f" {self._act.upper()} ({self._prec}) ", id="title-label")

        def fmt(v: float, spec: str = ".2e") -> str:
            if v < 0.01 or v >= 1000:
                return format(v, ".2e")
            elif v >= 1:
                return format(v, ".1f")
            else:
                return format(v, ".3f")

        # Stats for each dataset
        for i, (stats, label) in enumerate(zip(self._stats, self._labels)):
            color = COLOR_NAMES[i % len(COLOR_NAMES)]
            yield Static(
                f"[{color}]{label}[/]\n"
                f"Points  : {stats['n_points']}\n"
                f"Max ULP : {fmt(stats['max_ulp'])}\n"
                f"Mean ULP: {fmt(stats['mean_ulp'])}\n"
                f"p95 ULP : {fmt(stats['p95_ulp'])}\n"
                f"MAE     : {fmt(stats['mae'])}\n"
                f"Max Abs : {fmt(stats['max_abs'])}",
                id=f"stats{i}-box",
            )

        # Find winner
        if len(self._stats) >= 2:
            best_ulp_idx = min(range(len(self._stats)),
                              key=lambda i: self._stats[i]['max_ulp'])
            best_mae_idx = min(range(len(self._stats)),
                              key=lambda i: self._stats[i]['mae'])

            yield Static(
                f"[cyan]Winner[/]\n"
                f"Best ULP: {self._labels[best_ulp_idx]}\n"
                f"Best MAE: {self._labels[best_mae_idx]}",
                id="winner-box",
            )

        yield Label(" Legend ", id="legend-label")
        yield Static(
            "[cyan]--- 1 ULP 'great'[/]\n"
            "[yellow]--- 3 ULP 'accurate'[/]\n"
            "[red]--- 10 ULP 'approx'[/]",
            id="legend-box",
        )


# ── Main Application ───────────────────────────────────────────────────────────

class ErrorCompareApp(App):
    """Hardware Error Comparison TUI"""

    CSS = """
    Screen { layout: horizontal; }
    #sidebar {
        width: 30;
        border-right: thick $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    #title-label {
        text-align: center;
        color: $accent;
        text-style: bold;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    #legend-label {
        text-align: center;
        color: $secondary;
        text-style: bold;
        padding: 1 0 0 0;
        border-top: solid $primary-darken-2;
    }
    Static {
        border: round $secondary;
        padding: 0 1;
        margin: 1 0 0 0;
        background: $surface;
    }
    #main-area { width: 1fr; height: 1fr; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    PlotextPlot { height: 1fr; width: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "goto_tab('tab-ulp')", "ULP", show=True),
        Binding("2", "goto_tab('tab-abs')", "Abs", show=True),
        Binding("3", "goto_tab('tab-rel')", "Rel", show=True),
        Binding("4", "goto_tab('tab-out')", "Output", show=True),
    ]

    TITLE = "Hardware Error Comparison"

    def __init__(self, datasets: list[dict], stats_list: list[dict],
                 labels: list[str], activation: str, precision: str,
                 arch: str = None):
        super().__init__()
        self._datasets = datasets
        self._stats = stats_list
        self._labels = labels
        self._act = activation
        self._prec = precision
        n = len(datasets)
        arch_str = f"{arch} | " if arch else ""
        self.sub_title = f"{arch_str}{activation.upper()} | {precision.upper()} | {n} datasets | q=quit 1-4=tabs"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield StatsSidebar(
                self._stats, self._labels,
                self._act, self._prec,
                id="sidebar",
            )
            with Vertical(id="main-area"):
                with TabbedContent(id="tabs"):
                    with TabPane("ULP Error [1]", id="tab-ulp"):
                        yield ULPComparisonPlot(self._datasets, self._labels)
                    with TabPane("Abs Error [2]", id="tab-abs"):
                        yield AbsErrorComparisonPlot(self._datasets, self._labels)
                    with TabPane("Rel Error [3]", id="tab-rel"):
                        yield RelErrorComparisonPlot(self._datasets, self._labels)
                    with TabPane("Output [4]", id="tab-out"):
                        yield OutputComparisonPlot(self._datasets, self._labels)
        yield Footer()

    def action_goto_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare hardware output CSVs in a terminal TUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:

  1. Auto-discovery mode (like plot_hardware_error_analysis.py):
     Automatically finds SFPU fast/precise and LUT CSVs based on arch/activation/precision.

     python3 tui_error_compare.py --arch blackhole --activation gelu
     python3 tui_error_compare.py --arch blackhole --activation gelu --precision bf16
     python3 tui_error_compare.py --arch wormhole -a sigmoid -p fp32

  2. Manual CSV mode:
     Provide 2-3 CSV files explicitly.

     python3 tui_error_compare.py csv1.csv csv2.csv --activation gelu
     python3 tui_error_compare.py fast.csv precise.csv lut.csv -a gelu
     python3 tui_error_compare.py a.csv b.csv c.csv -a sigmoid \\
         --label1 "SFPU Fast" --label2 "SFPU Precise" --label3 "LUT"

OPTIONS:

  --arch         Architecture (wormhole, blackhole) - enables auto-discovery
  --activation   Activation function (required)
  --precision    bf16 or fp32 (default: bf16)
  --data-dir     Base directory for auto-discovery (default: data/hardware_outputs)
  --label1/2/3   Custom labels for datasets

Available activations:
    gelu, sigmoid, tanh, relu, leaky_relu, elu, selu, softplus,
    sin, cos, exp, exp2, erf, hardsigmoid, cosh, sinh, atanh, atan, abs, identity

Keyboard shortcuts:
    q     - Quit
    1     - ULP Error tab
    2     - Absolute Error tab
    3     - Relative Error tab
    4     - Output comparison tab

The plots automatically adapt to terminal width - resize your terminal for more detail!
        """
    )

    parser.add_argument('csvs', nargs='*', help='CSV files (2-3 files with input,output columns)')
    parser.add_argument('--arch', type=str, choices=['wormhole', 'blackhole'],
                       help='Architecture (enables auto-discovery mode)')
    parser.add_argument('-a', '--activation', required=True,
                       help='Activation function for reference computation')
    parser.add_argument('-p', '--precision', default='bf16',
                       choices=['bf16', 'fp32'],
                       help='Precision for ULP computation (default: bf16)')
    parser.add_argument('--data-dir', type=str, default='data/hardware_outputs',
                       help='Base directory for auto-discovery (default: data/hardware_outputs)')
    parser.add_argument('--label1', default=None, help='Label for first dataset')
    parser.add_argument('--label2', default=None, help='Label for second dataset')
    parser.add_argument('--label3', default=None, help='Label for third dataset')

    args = parser.parse_args()

    csv_paths = []
    labels = []
    arch = args.arch

    # Determine mode: auto-discovery vs manual
    if args.arch and not args.csvs:
        # Auto-discovery mode
        data_dir = Path(args.data_dir)
        print(f"Auto-discovering CSVs in {data_dir}/{args.arch}/{args.activation}/...")

        discovered = discover_csvs(data_dir, args.arch, args.activation, args.precision)

        if not discovered:
            print(f"Error: No CSVs found for {args.arch}/{args.activation}/{args.precision}", file=sys.stderr)
            print(f"Looking in: {data_dir / args.arch / args.activation}", file=sys.stderr)
            sys.exit(1)

        for path, label in discovered:
            csv_paths.append(path)
            labels.append(label)
            print(f"  Found: {path.name} -> {label}")

    elif args.csvs:
        # Manual mode
        if len(args.csvs) < 2:
            parser.error("At least 2 CSV files required in manual mode")
        if len(args.csvs) > 3:
            parser.error("Maximum 3 CSV files supported")

        csv_paths = [Path(p) for p in args.csvs]
        for p in csv_paths:
            if not p.exists():
                print(f"Error: CSV not found: {p}", file=sys.stderr)
                sys.exit(1)

        # Labels
        label_args = [args.label1, args.label2, args.label3]
        for i, p in enumerate(csv_paths):
            if i < len(label_args) and label_args[i]:
                label = label_args[i]
            else:
                label = p.stem
            # Truncate long labels
            if len(label) > 15:
                label = label[:12] + "..."
            labels.append(label)
    else:
        parser.error("Either provide --arch for auto-discovery or provide CSV files manually")

    # Apply custom labels if provided (override auto-discovered labels)
    label_args = [args.label1, args.label2, args.label3]
    for i in range(len(labels)):
        if i < len(label_args) and label_args[i]:
            labels[i] = label_args[i]
            if len(labels[i]) > 15:
                labels[i] = labels[i][:12] + "..."

    # Load and process data (with caching)
    datasets = []
    stats_list = []

    for i, p in enumerate(csv_paths):
        print(f"Loading {p}...")

        # Use cached errors if available
        error_data = load_or_compute_errors(p, args.activation, args.precision)
        if error_data is None:
            print(f"  Warning: Failed to load/compute errors for {p}")
            continue

        # Load raw outputs for OutputComparisonPlot
        inputs_raw, outputs_raw = load_hardware_outputs(p)
        ref_raw = compute_reference(inputs_raw, args.activation)

        # Build data dict expected by TUI widgets
        data = {
            'inputs': error_data['inputs'],
            'outputs': outputs_raw,  # Raw outputs for output comparison plot
            'reference': ref_raw,    # Raw reference for output comparison plot
            'abs_err': error_data['abs_errors'],
            'rel_err': error_data['rel_errors'],
            'ulp_err': error_data['ulp_errors'],
            'normal_mask': np.ones(len(error_data['inputs']), dtype=bool),  # Already filtered
        }
        stats = compute_stats(data)

        datasets.append(data)
        stats_list.append(stats)

        print(f"  {labels[i]}: Max ULP = {stats['max_ulp']:.1f}, MAE = {stats['mae']:.2e}")

    print("\nStarting TUI... (press 'q' to quit, resize terminal for more detail)")

    ErrorCompareApp(
        datasets, stats_list, labels,
        args.activation, args.precision,
        arch=arch
    ).run()


if __name__ == "__main__":
    main()
