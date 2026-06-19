#!/usr/bin/env python3
"""
Generate consolidated PDF reports for ULP error analysis and runtime plots.

Creates separate PDFs by compiling PNG files:
- ulp_report_<dtype>.pdf: All hardware_error_analysis ULP plots
- runtime_report_<dtype>.pdf: All error_vs_runtime plots

Usage:
    python generate_report.py                    # Generate both bf16 and fp32 reports
    python generate_report.py --dtype bf16       # Generate bf16 reports only
    python generate_report.py --activations gelu sigmoid  # Specific activations
"""

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image


def collect_plot_files(plots_dir: Path, dtype: str = "bf16") -> dict:
    """
    Collect all ULP and runtime plot PNG files.

    Returns dict with keys 'ulp' and 'runtime', each containing sorted list of paths.
    """
    hardware_dir = plots_dir / "hardware_error_analysis"
    runtime_dir = plots_dir / "error_vs_runtime"

    ulp_files = []
    runtime_files = []

    # Collect ULP plots
    if hardware_dir.exists():
        for act_dir in sorted(hardware_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            ulp_plot = act_dir / "ulp_only" / f"{act_dir.name}_{dtype}_ulp.png"
            if ulp_plot.exists():
                ulp_files.append(ulp_plot)

    # Collect runtime plots
    if runtime_dir.exists():
        for act_dir in sorted(runtime_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            runtime_plot = act_dir / "standalone" / "ulp_max" / f"yolov4_{dtype}.png"
            if runtime_plot.exists():
                runtime_files.append(runtime_plot)

    return {'ulp': ulp_files, 'runtime': runtime_files}


def compile_pdf(png_files: list[Path], output_path: Path) -> bool:
    """
    Compile a list of PNG files into a single PDF.

    Each PNG becomes one page in the PDF.

    Args:
        png_files: List of PNG file paths
        output_path: Output PDF path

    Returns:
        True if successful, False otherwise
    """
    if not png_files:
        return False

    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            # Convert to RGB if necessary (PDF doesn't support RGBA)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"  Warning: Failed to load {png_file.name}: {e}")

    if not images:
        return False

    # Save as multi-page PDF
    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=300)
    return True


def find_activations(plots_dir: Path, dtype: str = "bf16") -> list[str]:
    """Find all activation functions that have plots available."""
    hardware_dir = plots_dir / "hardware_error_analysis"

    activations = []
    if hardware_dir.exists():
        for act_dir in sorted(hardware_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            ulp_plot = act_dir / "ulp_only" / f"{act_dir.name}_{dtype}_ulp.png"
            if ulp_plot.exists():
                activations.append(act_dir.name)

    return activations


def filter_files_by_activations(files: list[Path], activations: list[str]) -> list[Path]:
    """Filter file list to only include specified activations."""
    filtered = []
    for f in files:
        # Extract activation name from path
        # ULP: hardware_error_analysis/<act>/ulp_only/<act>_bf16_ulp.png
        # Runtime: error_vs_runtime/<act>/standalone/ulp_max/yolov4_bf16.png
        parts = f.parts
        for i, part in enumerate(parts):
            if part in ('hardware_error_analysis', 'error_vs_runtime') and i + 1 < len(parts):
                act_name = parts[i + 1]
                if act_name in activations:
                    filtered.append(f)
                break
    return filtered


def create_reports(
    plots_dir: Path,
    output_dir: Path,
    activations: Optional[list[str]] = None,
    dtype: str = "bf16"
) -> list[Path]:
    """
    Create consolidated PDF reports for ULP and runtime plots.

    Args:
        plots_dir: Path to plots directory (e.g., plots/blackhole)
        output_dir: Directory for output PDFs
        activations: List of activations to include (None = all available)
        dtype: Data type (bf16 or fp32)

    Returns:
        List of created PDF paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all plot files
    files = collect_plot_files(plots_dir, dtype)

    # Filter by activations if specified
    if activations:
        files['ulp'] = filter_files_by_activations(files['ulp'], activations)
        files['runtime'] = filter_files_by_activations(files['runtime'], activations)

    created_pdfs = []

    # Create ULP report
    if files['ulp']:
        ulp_pdf = output_dir / f"ulp_report_{dtype}.pdf"
        print(f"Compiling {len(files['ulp'])} ULP plots...")
        if compile_pdf(files['ulp'], ulp_pdf):
            created_pdfs.append(ulp_pdf)
            print(f"  Created: {ulp_pdf} ({len(files['ulp'])} pages)")
    else:
        print(f"  No ULP plots found for {dtype}")

    # Create runtime report
    if files['runtime']:
        runtime_pdf = output_dir / f"runtime_report_{dtype}.pdf"
        print(f"Compiling {len(files['runtime'])} runtime plots...")
        if compile_pdf(files['runtime'], runtime_pdf):
            created_pdfs.append(runtime_pdf)
            print(f"  Created: {runtime_pdf} ({len(files['runtime'])} pages)")
    else:
        print(f"  No runtime plots found for {dtype}")

    return created_pdfs


def main():
    parser = argparse.ArgumentParser(
        description="Generate consolidated PDF reports for ULP and runtime plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports for all activations (both bf16 and fp32)
  python generate_report.py

  # Generate reports for specific activations
  python generate_report.py --activations gelu sigmoid tanh

  # Generate bf16 reports only
  python generate_report.py --dtype bf16

  # Custom output directory
  python generate_report.py -o my_reports/

Output:
  Creates separate PDFs in the output directory:
  - ulp_report_bf16.pdf / ulp_report_fp32.pdf
  - runtime_report_bf16.pdf / runtime_report_fp32.pdf
"""
    )

    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).parent / "plots" / "blackhole",
        help="Path to plots directory (default: plots/blackhole)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PDFs (default: reports/)"
    )

    parser.add_argument(
        "--activations", "-a",
        nargs="+",
        default=None,
        help="Specific activations to include (default: all available)"
    )

    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32", "both"],
        default="both",
        help="Data type for plots (default: both)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available activations and exit"
    )

    args = parser.parse_args()

    plots_dir = args.plots_dir.resolve()

    if not plots_dir.exists():
        print(f"Error: Plots directory not found: {plots_dir}")
        return 1

    if args.list:
        for dtype in ["bf16", "fp32"]:
            activations = find_activations(plots_dir, dtype)
            print(f"Available activations for {dtype} ({len(activations)}):")
            for act in activations:
                print(f"  - {act}")
            print()
        return 0

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "reports"
    else:
        output_dir = args.output_dir.resolve()

    # Determine which dtypes to generate
    if args.dtype == "both":
        dtypes = ["bf16", "fp32"]
    else:
        dtypes = [args.dtype]

    print("=" * 60)
    print("CONSOLIDATED PDF REPORT GENERATOR")
    print("=" * 60)
    print(f"Plots directory: {plots_dir}")
    print(f"Output directory: {output_dir}")
    print()

    all_pdfs = []
    for dtype in dtypes:
        print(f"Processing {dtype.upper()}...")
        pdfs = create_reports(
            plots_dir=plots_dir,
            output_dir=output_dir,
            activations=args.activations,
            dtype=dtype
        )
        all_pdfs.extend(pdfs)
        print()

    print("=" * 60)
    print(f"Generated {len(all_pdfs)} PDF file(s)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
