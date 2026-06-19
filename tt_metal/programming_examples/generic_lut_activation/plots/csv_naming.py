"""
Shared CSV naming helpers for hardware output files.

This module provides canonical naming functions for hardware output CSVs
that match the bash helper functions in sweep_helpers.sh.

Canonical formats:
- Polynomial: {activation}_{precision}_p{degree}_s{segments}_{segmentation}_{fitting}_tiles{count}.csv
- Rational: {activation}_{precision}_n{num}d{den}_s{segments}_{segmentation}_rational_tiles{count}.csv
"""


def degree_to_safe(degree):
    """Convert degree string to filesystem-safe form.

    Args:
        degree: Polynomial degree (str or int) or rational "num/den" string

    Returns:
        Safe string: "p{N}" for polynomial or "n{N}d{D}" for rational
    """
    degree_str = str(degree)
    if '/' in degree_str:
        num, den = degree_str.split('/')
        return f'n{num}d{den}'
    return str(degree)


def build_hw_poly_csv_filename(activation, precision, degree, segments, segmentation, fitting, tile_count):
    """Build hardware output CSV filename (polynomial).

    Canonical format: {activation}_{precision}_p{degree}_s{segments}_{segmentation}_{fitting}_tiles{count}.csv

    Args:
        activation: Activation function name (e.g., "gelu")
        precision: Data precision ("fp32" or "bf16")
        degree: Polynomial degree (int or str)
        segments: Number of segments (int or str)
        segmentation: Segmentation type ("uniform" or "adaptive")
        fitting: Fitting method ("lsq", "any", etc.)
        tile_count: Number of tiles (int or str)

    Returns:
        Canonical CSV filename string
    """
    return f'{activation}_{precision}_p{degree}_s{segments}_{segmentation}_{fitting}_tiles{tile_count}.csv'


def build_hw_rational_csv_filename(activation, precision, num_deg, den_deg, segments, segmentation, tile_count):
    """Build hardware output CSV filename (rational).

    Canonical format: {activation}_{precision}_n{num}d{den}_s{segments}_{segmentation}_rational_tiles{count}.csv

    Args:
        activation: Activation function name (e.g., "gelu")
        precision: Data precision ("fp32" or "bf16")
        num_deg: Numerator degree (int or str)
        den_deg: Denominator degree (int or str)
        segments: Number of segments (int or str)
        segmentation: Segmentation type ("uniform" or "adaptive")
        tile_count: Number of tiles (int or str)

    Returns:
        Canonical CSV filename string
    """
    return f'{activation}_{precision}_n{num_deg}d{den_deg}_s{segments}_{segmentation}_rational_tiles{tile_count}.csv'


def build_hw_poly_csv_pattern(activation, precision, degree, segments, segmentation, fitting):
    """Build hardware output CSV filename pattern (polynomial) for glob matching.

    Canonical format: {activation}_{precision}_p{degree}_s{segments}_{segmentation}_{fitting}_tiles*.csv

    Args:
        activation: Activation function name (e.g., "gelu")
        precision: Data precision ("fp32" or "bf16")
        degree: Polynomial degree (int or str)
        segments: Number of segments (int or str)
        segmentation: Segmentation type - ignored, uses wildcard (uniform, curvature, chebyshev, recursive_ulp)
        fitting: Fitting method - ignored, uses wildcard (lsq, any, adhoc, etc.)

    Returns:
        Glob pattern string for matching CSV files
    """
    return f'{activation}_{precision}_p{degree}_s{segments}_*_tiles*.csv'


def build_hw_rational_csv_pattern(activation, precision, num_deg, den_deg, segments, segmentation):
    """Build hardware output CSV filename pattern (rational) for glob matching.

    Canonical format: {activation}_{precision}_n{num}d{den}_s{segments}_{segmentation}_rational_tiles*.csv

    Args:
        activation: Activation function name (e.g., "gelu")
        precision: Data precision ("fp32" or "bf16")
        num_deg: Numerator degree (int or str)
        den_deg: Denominator degree (int or str)
        segments: Number of segments (int or str)
        segmentation: Segmentation type - ignored, uses wildcard

    Returns:
        Glob pattern string for matching CSV files
    """
    return f'{activation}_{precision}_n{num_deg}d{den_deg}_s{segments}_*_tiles*.csv'


def parse_hw_csv_filename(filename):
    """Parse a canonical hardware output CSV filename.

    Args:
        filename: CSV filename (with or without path)

    Returns:
        dict with parsed components or None if parsing fails

    Example:
        >>> parse_hw_csv_filename('gelu_fp32_p7_s16_uniform_any_tiles256.csv')
        {'activation': 'gelu', 'precision': 'fp32', 'type': 'polynomial',
         'degree': 7, 'segments': 16, 'segmentation': 'uniform',
         'fitting': 'any', 'tile_count': 256}
    """
    import os
    import re

    basename = os.path.basename(filename)

    # Try polynomial pattern first
    poly_pattern = r'^([a-z_]+)_(fp32|bf16)_p(\d+)_s(\d+)_([a-z]+)_([a-z]+)_tiles(\d+)\.csv$'
    match = re.match(poly_pattern, basename)
    if match:
        return {
            'activation': match.group(1),
            'precision': match.group(2),
            'type': 'polynomial',
            'degree': int(match.group(3)),
            'segments': int(match.group(4)),
            'segmentation': match.group(5),
            'fitting': match.group(6),
            'tile_count': int(match.group(7))
        }

    # Try rational pattern
    rat_pattern = r'^([a-z_]+)_(fp32|bf16)_n(\d+)d(\d+)_s(\d+)_([a-z]+)_rational_tiles(\d+)\.csv$'
    match = re.match(rat_pattern, basename)
    if match:
        return {
            'activation': match.group(1),
            'precision': match.group(2),
            'type': 'rational',
            'num_degree': int(match.group(3)),
            'den_degree': int(match.group(4)),
            'segments': int(match.group(5)),
            'segmentation': match.group(6),
            'fitting': 'rational',
            'tile_count': int(match.group(7))
        }

    return None


def get_hardware_output_dir(arch, activation, base_dir='data/hardware_outputs'):
    """Get hardware output directory path.

    Args:
        arch: Architecture name (e.g., "blackhole", "wormhole_b0")
        activation: Activation function name
        base_dir: Base directory for hardware outputs

    Returns:
        Path string to hardware output directory
    """
    # Strip _b0 suffix for cleaner paths
    arch_short = arch.replace('_b0', '')
    return f'{base_dir}/{arch_short}/{activation}'
