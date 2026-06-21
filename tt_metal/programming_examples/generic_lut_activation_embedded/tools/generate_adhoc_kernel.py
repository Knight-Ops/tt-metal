#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unified adhoc kernel generator for polynomial and rational approximations.

This script generates a single kernel file with embedded LUT constants,
eliminating the need for pre-generated kernel wrappers. Used by sweep scripts
to dynamically generate kernels at runtime.

Supports:
  - Polynomial: piecewise_generic.cpp (degrees 0-16)
  - Rational: piecewise_rational.cpp (num/den combinations)

Usage:
  # From coefficient CSV file
  python3 generate_adhoc_kernel.py --csv coeffs.csv --output adhoc.cpp

  # Polynomial from polynomial fitter coefficients directory
  python3 generate_adhoc_kernel.py --activation sigmoid --degree 4 --segments 16 \\
      --segmentation uniform --output adhoc.cpp

  # Rational approximation
  python3 generate_adhoc_kernel.py --activation exp --num-degree 2 --den-degree 2 \\
      --segments 4 --segmentation uniform --output adhoc.cpp

  # With custom coefficient directory
  python3 generate_adhoc_kernel.py --activation sigmoid --degree 4 --segments 16 \\
      --coeff-dir /path/to/coefficients --output adhoc.cpp
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Polynomial degree mapping (single source of truth)
# ============================================================================

POLY_DEGREE_MAP: Dict[int, str] = {
    0: "constant",
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
    5: "quintic",
    6: "hexic",
    7: "septic",
    8: "octic",
    9: "nonic",
    10: "decic",
    11: "undecic",
    12: "dodecic",
    13: "tredecic",
    14: "tetradecic",
    15: "pentadecic",
    16: "hexadecic",
    32: "duotrigesic",
}

POLY_METHOD_MAP: Dict[str, int] = {v: k for k, v in POLY_DEGREE_MAP.items()}


# ============================================================================
# Utility functions
# ============================================================================


def clamp_float32(value: float) -> float:
    """Clamp values to float32 representable range."""
    MAX_FLOAT32 = 3.4028234663852886e38
    MIN_FLOAT32_DENORM = 1.4e-45

    if abs(value) > MAX_FLOAT32:
        return MAX_FLOAT32 if value > 0 else -MAX_FLOAT32
    if abs(value) < MIN_FLOAT32_DENORM:
        return 0.0
    return value


def detect_segment_degree(coefficients: List[float], max_degree: int) -> int:
    """Detect actual polynomial degree by finding highest non-zero coefficient."""
    for deg in range(max_degree, -1, -1):
        if coefficients[deg] != 0.0:
            return deg
    return 0


def detect_affine_collapse(lut_info: Dict) -> Dict:
    """Detect when the WHOLE fit collapses to a single affine map y = c0 + c1*x.

    Generic (not hardcoded per-activation): a 1-segment polynomial whose effective
    degree is <= 1 with NO range reduction is exactly c0 + c1*x over the domain.
    Two sub-cases:
      - identity:  c0 == 0 and c1 == 1  -> y = x  (emit a PURE COPY, skip the SFPU
        eval entirely: copy_tile already places x in dst, pack stores it).
      - affine:    otherwise            -> y = c0 + c1*x (one SFPMAD per element).

    Any activation whose fit reduces to these shapes qualifies (abs does NOT — it
    is a 2-segment sign split, handled elsewhere). Returns {} when not applicable.
    """
    metadata = lut_info.get("metadata", {})
    method = str(metadata.get("range_reduction_method", "none") or "none").strip()
    if method not in ("", "none"):
        return {}  # range reduction means the poly is on a reduced domain, not affine in x
    if lut_info.get("num_segments", 0) != 1:
        return {}
    raw = lut_info.get("raw_coefficients") or []
    if len(raw) != 1:
        return {}
    coeffs = raw[0]
    # Effective degree must be <= 1 (all higher-order coeffs exactly zero).
    if any(c != 0.0 for c in coeffs[2:]):
        return {}
    c0 = float(coeffs[0]) if len(coeffs) >= 1 else 0.0
    c1 = float(coeffs[1]) if len(coeffs) >= 2 else 0.0
    is_identity = c0 == 0.0 and c1 == 1.0
    return {"c0": c0, "c1": c1, "identity": is_identity}


def get_affine_collapse_macros(lut_info: Dict) -> str:
    """Emit AFFINE_COLLAPSE / AFFINE_IDENTITY macros when the fit is affine in x."""
    info = detect_affine_collapse(lut_info)
    if not info:
        return ""
    if info["identity"]:
        print("AFFINE COLLAPSE: fit is identity (c0=0, c1=1) -> pure-copy bypass (no SFPU eval)")
        return (
            "\n// eval_method: affine_collapse / identity. fit is y = x -> pure tile copy (no SFPU eval).\n"
            "#define EVAL_METHOD_AFFINE_COLLAPSE\n#define AFFINE_COLLAPSE\n#define AFFINE_IDENTITY\n"
        )
    print(f"AFFINE COLLAPSE: fit is y = {info['c0']:.6g} + {info['c1']:.6g}*x -> single SFPMAD bypass")
    return (
        "\n// eval_method: affine_collapse. fit is y = c0 + c1*x over the whole domain. One SFPMAD.\n"
        "#define EVAL_METHOD_AFFINE_COLLAPSE\n"
        "#define AFFINE_COLLAPSE\n"
        f"#define AFFINE_C0 {clamp_float32(info['c0']):.10e}f\n"
        f"#define AFFINE_C1 {clamp_float32(info['c1']):.10e}f\n"
    )


def format_lut_array(values: List[float], indent: int = 4) -> str:
    """Format a list of floats as a C++ array literal."""
    lines = []
    for i in range(0, len(values), 6):
        chunk = values[i : i + 6]
        line = ", ".join(f"{clamp_float32(v):.10e}f" for v in chunk)
        lines.append(" " * indent + line)
    return ",\n".join(lines)


# ============================================================================
# CSV parsing functions
# ============================================================================


def parse_csv_metadata(csv_path: Path) -> Dict[str, str]:
    """Extract METADATA rows from coefficient CSV."""
    metadata = {}
    if not csv_path.exists():
        return metadata

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("segment_id", "").upper() == "METADATA":
                key = row.get("lo", "")
                value = row.get("hi", "")
                if key:
                    metadata[key] = value
    return metadata


def parse_polynomial_csv(csv_path: Path, degree: int) -> Dict:
    """
    Parse polynomial coefficient CSV.

    CSV format: segment_id,lo,hi,c0,c1,c2,...,error,method[,is_asymptotic,dominant_factor]
    LUT format: [boundaries..., coefficients...]
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Coefficient CSV not found: {csv_path}")

    boundaries = []
    coefficients = []
    segment_degrees = []
    asymptotic_flags = []  # per-segment: True if asymptotic
    dominant_factors = []  # per-segment: dominant factor string or ""

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("segment_id", "").upper() == "METADATA":
                continue

            # Boundaries
            if len(boundaries) == 0:
                boundaries.append(float(row["lo"]))
            boundaries.append(float(row["hi"]))

            # Coefficients c0, c1, ..., c{degree}
            seg_coeffs = []
            for j in range(degree + 1):
                coeff_key = f"c{j}"
                if coeff_key in row:
                    seg_coeffs.append(float(row[coeff_key]))
                else:
                    raise ValueError(f"Missing coefficient {coeff_key}")
            coefficients.extend(seg_coeffs)
            segment_degrees.append(detect_segment_degree(seg_coeffs, degree))

            # Asymptotic factoring metadata
            is_asym = row.get("is_asymptotic", "").strip().lower() == "true"
            dom_factor = row.get("dominant_factor", "").strip() if is_asym else ""
            asymptotic_flags.append(is_asym)
            dominant_factors.append(dom_factor)

    lut_values = boundaries + coefficients
    metadata = parse_csv_metadata(csv_path)

    # Store raw per-segment coefficient lists for parity detection
    raw_coefficients = []
    for i in range(0, len(coefficients), degree + 1):
        raw_coefficients.append(coefficients[i : i + degree + 1])

    return {
        "input_min": boundaries[0],
        "input_max": boundaries[-1],
        "num_segments": len(boundaries) - 1,
        "lut_size": len(lut_values),
        "lut_data": format_lut_array(lut_values),
        "segment_degrees": segment_degrees,
        "raw_coefficients": raw_coefficients,
        "metadata": metadata,
        "asymptotic_flags": asymptotic_flags,
        "dominant_factors": dominant_factors,
        "boundaries": boundaries,
    }


def parse_rational_csv(csv_path: Path, num_degree: int, den_degree: int) -> Dict:
    """
    Parse rational coefficient CSV.

    CSV format: segment_id,lo,hi,n0,n1,...,d0,d1,...,error,method
    LUT format: [boundaries..., (num_coeffs + den_coeffs per segment)...]
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Rational coefficient CSV not found: {csv_path}")

    segments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("segment_id", "").upper() == "METADATA":
                continue
            segments.append(row)

    if not segments:
        raise ValueError(f"No segments found in {csv_path}")

    # Boundaries
    boundaries = [float(seg["lo"]) for seg in segments]
    boundaries.append(float(segments[-1]["hi"]))

    # Coefficients: per segment interleaved num then den
    all_coeffs = []
    for seg in segments:
        # Numerator: n0, n1, ..., n{num_degree}
        for i in range(num_degree + 1):
            all_coeffs.append(float(seg[f"n{i}"]))
        # Denominator: d0, d1, ..., d{den_degree}
        for i in range(den_degree + 1):
            all_coeffs.append(float(seg[f"d{i}"]))

    lut_values = boundaries + all_coeffs
    metadata = parse_csv_metadata(csv_path)

    return {
        "input_min": boundaries[0],
        "input_max": boundaries[-1],
        "num_segments": len(segments),
        "num_degree": num_degree,
        "den_degree": den_degree,
        "lut_size": len(lut_values),
        "lut_data": format_lut_array(lut_values),
        "metadata": metadata,
        "raw_segments": segments,
    }


def parse_generic_csv(csv_path: Path) -> Tuple[str, Dict]:
    """
    Auto-detect CSV type (polynomial vs rational) and parse accordingly.

    Returns: (kernel_type, lut_info)
        kernel_type: "polynomial" or "rational"
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    # Detect type from column names
    has_numerator = any(h.startswith("n") and h[1:].isdigit() for h in headers)
    has_denominator = any(h.startswith("d") and h[1:].isdigit() for h in headers)
    has_polynomial = any(h.startswith("c") and h[1:].isdigit() for h in headers)

    if has_numerator and has_denominator:
        # Rational: detect degrees from columns
        num_cols = sorted([h for h in headers if h.startswith("n") and h[1:].isdigit()], key=lambda h: int(h[1:]))
        den_cols = sorted([h for h in headers if h.startswith("d") and h[1:].isdigit()], key=lambda h: int(h[1:]))
        num_degree = len(num_cols) - 1
        den_degree = len(den_cols) - 1
        print(f"Auto-detected: rational approximation n{num_degree}d{den_degree}")
        return "rational", parse_rational_csv(csv_path, num_degree, den_degree)

    elif has_polynomial:
        # Polynomial: detect degree from columns
        coeff_cols = sorted([h for h in headers if h.startswith("c") and h[1:].isdigit()], key=lambda h: int(h[1:]))
        degree = len(coeff_cols) - 1
        print(f"Auto-detected: polynomial degree {degree}")
        return "polynomial", parse_polynomial_csv(csv_path, degree)

    else:
        raise ValueError(f"Cannot determine CSV type. Expected c0,c1,... or n0,n1,.../d0,d1,... columns")


# ============================================================================
# Kernel generation functions
# ============================================================================


def detect_parity_from_coefficients(
    segments: List[Dict],
    num_degree: int,
    den_degree: int,
    threshold: float = 1e-30,
) -> Tuple[str, str]:
    """Detect parity by inspecting actual coefficient values across all segments.

    Returns (num_parity, den_parity) where each is "odd", "even", or "none".
    """
    num_even_all_zero = True
    num_odd_all_zero = True
    for seg in segments:
        for i in range(num_degree + 1):
            val = abs(float(seg[f"n{i}"]))
            if i % 2 == 0 and val > threshold:
                num_even_all_zero = False
            if i % 2 == 1 and val > threshold:
                num_odd_all_zero = False

    den_even_all_zero = True
    den_odd_all_zero = True
    for seg in segments:
        for i in range(den_degree + 1):
            val = abs(float(seg[f"d{i}"]))
            if i % 2 == 0 and val > threshold:
                den_even_all_zero = False
            if i % 2 == 1 and val > threshold:
                den_odd_all_zero = False

    num_parity = "odd" if num_even_all_zero else ("even" if num_odd_all_zero else "none")
    den_parity = "odd" if den_even_all_zero else ("even" if den_odd_all_zero else "none")
    return num_parity, den_parity


def detect_poly_parity_from_coefficients(
    segments_coeffs: List[List[float]],
    degree: int,
    threshold: float = 1e-30,
) -> str:
    """Detect polynomial parity from actual coefficient values.

    Returns "odd", "even", or "none".
    """
    even_all_zero = True
    odd_all_zero = True
    for seg in segments_coeffs:
        for i in range(degree + 1):
            if i % 2 == 0 and abs(seg[i]) > threshold:
                even_all_zero = False
            if i % 2 == 1 and abs(seg[i]) > threshold:
                odd_all_zero = False
    if even_all_zero:
        return "odd"
    if odd_all_zero:
        return "even"
    return "none"


def get_poly_parity_macros(lut_info: Dict, degree: int) -> str:
    """Generate POLY_PARITY_ODD/EVEN macros from actual coefficients.

    When a polynomial has parity structure (e.g., sin/tanh have odd parity,
    cos/cosh have even parity), half the coefficients are zero. With parity
    macros, the kernel evaluates in x² basis with stride-2 coefficient access,
    halving the Horner step count.
    """
    raw_coeffs = lut_info.get("raw_coefficients")
    if not raw_coeffs or degree < 2:
        return ""

    parity = detect_poly_parity_from_coefficients(raw_coeffs, degree)

    # Cross-check with metadata if present
    metadata = lut_info.get("metadata", {})
    meta_parity = metadata.get("poly_parity", "")
    if meta_parity and meta_parity != parity:
        print(f"WARNING: metadata says poly_parity={meta_parity} but coefficients show {parity}")

    if parity == "odd":
        print(f"Polynomial parity: ODD (c0=c2=c4=...=0) → x²-Horner enabled")
        return "\n// Polynomial parity: odd function (c0=c2=c4=...=0) → x²-Horner\n" "#define POLY_PARITY_ODD\n"
    elif parity == "even":
        print(f"Polynomial parity: EVEN (c1=c3=c5=...=0) → x²-Horner enabled")
        return "\n// Polynomial parity: even function (c1=c3=c5=...=0) → x²-Horner\n" "#define POLY_PARITY_EVEN\n"
    return ""


def get_parity_macros(
    metadata: Dict[str, str], segments: Optional[List[Dict]] = None, num_degree: int = 0, den_degree: int = 0
) -> str:
    """Generate parity macros for x²-Horner optimization.

    When a rational approximation has odd numerator / even denominator parity
    (common for odd functions like atanh, erfinv, silu), half the coefficients
    are zero. With parity macros, the kernel evaluates in x² basis using
    stride-2 coefficient access, halving the Horner step count.

    Parity is verified from actual coefficients (not just metadata).
    """
    # Detect from coefficients if available (ground truth)
    if segments and num_degree > 0:
        num_parity, den_parity = detect_parity_from_coefficients(segments, num_degree, den_degree)
        # Cross-check with metadata if present
        meta_num = metadata.get("num_parity", "")
        meta_den = metadata.get("den_parity", "")
        if meta_num and meta_num != num_parity:
            print(f"WARNING: metadata says num_parity={meta_num} but coefficients show {num_parity}")
        if meta_den and meta_den != den_parity:
            print(f"WARNING: metadata says den_parity={meta_den} but coefficients show {den_parity}")
    else:
        # Fallback to metadata only
        num_parity = metadata.get("num_parity", "none")
        den_parity = metadata.get("den_parity", "none")

    macros = ""
    if num_parity == "odd" and den_parity == "even":
        macros = (
            "\n// Parity optimization: odd numerator, even denominator\n"
            "// P(x) = x·Horner(odd_coeffs, x²), Q(x) = Horner(even_coeffs, x²)\n"
            "#define RATIONAL_NUM_PARITY_ODD\n"
            "#define RATIONAL_DEN_PARITY_EVEN\n"
        )
    elif num_parity == "even" and den_parity == "odd":
        macros = (
            "\n// Parity optimization: even numerator, odd denominator\n"
            "#define RATIONAL_NUM_PARITY_EVEN\n"
            "#define RATIONAL_DEN_PARITY_ODD\n"
        )
    if num_parity != "none" or den_parity != "none":
        print(f"Parity: num={num_parity}, den={den_parity}")
    return macros


def get_hw_exponent_alu_macros(method: str, lut_info: Dict) -> str:
    """Generate hardware-exponent-ALU range reduction macros (exp2 / log2 / pow).

    Contract (/tmp/exponent_alu_contract.md): the fitter tags
    range_reduction_method = exponent_alu_<kind> and emits NATURAL-basis poly
    coefficients on the reduced domain ([0,1) for exp2, [1,2) for log2/pow).
    The kernel owns the exman/exexp/setexp decompose, the 2^-23/2^-46 scale fold
    (exp2), and the recombine. We pull the coefficients from the first segment
    (these backends fit a single low-degree polynomial over the whole reduced
    domain — no piecewise cascade).
    """
    kind = method[len("exponent_alu_") :]
    metadata = lut_info.get("metadata", {})

    # newton_root: magic-seed + Newton, NO poly coeffs -- mirror run_csv.sh, fully metadata-driven.
    if kind == "newton_root":
        magic = metadata.get("newton_root_magic", "0x5f1110a0")
        c1 = float(metadata.get("newton_root_c1", "2.2825186"))
        c2 = float(metadata.get("newton_root_c2", "2.2533049"))
        root_n = int(float(metadata.get("newton_root_n", metadata.get("expalu_root_n", "2")) or "2"))
        recip = str(
            metadata.get("newton_root_reciprocal", metadata.get("expalu_reciprocal", "False"))
        ).strip().lower() in ("true", "1")
        iters = int(float(metadata.get("newton_root_iters", "3") or "3"))
        recip_macro = "#define NEWTON_ROOT_RECIPROCAL\n" if recip else ""
        return (
            "\n// eval_method: newton_root (magic-seed + Newton, NO polynomial fit). FIRST-CLASS\n"
            "// standalone method -- not a 'kind' of exponent_alu. Mirrors native sqrt/rsqrt/cbrt.\n"
            "#define EVAL_METHOD_NEWTON_ROOT\n"
            f"#define NEWTON_ROOT_MAGIC {magic}\n"
            f"#define NEWTON_ROOT_C1 {c1:.10e}f\n"
            f"#define NEWTON_ROOT_C2 {c2:.10e}f\n"
            f"#define NEWTON_ROOT_N {root_n}\n"
            f"#define NEWTON_ROOT_ITERS {iters}\n"
            f"{recip_macro}"
        )

    # Coefficients of the (single) reduced-domain polynomial.
    raw = lut_info.get("raw_coefficients")
    if not raw:
        raise ValueError(f"exponent_alu_{kind}: no polynomial coefficients found in CSV")
    coeffs = raw[0]  # first/only segment carries the reduced-domain fit
    degree = len(coeffs) - 1

    # ---- GENERIC constant-pool hoisting (data-driven, all kinds, any degree) ----
    # Collect every loop-invariant constexpr the kernel reads for this kind, ranked
    # by reuse. Top-3 -> vConstFloatPrgm0/1/2; next HOIST_BUDGET -> pre-loop hoisted
    # vFloat LREGs; anything beyond -> in-body literals, LOGGED (never dropped).
    PRGM_SLOTS, HOIST_BUDGET = 3, 5
    pool = []
    if kind == "exp2":
        pool = ["MULT(1/ln2)"] + [f"c{k}" for k in range(degree, -1, -1)] + ["clamp255"]
    elif kind == "log2":
        pool = ["LOG_HW_SCALE"] + [f"c{k}" for k in range(degree, -1, -1)]
    elif kind == "pow":
        pool = ["SQRT2", "round_magic"] + [f"c{k}" for k in range(degree, -1, -1)]
    spilled = pool[PRGM_SLOTS + HOIST_BUDGET :]
    hw_preload_macro = "#define HW_PRELOAD\n"
    if spilled:
        print(f"HW_PRELOAD constant-pool: {len(pool)} constants, {len(spilled)} spilled to in-body load: {spilled}")
        hw_preload_macro += f"// HW_PRELOAD_SPILL: {len(spilled)} constants spilled to in-body load: {spilled}\n"
    else:
        print(f"HW_PRELOAD constant-pool: {len(pool)} constants, 0 spilled (all in prgm/hoist budget)")

    if kind == "exp2":
        # exp2 expects a degree-N fit g(f)=2^f on [0,1). Emit the full natural
        # coeff array + degree; the kernel normalizes the exman fraction to a
        # float f in [0,1) and runs a plain degree-N Horner (mirrors pow). The
        # fitter also tags the log2-domain multiplier (1.0 -> 2^x, log2e -> e^x,
        # -log2e -> exp(-x)) and an optional compose post-transform.
        mult = metadata.get("expalu_log2_multiplier", "1.4426950408889634")
        compose = metadata.get("expalu_compose", "").strip()
        coeff_str = ", ".join(f"{clamp_float32(v):.10e}f" for v in coeffs)
        compose_macro = ""
        if compose == "sigmoid":
            compose_macro = "#define EXP_HW_COMPOSE_SIGMOID\n"
        elif compose == "minus_one":
            compose_macro = "#define EXP_HW_COMPOSE_MINUS_ONE\n"
        print(f"HW exponent-ALU exp2: degree {degree}, mult {mult}, compose {compose or 'none'}")
        return (
            "\n// eval_method: exponent_alu / exp2 (exman/exexp/setexp). STANDALONE evaluator."
            "\n// Natural [0,1)-basis coeffs for g(f)=2^f; kernel normalizes f then Horner.\n"
            "#define EVAL_METHOD_EXPONENT_ALU\n"
            "#define EXPONENT_ALU_EXP2\n"
            f"#define EXP_HW_MULT {clamp_float32(float(mult)):.10e}f\n"
            f"{compose_macro}"
            f"{hw_preload_macro}"
            f"constexpr uint32_t EXP_HW_DEGREE = {degree};\n"
            f"constexpr float EXP_HW_COEFFS[] = {{{coeff_str}}};\n"
        )

    if kind == "log2":
        # log2 expects h(m)=log2(m) on [1,2). Emit a coeff array + degree + scale.
        scale = metadata.get("expalu_log_scale", metadata.get("log_scale", "1.0"))
        basis = metadata.get("expalu_log2_basis", "natural")
        coeff_str = ", ".join(f"{clamp_float32(v):.10e}f" for v in coeffs)
        basis_macro = "#define LOG_HW_BASIS_M_MINUS_1\n" if basis == "m_minus_1" else ""
        # log1p decomposes (x + 1) before log2 -> expalu_input_offset = 1.0.
        offset = float(metadata.get("expalu_input_offset", "0.0") or "0.0")
        offset_macro = f"#define LOG_HW_INPUT_OFFSET {clamp_float32(offset):.10e}f\n" if offset != 0.0 else ""
        print(f"HW exponent-ALU log2: degree {degree}, scale {scale}, basis {basis}, input_offset {offset}")
        return (
            "\n// eval_method: exponent_alu / log2 (exexp -> e, exman -> m). STANDALONE evaluator."
            "\n// Natural-basis coeffs for h(m)=log2(m); result = (e + h(m + offset)) * scale.\n"
            "#define EVAL_METHOD_EXPONENT_ALU\n"
            "#define EXPONENT_ALU_LOG2\n"
            f"{basis_macro}"
            f"{offset_macro}"
            f"{hw_preload_macro}"
            f"constexpr uint32_t LOG_HW_DEGREE = {degree};\n"
            f"constexpr float LOG_HW_COEFFS[] = {{{coeff_str}}};\n"
            f"#define LOG_HW_SCALE {scale}f\n"
        )

    if kind == "pow":
        # pow expects p(m)=root_N(m) on [1,2). The fitter tags:
        #   expalu_root_n      - root order N (2=sqrt, 3=cbrt)
        #   expalu_reciprocal  - True -> final 1/result (rsqrt)
        #   expalu_pow_scale_c{r} - root_N(2^r) scale constants, r in {0..N-1}
        coeff_str = ", ".join(f"{clamp_float32(v):.10e}f" for v in coeffs)
        root_n = int(float(metadata.get("expalu_root_n", "2") or "2"))
        recip = str(metadata.get("expalu_reciprocal", "False")).strip().lower() in ("true", "1")
        recip_macro = "#define POW_HW_RECIPROCAL\n" if recip else ""
        scale_macros = f"#define POW_HW_ROOT_N {root_n}\n"
        for r in range(root_n):
            key = f"expalu_pow_scale_c{r}"
            if key in metadata:
                scale_macros += f"#define POW_HW_SCALE_C{r} {clamp_float32(float(metadata[key])):.10e}f\n"
        print(f"HW exponent-ALU pow: degree {degree}, root_n {root_n}, reciprocal {recip}")
        return (
            "\n// eval_method: exponent_alu / pow/root_N (exexp -> e, exman -> m). STANDALONE."
            "\n// Natural [1,2)-basis coeffs for root_N(m); recombine 2^(e/N)*root_N(2^r)*p(m).\n"
            "#define EVAL_METHOD_EXPONENT_ALU\n"
            "#define EXPONENT_ALU_POW\n"
            f"{scale_macros}"
            f"{recip_macro}"
            f"{hw_preload_macro}"
            f"constexpr uint32_t POW_HW_DEGREE = {degree};\n"
            f"constexpr float POW_HW_COEFFS[] = {{{coeff_str}}};\n"
        )

    print(f"WARNING: unknown exponent_alu kind '{kind}', skipping HW range reduction")
    return ""


def get_range_reduction_macros(lut_info: Dict) -> str:
    """Generate range reduction macros based on CSV metadata.

    Accepts the full lut_info (HW exponent-ALU paths need the fitted
    coefficients, not just metadata).
    """
    metadata = lut_info.get("metadata", {})
    method = metadata.get("range_reduction_method", "")
    # FIRST-CLASS newton_root (and legacy exponent_alu_newton_root) -> standalone
    # magic-seed + Newton evaluator. Reuse the HW macro builder (kind=newton_root).
    if method == "newton_root":
        return get_hw_exponent_alu_macros("exponent_alu_newton_root", lut_info)
    if method.startswith("exponent_alu_"):
        return get_hw_exponent_alu_macros(method, lut_info)
    # eval_method: reduced_poly -- Cody-Waite / mantissa reduce-then-poly. The
    # umbrella selector routes; the REDUCE_* sub-tag names the reduction.
    if method == "exp":
        return (
            "\n// eval_method: reduced_poly / exp (Cody-Waite, reduce to [-ln2/2, ln2/2])\n"
            "#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_EXP\n"
        )
    elif method == "trig":
        return (
            "\n// eval_method: reduced_poly / trig (Cody-Waite, reduce to [-pi/2, pi/2])\n"
            "#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_TRIG\n"
        )
    elif method == "log":
        # Read expansion constant from CSV metadata (defaults to ln(2) for natural log)
        expand_const = metadata.get("log_ln2_constant", "0.6931471805599453")
        return (
            f"\n// eval_method: reduced_poly / log (reduce to mantissa [1, 2))"
            f"\n#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_LOG"
            f"\n#define LOG_EXPAND_CONSTANT {expand_const}f\n"
        )
    elif method == "tan":
        return (
            "\n// eval_method: reduced_poly / tan (Cody-Waite, reduce to [-pi/4, pi/4])\n"
            "#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_TAN\n"
        )
    elif method == "cbrt":
        return (
            "\n// eval_method: reduced_poly / cbrt (exponent decomposition to mantissa [1, 2))\n"
            "#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_CBRT\n"
        )
    return ""


def get_adaptive_degree_macros(segment_degrees: List[int], max_degree: int) -> str:
    """Generate per-segment adaptive degree optimization macros."""
    if not segment_degrees or not any(d < max_degree for d in segment_degrees):
        return ""

    deg_array = ", ".join(str(d) for d in segment_degrees)
    return (
        "\n#ifndef DISABLE_ADAPTIVE_DEGREE\n"
        "#define HAS_SEGMENT_DEGREES\n"
        f"constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_array}}};\n"
        "#endif\n"
    )


# Dominant factor string → (define name, arg_scale, output_scale)
# arg_scale: multiplier for the exp argument (e.g., -0.5 for exp(-x²/2))
# output_scale: scalar multiplier on the dominant factor (e.g., -1/√(2π))
DOMINANT_FACTOR_MAP = {
    "-exp(-x^2/2) / sqrt(2*pi)": ("EXP_QUADRATIC", -0.5, -1.0 / math.sqrt(2 * math.pi)),
    "exp(-x^2/2) / sqrt(2*pi)": ("EXP_QUADRATIC", -0.5, 1.0 / math.sqrt(2 * math.pi)),
    "exp(x)": ("EXP_LINEAR", 1.0, 1.0),
    "exp(-x)": ("EXP_LINEAR", -1.0, 1.0),
    "-exp(-x)": ("EXP_LINEAR", -1.0, -1.0),
    "x * exp(x)": ("X_EXP_LINEAR", 1.0, 1.0),
    "x": ("X", 0.0, 1.0),
}


def get_asymptotic_macros(lut_info: Dict) -> str:
    """Generate asymptotic factoring macros from CSV metadata.

    When some segments are marked is_asymptotic=True with a dominant_factor,
    the kernel multiplies the correction polynomial by the dominant factor
    only for lanes that fall in the asymptotic region.

    Emits: ASYMPTOTIC_FACTOR_<CLASS>, ASYMPTOTIC_EXP_ARG_SCALE,
           ASYMPTOTIC_SCALE, ASYMPTOTIC_UPPER_BOUND / ASYMPTOTIC_LOWER_BOUND.
    """
    flags = lut_info.get("asymptotic_flags", [])
    factors = lut_info.get("dominant_factors", [])
    boundaries = lut_info.get("boundaries", [])

    if not flags or not any(flags):
        return ""

    # All asymptotic segments must share the same dominant factor
    active_factors = [f for f, is_asym in zip(factors, flags) if is_asym and f]
    if not active_factors:
        return ""
    unique_factors = set(active_factors)
    if len(unique_factors) > 1:
        print(f"WARNING: mixed dominant factors not supported: {unique_factors}")
        return ""

    dominant_str = active_factors[0]
    if dominant_str not in DOMINANT_FACTOR_MAP:
        print(f"WARNING: unknown dominant factor '{dominant_str}', skipping asymptotic")
        return ""

    factor_class, arg_scale, output_scale = DOMINANT_FACTOR_MAP[dominant_str]

    # Find the boundary between asymptotic and non-asymptotic segments.
    # Asymptotic segments are contiguous at the edges (left tail and/or right tail).
    num_segments = len(flags)
    macros = f"\n// Asymptotic factoring: {dominant_str}\n"
    macros += f"#define ASYMPTOTIC_FACTOR_{factor_class}\n"

    if factor_class != "X":
        macros += f"constexpr float ASYMPTOTIC_EXP_ARG_SCALE = {arg_scale:.16e}f;\n"
    macros += f"constexpr float ASYMPTOTIC_SCALE = {output_scale:.16e}f;\n"

    # Determine bound direction: left-tail (x < bound) or right-tail (x > bound)
    # Left tail: segments 0..k are asymptotic, boundary is boundaries[k+1]
    # Right tail: segments k..N-1 are asymptotic, boundary is boundaries[k]
    first_asym = next(i for i, f in enumerate(flags) if f)
    last_asym = next(i for i in range(num_segments - 1, -1, -1) if flags[i])

    if first_asym == 0 and last_asym < num_segments - 1:
        # Left tail: asymptotic segments start at 0
        bound = boundaries[last_asym + 1]
        macros += f"constexpr float ASYMPTOTIC_UPPER_BOUND = {bound:.16e}f;\n"
        print(f"Asymptotic factoring: {dominant_str} (left tail, x < {bound})")
    elif last_asym == num_segments - 1 and first_asym > 0:
        # Right tail: asymptotic segments end at N-1
        bound = boundaries[first_asym]
        macros += f"constexpr float ASYMPTOTIC_LOWER_BOUND = {bound:.16e}f;\n"
        print(f"Asymptotic factoring: {dominant_str} (right tail, x > {bound})")
    elif first_asym == 0 and last_asym == num_segments - 1:
        # All segments asymptotic (unusual but valid)
        # No bound needed — always apply
        macros += "// All segments are asymptotic — no bound check needed\n"
        macros += "constexpr float ASYMPTOTIC_UPPER_BOUND = 1.0e38f;\n"
        print(f"Asymptotic factoring: {dominant_str} (all segments)")
    else:
        print(f"WARNING: non-contiguous asymptotic segments, skipping")
        return ""

    return macros


def generate_polynomial_kernel(
    lut_info: Dict,
    degree: int,
    output_path: Path,
    activation: str = "unknown",
    segmentation: str = "unknown",
) -> None:
    """Generate polynomial embedded kernel."""
    degree_name = POLY_DEGREE_MAP.get(degree, f"degree_{degree}")

    # Optional macros
    range_reduction = get_range_reduction_macros(lut_info)
    adaptive_degree = get_adaptive_degree_macros(lut_info.get("segment_degrees", []), degree)
    poly_parity = get_poly_parity_macros(lut_info, degree)
    asymptotic = get_asymptotic_macros(lut_info)
    affine = get_affine_collapse_macros(lut_info)

    # Exactly one EVAL_METHOD_* selector is emitted. range_reduction emits
    # EXPONENT_ALU / NEWTON_ROOT / REDUCED_POLY; affine emits AFFINE_COLLAPSE.
    # When none of those fire, the method is the default poly_cascade. (Parity /
    # dual-eval / adaptive-degree / blend are ORTHOGONAL modifiers, not methods.)
    eval_method_default = (
        ""
        if ("EVAL_METHOD_" in range_reduction or "EVAL_METHOD_" in affine)
        else "\n// eval_method: poly_cascade (default piecewise polynomial cascade)\n#define EVAL_METHOD_POLY_CASCADE\n"
    )

    # Degree 0 (constant) is special case - uses different base kernel
    if degree == 0:
        base_kernel = "../piecewise_constant.cpp"
        degree_macro = ""
    else:
        base_kernel = "../piecewise_generic.cpp"
        degree_macro = f"constexpr uint32_t POLY_DEGREE = {degree};\n"

    content = f"""// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated adhoc kernel: {activation} piecewise {degree_name} ({lut_info['num_segments']} segments, {segmentation})
// Generated by tools/generate_adhoc_kernel.py

#include <array>
#include <cstdint>

#define EMBEDDED_LUT
{degree_macro}constexpr uint32_t NUM_SEGMENTS = {lut_info['num_segments']};

constexpr float INPUT_MIN = {lut_info['input_min']:.10e}f;
constexpr float INPUT_MAX = {lut_info['input_max']:.10e}f;

constexpr uint32_t LUT_SIZE_BF16 = {lut_info['lut_size']};
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {{{{
{lut_info['lut_data']}
}}}};

constexpr uint32_t LUT_SIZE_FP32 = {lut_info['lut_size']};
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {{{{
{lut_info['lut_data']}
}}}};

#ifdef USE_BF16
    constexpr auto& LUT_DATA = LUT_DATA_BF16;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_BF16;
#else
    constexpr auto& LUT_DATA = LUT_DATA_FP32;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_FP32;
#endif
{eval_method_default}{adaptive_degree}{poly_parity}{range_reduction}{asymptotic}{affine}
#include "{base_kernel}"
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Generated polynomial kernel: {output_path}")


def generate_rational_kernel(
    lut_info: Dict,
    output_path: Path,
    activation: str = "unknown",
    segmentation: str = "unknown",
) -> None:
    """Generate rational embedded kernel."""
    num_deg = lut_info["num_degree"]
    den_deg = lut_info["den_degree"]

    range_reduction = get_range_reduction_macros(lut_info)
    parity = get_parity_macros(
        lut_info.get("metadata", {}),
        segments=lut_info.get("raw_segments"),
        num_degree=num_deg,
        den_degree=den_deg,
    )

    # eval_method: rational_cascade is the base method for this kernel; a
    # reduced_poly reduction (REDUCE_EXP/TRIG/LOG) may be layered on top.
    eval_method_rational = (
        "\n// eval_method: rational_cascade (piecewise P(x)/Q(x))\n#define EVAL_METHOD_RATIONAL_CASCADE\n"
    )

    content = f"""// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated adhoc kernel: {activation} piecewise rational n{num_deg}d{den_deg} ({lut_info['num_segments']} segments, {segmentation})
// Generated by tools/generate_adhoc_kernel.py

#include <array>
#include <cstdint>

#define EMBEDDED_LUT

constexpr float INPUT_MIN = {lut_info['input_min']:.10e}f;
constexpr float INPUT_MAX = {lut_info['input_max']:.10e}f;

constexpr uint32_t NUM_DEGREE = {num_deg};
constexpr uint32_t DEN_DEGREE = {den_deg};
constexpr uint32_t NUM_SEGMENTS = {lut_info['num_segments']};
constexpr uint32_t LUT_SIZE = {lut_info['lut_size']};

constexpr std::array<float, LUT_SIZE> LUT_DATA = {{{{
{lut_info['lut_data']}
}}}};
{eval_method_rational}{parity}{range_reduction}
#include "../piecewise_rational.cpp"
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Generated rational kernel: {output_path}")


# ============================================================================
# Coefficient file resolution
# ============================================================================


def find_polynomial_csv(
    coeff_dir: Path,
    activation: str,
    degree: int,
    segments: int,
    segmentation: str,
    metric: str = "ulp",
) -> Path:
    """Locate polynomial coefficient CSV file.

    Uses canonical format from tt-polynomial-fitter/csv_filename.py:
        {activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}.csv

    Coefficients are metric-agnostic, so tries multiple metrics if needed.
    """
    # Try specified metric first, then fall back to others
    metrics_to_try = [metric] + [m for m in ["ulp", "max", "mae"] if m != metric]

    # Canonical format: {activation}_p{degree}_s{segments}_{segmentation}_*_{metric}.csv
    # Use glob to find any fitting method (any, remez, fpminimax, etc.)
    for file_metric in metrics_to_try:
        pattern = f"{activation}_p{degree}_s{segments}_{segmentation}_*_{file_metric}.csv"
        matches = list(coeff_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Polynomial CSV not found for {activation} p{degree} s{segments} {segmentation}\n"
        f"Searched in: {coeff_dir}\n"
        f"Expected canonical format: {activation}_p{degree}_s{segments}_{segmentation}_*_ulp.csv"
    )


def find_rational_csv(
    coeff_dir: Path,
    activation: str,
    num_degree: int,
    den_degree: int,
    segments: int,
    segmentation: str,
    metric: str = "ulp",
) -> Path:
    """Locate rational coefficient CSV file.

    Uses canonical format from tt-polynomial-fitter/csv_filename.py:
        {activation}_n{num}d{den}_s{segments}_{segmentation}_{fitting}_{metric}.csv

    Coefficients are metric-agnostic, so tries multiple metrics if needed.
    """
    # Try specified metric first, then fall back to others
    metrics_to_try = [metric] + [m for m in ["ulp", "max", "mae"] if m != metric]

    # Canonical format: {activation}_n{num}d{den}_s{segments}_{segmentation}_*_{metric}.csv
    for file_metric in metrics_to_try:
        pattern = f"{activation}_n{num_degree}d{den_degree}_s{segments}_{segmentation}_*_{file_metric}.csv"
        matches = list(coeff_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Rational CSV not found for {activation} n{num_degree}d{den_degree} s{segments} {segmentation}\n"
        f"Searched in: {coeff_dir}\n"
        f"Expected canonical format: {activation}_n{num_degree}d{den_degree}_s{segments}_{segmentation}_rational_{metric}.csv"
    )


# ============================================================================
# Main entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate adhoc embedded kernel for polynomial or rational approximation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV file (auto-detects polynomial vs rational)
  %(prog)s --csv coeffs.csv --output kernels/compute/adhoc/adhoc.cpp

  # Polynomial from coefficient directory
  %(prog)s --activation sigmoid --degree 4 --segments 16 --segmentation uniform \\
      --output kernels/compute/adhoc/adhoc.cpp

  # Rational approximation
  %(prog)s --activation exp --num-degree 2 --den-degree 2 --segments 4 \\
      --segmentation uniform --output kernels/compute/adhoc/adhoc.cpp

  # With custom coefficient directory
  %(prog)s --activation sigmoid --degree 4 --segments 16 \\
      --coeff-dir /path/to/coefficients --output adhoc.cpp
""",
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument("--csv", type=str, help="Direct path to coefficient CSV file")
    input_group.add_argument("--activation", type=str, help="Activation function name")

    # Polynomial options
    poly_group = parser.add_argument_group("Polynomial options")
    poly_group.add_argument("--degree", type=str, help="Polynomial degree (number or name: cubic, quartic, etc.)")

    # Rational options
    rational_group = parser.add_argument_group("Rational options")
    rational_group.add_argument("--num-degree", type=int, help="Numerator polynomial degree")
    rational_group.add_argument("--den-degree", type=int, help="Denominator polynomial degree")

    # Common options
    common_group = parser.add_argument_group("Common options")
    common_group.add_argument("--segments", type=int, help="Number of segments")
    common_group.add_argument(
        "--segmentation",
        type=str,
        default="uniform",
        help="Segmentation type: uniform|chebyshev|curvature (default: uniform)",
    )
    common_group.add_argument(
        "--metric",
        type=str,
        default="ulp",
        choices=["ulp", "mae", "max"],
        help="Error metric for coefficient selection (default: ulp)",
    )
    common_group.add_argument(
        "--coeff-dir", type=str, help="Coefficient directory (default: $TT_POLY_FIT_DIR/data/coefficients)"
    )

    # Output
    parser.add_argument("--output", "-o", type=str, required=True, help="Output kernel file path")

    args = parser.parse_args()

    # Determine coefficient directory
    # Support both local (macOS) and remote (Linux server) paths
    if os.environ.get("TT_POLY_FIT_DIR"):
        poly_fit_dir = os.environ["TT_POLY_FIT_DIR"]
    elif Path("/localdev/nkapre/tt-polynomial-fitter").exists():
        poly_fit_dir = "/localdev/nkapre/tt-polynomial-fitter"
    elif Path(os.path.expanduser("~/workspace/tt-polynomial-fitter")).exists():
        poly_fit_dir = os.path.expanduser("~/workspace/tt-polynomial-fitter")
    else:
        poly_fit_dir = "/localdev/nkapre/tt-polynomial-fitter"  # Fallback
    coeff_dir = Path(args.coeff_dir) if args.coeff_dir else Path(poly_fit_dir) / "data" / "coefficients"

    output_path = Path(args.output)

    # Mode 1: Direct CSV input
    if args.csv:
        csv_path = Path(args.csv)
        kernel_type, lut_info = parse_generic_csv(csv_path)

        if kernel_type == "polynomial":
            # Infer degree from lut_info
            degree = len(lut_info["segment_degrees"]) and max(lut_info["segment_degrees"]) or 0
            # Better: count coefficients per segment
            num_seg = lut_info["num_segments"]
            total_coeffs = lut_info["lut_size"] - (num_seg + 1)  # subtract boundaries
            coeffs_per_seg = total_coeffs // num_seg
            degree = coeffs_per_seg - 1
            generate_polynomial_kernel(
                lut_info, degree, output_path, activation=args.activation or "unknown", segmentation=args.segmentation
            )
        else:
            generate_rational_kernel(
                lut_info, output_path, activation=args.activation or "unknown", segmentation=args.segmentation
            )
        return

    # Mode 2: Lookup from coefficient directory
    if not args.activation:
        parser.error("--activation is required when not using --csv")

    # Determine if polynomial or rational
    is_rational = args.num_degree is not None or args.den_degree is not None

    if is_rational:
        # Rational mode
        if args.num_degree is None or args.den_degree is None:
            parser.error("Rational mode requires both --num-degree and --den-degree")
        if not args.segments:
            parser.error("--segments is required for rational mode")

        csv_path = find_rational_csv(
            coeff_dir, args.activation, args.num_degree, args.den_degree, args.segments, args.segmentation, args.metric
        )
        lut_info = parse_rational_csv(csv_path, args.num_degree, args.den_degree)
        generate_rational_kernel(lut_info, output_path, activation=args.activation, segmentation=args.segmentation)

    else:
        # Polynomial mode
        if not args.degree:
            parser.error("--degree is required for polynomial mode")
        if not args.segments:
            parser.error("--segments is required for polynomial mode")

        # Parse degree (can be number or name)
        if args.degree.isdigit():
            degree = int(args.degree)
        elif args.degree in POLY_METHOD_MAP:
            degree = POLY_METHOD_MAP[args.degree]
        else:
            parser.error(f"Invalid degree: {args.degree}. Use number (1-16) or name (linear, cubic, etc.)")

        csv_path = find_polynomial_csv(
            coeff_dir, args.activation, degree, args.segments, args.segmentation, args.metric
        )
        lut_info = parse_polynomial_csv(csv_path, degree)
        generate_polynomial_kernel(
            lut_info, degree, output_path, activation=args.activation, segmentation=args.segmentation
        )


if __name__ == "__main__":
    main()
