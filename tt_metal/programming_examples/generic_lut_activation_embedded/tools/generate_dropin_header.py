#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate a ckernel_sfpu_<activation>.h drop-in replacement header.

Reads the EXISTING header to extract function signatures, then generates
a new header with embedded LUT coefficients and calls to the shared
piecewise_rational_eval<>() evaluator.

Usage:
  python3 generate_dropin_header.py \\
      --activation softplus \\
      --bf16-csv coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \\
      --fp32-csv coefficients/softplus_n11d11_s2_uniform_rational_ulp.csv \\
      --output ckernel_sfpu_softplus.h

  # BF16-only (same coefficients for both dtypes):
  python3 generate_dropin_header.py \\
      --activation softplus \\
      --bf16-csv coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \\
      --output ckernel_sfpu_softplus.h
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_rational_csv(csv_path: str) -> Dict:
    """Parse a rational coefficient CSV. Returns dict with metadata."""
    boundaries = []
    num_coefficients = []
    den_coefficients = []
    metadata = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        num_cols = sorted([h for h in headers if h.startswith("n") and h[1:].isdigit()], key=lambda h: int(h[1:]))
        den_cols = sorted([h for h in headers if h.startswith("d") and h[1:].isdigit()], key=lambda h: int(h[1:]))

        if not num_cols or not den_cols:
            # Try polynomial columns
            coeff_cols = sorted([h for h in headers if h.startswith("c") and h[1:].isdigit()], key=lambda h: int(h[1:]))
            if not coeff_cols:
                raise ValueError(f"No coefficient columns found in {csv_path}")
            # Polynomial: treat as rational with den_degree=0, d0=1.0
            for row in reader:
                if row.get("segment_id", "").upper() == "METADATA":
                    key = row.get("lo", "")
                    val = row.get("hi", "")
                    if key:
                        metadata[key] = val
                    continue
                if not boundaries:
                    boundaries.append(float(row["lo"]))
                boundaries.append(float(row["hi"]))
                num_coefficients.append([float(row[c]) for c in coeff_cols])
                den_coefficients.append([1.0])

            return {
                "boundaries": boundaries,
                "num_coefficients": num_coefficients,
                "den_coefficients": den_coefficients,
                "num_degree": len(coeff_cols) - 1,
                "den_degree": 0,
                "num_segments": len(boundaries) - 1,
                "metadata": metadata,
                "is_rational": False,
            }

        for row in reader:
            if row.get("segment_id", "").upper() == "METADATA":
                key = row.get("lo", "")
                val = row.get("hi", "")
                if key:
                    metadata[key] = val
                continue
            if not boundaries:
                boundaries.append(float(row["lo"]))
            boundaries.append(float(row["hi"]))
            num_coefficients.append([float(row[c]) for c in num_cols])
            den_coefficients.append([float(row[c]) for c in den_cols])

    return {
        "boundaries": boundaries,
        "num_coefficients": num_coefficients,
        "den_coefficients": den_coefficients,
        "num_degree": len(num_cols) - 1,
        "den_degree": len(den_cols) - 1,
        "num_segments": len(boundaries) - 1,
        "metadata": metadata,
        "is_rational": True,
    }


def detect_parity(parsed: Dict) -> Optional[str]:
    """Detect rational parity from coefficient values.
    Returns 'odd_num_even_den' or None. Uses threshold=1e-30."""
    if not parsed["is_rational"]:
        return None

    threshold = 1e-30
    even_idx_all_zero = True
    odd_idx_all_zero = True
    for seg_num in parsed["num_coefficients"]:
        for i, c in enumerate(seg_num):
            if i % 2 == 0 and abs(c) > threshold:
                even_idx_all_zero = False
            if i % 2 == 1 and abs(c) > threshold:
                odd_idx_all_zero = False

    num_parity = "odd" if even_idx_all_zero else ("even" if odd_idx_all_zero else None)

    even_idx_all_zero_den = True
    odd_idx_all_zero_den = True
    for seg_den in parsed["den_coefficients"]:
        for i, c in enumerate(seg_den):
            if i % 2 == 0 and abs(c) > threshold:
                even_idx_all_zero_den = False
            if i % 2 == 1 and abs(c) > threshold:
                odd_idx_all_zero_den = False

    den_parity = "even" if odd_idx_all_zero_den else ("odd" if even_idx_all_zero_den else None)

    if num_parity == "odd" and den_parity == "even":
        return "odd_num_even_den"
    return None


def detect_range_reduction(parsed: Dict) -> Optional[str]:
    """Detect range reduction method from CSV metadata."""
    meta = parsed.get("metadata", {})
    rr = meta.get("range_reduction_method", "")
    if rr in ("exp", "trig", "log", "tan", "cbrt"):
        return rr
    return None


def detect_segment_degrees(parsed: Dict) -> Optional[List[int]]:
    """Detect effective per-segment degree for polynomial CSVs.
    Returns list of degrees if adaptive (not all same), else None."""
    if parsed["is_rational"]:
        return None
    max_degree = parsed["num_degree"]
    degrees = []
    for seg_coeffs in parsed["num_coefficients"]:
        # Find highest non-zero coefficient index
        eff_deg = 0
        for i, c in enumerate(seg_coeffs):
            if c != 0.0:
                eff_deg = i
        degrees.append(eff_deg)
    # Only emit if there's actual reduction (not all max degree)
    if all(d == max_degree for d in degrees):
        return None
    return degrees


def detect_dropin_correction(parsed: Dict) -> Optional[str]:
    """Detect drop-in correction from activation JSON (dropin_correction field).
    Returns correction name (elu, celu, selu) or None."""
    meta = parsed.get("metadata", {})
    correction = meta.get("dropin_correction", "")
    if correction in ("elu", "celu", "selu"):
        return correction
    return None


def load_boundary_metadata(parsed: Dict) -> Optional[Dict]:
    """Load boundary clamping metadata from activation JSON.

    Returns dict with 'left' and 'right' boundary specs, or None if no boundary.
    Resolves 'domain_min'/'domain_max' thresholds to actual float values.

    Boundary modes:
      - 'constant': return a fixed value outside threshold
      - 'identity': return x (passthrough) outside threshold
      - 'param_neg': return -param (e.g. -alpha for elu)
      - 'scale_identity': return scale*x outside threshold
      - 'none': no clamping (unbounded function)
    """
    meta = parsed.get("metadata", {})
    boundary = meta.get("boundary")
    if not boundary:
        return None

    left = boundary.get("left", {})
    right = boundary.get("right", {})

    # Skip if both sides are 'none'
    if left.get("mode") == "none" and right.get("mode") == "none":
        return None

    domain_min = parsed.get("metadata", {}).get("domain_min", -10.0)
    domain_max = parsed.get("metadata", {}).get("domain_max", 10.0)

    # Resolve 'domain_min'/'domain_max' string thresholds to floats
    def resolve_threshold(side):
        t = side.get("threshold")
        if t == "domain_min":
            side["threshold"] = domain_min
        elif t == "domain_max":
            side["threshold"] = domain_max
        elif t == "dynamic":
            side["threshold"] = None  # handled at runtime
        return side

    resolve_threshold(left)
    resolve_threshold(right)

    return {"left": left, "right": right, "comment": boundary.get("comment", "")}


def build_lut_array(parsed: Dict) -> List[float]:
    """Build flat LUT array.
    Rational: [boundaries..., seg0_num..., seg0_den..., seg1_num..., seg1_den..., ...]
    Polynomial: [boundaries..., seg0_coeffs..., seg1_coeffs..., ...]
    """
    lut = list(parsed["boundaries"])
    for i in range(parsed["num_segments"]):
        lut.extend(parsed["num_coefficients"][i])
        if parsed["is_rational"]:
            lut.extend(parsed["den_coefficients"][i])
    return lut


def clamp_float32(v: float) -> float:
    """Clamp to float32 representable range. Subnormals → 0."""
    if abs(v) < 1.175494e-38:  # FLT_MIN (smallest normal float32)
        return 0.0
    if abs(v) > 3.4028234663852886e38:
        return 3.4028234663852886e38 if v > 0 else -3.4028234663852886e38
    return v


def format_lut_cpp(lut: List[float], indent: int = 4) -> str:
    """Format LUT as C++ array initializer."""
    lines = []
    for i in range(0, len(lut), 5):
        chunk = lut[i : i + 5]
        line = ", ".join(f"{clamp_float32(v):.10e}f" for v in chunk)
        lines.append(" " * indent + line)
    return ",\n".join(lines)


def extract_signature(header_path: str, activation: str) -> Dict:
    """Extract function signatures from existing ckernel_sfpu header."""
    if not os.path.exists(header_path):
        return None

    content = Path(header_path).read_text()
    result = {}

    # Detect function name: "calculate_<activation>" or just "<activation>"
    # Check which one exists in the header
    if re.search(rf"\bcalculate_{activation}\s*\(", content):
        result["func_name"] = f"calculate_{activation}"
    elif re.search(rf"\b{activation}\s*\(", content):
        result["func_name"] = activation
    else:
        result["func_name"] = f"calculate_{activation}"  # default

    # Extract calculate_<activation> or <activation> signature
    fname = result["func_name"]
    pattern = rf"(template\s*<[^>]+>\s*\n\s*inline\s+void\s+{fname}\s*\([^)]*\))"
    m = re.search(pattern, content, re.MULTILINE)
    if m:
        result["calculate_sig"] = m.group(1)

    # Extract _body variant if exists
    pattern_body = rf"(template\s*<[^>]+>\s*\n\s*inline\s+void\s+{fname}_body\s*\([^)]*\))"
    m_body = re.search(pattern_body, content, re.MULTILINE)
    if m_body:
        result["body_sig"] = m_body.group(1)

    # Extract init signature
    pattern_init = rf"(template\s*<[^>]+>\s*\n\s*void\s+{activation}_init\s*\([^)]*\))"
    m_init = re.search(pattern_init, content, re.MULTILINE)
    if m_init:
        result["init_sig"] = m_init.group(1)

    # Detect if it has parameters (uint param0, param1, ...)
    result["has_params"] = "uint param0" in content or "uint param1" in content
    result["has_beta"] = "beta" in content
    result["has_threshold"] = "threshold" in content

    return result


def generate_header(
    activation: str,
    bf16_parsed: Dict,
    fp32_parsed: Optional[Dict],
    sig: Optional[Dict],
) -> str:
    """Generate the complete ckernel_sfpu_<activation>.h header."""
    NAME = activation.upper()
    name = activation.lower()

    bf16_lut = build_lut_array(bf16_parsed)
    bf16_size = len(bf16_lut)

    # FP32: use separate coefficients if provided, else reuse BF16
    if fp32_parsed:
        fp32_lut = build_lut_array(fp32_parsed)
        fp32_size = len(fp32_lut)
        has_fp32 = True
    else:
        fp32_lut = bf16_lut
        fp32_size = bf16_size
        fp32_parsed = bf16_parsed
        has_fp32 = False

    # Build the header
    lines = []
    lines.append("// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC")
    lines.append("//")
    lines.append("// SPDX-License-Identifier: Apache-2.0")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append('#include "ckernel.h"')
    lines.append('#include "ckernel_defs.h"')
    lines.append('#include "sfpu/ckernel_sfpu_converter.h"')
    lines.append("")

    # Detect and emit optimization defines BEFORE including shared evaluator
    # These must come first so the shared header enables the right code paths.
    # Parity must hold for BOTH precisions (defines are global, not per-#ifdef)
    bf16_parity = detect_parity(bf16_parsed)
    fp32_parity = detect_parity(fp32_parsed) if fp32_parsed else bf16_parity
    parity = bf16_parity if bf16_parity == fp32_parity else None

    range_red = detect_range_reduction(bf16_parsed)
    # FP32 may have different range reduction (check both)
    if fp32_parsed and not range_red:
        range_red = detect_range_reduction(fp32_parsed)

    if parity == "odd_num_even_den":
        if bf16_parsed["is_rational"]:
            lines.append("// Parity: odd num / even den → x²-Horner (~2x speedup)")
            lines.append("#define RATIONAL_NUM_PARITY_ODD")
            lines.append("#define RATIONAL_DEN_PARITY_EVEN")
        else:
            lines.append("// Parity: odd polynomial → x²-Horner (~2x speedup)")
            lines.append("#define POLY_PARITY_ODD")
        lines.append("")
    elif parity == "even":
        if not bf16_parsed["is_rational"]:
            lines.append("// Parity: even polynomial → x²-Horner (~2x speedup)")
            lines.append("#define POLY_PARITY_EVEN")
            lines.append("")

    if range_red == "exp":
        lines.append("#define RANGE_REDUCTION_EXP")
    elif range_red == "trig":
        lines.append("#define RANGE_REDUCTION_TRIG")
    elif range_red == "log":
        log_const = bf16_parsed.get("metadata", {}).get("log_ln2_constant", "0.6931471805599453")
        lines.append(f"#define RANGE_REDUCTION_LOG")
        lines.append(f"#define LOG_EXPAND_CONSTANT {log_const}f")
    if range_red:
        lines.append("")

    # Detect drop-in correction (from activation JSON dropin_correction field)
    dropin_correction = detect_dropin_correction(bf16_parsed)

    # Adaptive per-segment degree for polynomial (MUST come before shared header include)
    bf16_seg_degrees = detect_segment_degrees(bf16_parsed)
    fp32_seg_degrees = detect_segment_degrees(fp32_parsed) if fp32_parsed else None

    if not bf16_parsed["is_rational"] and (bf16_seg_degrees or fp32_seg_degrees):
        lines.append("// Adaptive per-segment degree — reduces Horner steps for low-degree segments")
        lines.append("#define HAS_SEGMENT_DEGREES")
        # Emit SEGMENT_DEGREES[] BEFORE the shared header include (it must be visible).
        # Both precisions MUST have a SEGMENT_DEGREES array when HAS_SEGMENT_DEGREES is
        # defined.  For the precision that lacks adaptive degrees, emit a uniform array
        # with all entries equal to the max degree (equivalent to no optimisation).
        if has_fp32:
            # Ensure both arrays exist — synthesize full-degree array for whichever
            # precision doesn't have adaptive degrees.
            effective_fp32_degrees = (
                fp32_seg_degrees if fp32_seg_degrees else [fp32_parsed["num_degree"]] * fp32_parsed["num_segments"]
            )
            effective_bf16_degrees = (
                bf16_seg_degrees if bf16_seg_degrees else [bf16_parsed["num_degree"]] * bf16_parsed["num_segments"]
            )

            lines.append("#ifdef INP_FLOAT32")
            deg_str = ", ".join(str(d) for d in effective_fp32_degrees)
            lines.append(f"constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_str}}};")
            lines.append("#else")
            deg_str = ", ".join(str(d) for d in effective_bf16_degrees)
            lines.append(f"constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_str}}};")
            lines.append("#endif")
        else:
            # No separate FP32 — BF16 coefficients used for both precisions.
            # bf16_seg_degrees must be non-None here (otherwise we wouldn't enter this block
            # since fp32_seg_degrees is None when has_fp32 is False).
            deg_str = ", ".join(str(d) for d in bf16_seg_degrees)
            lines.append(f"constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_str}}};")
        lines.append("")

    # Include the appropriate shared evaluator
    if bf16_parsed["is_rational"]:
        lines.append('#include "ckernel_sfpu_piecewise_rational.h"')
    else:
        lines.append('#include "ckernel_sfpu_piecewise_polynomial.h"')
    lines.append("")
    # Note: do NOT use 'using namespace sfpi;' — qualify types as sfpi::vFloat etc.
    lines.append("")
    lines.append("namespace ckernel::sfpu {")
    lines.append("")

    # Comment block
    lines.append(f"// {'=' * 70}")
    fitting_desc = "piecewise rational P(x)/Q(x)" if bf16_parsed["is_rational"] else "piecewise polynomial P(x)"
    lines.append(f"// LUT-based {name} via {fitting_desc}")
    lo = bf16_parsed["boundaries"][0]
    hi = bf16_parsed["boundaries"][-1]
    lines.append(f"//")
    lines.append(
        f"// BF16: n{bf16_parsed['num_degree']}/d{bf16_parsed['den_degree']}, "
        f"{bf16_parsed['num_segments']} segment(s), range [{lo}, {hi}]"
    )
    if has_fp32:
        lines.append(
            f"// FP32: n{fp32_parsed['num_degree']}/d{fp32_parsed['den_degree']}, "
            f"{fp32_parsed['num_segments']} segment(s), range [{fp32_parsed['boundaries'][0]}, {fp32_parsed['boundaries'][-1]}]"
        )
    lines.append(f"// {'=' * 70}")
    lines.append("")

    def emit_constexprs(parsed, lut, lut_size):
        """Emit constexpr declarations for a precision variant."""
        if parsed["is_rational"]:
            lines.append(f"constexpr uint32_t {NAME}_NUM_DEGREE = {parsed['num_degree']};")
            lines.append(f"constexpr uint32_t {NAME}_DEN_DEGREE = {parsed['den_degree']};")
        else:
            lines.append(f"constexpr uint32_t {NAME}_NUM_DEGREE = {parsed['num_degree']};")
        lines.append(f"constexpr uint32_t {NAME}_NUM_SEGMENTS = {parsed['num_segments']};")
        lines.append(f"constexpr uint32_t {NAME}_LUT_SIZE = {lut_size};")
        lines.append(f"constexpr std::array<float, {lut_size}> {NAME}_LUT = {{{{")
        lines.append(format_lut_cpp(lut))
        lines.append("}};")

    # FP32 coefficients
    if has_fp32:
        lines.append("#ifdef INP_FLOAT32")
        emit_constexprs(fp32_parsed, fp32_lut, fp32_size)
        lines.append("")
        lines.append("#else")
        lines.append("")

    # BF16 coefficients
    emit_constexprs(bf16_parsed, bf16_lut, bf16_size)

    if has_fp32:
        lines.append("")
        lines.append("#endif")

    lines.append("")

    # Load boundary metadata from activation JSON
    boundary = load_boundary_metadata(bf16_parsed)
    if boundary:
        lines.append(f"// Boundary clamping: {boundary['comment']}")
        lines.append("")

    def emit_boundary_clamp(lines, indent, boundary, result_var="result", x_var="x"):
        """Emit v_if boundary clamping lines.

        Emits v_if blocks that override `result_var` for inputs outside the LUT domain.
        Must be called AFTER result has been computed from the LUT eval.
        """
        if not boundary:
            return
        left = boundary["left"]
        right = boundary["right"]
        pad = " " * indent

        # Right boundary (checked first so left can override for x < left_thresh)
        if right.get("mode") == "identity" and right.get("threshold") is not None:
            thresh = right["threshold"]
            lines.append(f"{pad}v_if({x_var} >= {thresh:.1f}f) {{ {result_var} = {x_var}; }}")
            lines.append(f"{pad}v_endif;")
        elif right.get("mode") == "scale_identity" and right.get("threshold") is not None:
            thresh = right["threshold"]
            scale_p = right.get("scale_param", "scale_val")
            lines.append(f"{pad}v_if({x_var} >= {thresh:.1f}f) {{ {result_var} = {scale_p} * {x_var}; }}")
            lines.append(f"{pad}v_endif;")
        elif right.get("mode") == "constant" and right.get("threshold") is not None:
            thresh = right["threshold"]
            val = right["value"]
            lines.append(f"{pad}v_if({x_var} >= {thresh:.1f}f) {{ {result_var} = sfpi::vFloat({val:.10e}f); }}")
            lines.append(f"{pad}v_endif;")

        # Left boundary
        if left.get("mode") == "constant" and left.get("threshold") is not None:
            thresh = left["threshold"]
            val = left["value"]
            lines.append(f"{pad}v_if({x_var} < {thresh:.1f}f) {{ {result_var} = sfpi::vFloat({val:.10e}f); }}")
            lines.append(f"{pad}v_endif;")
        elif left.get("mode") == "param_neg" and left.get("threshold") is not None:
            thresh = left["threshold"]
            param = left.get("param", "alpha")
            lines.append(f"{pad}v_if({x_var} < {thresh:.1f}f) {{ {result_var} = -sfpi::vFloat({param}); }}")
            lines.append(f"{pad}v_endif;")

    # Function bodies — use the EXACT function name from the existing header
    func_name = sig.get("func_name", f"calculate_{name}") if sig else f"calculate_{name}"
    is_rational = bf16_parsed["is_rational"]

    # Build eval call strings based on rational vs polynomial (SINGLE LINE — no \n)
    if is_rational:
        eval_expr = f"piecewise_rational_eval<{NAME}_NUM_DEGREE, {NAME}_DEN_DEGREE, {NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, x)"
        eval_full_expr = f"piecewise_rational_eval_full<{NAME}_NUM_DEGREE, {NAME}_DEN_DEGREE, {NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, d)"
    else:
        eval_expr = f"piecewise_polynomial_eval<{NAME}_NUM_DEGREE, {NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, x)"
        eval_full_expr = (
            f"piecewise_polynomial_eval_full<{NAME}_NUM_DEGREE, {NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, d)"
        )

    # =========================================================================
    # DATA-DRIVEN function body generation from eval_regions[] in JSON.
    # NO per-activation special-casing. The JSON fully describes:
    #   - function_signature: template params, runtime params
    #   - eval_regions: condition → result expression
    #   - has_fp16b_convert: whether to truncate to fp16b
    # =========================================================================
    meta = bf16_parsed.get("metadata", {})
    eval_regions = meta.get("eval_regions")
    func_sig_json = meta.get("function_signature")
    has_fp16b = meta.get("has_fp16b_convert", False)
    domain_min = meta.get("domain_min", -10.0)
    domain_max = meta.get("domain_max", 10.0)

    # Softplus-style body_function pattern
    is_body_function = func_sig_json.get("body_function", False) if func_sig_json else False

    if eval_regions and func_sig_json:
        # --- Unified data-driven path: generate from eval_regions[] ---
        params = func_sig_json.get("params", [])
        template_str = func_sig_json.get("template", "bool APPROXIMATION_MODE, int ITERATIONS = 8")
        param_args = ", ".join(p["arg"] for p in params)

        default_region = next((r for r in eval_regions if r["condition"] == "default"), None)
        boundary_regions = [r for r in eval_regions if r["condition"] != "default"]

        def resolve_expr(expr_str):
            """Replace LUT(x)/LUT(x_rescaled) with actual eval call."""
            return (
                expr_str.replace("LUT(x_rescaled)", eval_expr.replace("x)", "x_rescaled)"))
                .replace("LUT(x)", eval_expr)
                .replace("{domain_min}", f"{domain_min:.1f}")
                .replace("{domain_max}", f"{domain_max:.1f}")
            )

        def resolve_condition(cond_str):
            return cond_str.replace("{domain_min}", f"{domain_min:.1f}").replace("{domain_max}", f"{domain_max:.1f}")

        if is_body_function:
            # Softplus-style: emit _body() helper + ITERATIONS wrapper
            lines.append(f"template <{template_str.split(',')[0].strip()}>")
            body_params = ", ".join(f"const float {p['var']}" for p in params)
            lines.append(f"inline void {func_name}_body({body_params}) {{")
            if default_region and default_region.get("input_rescale"):
                rescale = default_region["input_rescale"]
                lines.append(f"    sfpi::vFloat x = {rescale.replace('dst_reg', 'sfpi::dst_reg[0]')};")
            else:
                lines.append("    sfpi::vFloat x = sfpi::dst_reg[0];")
            passthrough = [br for br in boundary_regions if br["result"] == "PASSTHROUGH"]
            non_passthrough = [br for br in boundary_regions if br["result"] != "PASSTHROUGH"]
            if passthrough:
                cond = resolve_condition(passthrough[0]["condition"])
                inv_cond = cond.replace(">=", "<").replace("<=", ">")
                lines.append(f"    v_if({inv_cond}) {{")
                lines.append(f"        sfpi::dst_reg[0] = {resolve_expr(default_region['result'])};")
                lines.append("    }")
                lines.append("    v_endif;")
            else:
                lines.append(f"    sfpi::dst_reg[0] = {resolve_expr(default_region['result'])};")
            # Emit non-PASSTHROUGH boundary clamps (e.g. x < domain_min → 0)
            for br in non_passthrough:
                cond = resolve_condition(br["condition"])
                val = resolve_expr(br["result"])
                lines.append(f"    v_if({cond}) {{ sfpi::dst_reg[0] = {val}; }}")
                lines.append("    v_endif;")
            lines.append("}")
            lines.append("")
            lines.append(f"template <{template_str}>")
            lines.append(f"inline void {func_name}({param_args}) {{")
            for p in params:
                lines.append(f"    const float {p['var']} = {p['cast']};")
            lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
            call_args = ", ".join(p["var"] for p in params)
            lines.append(f"        {func_name}_body<APPROXIMATION_MODE>({call_args});")
            lines.append("        sfpi::dst_reg++;")
            lines.append("    }")
            lines.append("}")
        else:
            # Standard pattern: template + for loop + eval_regions
            lines.append(f"template <{template_str}>")
            sig_str = f"inline void {func_name}({param_args})" if param_args else f"inline void {func_name}()"
            lines.append(f"{sig_str} {{")
            for p in params:
                lines.append(f"    sfpi::vFloat {p['var']} = {p['cast']};")
            lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
            lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
            if default_region:
                rescale = default_region.get("input_rescale")
                if rescale:
                    lines.append(f"        sfpi::vFloat x_rescaled = {rescale};")
                lines.append(f"        sfpi::vFloat result = {resolve_expr(default_region['result'])};")
            for br in boundary_regions:
                cond = resolve_condition(br["condition"])
                val = resolve_expr(br["result"])
                lines.append(f"        v_if({cond}) {{ result = {val}; }}")
                lines.append("        v_endif;")
            if has_fp16b:
                lines.append("        if constexpr (!is_fp32_dest_acc_en) {")
                lines.append("            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));")
                lines.append("        }")
            lines.append("        sfpi::dst_reg[0] = result;")
            lines.append("        sfpi::dst_reg++;")
            lines.append("    }")
            lines.append("}")
    elif dropin_correction == "elu":
        # LEGACY — kept for backward compat until all JSONs have eval_regions
        # ELU: calculate_elu(uint slope) — slope=alpha
        lines.append("template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>")
        lines.append(f"inline void {func_name}(uint slope) {{")
        lines.append("    sfpi::vFloat alpha = Converter::as_float(slope);")
        lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
        lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
        lines.append(f"        sfpi::vFloat result = {eval_expr};")
        lines.append("        v_if(x < 0.0f) { result = alpha * result; }")
        lines.append("        v_endif;")
        emit_boundary_clamp(lines, 8, boundary, "result", "x")
        lines.append("        if constexpr (!is_fp32_dest_acc_en) {")
        lines.append("            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));")
        lines.append("        }")
        lines.append("        sfpi::dst_reg[0] = result;")
        lines.append("        sfpi::dst_reg++;")
        lines.append("    }")
        lines.append("}")
    elif dropin_correction == "celu":
        # CELU: calculate_celu(uint32_t param0, uint32_t param1) — alpha, alpha_recip
        # celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
        # For x>=0: result = x_orig (passthrough). For x<0: rescale by 1/alpha, eval, scale back.
        # Boundary: x<domain_min*alpha → -alpha (saturated exp)
        lines.append("template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>")
        lines.append(f"inline void {func_name}(uint32_t param0, uint32_t param1) {{")
        lines.append("    sfpi::vFloat alpha = Converter::as_float(param0);")
        lines.append("    sfpi::vFloat alpha_recip = Converter::as_float(param1);")
        lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
        lines.append("        sfpi::vFloat x_orig = sfpi::dst_reg[0];")
        lines.append("        sfpi::vFloat result = x_orig;  // positive passthrough")
        lines.append("        v_if(x_orig < 0.0f) {")
        lines.append("            sfpi::vFloat x = alpha_recip * x_orig;  // x/alpha")
        lines.append(f"            result = alpha * {eval_expr};")
        lines.append("        }")
        lines.append("        v_endif;")
        # Note: celu boundary is handled by the x>=0 passthrough above (right side)
        # and by the alpha rescaling for x<0 (which maps the input into LUT domain).
        # No additional boundary clamp needed — the rescaling IS the boundary handling.
        lines.append("        if constexpr (!is_fp32_dest_acc_en) {")
        lines.append("            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));")
        lines.append("        }")
        lines.append("        sfpi::dst_reg[0] = result;")
        lines.append("        sfpi::dst_reg++;")
        lines.append("    }")
        lines.append("}")
    elif dropin_correction == "selu":
        # SELU: calculate_selu(uint scale, uint alpha) — both fixed constants
        # LUT gives elu(x,1). x<0: scale*alpha*result, x>=0: scale*x.
        # Boundary: x>=0 → scale*x, x<domain_min → -scale*alpha (saturated)
        lines.append("template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS>")
        lines.append(f"inline void {func_name}(uint scale, uint alpha) {{")
        lines.append("    sfpi::vFloat scale_val = Converter::as_float(scale);")
        lines.append("    sfpi::vFloat alpha_val = Converter::as_float(alpha);")
        lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
        lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
        lines.append(f"        sfpi::vFloat result = {eval_expr};")
        lines.append("        v_if(x < 0.0f) { result = scale_val * alpha_val * result; }")
        lines.append("        v_else { result = scale_val * result; }")
        lines.append("        v_endif;")
        emit_boundary_clamp(lines, 8, boundary, "result", "x")
        lines.append("        if constexpr (!is_fp32_dest_acc_en) {")
        lines.append("            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));")
        lines.append("        }")
        lines.append("        sfpi::dst_reg[0] = result;")
        lines.append("        sfpi::dst_reg++;")
        lines.append("    }")
        lines.append("}")
    elif sig and sig.get("has_beta"):
        # Parametric activation (softplus-style: beta/threshold)
        lines.append("template <bool APPROXIMATION_MODE>")
        lines.append(
            f"inline void {func_name}_body(const float beta, const float beta_reciprocal, const float threshold) {{"
        )
        lines.append("    sfpi::vFloat x = beta * sfpi::dst_reg[0];")
        lines.append("    v_if(x < threshold) {")
        lines.append(f"        sfpi::dst_reg[0] = beta_reciprocal * {eval_expr};")
        lines.append("    }")
        lines.append("    v_endif;")
        lines.append("}")
        lines.append("")
        lines.append("template <bool APPROXIMATION_MODE, int ITERATIONS = 8>")
        lines.append(f"inline void {func_name}(uint param0, uint param1, uint param2) {{")
        lines.append("    const float beta = Converter::as_float(param0);")
        lines.append("    const float beta_reciprocal = Converter::as_float(param1);")
        lines.append("    const float threshold = Converter::as_float(param2);")
        lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
        lines.append(f"        {func_name}_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);")
        lines.append("        sfpi::dst_reg++;")
        lines.append("    }")
        lines.append("}")
    else:
        # Non-parametric activation (gelu, sigmoid, tanh, hardmish, erf, erfc, etc.)
        lines.append("template <bool APPROXIMATION_MODE, int ITERATIONS = 8>")
        lines.append(f"inline void {func_name}() {{")
        if range_red:
            lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
            lines.append(f"        {eval_full_expr};")
            lines.append("    }")
        elif boundary:
            # Need a result variable for boundary clamping
            lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
            lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
            lines.append(f"        sfpi::vFloat result = {eval_expr};")
            emit_boundary_clamp(lines, 8, boundary, "result", "x")
            lines.append("        sfpi::dst_reg[0] = result;")
            lines.append("        sfpi::dst_reg++;")
            lines.append("    }")
        else:
            lines.append("    for (int d = 0; d < ITERATIONS; d++) {")
            lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
            lines.append(f"        sfpi::dst_reg[0] = {eval_expr};")
            lines.append("        sfpi::dst_reg++;")
            lines.append("    }")
        lines.append("}")

    lines.append("")
    lines.append("template <bool APPROXIMATION_MODE>")
    lines.append(f"void {name}_init() {{")
    if is_rational:
        lines.append("    sfpu_reciprocal_init();")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace ckernel::sfpu")
    lines.append("")

    return "\n".join(lines)


def generate_erf_erfc_header(
    erf_bf16_parsed,
    erf_fp32_parsed,
    erfc_bf16_parsed,
    erfc_fp32_parsed,
):
    """Generate combined ckernel_sfpu_erf_erfc.h with independent LUTs."""

    def _emit_lut_block(lines, name, NAME, bf16_parsed, fp32_parsed):
        """Emit constexpr LUT + degree/segment constants for one activation."""
        bf16_lut = build_lut_array(bf16_parsed)
        has_fp32 = fp32_parsed is not None
        fp32_lut = build_lut_array(fp32_parsed) if has_fp32 else bf16_lut

        def emit_constexprs(parsed, lut, lut_size):
            if parsed["is_rational"]:
                lines.append(f"constexpr uint32_t {NAME}_NUM_DEGREE = {parsed['num_degree']};")
                lines.append(f"constexpr uint32_t {NAME}_DEN_DEGREE = {parsed['den_degree']};")
            else:
                lines.append(f"constexpr uint32_t {NAME}_NUM_DEGREE = {parsed['num_degree']};")
            lines.append(f"constexpr uint32_t {NAME}_NUM_SEGMENTS = {parsed['num_segments']};")
            lines.append(f"constexpr uint32_t {NAME}_LUT_SIZE = {lut_size};")
            lines.append(f"constexpr std::array<float, {lut_size}> {NAME}_LUT = {{{{")
            lines.append(format_lut_cpp(lut))
            lines.append("}};")

        if has_fp32:
            lines.append("#ifdef INP_FLOAT32")
            emit_constexprs(fp32_parsed, fp32_lut, len(fp32_lut))
            lines.append("")
            lines.append("#else")
            lines.append("")

        emit_constexprs(bf16_parsed, bf16_lut, len(bf16_lut))

        if has_fp32:
            lines.append("")
            lines.append("#endif")
        lines.append("")

    def _eval_expr(NAME, parsed):
        if parsed["is_rational"]:
            return (
                f"piecewise_rational_eval<{NAME}_NUM_DEGREE, {NAME}_DEN_DEGREE, "
                f"{NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, x)"
            )
        else:
            return (
                f"piecewise_polynomial_eval<{NAME}_NUM_DEGREE, " f"{NAME}_NUM_SEGMENTS, {NAME}_LUT_SIZE>({NAME}_LUT, x)"
            )

    # Determine which shared evaluators to include
    erf_rational = erf_bf16_parsed["is_rational"]
    erfc_rational = erfc_bf16_parsed["is_rational"]
    need_rational = erf_rational or erfc_rational
    need_polynomial = (not erf_rational) or (not erfc_rational)

    # Detect parity for each
    erf_bf16_parity = detect_parity(erf_bf16_parsed)
    erf_fp32_parity = detect_parity(erf_fp32_parsed) if erf_fp32_parsed else erf_bf16_parity
    erf_parity = erf_bf16_parity if erf_bf16_parity == erf_fp32_parity else None

    erfc_bf16_parity = detect_parity(erfc_bf16_parsed)
    erfc_fp32_parity = detect_parity(erfc_fp32_parsed) if erfc_fp32_parsed else erfc_bf16_parity
    erfc_parity = erfc_bf16_parity if erfc_bf16_parity == erfc_fp32_parity else None

    lines = []
    lines.append("// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC")
    lines.append("//")
    lines.append("// SPDX-License-Identifier: Apache-2.0")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append('#include "ckernel.h"')
    lines.append('#include "ckernel_defs.h"')
    lines.append("")

    # Include shared evaluators (no parity macros — erf uses sign-folding instead)
    if need_rational:
        lines.append('#include "ckernel_sfpu_piecewise_rational.h"')
    if need_polynomial:
        lines.append('#include "ckernel_sfpu_piecewise_polynomial.h"')
    lines.append("")

    # If erfc has different parity, undef erf parity before erfc LUT
    # (parity macros are global — only one set can be active)
    # For now, only apply parity to erf (the common case)

    # Note: do NOT use 'using namespace sfpi;' — qualify types as sfpi::vFloat etc.
    lines.append("")
    lines.append("namespace ckernel::sfpu {")
    lines.append("")

    # Comment block
    erf_lo = erf_bf16_parsed["boundaries"][0]
    erf_hi = erf_bf16_parsed["boundaries"][-1]
    erfc_lo = erfc_bf16_parsed["boundaries"][0]
    erfc_hi = erfc_bf16_parsed["boundaries"][-1]
    fitting_erf = "rational P(x)/Q(x)" if erf_rational else "polynomial P(x)"
    fitting_erfc = "rational P(x)/Q(x)" if erfc_rational else "polynomial P(x)"
    lines.append(f"// {'=' * 70}")
    lines.append(f"// LUT-based erf via piecewise {fitting_erf}")
    lines.append(
        f"// BF16: n{erf_bf16_parsed['num_degree']}/d{erf_bf16_parsed['den_degree']}, "
        f"{erf_bf16_parsed['num_segments']} seg, range [{erf_lo}, {erf_hi}]"
    )
    if erf_fp32_parsed:
        lines.append(
            f"// FP32: n{erf_fp32_parsed['num_degree']}/d{erf_fp32_parsed['den_degree']}, "
            f"{erf_fp32_parsed['num_segments']} seg, range [{erf_fp32_parsed['boundaries'][0]}, {erf_fp32_parsed['boundaries'][-1]}]"
        )
    lines.append(f"//")
    lines.append(f"// LUT-based erfc via piecewise {fitting_erfc}")
    lines.append(
        f"// BF16: n{erfc_bf16_parsed['num_degree']}/d{erfc_bf16_parsed['den_degree']}, "
        f"{erfc_bf16_parsed['num_segments']} seg, range [{erfc_lo}, {erfc_hi}]"
    )
    if erfc_fp32_parsed:
        lines.append(
            f"// FP32: n{erfc_fp32_parsed['num_degree']}/d{erfc_fp32_parsed['den_degree']}, "
            f"{erfc_fp32_parsed['num_segments']} seg, range [{erfc_fp32_parsed['boundaries'][0]}, {erfc_fp32_parsed['boundaries'][-1]}]"
        )
    lines.append(f"// {'=' * 70}")
    lines.append("")

    # ERF LUT
    _emit_lut_block(lines, "erf", "ERF", erf_bf16_parsed, erf_fp32_parsed)

    # ERFC LUT
    _emit_lut_block(lines, "erfc", "ERFC", erfc_bf16_parsed, erfc_fp32_parsed)

    # calculate_erf — direct LUT evaluation
    erf_eval = _eval_expr("ERF", erf_bf16_parsed)
    lines.append("template <bool APPROXIMATION_MODE>")
    lines.append("inline void calculate_erf() {")
    lines.append("    for (int d = 0; d < 8; d++) {")
    lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
    lines.append(f"        sfpi::dst_reg[0] = {erf_eval};")
    lines.append("        sfpi::dst_reg++;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # calculate_erfc — direct LUT evaluation (independent fit)
    erfc_eval = _eval_expr("ERFC", erfc_bf16_parsed)
    lines.append("template <bool APPROXIMATION_MODE>")
    lines.append("inline void calculate_erfc() {")
    lines.append("    for (int d = 0; d < 8; d++) {")
    lines.append("        sfpi::vFloat x = sfpi::dst_reg[0];")
    lines.append(f"        sfpi::dst_reg[0] = {erfc_eval};")
    lines.append("        sfpi::dst_reg++;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    lines.append("}  // namespace ckernel::sfpu")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate ckernel_sfpu drop-in header")
    parser.add_argument("--activation", "-a", required=True, help="Activation name (or 'erf_erfc' for combined)")
    parser.add_argument("--bf16-csv", required=True, help="BF16 coefficient CSV (erf CSV for erf_erfc mode)")
    parser.add_argument("--fp32-csv", default=None, help="FP32 coefficient CSV")
    parser.add_argument("--erfc-bf16-csv", default=None, help="ERFC BF16 CSV (erf_erfc mode only)")
    parser.add_argument("--erfc-fp32-csv", default=None, help="ERFC FP32 CSV (erf_erfc mode only)")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: ckernel_sfpu_<activation>.h)")
    args = parser.parse_args()

    # Combined erf+erfc mode
    if args.activation == "erf_erfc":
        if not args.erfc_bf16_csv:
            parser.error("--erfc-bf16-csv required for erf_erfc mode")
        erf_bf16 = parse_rational_csv(args.bf16_csv)
        erf_fp32 = parse_rational_csv(args.fp32_csv) if args.fp32_csv else None
        erfc_bf16 = parse_rational_csv(args.erfc_bf16_csv)
        erfc_fp32 = parse_rational_csv(args.erfc_fp32_csv) if args.erfc_fp32_csv else None
        print(
            f"ERF  BF16: n{erf_bf16['num_degree']}/d{erf_bf16['den_degree']}, "
            f"{erf_bf16['num_segments']} seg, {len(build_lut_array(erf_bf16))} floats"
        )
        if erf_fp32:
            print(
                f"ERF  FP32: n{erf_fp32['num_degree']}/d{erf_fp32['den_degree']}, "
                f"{erf_fp32['num_segments']} seg, {len(build_lut_array(erf_fp32))} floats"
            )
        print(
            f"ERFC BF16: n{erfc_bf16['num_degree']}/d{erfc_bf16['den_degree']}, "
            f"{erfc_bf16['num_segments']} seg, {len(build_lut_array(erfc_bf16))} floats"
        )
        if erfc_fp32:
            print(
                f"ERFC FP32: n{erfc_fp32['num_degree']}/d{erfc_fp32['den_degree']}, "
                f"{erfc_fp32['num_segments']} seg, {len(build_lut_array(erfc_fp32))} floats"
            )
        # Generate two separate headers so erf can use parity independently
        output_dir = Path(args.output).parent if args.output else Path(".")

        # erf: generate header, then replace macro parity with USE_PARITY template param
        # (macros would contaminate erfc which shares the compile unit)
        erf_header = generate_header("erf", erf_bf16, erf_fp32, sig=None)
        # Remove macro defines
        erf_header = erf_header.replace("#define RATIONAL_NUM_PARITY_ODD\n", "")
        erf_header = erf_header.replace("#define RATIONAL_DEN_PARITY_EVEN\n", "")
        erf_header = erf_header.replace("// Parity: odd num / even den → x²-Horner (~2x speedup)\n", "")
        # Replace eval call with USE_PARITY=true (5th template param)
        erf_header = erf_header.replace(
            "piecewise_rational_eval<ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE>",
            "piecewise_rational_eval<ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE, true>",
        )
        erf_path = output_dir / "ckernel_sfpu_erf.h"
        erf_path.write_text(erf_header)
        print(f"Generated: {erf_path}")

        # erfc: use generate_header without parity
        erfc_header = generate_header("erfc", erfc_bf16, erfc_fp32, sig=None)
        erfc_path = output_dir / "ckernel_sfpu_erfc.h"
        erfc_path.write_text(erfc_header)
        print(f"Generated: {erfc_path}")
        return

    # Parse CSVs
    bf16_parsed = parse_rational_csv(args.bf16_csv)
    fp32_parsed = parse_rational_csv(args.fp32_csv) if args.fp32_csv else None

    print(
        f"BF16: n{bf16_parsed['num_degree']}/d{bf16_parsed['den_degree']}, "
        f"{bf16_parsed['num_segments']} seg, {len(build_lut_array(bf16_parsed))} floats"
    )
    if fp32_parsed:
        print(
            f"FP32: n{fp32_parsed['num_degree']}/d{fp32_parsed['den_degree']}, "
            f"{fp32_parsed['num_segments']} seg, {len(build_lut_array(fp32_parsed))} floats"
        )

    # Load activation JSON for dropin_correction
    act_json = None
    poly_fit_dir = os.environ.get("TT_POLY_FIT_DIR", "/localdev/nkapre/tt-polynomial-fitter")
    act_json_path = os.path.join(poly_fit_dir, "activations", f"{args.activation}.json")
    if os.path.exists(act_json_path):
        import json as json_mod

        with open(act_json_path) as jf:
            act_json = json_mod.load(jf)
        dropin_correction = act_json.get("dropin_correction")
        if dropin_correction:
            # Inject into parsed metadata so generate_header can find it
            bf16_parsed.setdefault("metadata", {})["dropin_correction"] = dropin_correction
            if fp32_parsed:
                fp32_parsed.setdefault("metadata", {})["dropin_correction"] = dropin_correction
            print(f"Drop-in correction: {dropin_correction}")

        # Inject eval_regions and function_signature from activation JSON
        eval_regions = act_json.get("eval_regions")
        func_sig_json = act_json.get("function_signature")
        has_fp16b = act_json.get("has_fp16b_convert", False)
        if eval_regions:
            bf16_parsed.setdefault("metadata", {})["eval_regions"] = eval_regions
            bf16_parsed["metadata"]["function_signature"] = func_sig_json
            bf16_parsed["metadata"]["has_fp16b_convert"] = has_fp16b
            if fp32_parsed:
                fp32_parsed.setdefault("metadata", {})["eval_regions"] = eval_regions
                fp32_parsed["metadata"]["function_signature"] = func_sig_json
                fp32_parsed["metadata"]["has_fp16b_convert"] = has_fp16b
            n_regions = len(eval_regions)
            n_params = len(func_sig_json.get("params", [])) if func_sig_json else 0
            print(f"Eval regions: {n_regions} regions, {n_params} params (data-driven)")

        # Inject boundary metadata from activation JSON
        boundary_meta = act_json.get("boundary")
        if boundary_meta:
            domain = act_json.get("domain", {})
            bf16_parsed.setdefault("metadata", {})["boundary"] = boundary_meta
            bf16_parsed["metadata"]["domain_min"] = domain.get("min", -10.0)
            bf16_parsed["metadata"]["domain_max"] = domain.get("max", 10.0)
            if fp32_parsed:
                fp32_parsed.setdefault("metadata", {})["boundary"] = boundary_meta
                fp32_parsed["metadata"]["domain_min"] = domain.get("min", -10.0)
                fp32_parsed["metadata"]["domain_max"] = domain.get("max", 10.0)
            left_mode = boundary_meta.get("left", {}).get("mode", "none")
            right_mode = boundary_meta.get("right", {}).get("mode", "none")
            if left_mode != "none" or right_mode != "none":
                print(f"Boundary clamping: left={left_mode}, right={right_mode}")

    # Detect optimizations
    parity = detect_parity(bf16_parsed)
    range_red = detect_range_reduction(bf16_parsed)
    if fp32_parsed and not range_red:
        range_red = detect_range_reduction(fp32_parsed)
    if parity:
        print(f"Parity: {parity} → x²-Horner enabled")
    if range_red:
        print(f"Range reduction: {range_red}")

    # Detect adaptive degree
    bf16_seg_deg = detect_segment_degrees(bf16_parsed)
    fp32_seg_deg = detect_segment_degrees(fp32_parsed) if fp32_parsed else None
    if bf16_seg_deg:
        avg_deg = sum(bf16_seg_deg) / len(bf16_seg_deg)
        print(f"BF16 adaptive degree: {bf16_seg_deg} (avg {avg_deg:.1f} vs max {bf16_parsed['num_degree']})")
    if fp32_seg_deg:
        avg_deg = sum(fp32_seg_deg) / len(fp32_seg_deg)
        print(f"FP32 adaptive degree: {fp32_seg_deg} (avg {avg_deg:.1f} vs max {fp32_parsed['num_degree']})")

    # Try to extract existing signature
    repo_root = os.environ.get("REPO_ROOT", "")
    if not repo_root:
        # Try to find it
        d = Path(__file__).resolve().parent
        while d != d.parent:
            if (d / ".git").exists():
                repo_root = str(d)
                break
            d = d.parent

    # Extract signature from the ORIGINAL header (upstream/main), not the local copy
    # which may already be our generated version from a previous run.
    header_rel_path = os.path.join(
        "tt_metal",
        "hw",
        "ckernels",
        "blackhole",
        "metal",
        "llk_api",
        "llk_sfpu",
        f"ckernel_sfpu_{args.activation}.h",
    )
    sig = None
    # Try upstream/main first
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", repo_root, "show", f"upstream/main:{header_rel_path}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Write to temp file for extract_signature
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as tmp:
                tmp.write(result.stdout)
                tmp_path = tmp.name
            sig = extract_signature(tmp_path, args.activation)
            os.unlink(tmp_path)
            if sig:
                print(f"Extracted signature from upstream/main:{header_rel_path}")
    except Exception:
        pass

    # Fallback: use local file
    if not sig:
        local_header = os.path.join(repo_root, header_rel_path)
        sig = extract_signature(local_header, args.activation)
        if sig:
            print(f"Extracted signature from {local_header}")
    if sig:
        print(f"  func_name={sig.get('func_name')}, has_params={sig.get('has_params')}, has_beta={sig.get('has_beta')}")
    else:
        print(f"No existing header found, using default signature")

    # Generate
    header = generate_header(args.activation, bf16_parsed, fp32_parsed, sig)

    # Write
    output = args.output or f"ckernel_sfpu_{args.activation}.h"
    Path(output).write_text(header)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
