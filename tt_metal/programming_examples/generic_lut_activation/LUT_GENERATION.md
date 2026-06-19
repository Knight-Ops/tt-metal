# LUT Generation Workflow

**Note:** Polynomial fitting code has been moved to standalone project: `~/tt-polynomial-fitter/`

## Two-Step Pipeline

### Step 1: Generate Coefficients (CSV)
```bash
./generate_coefficients.sh \
    --activation sigmoid \
    --method chebyshev \
    --degree 3 \
    --segments 16 \
    --precision bf16
```

**Output:** `coefficients/sigmoid_bf16_16_3_chebyshev.csv`

### Step 2: Generate Binary LUTs
```bash
./generate_luts.sh --input-dir coefficients/
```

**Output:** `luts_output/sigmoid_bf16_16_3_chebyshev.lut`

## Naming Convention

```
{activation}_{precision}_{segments}_{degree}_{method}.[csv|lut]
```

**Components:**
- `activation`: sigmoid, tanh, gelu, etc.
- `precision`: bf16, fp32, fp16
- `segments`: 4, 8, 16, 32, 64, 128
- `degree`: 0-8 (polynomial degree)
- `method`: chebyshev, fpminimax, remez

**Examples:**
- `sigmoid_bf16_16_3_chebyshev.csv`
- `tanh_fp32_32_5_fpminimax.lut`
- `gelu_bf16_8_2_chebyshev.csv`

## Standalone Usage

The scripts in this directory are wrappers. For direct use:

```bash
cd ~/tt-polynomial-fitter

# Generate coefficients
./generate_coefficients.sh --activation sigmoid

# Convert to LUT
./generate_luts.sh --input-dir coefficients/
```

## Project Structure

- **tt-polynomial-fitter/** - Standalone fitting library
  - `segmentation/` - Fitting algorithms
  - `luts/` - Binary LUT generation
  - `generate_coefficients.sh` - Main coefficient generator
  - `generate_luts.sh` - CSV → LUT converter

- **tt-metal/.../generic_lut_activation/** - Hardware integration
  - `generate_coefficients.sh` - Wrapper script
  - `generate_luts.sh` - Wrapper script
  - Legacy LUT generation scripts

## File Locations

- **Coefficients:** `coefficients/` (CSV tables)
- **Binary LUTs:** `luts_output/` (hardware-ready)
- **Legacy script:** `generate_luts_legacy.sh` (old 598-line version)

## Quick Reference

| Option | Values | Default |
|--------|--------|---------|
| `--activation` | sigmoid, tanh, gelu, ... | (required) |
| `--precision` | bf16, fp32, fp16 | bf16 |
| `--segments` | 4, 8, 16, 32, 64, 128 | 16 |
| `--degree` | 0-8 | 3 |
| `--method` | chebyshev, fpminimax | chebyshev |
| `--domain` | min max | -5 5 |
| `--grid-size` | N | 100 |
