# Parallel Sweeps with Per-Activation CSVs

This document explains how to run sweeps in parallel (across multiple machines) and consolidate results.

## Directory Structure

```
data/
├── blackhole/
│   ├── best/
│   │   ├── gelu.csv           # Per-activation results
│   │   ├── sigmoid.csv
│   │   └── tanh.csv
│   ├── polynomial/
│   │   ├── gelu.csv
│   │   └── ...
│   ├── rational/
│   │   ├── gelu.csv
│   │   └── ...
│   ├── native_sfpu/
│   │   ├── gelu.csv
│   │   └── ...
│   ├── best_results.csv         # Consolidated (all activations)
│   ├── polynomial_results.csv   # Consolidated
│   ├── rational_results.csv     # Consolidated
│   └── native_sfpu_results.csv  # Consolidated
├── wormhole_b0/
│   └── ... (same structure)
└── hardware_outputs/
    └── ... (detailed output CSVs)
```

## Usage

### Single Activation (Sequential)

Run a single activation - results go to per-activation CSV:

```bash
# Writes to: data/blackhole/best/gelu.csv
./sweep_best.sh --activation gelu

# Writes to: data/blackhole/polynomial/sigmoid.csv
./sweep_polynomial.sh --activation sigmoid --degrees 3,5,7

# Writes to: data/blackhole/rational/tanh.csv
./sweep_rational.sh --activation tanh
```

### All Activations (Sequential)

Run all activations - results go to consolidated CSV:

```bash
# Writes to: data/blackhole/best_results.csv
./sweep_best.sh

# Writes to: data/blackhole/polynomial_results.csv
./sweep_polynomial.sh --degrees 3,5,7,9
```

### Distributed Execution (CI/Multiple Machines)

On each machine, run a subset of activations:

```bash
# Machine 1 with Blackhole card
export ARCH_NAME=blackhole
./sweep_best.sh --activation gelu --skip-build
./sweep_best.sh --activation sigmoid --skip-build
./sweep_best.sh --activation tanh --skip-build

# Machine 2 with Blackhole card
export ARCH_NAME=blackhole
./sweep_best.sh --activation relu --skip-build
./sweep_best.sh --activation silu --skip-build
./sweep_best.sh --activation softplus --skip-build
```

Each machine writes to its own per-activation CSV:
- Machine 1: `data/blackhole/best/gelu.csv`, `data/blackhole/best/sigmoid.csv`, ...
- Machine 2: `data/blackhole/best/relu.csv`, `data/blackhole/best/silu.csv`, ...

### Consolidation

After distributed runs complete, consolidate all per-activation CSVs:

```bash
# Merge all data/blackhole/{type}/*.csv into data/blackhole/{type}_results.csv
./consolidate_results.sh --arch blackhole
```

Output:
```
Processing best...
  Input: data/blackhole/best/*.csv (6 files)
  Output: data/blackhole/best_results.csv
  ✓ Consolidated 12 rows

Processing polynomial...
  Input: data/blackhole/polynomial/*.csv (6 files)
  Output: data/blackhole/polynomial_results.csv
  ✓ Consolidated 48 rows

...
```

### GitHub Actions Example

```yaml
jobs:
  sweep-blackhole:
    runs-on: self-hosted-blackhole
    strategy:
      matrix:
        activation: [gelu, sigmoid, tanh, relu, silu, softplus]
    steps:
      - name: Run sweep for ${{ matrix.activation }}
        run: |
          export ARCH_NAME=blackhole
          ./sweep_best.sh --activation ${{ matrix.activation }} --skip-build

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: blackhole-best-${{ matrix.activation }}
          path: data/blackhole/best/${{ matrix.activation }}.csv

  consolidate:
    needs: sweep-blackhole
    runs-on: ubuntu-latest
    steps:
      - name: Download all results
        uses: actions/download-artifact@v3
        with:
          path: artifacts/

      - name: Copy to data directory
        run: |
          mkdir -p data/blackhole/best
          cp artifacts/blackhole-best-*/*.csv data/blackhole/best/

      - name: Consolidate
        run: ./consolidate_results.sh --arch blackhole

      - name: Upload consolidated results
        uses: actions/upload-artifact@v3
        with:
          name: blackhole-consolidated
          path: data/blackhole/*_results.csv
```

## Backward Compatibility

The old consolidated file paths still work:

```bash
# Old path (deprecated but still works after consolidation)
data/blackhole_best_results.csv

# New path (preferred)
data/blackhole/best_results.csv
```

Scripts will automatically create the new directory structure. The `compare_best_results.py` script has been updated to use the new paths.

## Best Practices

1. **Local development**: Use `--activation` flag for faster iteration
2. **CI/Multiple machines**: Use `--activation` flag + consolidation script
3. **Full sweeps**: Run without `--activation` flag (writes directly to consolidated file)
4. **Always run consolidation** after distributed execution

## Notes

- Per-activation CSVs only created when using `--activation` flag
- Consolidated CSVs created when running all activations at once
- Architecture auto-detected from hostname or `ARCH_NAME` environment variable
- Both `generic_lut_activation` and `generic_lut_activation_embedded` use same structure
