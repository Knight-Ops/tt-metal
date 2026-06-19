# Data Management Strategy

## Overview

All sweep results are committed to the git repository for easy analysis and plotting. This is the cleanest approach for benchmark data.

## Directory Structure

```
data/
├── blackhole/
│   ├── best/
│   │   ├── gelu.csv          # Per-activation results
│   │   ├── sigmoid.csv
│   │   └── ...
│   ├── polynomial/
│   │   ├── gelu.csv
│   │   └── ...
│   ├── rational/
│   │   └── ...
│   ├── native_sfpu/
│   │   └── ...
│   ├── best_results.csv      # Consolidated
│   ├── polynomial_results.csv
│   ├── rational_results.csv
│   └── native_sfpu_results.csv
└── wormhole_b0/
    └── ... (same structure)

plots/
├── blackhole/
│   ├── error_vs_runtime/
│   │   └── *.png
│   └── pareto/
│       └── *.png
└── wormhole_b0/
    └── ...
```

## What Gets Committed

✅ **Per-activation CSVs** (`data/{arch}/{type}/{activation}.csv`)
- Individual results for each activation
- Used for detailed analysis
- Enables per-activation comparisons

✅ **Consolidated CSVs** (`data/{arch}/{type}_results.csv`)
- Merged results for all activations
- Used for cross-activation comparisons
- Easier to load in analysis scripts

✅ **Comparison plots** (`plots/{arch}/*.png`)
- Generated visualizations
- Pareto frontiers
- Error vs runtime plots

❌ **Hardware output logs** (`data/hardware_outputs/`)
- Too large and verbose
- Not needed for analysis
- Available in CI artifacts if needed

## Git Configuration

See `.gitattributes` for configuration:
- CSVs marked as binary (-diff) to avoid merge conflicts
- Can enable Git LFS if files become large (>1MB)
- Hardware outputs excluded from diffs

## CI Workflow

When `commit-results: true`:

1. **Sweep jobs** generate per-activation CSVs in parallel
2. **Consolidate job** merges into consolidated CSVs
3. **Compare job** generates plots
4. **Commit job** adds all files to git and creates PR

**Result:** PR with all data ready for analysis!

## Local Usage

### Generate and commit results locally:

```bash
# Run sweep
./sweep_best.sh --activation gelu

# Result is at: data/blackhole/best/gelu.csv

# Commit it
git add data/blackhole/best/gelu.csv
git commit -m "Add gelu sweep results"
```

### Consolidate multiple runs:

```bash
# Run multiple activations
for act in gelu sigmoid tanh; do
  ./sweep_best.sh --activation $act
done

# Consolidate
./consolidate_results.sh --arch blackhole

# Commit everything
git add data/blackhole/
git commit -m "Update blackhole sweep results"
```

## Analysis Workflow

All data is in git, so analysis is straightforward:

```python
import pandas as pd

# Load per-activation data
gelu = pd.read_csv('data/blackhole/best/gelu.csv')
sigmoid = pd.read_csv('data/blackhole/best/sigmoid.csv')

# Or load consolidated
all_results = pd.read_csv('data/blackhole/best_results.csv')

# Compare
import matplotlib.pyplot as plt
plt.plot(gelu['runtime_ms'], gelu['max_error'], label='gelu')
plt.plot(sigmoid['runtime_ms'], sigmoid['max_error'], label='sigmoid')
plt.legend()
plt.savefig('plots/blackhole/my_analysis.png')
```

## Benefits of This Approach

✅ **Simple** - Everything in one place
✅ **Trackable** - Git history shows performance changes
✅ **Shareable** - Just clone the repo
✅ **Analyzable** - CSVs ready for pandas/matplotlib
✅ **Reviewable** - PRs show data changes

## Size Management

### Current size (estimate):
- Per-activation CSV: ~2-10 KB
- Consolidated CSV: ~50-200 KB
- PNG plots: ~50-200 KB each

**Total per architecture:** ~5-10 MB

### If files grow large:

1. **Enable Git LFS:**
   ```bash
   # Install git-lfs
   git lfs install

   # Enable for CSVs (uncomment in .gitattributes)
   git lfs track "data/**/*.csv"

   # Commit .gitattributes
   git add .gitattributes
   git commit -m "Enable Git LFS for CSV files"
   ```

2. **Or compress old results:**
   ```bash
   # Archive old data
   tar czf data/archive/blackhole-2025-01.tar.gz data/blackhole/
   git add data/archive/blackhole-2025-01.tar.gz
   git rm data/blackhole/best/*.csv
   git commit -m "Archive January 2025 results"
   ```

## Best Practices

### 1. Commit regularly
After significant sweeps, commit results:
```bash
git add data/ plots/
git commit -m "Update sweep results: all activations on blackhole"
```

### 2. Use descriptive commit messages
```bash
git commit -m "Sweep results: baseline before optimization

- All 30 activations on blackhole
- Best configurations from polynomial fitter
- Comparison with native SFPU included"
```

### 3. Tag important baselines
```bash
git tag -a baseline-v1.0 -m "Pre-release baseline"
git push origin baseline-v1.0
```

### 4. Keep plots updated
Regenerate plots when data changes:
```bash
python compare_best_results.py --arch blackhole
git add plots/blackhole/
git commit -m "Update comparison plots"
```

### 5. Document changes
Use PR descriptions to explain significant changes:
- Why results changed
- What was optimized
- Performance improvements

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Commit all CSVs** ✅ | Simple, trackable, analyzable | Repo size grows |
| Artifacts only | Clean repo | Expires, not trackable |
| Separate data repo | Clean separation | Complex, extra repo |
| Git LFS | Efficient storage | Requires setup |

**Our choice:** Commit all CSVs directly
- Simplest for users
- Best for analysis workflow
- Size is manageable
- Can enable LFS later if needed

## Migration from Artifacts

If you have artifact-only results, migrate them:

```bash
# Download artifacts from GitHub Actions
gh run download 123456789

# Copy to data directory
mkdir -p data/blackhole/best/
cp artifacts/blackhole-best-*/*.csv data/blackhole/best/

# Consolidate
./consolidate_results.sh --arch blackhole

# Commit
git add data/
git commit -m "Import historical sweep results"
```

## Troubleshooting

### "CSV files are too large for diff"
CSVs are marked as binary in `.gitattributes` to avoid this.

### "Git push rejected - file too large"
Enable Git LFS (see above) or compress large files.

### "Merge conflict in CSV"
CSVs shouldn't have merge conflicts (binary diffs disabled).
If it happens, choose the newer version:
```bash
git checkout --theirs data/blackhole/best/gelu.csv
```

### "Old results are cluttering the repo"
Archive old data periodically:
```bash
# Create monthly archives
mkdir -p data/archive/
tar czf data/archive/2025-01.tar.gz data/*/best/*.csv
git rm data/*/best/*.csv
git add data/archive/2025-01.tar.gz
```
