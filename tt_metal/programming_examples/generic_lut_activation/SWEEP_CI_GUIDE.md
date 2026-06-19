# Complete CI/CD Guide for LUT Activation Sweeps

## Overview

This guide covers the complete workflow for running parallel activation sweeps in CI, from triggering builds to analyzing results.

## Quick Start

### 1. Trigger a Sweep Run

Go to: **GitHub Actions → LUT Activation Sweeps → Run workflow**

**Minimal configuration:**
```yaml
arch: blackhole
sweep-type: best
activations: all
skip-build: false
embedded: false
commit-results: false
```

### 2. Monitor Progress

- **Build job:** ~10-15 minutes (builds tt-metal once)
- **Sweep jobs:** 8 parallel jobs at a time (matrix strategy)
- **Consolidate:** ~2 minutes (merges all CSVs)
- **Compare:** ~3 minutes (generates plots)

**Total time for 30 activations:** ~25-30 minutes (vs 150 minutes sequential)

### 3. Download Results

After completion, find **Artifacts** section at bottom of workflow page:
- `{arch}-consolidated-results` - Final CSVs (90 day retention)
- `{arch}-comparison-plots` - PNG plots (30 day retention)
- `{arch}-{sweep_type}-{activation}` - Individual CSV per activation (30 days)

## Compilation Optimization

### Non-Embedded (generic_lut_activation)

**Build once, run many:**
- Targets are activation-agnostic
- One binary works for all activations
- Fast compile time (~5 min for all targets)

```bash
# Single build for all activations
./build_metal.sh --build-programming-examples

# Run any activation
./sweep_best.sh --activation gelu --skip-build
./sweep_best.sh --activation sigmoid --skip-build
./sweep_best.sh --activation tanh --skip-build
```

### Embedded (generic_lut_activation_embedded)

**Activation-specific builds:**
- Each activation needs separate targets
- Generate only what you need with `--activation`
- Dramatically faster compile times

```bash
# Generate CMake targets for ONE activation
python3 tools/generate_cmake_embedded.py \
  --activation gelu \
  --best \
  --output cmake/EmbeddedTargets.cmake

# Build only gelu targets (fast!)
cmake --build build_Release \
  --target programming_examples_generic_lut_activation_embedded_gelu

# Run sweep
./sweep_best.sh --activation gelu --skip-build
```

**Compile time comparison:**
- All activations (29): ~30-45 minutes
- Single activation: ~2-3 minutes
- **Speedup: 10-15x faster**

### CI Strategy

The workflow automatically uses activation-specific builds for embedded:

```yaml
# For each activation in parallel:
1. Generate CMake for this activation only
2. Build only this activation's targets
3. Run sweep
4. Upload results
```

## Data Management Options

### Option 1: Artifacts Only (Default) ✅

**How it works:**
- CSVs stored as GitHub Actions artifacts
- Retention: 30 days (per-activation), 90 days (consolidated)
- Download from workflow run page

**Pros:**
- Clean repo, no git bloat
- Fast workflow execution
- Easy cleanup

**Cons:**
- Artifacts expire after retention period
- Need to download manually for analysis

**When to use:**
- Development/iteration
- Temporary benchmarks
- CI validation

### Option 2: Commit Consolidated Results

**How it works:**
- Set `commit-results: true` in workflow
- Creates PR with consolidated CSVs and plots
- Only commits final results (not per-activation)

**Pros:**
- Historical tracking in git
- Easy comparison across commits
- Reviewable changes

**Cons:**
- Adds data to git history
- Requires PR merge
- Slower workflow

**When to use:**
- Release benchmarks
- Performance tracking
- Baseline establishment

**Workflow:**
```yaml
commit-results: true  # Enable in workflow inputs
```

**Result:**
- Auto-created PR: "[blackhole] Update sweep results - best"
- Contains: `data/blackhole/*_results.csv` + plots
- Labeled: `benchmark`, `sweep-results`, `blackhole`
- Assigned: workflow trigger user

### Option 3: Separate Data Repository

**How it works:**
- Push results to `tt-metal-benchmarks` repo
- Keep code and data separate
- Unlimited retention

**Setup required:**
```yaml
# Add to workflow (custom implementation)
- name: Push to data repo
  run: |
    git clone git@github.com:tenstorrent/tt-metal-benchmarks.git
    cp data/blackhole/*.csv tt-metal-benchmarks/sweeps/
    cd tt-metal-benchmarks
    git add . && git commit -m "Update"
    git push
```

**Pros:**
- Clean separation
- Unlimited history
- No main repo bloat

**Cons:**
- Extra repo to manage
- More complex workflow
- Requires setup

**When to use:**
- Long-term benchmarking
- Public performance data
- Cross-team sharing

## Workflow Configuration

### Input Parameters

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `arch` | choice | blackhole, wormhole_b0, both | blackhole | Architecture to test |
| `sweep-type` | choice | best, polynomial, rational, native_sfpu, all | best | Which sweep to run |
| `activations` | string | comma-separated or "all" | all | Activations to test |
| `skip-build` | boolean | true/false | false | Skip build phase |
| `embedded` | boolean | true/false | false | Use embedded variants |
| `commit-results` | boolean | true/false | false | Create PR with results |

### Example Configurations

**1. Quick validation (single activation):**
```yaml
arch: blackhole
sweep-type: best
activations: gelu
skip-build: false
embedded: false
commit-results: false
```
**Time:** ~8 minutes
**Use:** Validate changes, quick test

**2. Full sweep (all activations):**
```yaml
arch: blackhole
sweep-type: all
activations: all
skip-build: false
embedded: false
commit-results: false
```
**Time:** ~25-30 minutes
**Use:** Comprehensive testing

**3. Embedded sweep with commit:**
```yaml
arch: both
sweep-type: best
activations: all
skip-build: false
embedded: true
commit-results: true
```
**Time:** ~40-50 minutes
**Use:** Release benchmarks

**4. Targeted sweep (multiple activations):**
```yaml
arch: blackhole
sweep-type: polynomial
activations: gelu,sigmoid,tanh,relu
skip-build: false
embedded: false
commit-results: false
```
**Time:** ~10 minutes
**Use:** Focused testing

## Self-Hosted Runner Setup

### Required Labels

- `blackhole` - For Blackhole hardware
- `wormhole_b0` - For Wormhole B0 hardware
- `self-hosted` - Standard GitHub label

### Installation

**On Blackhole server:**
```bash
# Install runner
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure (get token from: Settings → Actions → Runners → New runner)
./config.sh \
  --url https://github.com/tenstorrent/tt-metal \
  --token YOUR_REGISTRATION_TOKEN \
  --labels blackhole,self-hosted \
  --name blackhole-runner-1

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

**Verify:**
```bash
sudo ./svc.sh status
# Should show: "Active: active (running)"
```

### Runner Requirements

**Per runner:**
- Tenstorrent hardware (Blackhole or Wormhole)
- Ubuntu 22.04 or 24.04
- 16GB+ RAM
- 100GB+ disk space
- Network access to GitHub
- `/opt/venv/bin/tt-smi` available

**For parallel execution:**
- 4-8 runners per architecture recommended
- Adjust `max-parallel` in workflow based on available runners

### Monitoring Runners

```bash
# Check runner status
sudo ./svc.sh status

# View runner logs
sudo journalctl -u actions.runner.* -f

# Check hardware
/opt/venv/bin/tt-smi

# Check disk space
df -h
```

### Maintenance

```bash
# Stop runner
sudo ./svc.sh stop

# Clean up old artifacts
rm -rf ~/actions-runner/_work/_temp/*
rm -rf ~/actions-runner/_diag/*

# Update runner
cd ~/actions-runner
./config.sh remove
# Download new version
./config.sh --url ... --token ...
sudo ./svc.sh install
sudo ./svc.sh start
```

## Parallelization Strategy

### Matrix Configuration

```yaml
strategy:
  fail-fast: false
  max-parallel: 8
  matrix:
    arch: [blackhole, wormhole_b0]
    activation: [gelu, sigmoid, tanh, ...]
```

### Execution Flow

**With 8 runners, 30 activations:**

```
Round 1: [gelu]      [sigmoid]    [tanh]      [relu]      [silu]      [softplus]  [hardswish] [swish]
Round 2: [mish]      [selu]       [elu]       [celu]      [leaky]     [prelu]     [relu6]     [hardtanh]
Round 3: [softsign]  [tanhshrink] [softshrink][hardshrink][threshold] [logsigmoid][exp]       [sin]
Round 4: [cos]       [sinh]       [cosh]      [atanh]     [erf]       [hardsigmoid]
```

**Time per round:** ~5 minutes
**Total:** 4 rounds × 5 min = 20 minutes + consolidation (2 min) = **22 minutes**

### Scaling Recommendations

| Activations | Runners | Parallel Jobs | Estimated Time |
|-------------|---------|---------------|----------------|
| 30 | 4 | 4 | ~35 min |
| 30 | 8 | 8 | ~22 min |
| 30 | 16 | 16 | ~15 min |
| 60 (2 archs) | 8 | 8 | ~40 min |
| 60 (2 archs) | 16 | 16 | ~22 min |

## Local Testing

Test the exact same workflow locally:

```bash
# 1. Single activation (mimics one CI job)
export ARCH_NAME=blackhole
./sweep_best.sh --activation gelu --skip-build

# 2. Check result
ls -lh data/blackhole/best/gelu.csv

# 3. Run multiple activations (simulate parallel)
for act in gelu sigmoid tanh; do
  ./sweep_best.sh --activation $act --skip-build &
done
wait

# 4. Consolidate (mimics consolidate job)
./consolidate_results.sh --arch blackhole

# 5. Compare (mimics compare job)
python compare_best_results.py --arch blackhole
```

## Troubleshooting

### Common Issues

**1. "No runner available"**
```
Error: No self-hosted runner found with label 'blackhole'
```
**Solution:**
- Check runner status: `Settings → Actions → Runners`
- Restart runner: `sudo ./svc.sh restart`
- Verify labels match workflow

**2. "Build cache not found"**
```
Error: Cache not found for key: tt-metal-build-blackhole-abc123
```
**Solution:**
- Don't use `skip-build: true` on first run
- Cache expires after 7 days
- Re-run build job if needed

**3. "Consolidation failed - no CSVs"**
```
Error: No CSV files found in data/blackhole/best/
```
**Solution:**
- Check sweep jobs completed successfully
- Verify artifacts uploaded correctly
- Check artifact download pattern matches

**4. "Compile error - target not found"**
```
Error: No rule to make target 'programming_examples_..._gelu'
```
**Solution:**
- For embedded: ensure CMake regenerated
- Check activation name matches CSV
- Rebuild with correct --activation flag

**5. "Out of disk space"**
```
Error: No space left on device
```
**Solution:**
```bash
# On runner
sudo ./svc.sh stop
du -sh ~/actions-runner/_work/*  # Find large directories
rm -rf ~/actions-runner/_work/_temp/*
rm -rf ~/actions-runner/_diag/*
sudo ./svc.sh start
```

### Debug Mode

Enable debug logging in workflow:

```yaml
# In workflow file, add:
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Best Practices

### 1. Start Small
✅ Test with 1-2 activations first
✅ Verify artifacts download correctly
✅ Check consolidation works

### 2. Use Descriptive Names
When running manually, add context:
- "Pre-release validation - gelu,sigmoid,tanh"
- "Full sweep for performance baseline"
- "Debug polynomial issue on blackhole"

### 3. Monitor Resources
```bash
# On runner, watch:
htop              # CPU/memory usage
watch -n 1 nvidia-smi  # GPU if applicable
df -h             # Disk space
/opt/venv/bin/tt-smi  # Tenstorrent hardware
```

### 4. Clean Regularly
```bash
# Weekly cleanup on runners:
sudo ./svc.sh stop
find ~/actions-runner/_work -type f -mtime +7 -delete
docker system prune -af  # If using Docker
sudo ./svc.sh start
```

### 5. Version Control
- Keep workflow file in sync with scripts
- Test changes in feature branch first
- Document major configuration changes

## Integration with Main CI

Add to post-commit workflow:

```yaml
# In .github/workflows/all-post-commit-workflows.yaml

lut-activation-validation:
  needs: find-changed-files
  if: ${{ needs.find-changed-files.outputs.test-programming-examples == 'true' }}
  uses: ./.github/workflows/lut-activation-sweeps.yaml
  with:
    arch: blackhole
    sweep-type: best
    activations: gelu,sigmoid  # Quick validation only
    skip-build: false
    embedded: false
    commit-results: false
```

## Performance Metrics

Track these metrics for optimization:

- **Build time:** Target < 15 min
- **Per-activation time:** Target < 5 min
- **Consolidation time:** Target < 2 min
- **Total workflow time:** Target < 30 min (30 activations)

## Future Enhancements

Potential improvements:

1. **Dynamic matrix generation** - Read activations from CSV
2. **Failure retry** - Auto-retry failed activations
3. **Caching improvements** - Per-activation build caches
4. **Result comparison** - Auto-comment on PRs with comparisons
5. **Performance alerts** - Notify on regressions
