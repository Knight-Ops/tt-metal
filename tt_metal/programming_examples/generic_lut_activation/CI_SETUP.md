# CI/CD Setup for LUT Activation Sweeps

This document explains how to run parallel sweeps using GitHub Actions CI.

## Quick Start

### Trigger a Manual Sweep

Go to: **Actions → LUT Activation Sweeps → Run workflow**

**Example configurations:**

1. **Quick test (single activation):**
   - Architecture: `blackhole`
   - Sweep type: `best`
   - Activations: `gelu`
   - Skip build: `false`

2. **Full sweep (all activations, parallel):**
   - Architecture: `blackhole`
   - Sweep type: `all`
   - Activations: `all` (default)
   - Skip build: `false`

3. **Targeted sweep (multiple activations):**
   - Architecture: `both`
   - Sweep type: `polynomial`
   - Activations: `gelu,sigmoid,tanh,relu`
   - Skip build: `false`

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Setup Job                              │
│  - Parse activation list                                     │
│  - Determine architecture(s)                                 │
│  - Generate matrix configuration                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    2. Build Job (Optional)                   │
│  Runs on: self-hosted runners (per architecture)            │
│  - Checkout code                                             │
│  - Build TT-Metal with programming examples                 │
│  - Cache build artifacts                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    3. Sweep Jobs (Parallel)                  │
│  Matrix: arch × activation                                   │
│  Runs on: self-hosted runners (per architecture)            │
│  Max parallel: 8 jobs at once                               │
│                                                              │
│  For each (arch, activation) pair:                          │
│  - Restore build cache                                       │
│  - Run: sweep_X.sh --activation Y --skip-build              │
│  - Upload: data/{arch}/{type}/{activation}.csv              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    4. Consolidate Job                        │
│  Runs on: ubuntu-latest                                      │
│  Per architecture:                                           │
│  - Download all per-activation CSVs                          │
│  - Run consolidate_results.sh                               │
│  - Upload: data/{arch}/{type}_results.csv                   │
│  - Generate summary                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    5. Compare Job (Optional)                 │
│  Runs on: ubuntu-latest                                      │
│  - Download consolidated results                             │
│  - Compare with RVV baseline                                 │
│  - Generate plots                                            │
│  - Upload comparison artifacts                               │
└──────────────────────────────────────────────────────────────┘
```

## Parallelization Strategy

### Example: 30 activations × blackhole

Without parallelization:
- **Sequential:** 30 activations × 5 min/activation = **150 minutes**

With CI parallelization (max 8 parallel):
- **Round 1:** 8 activations in parallel (5 min)
- **Round 2:** 8 activations in parallel (5 min)
- **Round 3:** 8 activations in parallel (5 min)
- **Round 4:** 6 activations in parallel (5 min)
- **Total:** ~20-25 minutes + consolidation (2 min) = **~25 minutes**

**Speedup:** ~6x faster

## Self-Hosted Runners

The workflow requires self-hosted runners with labels:
- `blackhole` - Runners with Blackhole hardware
- `wormhole_b0` - Runners with Wormhole B0 hardware

### Setup Self-Hosted Runners

1. **On your Blackhole server:**
   ```bash
   # Install GitHub runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
     https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

   # Configure runner
   ./config.sh --url https://github.com/YOUR_ORG/tt-metal \
     --token YOUR_TOKEN --labels blackhole,self-hosted

   # Install and start service
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

2. **On your Wormhole server:**
   ```bash
   # Same steps, but use --labels wormhole_b0,self-hosted
   ```

## Artifacts

All artifacts are uploaded and available in the workflow summary:

### Per-Activation Results
- **Name:** `{arch}-{sweep_type}-{activation}`
- **Contents:** Single CSV file with results for one activation
- **Retention:** 30 days

### Consolidated Results
- **Name:** `{arch}-consolidated-results`
- **Contents:** All `*_results.csv` files (best, polynomial, rational, native_sfpu)
- **Retention:** 90 days

### Comparison Plots
- **Name:** `{arch}-comparison-plots`
- **Contents:** PNG plots comparing hardware vs RVV baseline
- **Retention:** 30 days

## Advanced Usage

### Custom Activation Lists

Test only specific activations:
```yaml
activations: "gelu,sigmoid,tanh"
```

### Skip Build Phase

If binaries are already built (e.g., running multiple sweep types):
```yaml
skip-build: true
```

### Embedded Variants

Run embedded LUT sweeps instead:
```yaml
embedded: true
```

### Both Architectures

Run on both Blackhole and Wormhole:
```yaml
arch: both
```

## Local Testing

Test the same workflow locally:

```bash
# Single activation (mimics one CI job)
./sweep_best.sh --activation gelu --skip-build

# Upload result (mimics artifact upload)
# Result is at: data/blackhole/best/gelu.csv

# Consolidate (mimics consolidate job)
./consolidate_results.sh --arch blackhole
```

## Monitoring

### Check Progress

1. Go to **Actions** tab
2. Click on your workflow run
3. Click on **sweep** job
4. Expand matrix to see all parallel jobs

### View Results

1. Wait for workflow to complete
2. Scroll down to **Artifacts** section
3. Download `{arch}-consolidated-results`

### Debugging Failures

If a specific activation fails:
1. Click on the failed job in the matrix
2. Expand the "Run sweep" step
3. Check error logs
4. Re-run just that activation:
   ```bash
   # On the specific runner
   ./sweep_best.sh --activation FAILED_ACTIVATION --skip-build
   ```

## Cost Optimization

### Minimize Build Time

- Use `--skip-build` when running multiple sweep types
- Cache build artifacts between workflow runs

### Minimize Sweep Time

- Use `--timeout` to prevent hanging tests
- Use `--activation` to target specific functions

### Resource Limits

- Adjust `max-parallel` in workflow (default: 8)
- Lower for fewer runners, higher for more

## Integration with Existing CI

Add to your post-commit workflow:

```yaml
# In .github/workflows/all-post-commit-workflows.yaml

lut-activation-sweeps:
  needs: find-changed-files
  if: ${{ needs.find-changed-files.outputs.test-programming-examples == 'true' }}
  uses: ./.github/workflows/lut-activation-sweeps.yaml
  with:
    arch: blackhole
    sweep-type: best
    activations: all
    skip-build: false
```

## Troubleshooting

### Runner Not Found

**Error:** `No self-hosted runner found with label 'blackhole'`

**Solution:**
- Verify runners are online: Settings → Actions → Runners
- Check runner labels match workflow requirements

### Build Cache Miss

**Error:** `Cache not found for key: tt-metal-build-blackhole-abc123`

**Solution:**
- Don't use `--skip-build` on first run
- Re-run build job if cache expired

### Consolidation Fails

**Error:** `No CSV files found in data/blackhole/best/`

**Solution:**
- Check that sweep jobs completed successfully
- Verify artifact uploads succeeded
- Check artifact download patterns match

### Out of Disk Space

**Error:** `No space left on device`

**Solution:**
```bash
# On runner, clean old artifacts
sudo ./svc.sh stop
rm -rf _work/_temp/*
rm -rf _diag/*
sudo ./svc.sh start
```

## Best Practices

1. **Start small:** Test with 1-2 activations first
2. **Use descriptive names:** Include date/purpose in manual runs
3. **Monitor resources:** Check runner CPU/memory during sweeps
4. **Clean regularly:** Remove old artifacts to save space
5. **Document changes:** Update this file when modifying workflows
