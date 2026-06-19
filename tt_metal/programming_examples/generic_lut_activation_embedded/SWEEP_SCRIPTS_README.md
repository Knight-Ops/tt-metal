# Sweep Scripts: Embedded vs CB Mode

## Quick Answer: Should These Be Symlinks to generic_lut_activation?

**NO** - The embedded sweep scripts are **unique** and serve different purposes:

| Aspect | generic_lut_activation | generic_lut_activation_embedded |
|--------|----------------------|--------------------------------|
| LUT Mode | CB mode (runtime LUT) | EMBEDDED mode (compile-time LUT) |
| WORK_DIR | `generic_lut_activation` | `generic_lut_activation_embedded` |
| sweep_all.sh | Local comprehensive sweep | Dual-mode (orchestrator/executor) |
| Scripts Called | sweep_piecewise.sh, sweep_rational.sh | sweep_embedded.sh |

## Status: ✅ Ready for Remote Execution

The embedded sweep scripts are properly configured for remote server execution.

## Quick Start

### Full Sweep on Both Architectures

```bash
cd generic_lut_activation_embedded
./sweep_all.sh --arch both
```

This will:
- Pull latest code from GitHub to remote servers
- Launch tmux sessions (`wormhole_sweep_embedded`, `blackhole_sweep_embedded`)
- Run constant, linear, quadratic, cubic, hexic sweeps
- Continue running after you disconnect

### Targeted Sweep

```bash
# Single activation at specific depth
./sweep_all.sh --arch wormhole_b0 --activation sigmoid --depth 32

# Specific degree for all activations
./sweep_all.sh --arch blackhole --degree cubic
```

### Best Configurations Only

```bash
./sweep_best.sh --activation sigmoid
```

## Monitoring Progress

```bash
# Check tmux sessions
ssh -p $WORMHOLE_PORT $WORMHOLE_HOST 'tmux ls'

# Attach to session
ssh -A -p $WORMHOLE_PORT $WORMHOLE_HOST -t 'tmux attach -t wormhole_sweep_embedded'

# Tail log file
ssh -p $WORMHOLE_PORT $WORMHOLE_HOST 'tail -f /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/sweep_wormhole_b0.log'
```

## Architecture Detection

Scripts auto-detect architecture from hostname:
- Hostname contains "bh" → `ARCH_NAME=blackhole`
- Otherwise → `ARCH_NAME=wormhole_b0`

## Pre-flight Checklist

```bash
# 1. Verify hosts.sh exists
ls -la ../generic_lut_activation/hosts.sh

# 2. Check SSH agent
ssh-add -l

# 3. Verify remote access
source ../generic_lut_activation/hosts.sh
sshpass -p "$WORMHOLE_PASSWORD" ssh -o StrictHostKeyChecking=no -p $WORMHOLE_PORT $WORMHOLE_HOST 'hostname'
```

## Key Differences: Embedded vs CB Mode

| Feature | CB Mode | EMBEDDED Mode |
|---------|---------|---------------|
| **LUT Storage** | L1 circular buffer (~1KB) | Compiled into kernel (0 bytes L1) |
| **Compile Time** | Fast (1 config) | Slow (1 per config) |
| **Flexibility** | Runtime LUT changes | Must recompile |
| **Use Case** | Production | Benchmarking, minimal overhead |

## Script Responsibilities

### sweep_all.sh
- **Local mode**: Orchestrates remote execution via SSH
- **Remote mode**: Executes sweeps by calling sweep_embedded.sh
- Handles tmux session management
- Auto-detects architecture

### sweep_best.sh
- Runs only best configurations from `$TT_POLY_FIT_DIR/best.csv`
- No remote orchestration (runs locally)
- Tests optimal degree/depth for each activation

### sweep_embedded.sh
- Called by sweep_all.sh
- Performs actual embedded LUT sweeps
- Builds and runs embedded binaries

## Expected Runtime

| Sweep Type | Duration |
|-----------|----------|
| Targeted (1 activation, 1 depth) | 5-10 min |
| Single degree (all activations) | 1-2 hours |
| Full sweep (5 degrees) | 6-10 hours |

## Output Files

Results saved to:
```
generic_lut_activation_embedded/
├── wormhole_*.csv          (Wormhole results)
├── blackhole_*.csv         (Blackhole results)
├── sweep_wormhole_b0.log   (Wormhole log)
├── sweep_blackhole.log     (Blackhole log)
└── data/hardware_outputs/  (Raw outputs)
```

## Retrieving Results

```bash
# Download from remote server
scp -P $WORMHOLE_PORT $WORMHOLE_HOST:/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/wormhole_*.csv ./
```

## Troubleshooting

### hosts.sh not found
**Fix:** Ensure you're in `generic_lut_activation_embedded/` and hosts.sh exists in `../generic_lut_activation/hosts.sh`

### Branch doesn't exist on remote
**Fix:**
```bash
git push origin $(git branch --show-current)
```

### SSH connection fails
**Fix:** Verify credentials in hosts.sh:
```bash
source ../generic_lut_activation/hosts.sh
echo "$WORMHOLE_HOST:$WORMHOLE_PORT"
```

## Summary

✅ Scripts are ready to run
✅ No symlinks needed (intentionally unique)
✅ Remote orchestration configured
✅ Use for EMBEDDED_LUT benchmarking
✅ For CB mode sweeps, use `../generic_lut_activation/sweep_*.sh`
