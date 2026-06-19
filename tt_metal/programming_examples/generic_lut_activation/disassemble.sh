#!/usr/bin/env bash
# Disassemble a kernel (trisc1 = SFPU math core)
# Usage 1 (non-embedded): ./disassemble.sh <degree> <segments> <csv_file> [binary_args...]
# Usage 2 (any binary):   ./disassemble.sh --binary <binary_path> <csv_file> [binary_args...]
# Example: ./disassemble.sh 8 16 /localdev/nkapre/tt-polynomial-fitter/data/coefficients/gelu_16_8_curvature_any.csv --activation gelu --precision fp32 --range-min -10 --range-max 10
# Example: ./disassemble.sh --binary build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_octic_16 data/coefficients/gelu_16_8_curvature_any.csv

set -euo pipefail

REPO=/localdev/nkapre/tt-metal
OBJDUMP="$REPO/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"

if [[ "${1:-}" == "--binary" ]]; then
    BINARY="${2:?--binary requires a path}"
    # Resolve relative paths against repo root
    [[ "$BINARY" != /* ]] && BINARY="$REPO/$BINARY"
    shift 2
    CSV=${1:?Usage: $0 --binary <path> <csv_file> [binary_args...]}
    shift
    LABEL=$(basename "$BINARY")
    OUTFILE="$REPO/disasm_${LABEL}_trisc1.txt"
    KERNEL_PATTERN="*"
else
    DEGREE=${1:?Usage: $0 <degree> <segments> <csv_file> [binary_args...]}
    SEGMENTS=${2:?}
    CSV=${3:?}
    shift 3
    BINARY="$REPO/build_Release/programming_examples/programming_examples_generic_lut_activation_p${DEGREE}_s${SEGMENTS}"
    OUTFILE="$REPO/disasm_p${DEGREE}_s${SEGMENTS}_trisc1.txt"
    KERNEL_PATTERN="*/piecewise_generic/*"
fi

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found: $BINARY" >&2
    exit 1
fi
if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV" >&2
    exit 1
fi

# Activate venv
source "$REPO/python_env/bin/activate"

export ARCH_NAME=wormhole_b0
export TT_METAL_SKIP_DELETING_BUILT_CACHE=1

cd "$REPO"  # binary needs to run from repo root for soc descriptor paths
echo "Running: $BINARY $CSV $*"
"$BINARY" "$CSV" "$@" 2>&1 | grep -E "(cache|Using kernel|PASSED|FAILED|Error|exception)" || true

# Find the most recently built trisc1.elf for this kernel
ELF=$(find /home/nkapre/.cache/tt-metal-cache -name "trisc1.elf" -path "$KERNEL_PATTERN" \
      -newer "$BINARY" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

if [[ -z "$ELF" ]]; then
    # Fallback: most recent trisc1.elf anywhere in cache
    ELF=$(find /home/nkapre/.cache/tt-metal-cache -name "trisc1.elf" -path "$KERNEL_PATTERN" \
          -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
fi

if [[ -z "$ELF" ]]; then
    echo "ERROR: No trisc1.elf found in cache" >&2
    exit 1
fi

echo "ELF: $ELF"
echo "Disassembling to: $OUTFILE"

"$OBJDUMP" -D "$ELF" > "$OUTFILE" 2>/dev/null

# Summary
TOTAL=$(wc -l < "$OUTFILE")
SFPU=$(grep -c "sfp" "$OUTFILE" || true)
echo "Done: $TOTAL lines, $SFPU SFPU instructions"
echo ""
echo "SFPU instructions:"
grep "sfp" "$OUTFILE" | grep -v "piecewise_generic" | head -40 || true

# Extract repeating Horner core via inter-MAD spacing analysis
python3 - "$OUTFILE" << 'PYEOF'
import re, sys
from collections import Counter

outfile = sys.argv[1]
with open(outfile) as f:
    content = f.read()

# Extract SFPU mnemonics in program order
sfpu = re.findall(r'\t(sfp\w+)', content)
n = len(sfpu)
if not sfpu:
    print("No SFPU instructions found")
    sys.exit(0)

mad_pos = [i for i, m in enumerate(sfpu) if m == 'sfpmad']
if len(mad_pos) < 2:
    print(f"Only {len(mad_pos)} sfpmad — too few to analyse")
    sys.exit(0)

# Gap between consecutive MADs = one Horner step
gaps = [mad_pos[i+1] - mad_pos[i] for i in range(len(mad_pos) - 1)]
gap_counts = Counter(gaps)

print(f"\nTotal SFPU instrs : {n}")
print(f"Total sfpmad      : {len(mad_pos)}")
print(f"\nInter-MAD gap distribution (gap → count):")
for gap, cnt in sorted(gap_counts.items()):
    bar = '█' * min(cnt, 60)
    print(f"  {gap:4d}  {bar} {cnt}")

# Most common gap = canonical Horner step
step_gap, step_freq = gap_counts.most_common(1)[0]
step_frac = step_freq / len(gaps) * 100

# Extract the instruction window [prev_MAD+1 .. this_MAD] for a representative instance
# (instructions that load the coefficient + the MAD itself)
for p1, p2 in zip(mad_pos, mad_pos[1:]):
    if p2 - p1 == step_gap:
        step_instrs = sfpu[p1 + 1 : p2 + 1]   # loads before next MAD, plus that MAD
        break

counts = Counter(step_instrs)
loads  = counts.get('sfpload', 0) + counts.get('sfploadi', 0)
mads   = counts.get('sfpmad', 0)
ccs    = counts.get('sfpsetcc', 0) + counts.get('sfpcompc', 0) + counts.get('sfpencc', 0)
other  = len(step_instrs) - loads - mads - ccs

print(f"\n--- Horner step  (gap={step_gap}, {step_freq}/{len(gaps)} = {step_frac:.0f}% of transitions) ---")
for i, instr in enumerate(step_instrs):
    print(f"  {i:3d}  {instr}")
print(f"\n  loads={loads}  MADs={mads}  cond-codes={ccs}  other={other}")
print(f"  → {loads} loads + 1 MAD per coefficient  (implied: {loads}-word coefficient encoding)")
PYEOF
