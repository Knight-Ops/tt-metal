#!/usr/bin/env bash
# Clean exp-body disassembly vs native TTNN exp — instruction-level diff.
#
# Settles where the ~0.4us residual vs native exp comes from. BOTH our kernel and
# native use the replay buffer (ttreplay record+replay, confirmed by disasm) and the
# same exponent-ALU math — so the gap is the per-pass BODY INSTRUCTION COUNT.
#
# Native target (from ckernel_sfpu_exp.h::_sfpu_exp_21f_bf16_tti_):
#   BODY_LEN = 21 TTI instrs, recorded once, replayed (ADDR_MOD_6/7 auto-increment).
#   Sequence: SFPLOAD, [SFPMULI scale], SFPMAD, SFPLOADI, SFPSWAP(clamp),
#             SFPEXEXP, SFPEXMAN, SFPSHFT, SFPEXMAN, SFPCAST, SFPMAD(poly1),
#             SFPGT(sign), SFPMAD(poly2), SFPAND, SFPSETEXP, SFP_STOCH_RND, SFPSTORE.
#   Degree-2 poly => 2x SFPMAD for the polynomial.
#
# Usage: ./disasm_exp_body_vs_native.sh   (run from anywhere; needs the device free)
set -euo pipefail
REPO=/localdev/nkapre/tt-metal
PF=/localdev/nkapre/tt-polynomial-fitter
EX=$REPO/tt_metal/programming_examples/generic_lut_activation_embedded
OBJDUMP=$REPO/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
CSV=$(ls $PF/data/coefficients/exp_expalu_*.csv 2>/dev/null | head -1)
export TT_POLY_FIT_DIR=$PF TT_METAL_HOME=$REPO

echo "== building clean exp exponent-ALU kernel (single-seg, min-degree) =="
rm -rf /home/nkapre/.cache/tt-metal-cache 2>/dev/null || true
"$EX/run_csv.sh" "$CSV" --activation exp --precision bf16 --tiles 256 --runs 1 2>&1 | grep -E "^custom_|FAILED" | tail -1

ELF=$(find /home/nkapre/.cache/tt-metal-cache -name trisc1.elf -path "*adhoc*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
echo "ELF: $ELF"
DIS=/tmp/exp_clean_disasm.txt
"$OBJDUMP" -D "$ELF" > "$DIS" 2>/dev/null

echo "== OUR recorded replay body (between 'ttreplay ...,1,1' record and first replay) =="
# record line: ttreplay <s>,<len>,1,1  ; the next <len> SFPU instrs are the body
awk '
  /\tttreplay\t.*,1,1$/ {rec=1; next}
  rec && /\tttreplay\t.*,0,0$/ {exit}
  rec && /\t(sfp|ttsfp)[a-z0-9_]+/ {match($0,/\t((sfp|tt)[a-z0-9_]+)/,m); print m[1]}
' "$DIS" > /tmp/our_body.txt || true
OUR=$(wc -l < /tmp/our_body.txt)
echo "OUR body length: $OUR SFPU instrs (native = 21)"
echo "--- our body mnemonics ---"; cat /tmp/our_body.txt | nl
echo "--- our body histogram ---"; sort /tmp/our_body.txt | uniq -c | sort -rn

echo
echo "== DIFF vs native 21-instr body =="
echo "native: SFPLOAD SFPMULI SFPMAD SFPLOADI SFPSWAP SFPEXEXP SFPEXMAN SFPSHFT SFPEXMAN SFPCAST SFPMAD SFPGT SFPMAD SFPAND SFPSETEXP SFP_STOCH_RND SFPSTORE (=21 w/ degree-2)"
echo "delta = OUR($OUR) - native(21) = $((OUR-21)) extra instrs/pass  (x32 passes x tiles = the residual us)"
echo "ACTION: identify the extra mnemonics above (likely sfpsetman/sfpconfig/sfpstochrnd/extra sfpmad from higher degree or format conversion) and trim them in piecewise_generic.cpp exp_hw_eval."
