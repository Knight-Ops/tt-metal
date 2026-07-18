# TT Performance Debugging Guide

**A one-stop shop for taking any Tenstorrent (tt-metal / TT-NN) implementation, finding out *why* it is
slow, and identifying *what to fix* to get the speedup.**

This lives in the Qwen3.6 model dir and uses it for concrete examples, but the tooling
(Tracy, TT-NN reports, NoC/perf-counter capture, memory triage) is generic — the same commands work
for any model or op by swapping the target script.

Contents:
- [§0 Performance triage playbook](#0-performance-triage-playbook) — the decision flow: slow → cause → fix
- [§1 Tracy — performance profiling](#1-tracy--performance-profiling) (host + device + **NoC** + perf counters)
- [§2 TT-NN reports](#2-tt-nn-reports--op-graph-buffers-tensors-generatedttnn) — op graph / buffers / tensors
- [§3 Device memory (OOM triage)](#3-device-memory-oom-triage)
- [§4 Numerical debugging (PCC)](#4-numerical-debugging-pcc--comparison-mode)
- [§5 Discipline / gotchas](#5-discipline--gotchas)
- [§6 Related docs](#6-related-docs)

### The toolchains at a glance
Three **independent** systems live here; the most common mistake is confusing them:

| Tool | Turns on via | Writes to | Answers |
|------|-------------|-----------|---------|
| **Tracy** (host/device/NoC perf) | `python -m tracy` wrapper | `generated/profiler/` | "Where is time going? per-op timing, host-dispatch gaps, NoC/DRAM BW" |
| **TT-NN reports** (op-graph / buffers) | `ttnn.CONFIG` / `TTNN_CONFIG_OVERRIDES` | `generated/ttnn/reports/<name>/db.sqlite` | "What ops ran, what buffers/tensors, op graph" |
| **Device memory state** (live alloc) | `ttnn.dump_device_memory_state()` API | stdout / log | "L1/DRAM high-water mark right now (OOM triage)" |

> **Key fact:** running under `python -m tracy` does **nothing** to produce `generated/ttnn` reports,
> and setting `TTNN_CONFIG_OVERRIDES` does **nothing** to produce Tracy traces. They are separate.
> Neither requires `pytest` — they key off env vars / `import ttnn`, so they work with `demo.py`,
> `server.py`, or any script.

All commands assume the repo venv: `./python_env/bin/python`. Qwen examples use `QWEN36_LAYERS=4` for
fast iteration (per-op-*type* timing generalizes; scale by 30 gated-delta + 10 attn + 40 MoE for a
per-token estimate) and `QWEN36_LAYERS=40` for the authoritative number.

---

## 0. Performance triage playbook

The goal of profiling is not "collect data" — it's to answer, in order: **(1) is the bottleneck host or
device? (2) which op dominates? (3) is that op compute-, bandwidth-, or overhead-bound? (4) what changes
that?** Work top-down; each step tells you which tool to reach for next.

### Step 1 — Establish the real baseline (no profiler)
Before profiling, get the honest wall-clock number with a plain run — the profiler inflates absolutes
~1.8×, so never optimize against profiled µs.
- Qwen: `tests/bench_decode.py` (ms/token) / `tests/bench_prefill.py`.
- Generic: time the region you care about with `time.perf_counter()` around a warmed-up call (discard
  the first call — it compiles kernels).

### Step 2 — Host-bound or device-bound? (the single most important split)
Capture with Tracy (§1.3) and compare, per region, **Σ device kernel time** vs **wall time**.
- **Wall ≫ Σ kernel time → host-bound.** The device is idle waiting for the host to dispatch work.
  In the ops CSV this shows as large `OP TO OP LATENCY` gaps between ops. Dominant in **eager decode**
  (often ~80% gaps).
  → **Fix: metal trace** (capture the op sequence once, replay with no host dispatch). This is usually
  the single biggest decode win. Also: reduce op count, avoid host round-trips (`.cpu()`,
  reshapes that fall back to host), enlarge batches.
- **Wall ≈ Σ kernel time → device-bound.** The host is keeping up; the device is the wall. Go to Step 3.

Rule of thumb: if you have not yet enabled trace, do that first — most other device-op tuning is noise
next to removing host-dispatch bubbles.

### Step 3 — Which op dominates? (attribute the device time)
From the traced region, sort ops by **Device Kernel Duration** (not Total %, which includes one-time
compile stalls). Pareto: the top 1–3 op *types* usually own most of the time. For Qwen that's the MoE
`sparse_matmul`, the attention SDPA, and the gated-delta recurrence. Now diagnose *that* op (Step 4).

### Step 4 — Is the dominant op compute-, bandwidth-, or overhead-bound?
This decides the fix. Use the utilization columns (official report + `--collect-noc-traces`, §1.8):

| Signature | Meaning | Fix direction |
|-----------|---------|---------------|
| High **FLOPS %**, high **DEVICE KERNEL DURATION** | compute-bound | better matmul config (grid, in0/in1 block, subblock), lower-precision inputs (bf8/bf4), fuse activations |
| High **DRAM BW UTIL %** / **NOC UTIL %**, low FLOPS % | bandwidth-bound | shard to L1, reduce reads (dedup weights, cache), smaller dtype for the *moved* tensor, better DRAM interleave / grid |
| Low kernel time but big **OP TO OP LATENCY** even in trace | on-device **overhead-bound** (setup/sync/tiny op) | fuse adjacent ops, remove reshards/format conversions, merge small ops, cut unnecessary syncs |
| `>100% DRAM` on a sparse/gather op | it is *skipping* most nominal reads → scan/dispatch-bound, **not** BW-bound | attack op count / gather structure, not bandwidth |
| High `NPE CONG IMPACT %` | NoC congestion (too many cores hitting same DRAM bank/route) | change sharding/grid to spread traffic; check multicast patterns |

### Step 5 — Confirm the fix moved the needle
Re-run Step 1's baseline (plain, no profiler). Trust the wall-clock delta, not the profiled delta.
Keep a before/after log. If the profiled ranking changed but wall-clock didn't, you optimized something
that wasn't on the critical path — go back to Step 3.

### Quick tool-picker
- "Is it host or device?" → Tracy eager capture, look at gaps (§1.3, §1.7).
- "Which op?" → Tracy ops CSV / visualizer, sort by Device Time (§1.5–1.7).
- "Is that op BW-bound? NoC congested?" → Tracy `--collect-noc-traces` (§1.8).
- "Which pipe inside the core is hot (FPU/pack/unpack)?" → Tracy `--profiler-capture-perf-counters` (§1.9).
- "What's the op graph / buffer layout?" → TT-NN reports (§2).
- "Why am I OOMing?" → device memory dump (§3).
- "Is the output even correct?" → comparison mode / PCC (§4).

---

## 1. Tracy — performance profiling

### 1.1 What it is / when to use it
[Tracy](https://github.com/wolfpld/tracy) is the primary profiler. tt-metal uses a fork adapted to
Tensix. It captures **host** zones (C++ and Python), **device** zones (per-RISC-V-core kernel timing on
every Tensix + Ethernet tile), and — with extra flags — **NoC events** and **hardware perf counters**.
Use it for any "where is time going" question. It is the right tool for prefill/decode speedup work.

### 1.2 One-time build requirement
The profiler must be compiled in (it is in stock `build_metal.sh` builds). Check:
```bash
grep ENABLE_TRACY build/CMakeCache.txt   # want: ENABLE_TRACY:BOOL=ON
```
If not: `./build_metal.sh --enable-profiler` (or `-DENABLE_TRACY=ON`). Device profiling is compiled in
by default but **disabled at runtime** unless enabled (the `-m tracy` wrapper sets
`TT_METAL_DEVICE_PROFILER=1` for you). The **first** profiled run JIT-recompiles every kernel with
device-zone instrumentation (slow — minutes, especially the E=256 MoE build); disk-cached after.

> **Mutual exclusion:** the device profiler, kernel DPRINT, and Watcher all use scarce core SRAM and
> **cannot run together**. Ensure `TT_METAL_DPRINT_CORES` and `TT_METAL_WATCHER` are unset when
> profiling. Same applies to `TT_METAL_NOC_DEBUG_DUMP` (§1.8).

### 1.3 Running it — any script, module, pytest, or command
Always go **through the wrapper**, never plain `python` — the wrapper sets `TT_METAL_DEVICE_PROFILER=1`
and does post-processing.
```bash
# A plain script
./python_env/bin/python -m tracy -r your_script.py

# A python module
./python_env/bin/python -m tracy -r -m your.module

# A pytest test
./python_env/bin/python -m tracy -r -m pytest path/to/test.py::test_name

# Qwen concrete example (prefill + eager-decode + traced-decode in one run)
QWEN36_LAYERS=4 QWEN36_SEQ=32 ./python_env/bin/python -m tracy -r --op-support-count 6000 \
    models/demos/qwen3_6_a3b/tests/prof_decode.py
```
`prof_decode.py` wraps each region in signposts (`prefill_start/stop`, `eager_start/stop`,
`trace_start/stop`) with warmup before each measured region so compiles aren't timed — a good template
for your own profiling harness.

### 1.4 Flag reference (`python -m tracy`)
| Flag | Effect |
|------|--------|
| `-r`, `--report` | **Generate the ops report CSV** (the thing you parse). Almost always want this. |
| `-p`, `--partial` | Only profile zones between `profiler.enable()`/`disable()` (partial python profiling). |
| `-l`, `--lines` | Line-level python profiling (lots of data; for host hotspots). |
| `-o DIR`, `--output-folder DIR` | Redirect all artifacts (else `generated/profiler`). |
| `--op-support-count N` | Max ops the profiler buffers. **Must exceed total op count** or ops are dropped (4-layer Qwen ≈ 1.7k; 40 layers needs much more). |
| `--collect-noc-traces` | **Capture NoC events** → per-op NoC/DRAM/ETH BW + congestion (see §1.8). |
| `--profiler-capture-perf-counters LIST` | Capture HW perf counters: `fpu,pack,unpack,l1,instrn,all` (see §1.9). |
| `--device-memory-profiler` | Profile device L1/DRAM buffer allocations (sets `TT_METAL_MEM_PROFILER=1`). |
| `--device-trace-profiler` | Profile durations of ops **inside a metal trace** (sets `TT_METAL_TRACE_PROFILER=1`). |
| `--profile-dispatch-cores` | Include dispatch-core profiling data. |
| `--no-device` | Host-only profiling (skip device data). |
| `--sync-host-device` | Sync host clock with all devices (better cross-timeline alignment). |
| `-a TYPE`, `--device-analysis-types` | Extra device analysis passes (repeatable). |
| `--no-runtime-analysis` | Disable C++ post-processing (falls back to legacy python processing). |
| `-v`, `--verbose` | More stdout, including the capture command. |
| `--check-exit-code` | Abort (skip post-processing) if the profiled command fails. |

### 1.5 Signposts — scope your measurement
```python
from tracy import signpost
signpost(header="decode_start")        # appears as a row in the ops CSV + a marker in the GUI
model.decode_step()
signpost(header="decode_stop")
```
Put warmup calls *before* the first signpost so kernel compiles aren't counted. Filter the report to
rows between a signpost pair (`OP TYPE == "signpost"`) to isolate a region.

For manual partial python profiling (with `-p`):
```python
from tracy import Profiler
p = Profiler(); p.enable()
function_under_test()
p.disable()
```

#### 1.5.1 Per-component signposts (built into the model — `QWEN36_SIGNPOST=1`)
The model is instrumented with nested signpost regions at every component boundary in **both prefill
and decode**, via the helper `tt/signpost.py` (`sp.region("name")`). They are **off by default and a
complete no-op** (no f-string, no call) unless `QWEN36_SIGNPOST=1` — the gate matters because on a
Tracy-enabled build even a bare `signpost()` records a host message per call, and there are hundreds
per decode step across 40 layers. Regions are pure host-side messages (no `synchronize_device`), so
unlike `prof.phase` (`QWEN36_PROFILE_PHASES=1`, device-synced, prefill-only) they are trace-safe and
work on the decode path.

Region names (each brackets `name` … `name/end`, nested):
- **model:** `embed`, `rope`, `head`(`.norm`/`.lm_head`/`.d2h`); decode: `dec.rope_idx`, `dec.embed`,
  `dec.final_norm`, `dec.select`(→`head.lm_head` + `head.argmax` **or** `head.samp.topk`/`.sample`/
  `.penalty`), `dec.pos_advance`, `dec.lm_head`/`dec.d2h` (vLLM logits path).
- **per layer** (`decoder.py`, aggregated across all 40): `layer.input_norm`, `layer.mixer`,
  `layer.residual1`, `layer.post_norm`, `layer.moe`, `layer.residual2`.
- **gated-delta** (`delta.*`): `in_proj`, `conv`, `split`, `gate`, `qknorm`, `recurrence` (the fused
  `decode_step_tt` kernel or the chunked/scan path), `w_out`, `norm_out`.
- **full attention** (`attn.*`): `qkv`, `qk_norm`, `rope`, `kv_write`, `sdpa`, `out`.
- **MoE** (`moe.*`): `shared`, `router`, `scatter`, `experts`, `combine` (+ dense-prefill
  `repeat`/`gate_up_mm`/`swiglu`/`down_mm`/`reduce`, which also carry the matching `prof.phase`).

**Decode caveat:** host signposts fire during trace **capture**, not `execute_trace` **replay**, so
profile the **eager** step. Per-op `DEVICE KERNEL DURATION` there is identical to traced replay (same
kernels); only the op-to-op gaps differ. Cross-check totals against `bench_decode.py::bench_components`.

**Scale caveat:** the device profiler aborts (device-data readback overflow — >1M zones) above ~4–8
layers, and **do NOT pass `-p`** (it truncates the capture before the measured window). So profile the
signpost breakdown at a **small layer count**; per-layer op shapes are identical at any depth, so the
component *proportions* hold for the full 40-layer model. For real 40-layer per-component ms, use the
un-profiled traced replays in `bench_decode.py::bench_components` (also surfaced in `CURRENT_BENCHMARK.md` §2).

Run + read:
```bash
# prof_decode.py sets QWEN36_SIGNPOST=1 by default and brackets the eager step with
# prefill_start/stop (prefill) and eager_start/stop (decode). Keep layers small; no -p.
QWEN36_LAYERS=4 QWEN36_SEQ=128 python -m tracy -r -v --op-support-count 20000 \
    models/demos/qwen3_6_a3b/tests/prof_decode.py
# Per-component device-kernel breakdown (incl + self time, aggregated over layers) per window:
python models/demos/qwen3_6_a3b/tests/signpost_report.py --between prefill_start prefill_stop
python models/demos/qwen3_6_a3b/tests/signpost_report.py --between eager_start eager_stop
# Or the full end-to-end doc (throughput + real traced breakdown + signposts) in one command:
QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/tests/gen_current_benchmark.py
```

### 1.6 Output files
- `generated/profiler/.logs/` — raw capture. `tracy_profile_log_host.tracy` is the file you load in the
  Tracy GUI. **This folder is wiped at the start of every profiled run** — copy off anything to keep.
- `generated/profiler/.logs/cpp_device_perf_report.csv` — **per-op device summary** (fast to parse).
  Columns: `DEVICE KERNEL DURATION [ns]` (pure Tensix compute), `DEVICE FW DURATION [ns]`
  (compute + on-device setup), `OP TO OP LATENCY [ns]` (inter-op gap = host-dispatch bubble when eager),
  `METAL TRACE ID` (set ⇒ traced op, empty ⇒ eager). OP NAME is blank here.
- `generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv` — official report **with op names +
  signpost rows + DRAM/FLOPS utilization** (and NoC columns if `--collect-noc-traces`). This is the
  main artifact for attribution.
- `generated/profiler/reports/<ts>/npe_viz/` — NoC visualization (only with `--collect-noc-traces`).

Change the root with `-o <dir>` or `export TT_METAL_PROFILER_DIR=<dir>`.

### 1.7 Reading the results
- **Eager region:** per-op kernel time **+ host-dispatch gaps**. Big gaps ⇒ host-bound ⇒ use trace.
- **Traced region:** same kernels, but gaps are the **on-device** inter-op latency that survives a
  trace (the only overhead op-fusion can attack). Split: Σ kernel vs Σ(FW−kernel) vs Σ gaps.
- **Instrumentation inflates absolutes** (~1.8×) and inflates overhead more than kernel time. **Trust
  ratios and op ranking, not absolute µs.** Real ms/token comes from a non-profiled bench run.
- **Utilization:** high FLOPS% ⇒ compute-bound; high DRAM/NoC BW% + low FLOPS% ⇒ bandwidth-bound;
  `>100% DRAM` on a `sparse_matmul` ⇒ it is *skipping* most of its nominal (full-256-expert) reads →
  scan/overhead-bound, not bandwidth-bound. `SLOW`-flagged dense matmuls show true DRAM%/FLOPS%.

### 1.8 NoC event profiling (interconnect / bandwidth)
Add `--collect-noc-traces` to capture every NoC transaction (type, src/dst coords, counters, size) on
NCRISC/BRISC and analyze it:
```bash
QWEN36_LAYERS=4 ./python_env/bin/python -m tracy -r --collect-noc-traces \
    models/demos/qwen3_6_a3b/tests/prof_decode.py
```
This sets `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` and a report path, then feeds the traces to
**tt-npe** (Tenstorrent's NoC performance estimator). The official ops CSV gains per-op columns:
`NOC UTIL (%)`, `MULTICAST NOC UTIL (%)`, `DRAM BW UTIL (%)`, `DRAM BW UTIL PER CTRL (%)`,
`ETH BW UTIL (%)` (multi-chip), `NPE CONG IMPACT (%)` (congestion). An `npe_viz/` directory is written
into the report folder for visualization.

- **Requires `tt-npe` importable on `$PYTHONPATH`.** If it can't import, capture still runs but the NoC
  columns are silently skipped — verify `python -c "import npe_analyze_noc_trace_dir"` works first.
- **Use it when** Step 4 points at bandwidth/interconnect: confirm whether an op is truly DRAM/NoC-bound
  (vs the `>100% DRAM` scan-bound illusion), and whether `NPE CONG IMPACT` says cores are colliding on a
  DRAM bank/route (→ fix sharding/grid, not the matmul config).

**Distinct tool — NoC *correctness* dump (not perf):** `TT_METAL_NOC_DEBUG_DUMP=1` instruments every
transaction to detect missing NoC barriers / ordering bugs and prints issues per core. It is
experimental and **mutually exclusive** with the profiler/Watcher/DPrint, so it cannot run in the same
pass as the perf capture above. Reach for it only when chasing a hang/corruption, not slowness.
(See `docs/source/tt-metalium/tools/noc_debug_dump.rst`.)

### 1.9 Hardware performance counters (inside-the-core hotspots)
When an op is device-bound and you need to know *which pipe* is hot:
```bash
./python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu,pack,unpack your_script.py
```
Groups: `fpu`, `pack`, `unpack`, `l1` (`l1_0`/`l1_1`; `l1_2`/`l1_3`/`l1_4` are Blackhole-only),
`instrn`, or `all`. Sets `TT_METAL_PROFILE_PERF_COUNTERS` to a bitfield. Tells you e.g. whether a matmul
is pack/unpack-bound (data movement in/out of the FPU) vs FPU-bound (raw compute) — different fixes.

### 1.10 Custom device zones (instrumenting your own kernel)
To time a section *inside* a C++ kernel (not just whole ops), use `DeviceZoneScopedN`:
```cpp
#include <tools/profiler/kernel_profiler.hpp>
void kernel_main() {
    DeviceZoneScopedN("MyCustomZone");   // shows up as a device zone in the report/GUI
    // ... code to measure ...
}
```
The annotation has overhead — use it selectively in hot sections. Host-side C++ uses Tracy's
`ZoneScoped;` / `ZoneScopedN("name")`. (See `docs/source/tt-metalium/tools/device_program_profiler.rst`.)

### 1.11 Viewing — visualizer (CSV) and Tracy GUI (timeline)
`ttnn-visualizer` (in `./python_env/bin`) reads the profiler CSV **directly** — no DB needed — for the
per-op timing / eager-vs-traced / utilization view:
```bash
./python_env/bin/ttnn-visualizer --performance-path generated/profiler/reports/<ts> --port 8000
```
Scope to a signpost region and sort by **Device Time** (not Total %, which is wall-time-incl-gap and
gets dominated by one-time compile stalls). This is all you need for overhead analysis.

For the full Tracy **timeline GUI** (flame-graph of host+device zones), build `Tracy-release` from
`github.com/tenstorrent/tracy` and either load a `.tracy` capture file (produced by
`build/tools/profiler/bin/capture-release -o out.tracy`) or connect live. The GUI is a TCP **server** on
port 8086; the profiled app is the client:
```bash
ssh -NL 8086:127.0.0.1:8086 user@remote-machine   # then point the GUI at 127.0.0.1:8086
```

---

## 2. TT-NN reports — op graph, buffers, tensors (`generated/ttnn`)

### 2.1 When to use it
For the **op graph, per-op input/output buffer layouts, or tensor-level comparison** in the TT-NN
visualizer's `--profiler-path` view. Heavier and slower than Tracy — a debug/inspection mode, not a
serving mode. (For *timing*, use Tracy; this answers *"what ran and with what buffers."*)

### 2.2 The config chain (why nothing appears)
The report path is `generated/ttnn/reports/`, but a report is only written when **all** of these hold
(defaults are all against you — see `ttnn/api/ttnn/config.hpp`):

| Setting | Default | Needed |
|---------|---------|--------|
| `enable_fast_runtime_mode` | `true` | **`false`** — logging is refused while fast mode is on |
| `enable_logging` | `false` | **`true`** — the master switch; also initializes the report DB path |
| `report_name` | `None` | **set** — with no name, `report_path` resolves to `null` and nothing is written |
| `enable_detailed_buffer_report` | `false` | `true` for buffer/memory view |
| `enable_graph_report` | `false` | `true` for the op-graph SVG |

Ordering matters: set `enable_fast_runtime_mode = false` **before** `enable_logging = true`. If you see
`"Logging cannot be enabled in fast runtime mode"` or `"Running without logging. Please enable logging
to save detailed buffer report"`, that's this chain being violated.

Output folder: `generated/ttnn/reports/<snake_cased_report_name>_<mon><dd>_<HHMM>/db.sqlite`.

### 2.3 How to turn it on (no pytest needed)
`TTNN_CONFIG_OVERRIDES` is parsed at `import ttnn` (see `ttnn/ttnn/__init__.py`), so it applies to any
process — set it **before** the process starts:
```bash
export TTNN_CONFIG_OVERRIDES='{
  "enable_fast_runtime_mode": false,
  "enable_logging": true,
  "report_name": "qwen36_server",
  "enable_detailed_buffer_report": true,
  "enable_graph_report": true
}'
./python_env/bin/python models/demos/qwen3_6_a3b/demo/server.py
```
Equivalent alternatives: `TTNN_CONFIG_PATH=<file.json>` (a JSON file with the same keys), or in-code
`ttnn.CONFIG.enable_logging = True` etc. before the model runs.

### 2.4 The Qwen-specific gotcha — scope it to the decode step
Leaving the graph/buffer reports on **during model build** snapshots the weight tensors and dies with
`parse_py_tensor: ndarray_import failed`. So: keep `enable_logging` ON at load (it initializes the DB
path), but toggle the detailed reports on **only around one decode step** with `ttnn.manage_config`.
`tests/gen_ttnn_report.py` does exactly this — prefer it over hand-rolling:
```bash
TTNN_CONFIG_OVERRIDES='{"enable_logging": true, "report_name": "qwen36_decode", \
    "enable_fast_runtime_mode": false}' QWEN36_LAYERS=4 \
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/gen_ttnn_report.py
```
The pattern inside:
```python
model.decode_step_eager()          # warmup, reports OFF
ttnn.synchronize_device(mesh)
with ttnn.manage_config("enable_graph_report", True), \
     ttnn.manage_config("enable_detailed_buffer_report", True):
    model.decode_step_eager()      # ONE step captured -> db.sqlite
    ttnn.synchronize_device(mesh)
```
Must be **eager** (the graph/buffers are captured at dispatch, not inside a trace). Note most decode
ops are C++ FastOperations, so this DB is sparser than the perf CSV.

### 2.5 View it
```bash
./python_env/bin/ttnn-visualizer --profiler-path generated/ttnn/reports/<name>_<ts> --port 8000
```
`--profiler-path` expects a **prebuilt `db.sqlite`** — the visualizer does not build it from the perf
CSV. (`--performance-path` from §1.11 is the other, independent source.)

### 2.6 Cost warning
`enable_fast_runtime_mode: false` + `enable_logging: true` records every op to sqlite and disables fast
runtime — **dramatically** slower, and the DB balloons on a long server run. Capture a short run, then
turn it off. Never leave it on in a serving path.

---

## 3. Device memory (OOM triage)

For "where is my HBM/L1 going / why am I OOMing," the TT-NN visualizer is overkill. Use the direct
allocator APIs (in `ttnn/ttnn/device.py`) — no logging mode, negligible overhead:
```python
ttnn.dump_device_memory_state(mesh, prefix="after_prefill_")   # dumps L1 + DRAM usage
view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)        # programmatic high-water mark
```
Drop these at suspect points (after model build, after prefill, mid-decode). For per-op buffer
profiling over a whole run, add `--device-memory-profiler` to the Tracy capture (§1.3), which sets
`TT_METAL_MEM_PROFILER=1`.

See memory `qwen36-server-oom-fix` for the known server OOM cause (per-request cache/trace leak).

---

## 4. Numerical debugging (PCC / comparison mode)

To catch a divergence between a TT op and its PyTorch golden, TT-NN comparison mode runs both and
compares:
```bash
export TTNN_CONFIG_OVERRIDES='{
  "enable_fast_runtime_mode": false,
  "enable_logging": true,
  "report_name": "qwen36_cmp",
  "enable_comparison_mode": true,
  "comparison_mode_pcc": 0.999
}'
```
Set `comparison_mode_should_raise_exception: true` to hard-fail on the first op below the PCC
threshold. Same fast-runtime-mode caveat as §2. For model-level PCC, prefer the existing per-module
tests in `tests/` and `tests/cross_validate_reference.py` (reference venv at `~/qwen_ref_venv`).

---

## 5. Discipline / gotchas

- **A profiled or logged run killed mid-compile leaves the device HUNG.** Run `tt-smi -r` before the
  next run. Use generous timeouts — the E=256 MoE build + first instrumented compile is minutes.
- **Profiler ⊥ Watcher ⊥ DPRINT ⊥ NoC-debug-dump** — these fight over core SRAM; run one at a time.
- `generated/profiler/.logs/` is wiped every profiled run; `generated/ttnn/reports/` accumulates
  timestamped dirs (clean up manually).
- Tracy tools are only fully supported on **source builds**.
- **Always validate wins against a non-profiled bench run** — profiled absolutes are inflated ~1.8×.
  Qwen: `tests/bench_decode.py` is the before/after ms/token loop; use `QWEN36_LAYERS=4` to iterate,
  `=40` for the authoritative figure.
- `--op-support-count` too low silently drops ops (looks like the tail of the model vanished).

## 6. Related docs
- `PREFILL.md`, `FUTURE_OPTIMIZATIONS.md` — perf findings these tools produced.
- Upstream tt-metal docs:
  - `docs/source/tt-metalium/tools/tracy_profiler.rst` — Tracy overview / GUI setup.
  - `docs/source/tt-metalium/tools/device_program_profiler.rst` — device zones, `DeviceZoneScopedN`.
  - `docs/source/tt-metalium/tools/noc_debug_dump.rst` — NoC correctness dump.
  - `ttnn/tutorials/ttnn_visualizer.md`, https://docs.tenstorrent.com/ttnn-visualizer/.
