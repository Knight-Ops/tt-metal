# Implementing models on Tenstorrent hardware — field notes

Hard-won rules from bringing up Qwen3.6-35B-A3B on Blackhole (P150a) and then debugging a series of
device hangs in a real serving loop. Everything here is **measured**, not inferred, and most of it is
architecture-agnostic: it is about `ttnn`, traces, the allocator and the program cache, not about this
model. Companion docs: `README.md` (the model), `MEMORY.md` (the trace/allocator aliasing rule in
depth), `PREFILL.md` / `FUTURE_OPTIMIZATIONS.md` (perf levers).

---

## 1. The allocator and traces are one system. Order of allocation is load-bearing.

A captured trace **bakes the addresses** of every buffer it touches, and nothing pins its in-graph
transients — they are freed the moment `end_trace_capture` returns. Anything you allocate afterwards
can land in that hole, and the next replay writes over it.

**Rule: allocate every persistent buffer before anything captures a trace. Biggest first.**

Three separate bugs in this repo, all the same fault:

| what | symptom |
|---|---|
| multi-bucket prefill traces (`MEMORY.md` §1-3) | later bucket replays at PCC ~0; `S0` came back \|max\|=33.19, output NaN |
| gated-delta checkpoint ring allocated lazily, after the first prefill | **silent degradation** — prefill 149 vs 743 tok/s, decode 5.5 vs 24.5 tok/s. At 40 layers (180 MiB ring) it wedged the device outright on request two |
| sampling tail's buffers allocated between the greedy and sampling captures | wedged the device on a *repeated prompt* |

Note the second row: this class of bug does **not** always crash. It can just make everything 5×
slower while producing plausible output. If throughput drops for no reason, suspect allocation order
before you suspect kernels.

`allocator.cpp` warns about this itself ("buffers allocated when a trace is active may be corrupted
once a trace is executed"). Treat that warning as an error in a serving process.

### Capture the LARGEST trace first

`capture_decode_trace` releases the previous trace, then captures the new one. Capturing a *larger*
graph into the hole left by releasing a *smaller* one leaves the region in a state a later large
eager allocation collides with. Measured, 4 layers, prefilling T=636 twice:

```
warmup rounds                            2nd prefill of the same shape
none / greedy only / sampling only       pass
greedy -> greedy (release + recapture)   pass      <- so it is not the release
greedy -> sampling  (was shipped)        HANG
sampling -> greedy  (the fix)            pass
```

This shipped for months and hid behind a coincidence: a growing chat transcript gives every turn a
different prompt length, so the *second prefill of the same shape* almost never happened.

---

## 2. Bound your shape space, or you will pay for it twice

Every distinct tensor shape compiles its own device programs. Measured on this model, **one distinct
prefill block width costs ~85–110 programs; repeating a width costs 0.**

```
single m=1   -> 93 programs      single m=1 again -> 0
single m=15  -> 109              single m=32 -> 85       single m=47 -> 100
ragged m=124 -> 94               m=254 -> 85             m=384 -> 88
```

That bites twice:

1. **Latency.** Each new shape JIT-compiles. Measured: a 5-turn conversation where every turn had a
   new prompt length cost **15–18 s per turn**, nearly all of it compilation.
2. **Stability.** The count grows without bound in a long-running server, and past some point the
   device hangs — see §5, which is the single most important section here.

**Rule: make every device-visible length come from a small fixed set.** Pad to a bucket and mask,
never slice to the natural length.

The wrong way (what this model did): a prompt of length `T` ends in a ragged block of `T mod chunk`
real tokens, and the recurrence sliced itself down to that count — a different value every turn.

The right way, and it is cheap:

- Pad the block to a bucket (here, a 128-multiple).
- Zero the *gates* on the pad rows so they are no-ops in the recurrence. For a gated-delta / Mamba-style
  scan that is `beta = 0` (the row contributes nothing) and `g = 0` in log space (it decays nothing).
  For attention, nothing is needed: pad rows only corrupt their own outputs, which nothing reads.
- Build the mask with a shape that depends only on the bucket, never on the real length —
  a `[T, 1]` column of ones-then-zeros. The *values* vary per call; the *shape* must not.
- Keep any carried state coming from the last **real** rows, not the padding (here, the depthwise
  conv window: `_conv_silu(state_at=valid_len)`).

**How short should the ladder be? Shorter than "every possible length", but not as short as we first
thought.** Each width is ~90-110 one-time programs, so four widths {128, 256, 384, 512} is ~384 of a
~550 working set. That *looked* like the cause of §5's hang, and narrowing to two widths {128, 512}
did make the hang much rarer — so the ladder was cut to two.

**That was the wrong conclusion**, and it is a good example of mistaking a correlate for a cause. §5
turned out to be per-width constants corrupted under a live decode trace; a narrow ladder helped only
because fewer widths were ever *reused*, and reuse is what exposes the corruption. With the real fix
in, program count does not bind (801 cached programs measured fine) and the narrow ladder costs real
time: a reuse turn re-ingesting ~130 tokens pads to 512 and costs ~614 ms instead of padding to 256
for ~370 ms. Measured end to end at 40 layers, prefill per reuse turn: **0.34 s on the four-width
ladder against 0.56 s on the two-width one.**

So: keep the ladder short enough that every width's programs are built up front
(`prewarm_prefill_shapes`), and add a width only for a length range that actually recurs — but size
it by *padding waste*, which is measurable, not by a program budget, which turned out not to bind.

Cap the padded width by the room left in the KV cache — padding writes real KV rows, so a block near
the end of the context must fall back to a narrower width rather than run past `max_seq`.

Result on this model: per-turn program growth dropped from ~90 to **0–4** once each bucket width had
been seen once. PCC gate in `tests/test_ragged_mask.py`: masked vs sliced is **exactly** equal
(1.000000 on logits, recurrent state and conv state), and masked vs a full single-shot prefill is
0.992–0.998 across remainders {1, 127, 129, 255}.

### `T == 1` is usually a different code path

Watch out when a "pad to bucket vs slice" change alters whether `T` reaches 1. In this model `T == 1`
is the decode signature everywhere — the add-chain conv, decode program configs, the fused
single-step kernel. Slicing a ragged prefill block down to one real row silently routed it through
the *decode* kernels (PCC 0.88 against the masked path, for that reason alone). Padding never
narrows, so it cannot trip this.

---

### Check whether an op takes your offset at COMPILE time or RUN time

A scalar op attribute is usually part of the program-cache key, so a value that changes every request
silently compiles a new program every request. The one that mattered here is chunked SDPA's KV
offset:

```python
# baked into the program: +1 program per distinct offset, forever
ttnn.transformer.chunked_scaled_dot_product_attention(..., chunk_start_idx=P)

# read from device at runtime: ONE program for every offset
ttnn.transformer.chunked_scaled_dot_product_attention(..., chunk_start_idx_tensor=t)  # int32[1]
```

Measured across six offsets (`hangdbg/flex.py`): legacy `+1` program at every offset; tensor form
`+1` at the first and then **0**, at 0.3 ms per call, with **bit-identical output**. In a chat the
offset grows forever, so the legacy form never amortises — and the cost is far worse in a real
serving loop than in isolation: a ragged block at a *new* offset took **1.63 s** with 40 layers of
work and a decode trace resident, against **0.27 s** at an offset already seen (isolated, that same
new program costs ~8 ms). That one difference was the entire reason a finer checkpoint grain first
looked like a 3x regression.

Two corollaries worth internalising:

- **The first use of the runtime form costs a real compile (~1.4 s here).** Warm it before serving,
  and warm *every* variant: the same op with `q_chunk_size = width` and with `q_chunk_size = 128`
  are different programs. Warming only the tail shape left a **2.36 s** spike on the first request
  that used the wide-block shape.
- **Do not assume the other index args are compile-time.** `fill_cache`'s `compute_program_hash`
  explicitly *excludes* `batch_idx`/`update_idx`/`batch_offset` and re-applies them in
  `override_runtime_arguments()`, so it is already offset-independent. Read the op's
  `compute_program_hash` instead of guessing; a probe that counts
  `mesh.num_program_cache_entries()` around one op answers it in seconds.

## 3. Alignment: the rules the kernels actually enforce

Tile geometry is 32×32, but the constraints that matter in practice are per-op and stricter.

- **The loud ones assert.** `chunked_scaled_dot_product_attention` says exactly what it wants:
  `TT_FATAL: chunk_start_idx must be a multiple of q_chunk_size`. A violation of a *documented*
  rule is a clean exception, not a hang. If you get a hang, stop looking for an alignment bug.
- **The documented rule is necessary, not sufficient.** Two configurations that satisfy it hung the
  device outright: `P=640, width=256, q_chunk=128` and `P=768, width=256, q_chunk=256`, while
  `P=512/640/768` at width 128 were all fine. No rule short of "only the shapes the shipped path
  already emits" survived contact with the device.
- **So: stay inside shapes that already ship.** The chunked-prefill planner here emits only
  chunk-wide blocks at chunk-multiple offsets plus one ragged remainder — exactly what the
  pre-existing long-prompt path emitted. Novel-but-legal geometry is where the wedges live.
- Non-tile-aligned *logical* shapes are fine per se. `conv_state` is `[3, 8192]` — 3 rows in a
  32-row tile — and `clone`/`copy` round-trip it exactly with neighbouring buffers untouched.

---

## 4. Debugging a device hang

A hang surfaces at the **next readback**, which is wherever the queue happens to drain — usually a
`ttnn.to_torch` far from the actual fault. py-spy will show you that readback and it tells you almost
nothing. Do this instead:

1. **`py-spy dump --pid <server>` first**, before killing anything. A wedged device holds the GIL in
   C++, so every Python thread freezes and `/health` stops answering — that alone distinguishes
   "device hung" from "slow".
2. **Bisect inside the forward pass with syncs.** Add an env-gated `ttnn.synchronize_device()` + log
   after each layer and each major sub-op (this repo: `QWEN36_PREFILL_LAYER_SYNC=1`). The hang then
   names itself: `layer 0 gated_delta ok / post_norm ok / moe ...` and nothing after.
3. **Measure the allocator, don't guess.** `memstat.region(mesh, ttnn.BufferType.DRAM | L1 | TRACE)`
   is host-side book-keeping — no device round-trip, safe to call mid-forward — and
   `mesh.num_program_cache_entries()` is the other half. At the moment of one hang: DRAM 10.6 GB
   free, L1 1% used, TRACE 90% free with one 19 MiB trace resident. That killed the obvious
   "we ran out of memory / can only fit one trace" hypothesis in one run.
4. **Reproduce out of the server.** Drive the engine class in-process; if it reproduces, iteration
   goes from ~4 minutes to ~1, and you can instrument freely.
5. **`TT_METAL_WATCHER=1` is often unusable.** Its instrumentation inflates programs past the kernel
   config buffer (`Program size (71888) too large for kernel config buffer (70656)`), so the run dies
   before reaching your bug.
6. **Budget for board resets.** A wedge needs `tt-smi -r`; a hung process holds the PCIe lock and the
   next run fails with `Read 0xffffffff over PCIe` or `Waiting for lock 'CHIP_IN_USE_0_PCIe'`. Kill
   with `-9`, then reset, then re-run.

### Test at real model size. Always.

**This is the single most expensive lesson here.** A 4-layer model has so much headroom that every
fault in this document disappears. Work bisected, fixed and validated at `QWEN36_LAYERS=4` — with
passing PCC gates and a green end-to-end run — failed immediately at 40 layers. Small-layer runs are
for *correctness* (PCC vs reference); they are worthless for anything touching memory, traces, the
allocator or the program cache.

### Why a device hang looks like a *host* hang

A wedged device does not just hang your process — it makes the whole box appear dead, and the
temptation is to reboot it from the console. Don't; nothing is actually wrong with the host. The
mechanism, all confirmed in `tt_metal/impl/dispatch/system_memory_manager.cpp`:

1. The host waits for the device in `loop_and_wait_with_timeout`. With no timeout configured — **the
   default**, `TT_METAL_OPERATION_TIMEOUT_SECONDS` unset — that reduces to

   ```cpp
   do { func_body(); } while (wait_condition());   // func_body() = a PCIe register read
   ```

   No sleep, no yield, no exit. One core pinned at 100% issuing MMIO reads at full rate against a
   device that will never answer. (The timeout branch at least calls `std::this_thread::yield()`.)
2. That loop is inside C++ holding the GIL, so Python signal handlers never run: **Ctrl-C does
   nothing** and the session that launched it is unrecoverable. This is the part that reads as "the
   server is hung". Measured at the wedge: main thread `STAT=R`, `/proc/<pid>/syscall` = `running`,
   95% of one core, every other thread parked in `futex_wait_queue`. Note what that rules out — it
   is *not* stuck in an uninterruptible driver ioctl (never `D` state), so `SIGKILL` does work, and
   at ~1 of 12 cores it is not starving the host either. The box is fine; only your terminal is
   gone. Open a second SSH session and kill it rather than rebooting.
3. Setting `TT_METAL_OPERATION_TIMEOUT_SECONDS` is better but **not** sufficient, and can make
   things worse. It throws — and the throw unwinds into `MeshDevice::close()` →
   `~FDMeshCommandQueue()` → `clear_expected_num_workers_completed()`, which enqueues *another*
   command at the dead device, times out again, and throws **from a destructor**: `std::terminate`
   → `SIGABRT`. Measured: first timeout 15:01:55, second 15:03:26, then abort. It also leaves the
   card worse than it found it — the next `tt-smi -r` failed with
   `Read 0xffffffff over PCIe ID 1: the board should be reset` (the second reset attempt worked).

   This is a genuine tt-metal bug, in `tt_metal/distributed/fd_mesh_command_queue.cpp`
   (`~FDMeshCommandQueue`, ~line 245). The destructor is guarded against a *worker-thread*
   failure — `if (in_use_ && !thread_exception_state_.load(...))` — and the checks after it
   deliberately "log warnings instead of aborting - destructors should never throw or abort". But a
   dispatch timeout raised on the **main** thread never sets `thread_exception_state_`, so
   `in_use_` is still true, the guard passes, and the unprotected blocking call runs. It wants a
   `try/catch` that downgrades to a warning, like the checks below it already do.

### The device hang really can take the host down — via the driver's ARC path

**Correction to the above:** a wedged card *does* hard-reboot the host, and the mechanism is in the
KMD, not in tt-metal. Measured on 2026-09-08: the boot that died at 15:15:37 ends with

```
tenstorrent 0000:02:00.0: Timeout waiting for ARC response
tenstorrent 0000:02:00.0: Failed to set initial power state: -110
```

and then nothing — no shutdown sequence, no panic text, just gone. Those errors first appeared at
15:03:28, **two seconds after** the destructor `SIGABRT` above, and then recurred at every single
point something touched the card (`tt-smi -r` 15:04:12, again 15:08:07, watchdog reset 15:12:44-51,
`tt-triage` 15:15:37 → dead). Note they are on `0000:02:00.0` — the *second* card, which a 1×1 mesh
still enumerates (`Fabric | ... intra-mesh degree histograms mesh0 {1:2}`).

Why it escalates from "one wedged chip" to "reboot the box", reading
`/usr/src/tenstorrent-<ver>/`:

- `arc_msg_send_sync()` (`msgqueue.c`) takes `arc_msg_mutex` and **sleeps up to 1 s** polling for an
  ARC response (`ARC_MSG_TIMEOUT_MS 1000`), or up to 2 s when it first has to
  `arc_msg_flush_inflight()`. So with ARC unresponsive, every ARC transaction costs seconds.
- `tt_cdev_open()` (`chardev.c`) holds `reset_rwsem` **shared** across both
  `cancel_delayed_work_sync(&tt_dev->power_down_work)` and the initial power-state transition, and
  that transition also takes the per-device `chardev_mutex`. So *every open of the char device*
  serialises on a multi-second ARC wait.
- `tt_cdev_release()` does the same on close, and last-close arms `power_down_work` on the **shared
  `system_wq`** (`idle_power_down_grace_ms`, default 5000). That work item then sleeps 1-2 s inside
  an ARC transaction on a shared kernel workqueue.
- The reset ioctl takes `reset_rwsem` for **write**. Linux rwsems are fair, so one pending writer
  (`tt-smi -r`) blocks every subsequent reader — i.e. every `open()`/`release()` on that card.
- `device.h` says it outright: `arc_msg_mutex; // Always innermost; don't take reset_rwsem while
  holding.` The ordering is known to be delicate.

Put together: once ARC stops answering, each of `tt-smi`, `tt-triage` and each new run opens *both*
`/dev/tenstorrent/{0,1}`, and each open parks in uninterruptible sleep behind a chain of 1-2 s ARC
waits, a pending exclusive `reset_rwsem` writer, and a blocked `system_wq`. That accumulation — not
a panic — is what stops the host. (The pile-up is the one inferred link: the journal dies with the
box, so there is no hung-task dump to quote.)

And the first domino: the driver arms a **10 s ARC watchdog on the card itself** at init —
`blackhole_init_hardware()` sends `ARC_MSG_TYPE_SET_WDT_TIMEOUT` with
`1000 * auto_reset_timeout` (module param, default 10 s) — and nothing in the driver ever pets it.
Wormhole's comment for the unresponsive case is literally "If not responsive, wait for the
watchdog." So a chip left saturated by a wedged dispatch can starve its own ARC, the card watchdog
resets the ASIC underneath the driver, and from then on ARC messages time out and MMIO reads come
back `0xffffffff` — exactly the `Read 0xffffffff over PCIe ID 1` that `tt-smi -r` reported.

**Two concrete mitigations, both from the parameter list (`/sys/module/tenstorrent/parameters/`):**

- `power_policy` (default `Y`) is documented as "low power at probe, re-aggregate on close". That is
  *precisely* the two ARC transitions that fail. Loading with `power_policy=N` removes them from the
  open and release paths.
- Do **not** `SIGKILL` a wedged run and let the fds close: `tt_cdev_release()` then does NOC cleanup
  and a power transition against a chip that cannot answer, which is what poisons ARC. Reset the
  card *first*, then kill.

**What to do instead** — supervise from outside the process (`hangdbg/watchdog.sh` here):

- Run the workload detached (`setsid nohup`) writing to a log file, never attached to your only SSH
  session, and with `PYTHONUNBUFFERED=1` so the log is truthful up to the instant it stops.
- Detect the wedge by the *log going quiet* while the pid is alive, not by a device query.
- `SIGSTOP` the process first. That halts the PCIe read storm while preserving both the device state
  and the process's handle to it — exactly what you need for a snapshot, and it stops loading the
  host.
- Then snapshot, then `SIGKILL`, then `tt-smi -r` **with retries** (the first reset after an abort
  often fails; the second usually works).
- `faulthandler.dump_traceback_later(60, repeat=True, file=...)` in the workload costs nothing and
  gives you the Python frame at the wedge without py-spy.

---

## 4b. Localising the grow4 wedge (worked example)

Reproduced 6/6 at the same block, so every step below is a real measurement, not a sample of one.
The point of the sequence is that each step *removes* a suspect rather than confirming a guess.

1. **Where it stops.** `QWEN36_PREFILL_LAYER_SYNC=1` (a `synchronize_device` + log after every layer
   and sub-stage) is the single most valuable instrument here. It printed:

   ```
   [inc] P=1024 M=256 embed ok
   [inc] P=1024 M=256 rope ok
   [inc]   layer0 input_norm ok
   [inc]   layer0 init_state reshape ok        <- next sub-stage is gated_delta; never logged
   ```

   So: **layer 0's gated-delta**, not something deep in the model. Note this contradicts the
   unsynced run's Python frame (`gated_delta.py:1194`, near the *end* of the mixer) — that frame is
   only where the host's fetch queue filled up, which is generally *not* the faulting op. Trust the
   sync log, not the traceback.

2. **What actually differs.** `decoder.forward_prefill_incremental` passes `P` only to attention;
   a linear layer gets `(h, cache, init_state, valid_len)`. Between the block that works
   (grow1: `P=512, T=256, valid_len=254`) and the one that wedges (grow4: `P=1024, T=256,
   valid_len=132`), gated-delta's inputs differ *only in `valid_len`*. P is a red herring.

3. **Attention exonerated.** A standalone `chunked_scaled_dot_product_attention` with grow4's exact
   arguments (`q[1,16,256,256]`, flat-as-paged cache, `chunk_start_idx=1024`, `q_chunk=128`,
   `k_chunk=64`) returns finite output in ~64 MiB. Two gotchas while building that harness:
   `_PREFILL_KCHUNK_MAX` defaults to **64**, and using 512 instead fails loudly with
   `Statically allocated circular buffers ... grow to 1604608 B which is beyond max L1 size of
   1572864 B` — worth knowing, because it means the k-chunk ladder never actually varies here.

4. **`valid_len` exonerated as an op-level fault.** A 1-layer model (`QWEN36_LAYERS=1`, layer 0 is
   linear) running `_prefill_single(512)` then `forward_incremental(tail, ragged=True, q_chunk=128)`
   returns finite logits for `valid_len=254` **and** `valid_len=132`. So the shape that wedges 40
   layers is fine on 1 layer.

**Therefore the fault is contextual, not shape-driven**: same op, same shapes, same `valid_len`,
wedges only after five turns of a 40-layer model with a resident decode trace and a 553-entry
program cache. That is consistent with the "one intervention buys exactly one turn" signature in §5
— the fingerprint of a deterministic *address* collision, not a resource limit.

---

## 5. The long-conversation hang — SOLVED: lazily-built per-shape constants

**Symptom.** In a long serving loop the device wedges inside an ordinary prefill op and needs
`tt-smi -r`. Reproduced deterministically 6/6 at 40 layers on the same block (turn 5,
`ragged P=1024 m=132`).

**Cause, exactly.** The prefill path caches **read-only device constants keyed by block width** and
builds them on first use of that width:

- `GatedDelta._conv_stack_const(T)` — five 0/1 selection/summation matrices per width, per linear
  layer (30 layers here).
- `MoE._rowsel(K)` — a per-K one-hot.

After warmup, the first use of any width happens with the **decode trace resident**, so those
constants are allocated under an active trace. tt-metal warns about precisely this, once per device
and then never again:

```
Allocating device buffers is unsafe due to the existence of an active trace.
These buffers may be corrupted once a trace is executed.        (allocator.cpp:123)
```

A replay then stomps them and — because they are **read-only, written once at build time and never
again** — the corruption is permanent. The wedge appears on the next turn that *reuses* that width.
That is why the failure looked so arbitrary; follow the widths in the reproduction:

| turn | block | padded width | what happens |
|---|---|---|---|
| grow1 | `ragged P=512 m=254` | **256** | constants for 256 built **under the live trace**, then stomped by this turn's decode |
| grow2 | `ragged P=512 m=384` | 384 | fine, different width |
| grow3 | `block P=512 m=512` + `ragged P=1024 m=2` | 512, 128 | fine, different widths |
| grow4 | `ragged P=1024 m=132` | **256** | re-reads the corrupted 256 constants → **wedge** |

**Fix.** `TtModel.prewarm_prefill_shapes()` runs one real prefill per ladder width during warmup,
*before any trace is captured*, so every lazy per-shape constant already exists. Called from
`Qwen36Engine.warmup()` right after the checkpoint ring; `QWEN36_PREWARM_SHAPES=0` disables. It runs
one prefill per width rather than enumerating buffers deliberately — the set of lazy per-shape
constants is not closed, and any future one inherits the same hazard.

`_release_traces_for_prefill()` remains as the backstop for a width that was never pre-warmed (a
prompt whose remainder falls off the ladder): it drops the decode-side traces only when a prefill is
about to build constants for an unseen width, and records the width so no later turn pays again.
In the validated 10-turn run it fired **zero** times.

**Measured.** Decode trace resident for the whole conversation, all 10 turns out to T=1806, 66-92%
of each prompt reused. And it is *faster* than the pre-fix baseline, because the constants are no
longer built mid-request: turn wall time at T=896 goes 1.02 s → **0.67 s**, at T=766 1.08 s →
**0.80 s**. Releasing the trace on every prefill also fixes the wedge but costs a flat **+0.33 s per
turn** in re-capture, so it is the fallback, not the default.

**This is the same fault as MEMORY.md §1-3**, which was found and fixed for the *prefill* traces
(an earlier bucket's replay writing into a later bucket's pool; fixed by allocating every bucket's
buffers before any capture). It was simply never applied to the constants the eager prefill builds.

### What was falsified along the way — and why each failure was informative

| hypothesis | evidence against |
|---|---|
| the block shape / `valid_len` | standalone `chunked_scaled_dot_product_attention` with the wedging block's exact arguments returns finite output; so does a **1-layer** model at `valid_len=132` |
| capture-time placement (trace captured on a quiet allocator) | releasing the trace **once** and re-capturing it *after* a real prefill still wedges at grow4 |
| prefill outgrowing the trace's addresses | a probe sampling both allocators after every heavy op puts prefill's peak at **+252 MiB DRAM / +9.45 MiB L1** above baseline; a capture-time ballast of **512 MiB DRAM + 13.8 MiB L1** — comfortably above both — wedges at grow4 unchanged |
| program-cache **count** limit | a synthetic loop reached **801** cached programs with 21.5 GB of ballast and a replaying trace; the model wedges at 553 |
| program-cache **bytes** | the whole cache is **31.2 MiB** (69 KiB/program) against 10.6 GB free |
| a size limit in tt-metal | only a per-program cap (`get_ringbuffer_size`, 70656 B, raises `TT_FATAL` — loudly) and an 8-entry config ring that recycles. `clear_program_cache()` is just `program_cache_->clear()` |
| memory exhaustion / fragmentation | at the wedge: DRAM 10.6 GB free, L1 ~1%, trace region 19.1 of 200 MiB, largest contiguous free per bank flat at 1330 MiB |
| greedy vs sampling decode tails | they differ by **one** program |

The clue that mattered: every intervention that merely *shifted addresses* (DRAM ballast,
`QWEN36_GDN_L1=0`, clearing the program cache) bought **exactly one extra turn**. That is not a
series of near misses — it is the fingerprint of a *relocating* collision, and a resource limit does
not behave that way.

`QWEN36_PROGRAM_CACHE_LIMIT` is **off by default** now: it only ever existed to bound the cache while
this wedge was unexplained, and leaving it on is actively harmful once the padding ladder is four
widths wide. The steady-state working set is ~544 entries, so a 450 limit fires mid-conversation,
drops every trace and every kernel binary, and the next request recompiles — measured as **0.55 s
and 0.63 s** prefills against **0.34 s** on the turns either side, with the valve firing twice in a
10-turn chat. If you ever do need it, set it *above* the working set; below is a cure worse than the
disease.

**Generalise this.** Any read-only device constant built lazily on first use is a landmine on
Tenstorrent hardware, because "first use" almost always lands after a trace exists. Build them
eagerly, before capture. The hazard is invisible in short tests: it needs the width to be used once
(to be corrupted), then a replay, then used *again*.

## 5b. A trace bakes addresses, not contents — which is a *feature*

The rule in §1 has a constructive half that is easy to miss. A captured trace bakes the **address**
of every buffer it touches, so anything you allocate afterwards is a hazard — but anything you
**rewrite in place** is free, including between replays.

That is the whole mechanism behind parked conversations here (`PREFILL.md` §0.95). The model needs
several independent conversations resident so an interleaved request from one does not destroy
another's cached prefix. The obvious implementations all fight the trace:

| approach | why it fails |
|---|---|
| N flat KV caches, swap which one the layers read | the trace baked the addresses of cache set 0; swapping needs a re-capture (~0.33 s/turn) |
| snapshot/restore the KV prefix region per conversation | unbounded — 10 attention layers x 2 x 2 heads x P x 256 is 68 MB at P=6.3K and 1.4 GB at 128K |
| a shared LRU pool of checkpoints | a 200-token side request evicts the 6.3K conversation's only checkpoint every turn |

The one that works costs **nothing**: page the KV, keep the device page table at its single baked
address, and give each conversation its own **host-side** mapping row. Switching conversations is one
`copy_host_to_device_tensor` into that table — the identical in-place write `PagedKV.ensure` already
performs on block boundaries with a trace resident. No allocation, no re-capture, and the device
memory is unchanged because the pool is sized by aggregate demand rather than per conversation.

Generalise it: **when you need per-context state under a resident trace, look for the one small
indirection buffer you can rewrite instead of the large buffers you would otherwise have to swap.**
A page table, an index tensor, an offset tensor. `TtAttention._csi` (§2, runtime `chunk_start_idx`)
is the same trick applied to a scalar. The corollary is the constraint: anything whose *count* is
baked into a shape — the page table's row count, the checkpoint ring's slot count — must be sized
before the first capture and can never be grown, which is why `alloc_conversation_slots` raises
rather than resizing if it is called late.

## 6. Miscellaneous, all measured

- `ttnn.deallocate` defaults to `force=True` and will free a buffer another tensor shares. Never pass
  it a `ttnn.reshape` **view** — the expert-weight views in the MoE are views of the *weights*, and
  freeing those destroys the model.
- Write in-place rather than reallocating for anything a trace reads. `ttnn.multiply(x, 0.0,
  output_tensor=x)` to zero a buffer; `ttnn.copy(src, dst)` to restore one. A fresh allocation moves
  the address and silently invalidates the trace — which is why per-request re-capture used to be
  needed at all, and it costs ~225 ms of drained pipeline each time.
- A `[1, vocab]` TILE tensor is physically `[32, vocab]`. Convert to ROW_MAJOR before a D2H or you
  move 32× the bytes (~15.9 MB vs 0.5 MB at vocab=248320; worth ~8 ms per prefill).
- Trace capture forbids host writes: no in-graph `ttnn.zeros`/`fill`/`tril`. Pre-allocate pools and
  use mask-multiplies instead.
- First-iteration timings are meaningless — one measured pair differed **3.0 s vs 21.2 s** for
  identical work. Always compare settled iterations, and exclude several steps after a trace capture
  (its snapshot/restore drains into the following replays: 43–46 ms/token against a settled ~29).
- Kernels can hang instead of failing. The fused gated-delta op silently wedges at small head dims
  (measured at `Dk=Dv=32`) with no `TT_FATAL`. Guard unverified geometry rather than risking it.
