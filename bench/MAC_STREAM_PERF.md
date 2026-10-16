# Mac stream performance

Campaign log for scheduling changes on the fork [Boundless-Design/Strata](https://github.com/Boundless-Design/Strata), branch `mac-stream-perf`, cut from `296b59d2f73c8f04a805df29c2fa97655adace78`. StrataMetal stays the verification harness. Its engine is not modified. Each step below is one commit, and the gate runs after that step.

The machine is the Apple M5 Max from `bench/UPSTREAM_MAC.md` (StrataMetal) and from the arena section of `bench/UPSTREAM_MAC_GAPS.md`: 18 CPUs, `hw.memsize` 38654705664, engine RAM print 38.7 GB, GPU working set 30.2 GB. Page size 16384. The file cache was not purged. One `build-metal/strata` at a time. The arena is `STRATA_METAL_ARENA_GB=12` on every run in this file (decimal GB, `kGB = 1e9`). The engine prints that as 12.00 GB, 8616 slots. `--expert-cache 64` is the separate 64-slot, 0.08 GiB profile cache. `STRATA_METAL_WIRE_ARENA` is unset, and this Mac is under 64 GB, so the arena is not wired.

The binary is the fork's Release `build-metal/strata`, built from that commit with no engine diff. `third_party/metal-cpp` in the fork checkout is an untracked symlink to the existing tree and is not part of any commit.

## Gate

A scheduling change has to leave the math alone.

Bit-identical, against the step 0 captures:

- Two `STRATA_LOGPOS` serves of p1–p5, same command as `bench/UPSTREAM_MAC.md`, plus `STRATA_METAL_ARENA_GB=12`. The reference file is 2,041,179 bytes, 492 lines, sha256 `5a9f6f5320876493934df4063c20baf8095338408e25d9c5fde9dfccc4ed739b`. The two new files must match each other and that reference. Log probabilities are `%.6f`.
- A 200-token greedy generation of the 307-id p5 prompt. This binary prints the generated ids on one `output  :` line, not as `T <id>` lines. The reference line is 921 bytes, 200 ids, sha256 `d6d12f1383826252ac6500020a2aef6a918fba6b4700d1a18e72ab9c3eeadd11`. The three runs of a step must match each other and that line.

Any difference is a bug. The step is reverted and reported.

Fixtures: `metal/tools/compare_fixtures.py` is not edited. A native pack still writes an empty `--dump-logits` file, so the score is the same pick rule and the same p5 prompt-NLL rule applied to the LOGPOS rows. Exact argmax, or the Mac id is in CUDA's top 3 and CUDA's gap from its own first choice to that id is at most 0.5. p5 prompt NLL uses the 293 prompt rows whose next prompt token is in CUDA's top 64, within 1% of CUDA and not more than 1% worse. Spread is not run, because there is no one-ulp dump.

Speed: p5 prompt plus 200 tokens, three fresh processes, median of the three decode rates. `STRATA_METAL_STREAM=1`, `STRATA_METAL_ARENA_GB=12`. Report tok/s, the ledger below, and the memory-pressure state.

Pressure samples are about twice a second: `kern.memorystatus_vm_pressure_level`, `kern.memorystatus_level`, `vm.swapusage`, and `vm_stat` (Swapouts, compressor pages, Compressions). Idle on this Mac is pressure level 1. A run is killed on swap-used growth, on Swapouts growth, or when pressure level 2 or higher holds across two consecutive samples. A single level-2 sample is recorded and the run continues. A run is not started when the level is already 2 or higher. The arena is not pushed past 12 GB. The 16 GB and 20 GB sizes stay unstarted, as in the arena section of `bench/UPSTREAM_MAC_GAPS.md`.

## Ledger

Stock counters only. `STRATA_METAL_PROFILE` is unset, and the debug patch from the gaps note is not in this binary.

Decode is one window per generated token (drafts accepted 0 of 199, 1.00 tokens per round). Steady state is the T=2 line: 199 of the 200 windows. The expert-stream line averages all 200 windows, including the one T=1 window.

- SSD wait is the expert-stream "of it waiting" milliseconds. That is the `done_cv_` wait inside `ExpertStream::at_layer`.
- The host turn is the expert-stream "host" milliseconds. It includes the SSD wait, the locked lookup, and `ahead()`.
- Residual is host minus waiting. On the earlier 12 GB sweep that residual sat at 0.69 ms.
- The T=2 GPU figure is `GPUEndTime − GPUStartTime` for those 199 windows. The engine's own label is "GPU incl. the host's turns": the GPU waits on the shared event until the host stores the release after `at_layer` returns, so the host turn sits inside that timestamp.
- Encoder time is not a separate counter. T=2 GPU timestamp minus the 200-window host turn is resident kernel time and the per-layer encoder-boundary tax added together. The kprof split of those two, 16.071 ms and 6.310 ms, is the other capture in `bench/UPSTREAM_MAC_GAPS.md`. It is not re-measured here.
- The T=2 host wall minus the T=2 GPU timestamp is CPU outside the GPU span, including the encode of the command buffer before commit.

The engine's "MB read per window" is `bytes_read_ / windows`. `reset_stats()` has no caller, so the counter still holds the arena fill. Each completed expert is 1,382,400 bytes. The after-fill column subtracts `8616 × 1,382,400` and divides by 200 generated tokens. That remainder still includes the prompt chunk.

## Step 0 — unmodified baseline

No source change. This is the number later steps have to beat. The earlier controlled 12 GB median in the arena sweep was 30.59 tok/s, hits 0.9508, wait 7.76 ms, host 8.45 ms, peak footprint 18249108872 bytes, with swap used constant at 1024458752 bytes. Those runs are a different machine state. Swap used on every run below is 1640434237 bytes (`vm.swapusage` 1564.44M), left over from the killed 16 GB fill, and it does not grow. The target is the median measured here.

### Correctness

Same serve as `bench/UPSTREAM_MAC.md`, from the fork checkout, with `STRATA_METAL_ARENA_GB=12` added. `STRATA_HC_Q8=0`, so this path is BF16. Two processes, `base-log-a` and `base-log-b`. Both exited 0. The two LOGPOS files are byte-identical.

```sh
/usr/bin/time -l -o TIMEFILE \
  env STRATA_METAL_STREAM=1 STRATA_METAL_ARENA_GB=12 \
      STRATA_ROUTE_TAIL_SKIP=0 STRATA_HC_Q8=0 STRATA_PREFILL_CPU_SHARE=0 \
      STRATA_METAL_PROMPT=window \
      STRATA_LOGPOS=LOGPOS.tsv STRATA_LOGPOS_TOPK=256 \
  ./build-metal/strata \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile ./data/expert-profile.bin \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --mmap-experts --kv fp16 \
  --max-context 4096 --short-read 8192 --prompt-cache 0 --adapt-every 0
```

Stdin is the five `GEN 1 ckpt=0 …` lines plus `QUIT`. The profile path is the fork's `data/expert-profile.bin` (24576 ranked pairs). Arena on `base-log-a`: 12.00 GB, 8616 slots, filled in 2.6 s at 4.64 GB/s. Planned footprint 18.5 GB = arena 12.0 + 6.0 held + 0.5. Available RAM printed 19.7 GB. Prompt buffers 1.36 GB in the arena.

`time -l` peak footprint: `base-log-a` 18137467840 bytes (18.137 GB), max RSS 13893992448 (12.940 GiB). `base-log-b` 18104192056 (18.104 GB), max RSS 15886237696 (14.795 GiB).

Score against the CUDA fixtures, same rules as `compare_fixtures.py`:

| Prompt | Positions | Exact | Accepted | Pick failures |
| --- | ---: | ---: | ---: | ---: |
| p1 | 36 | 36 | 0 | 0 |
| p2 | 42 | 41 | 1 | 0 |
| p3 | 35 | 34 | 1 | 0 |
| p4 | 41 | 40 | 1 | 0 |
| p5 | 338 | 328 | 10 | 0 |

492 positions, 479 exact, 13 accepted, 0 pick failures. p5 prompt NLL: CUDA mean 1.593251, Mac mean 1.590797, relative 0.001540. Mac lower on 143 rows, CUDA lower on 150, no ties. Both NLL checks pass. Spread was not checked. The relative NLL matches the auto-arena score in `bench/UPSTREAM_MAC.md`.

Pressure on both serves: level stayed 1, swap used stayed 1640434237, Swapouts stayed 1785829. `base-log-a` compressor pages peaked at 907120 from a base of 65801 (Δ 841319), Compressions Δ 1072125, `memorystatus_level` 33–85. `base-log-b` compressor peak Δ 396329, Compressions Δ 609265, `memorystatus_level` 33–81.

### Speed

Default speed settings. No `STRATA_HC_Q8`, no `STRATA_ROUTE_TAIL_SKIP`, no `STRATA_PREFILL_CPU_SHARE`, no `STRATA_METAL_PREFETCH2` (the code default is 0; the log's "read-ahead top 10" is `pre_k_`). Three fresh processes, in order `base-s1`, `base-s2`, `base-s3`. All exited 0.

```sh
/usr/bin/time -l -o TIMEFILE \
  env STRATA_METAL_STREAM=1 STRATA_METAL_ARENA_GB=12 \
  ./build-metal/strata \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile ./data/expert-profile.bin \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --max-context 32768 --mmap-experts --kv fp16 \
  --tokens "$(cat /tmp/upstream-mac/p5.prompt)" --max-new 200
```

The three `output  :` lines are byte-identical to each other and to the reference hash above. HC read in Q8_0. Planned footprint 21.7 GB = arena 12.0 + 9.2 held + 0.5, under the 30.2 GB cap. After the fill the engine says the process holds 18.2 GB. Prompt buffers 1.39 GB, 997 slots. Prefill reports 306 tokens.

| Run | Decode tok/s | Decode ms | Prefill tok/s | Prefill ms | TTFT ms | T=2 wall ms | T=2 GPU ms | Host ms | SSD wait ms | Residual ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base-s1 | 21.75 | 9196.6 | 89.71 | 3411.2 | 3759.4 | 45.83 | 44.34 | 12.65 | 11.95 | 0.70 |
| base-s2 | 24.92 | 8027.1 | 88.69 | 3450.2 | 3829.8 | 39.82 | 38.61 | 8.41 | 7.70 | 0.71 |
| base-s3 | 25.02 | 7993.0 | 83.60 | 3660.5 | 4031.4 | 39.78 | 38.58 | 8.56 | 7.87 | 0.69 |

Median of the three decode rates: **24.92 tok/s** (`base-s2`). That run's ledger is the one later steps quote beside the new median.

`base-s2` per token, from the stock counters:

- SSD wait 7.70 ms (200-window average).
- Host turn 8.41 ms, residual 0.71 ms.
- T=2 GPU timestamp 38.61 ms. Minus the host turn, 30.20 ms, which is resident kernels and the per-layer encoder boundary together.
- T=2 host wall 39.82 ms. Minus the GPU timestamp, 1.21 ms of CPU outside the GPU span.
- Decode wall is 8027.1 / 200 = 40.1355 ms per token. The single T=1 window on this run was 96.48 ms (GPU timestamp 92.10 ms), which is why the decode line is slower than the T=2 mean.

Hits on all three runs: 0.9508. Read-ahead hits 0.0271. Misses 0.0221 (19.3 per window). Late read-aheads: 1626, 1604, 1709. Read-aheads issued: 6909, 6910, 6908 (34.5 per window). Engine MB/window: 185.5, 186.3, 186.2 (reads 26832, 26959, 26940). After-fill MB per generated token: 125.909, 126.787, 126.655. Miss bytes at the printed 19.3 misses per window are 26.68 MB per token.

`time -l` peak footprint and max RSS:

| Run | Footprint | RSS |
| --- | ---: | ---: |
| base-s1 | 18248371616 (18.248 GB) | 17361600512 (16.169 GiB) |
| base-s2 | 18248420768 (18.248 GB) | 16956981248 (15.792 GiB) |
| base-s3 | 18248650144 (18.249 GB) | 16095657984 (14.990 GiB) |

Available RAM printed at arena time: 20.3 GB, 17.1 GB, 16.8 GB. Fill: 1.6 s at 7.25 GB/s, 1.7 s at 7.02 GB/s, 1.7 s at 6.82 GB/s. The footprint matches the earlier 12 GB sweep (18.249 GB), and the hit rate and the miss count match it. The wait on the two warmer runs, 7.70 and 7.87 ms, sits on that sweep's 7.76 ms. Decode does not: 24.92 tok/s here, 30.59 tok/s there. The extra time is inside the T=2 GPU timestamp, not in the SSD wait.

Pressure. Level stayed 1 on all three. Swap used stayed 1640434237. Swapouts stayed 1785829. Nothing was killed. The single-sample level-2 rule did not come into play on the runs in this table.

| Run | Compressions Δ | Compressor pages, peak Δ | Pages at the peak, bytes | memorystatus_level |
| --- | ---: | ---: | ---: | --- |
| base-s1 | +291527 | +152829 | 2503950336 (2.504 GB) | 32–80 |
| base-s2 | +253912 | +140712 | 2305425408 (2.305 GB) | 32–80 |
| base-s3 | +346229 | +174866 | 2865004544 (2.865 GB) | 33–80 |

Compressor bases were 178954, 187227, and 187928 pages. At exit the page count was within a few thousand of that base.

## Step 1 — `STRATA_METAL_PREFETCH2`, stopped before a speed number

No source change. The variable is the top-k for the layer `l+2` prediction. The code default remains 0. The binary does not print the value. The log line "read-ahead top 10" is still `pre_k_`.

`STRATA_METAL_PREFETCH2=10` was applied to the same two-serve LOGPOS gate as step 0. Both files match the step 0 reference (2,041,179 bytes, sha256 `5a9f6f5320876493934df4063c20baf8095338408e25d9c5fde9dfccc4ed739b`). The fixture score matches step 0: 479 exact, 13 accepted, 0 pick failures, p5 prompt NLL CUDA 1.593251 and Mac 1.590797, relative 0.001540. Both NLL checks pass. Spread was not checked.

Swap used on these runs is 1632045629 bytes (`vm.swapusage` 1556.44M), 8388608 bytes under the step 0 constant. It does not grow. Swapouts stay 1785829.

| Serve | Available | Fill | Level | Compressor pages | jetsam | Footprint | RSS |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| p10-log-a | 15.8 GB | 2.5 s at 4.78 GB/s | 1 the whole serve | base 106650, peak 886668 | 30–82 | 18105699168 (18.106 GB) | 13119029248 (12.218 GiB) |
| p10-log-b | 17.6 GB | 2.1 s at 5.81 GB/s | one sample at 2, jetsam 33, compressor 792816 | base 219849, peak 814814 | 31–79 | 18105175120 (18.105 GB) | 15846850560 (14.759 GiB) |

The single level-2 sample on `p10-log-b` did not repeat on the next sample, so the serve was not killed. Swap delta on that sample was 0.

The speed gate did not produce a tok/s. Two fresh `PREFETCH2=10` processes were killed after the arena-fill line and before prefill. Both printed arena 12.00 GB, 8616 slots, planned footprint 21.7 GB, and "the process holds 18.2 GB" after the fill. Resident at that line was 2.9 GB on the first attempt and 3.1 GB on the second. The step 0 speed fills reported 15.4–17.4 GB resident at the same line, and those fills ran at 6.82–7.25 GB/s. These two ran at 4.04 GB/s (2.9 s, 16.8 GB available) and 3.97 GB/s (3.0 s, 16.6 GB available).

Both kills were two consecutive level-2 samples. Swap did not grow on either. After each kill the next sample was level 1 and jetsam was back at 79–80.

| Attempt | Level-2 samples | jetsam on those samples | Compressor pages on those samples | Peak compressor pages while level was still 1 |
| --- | --- | --- | ---: | ---: |
| First | 2 | 43, then 37 | 830353, then 986356 | 609615 |
| Second | 2 | 40, then 36 | 981723, then 722794 | 1006106 |

`STRATA_METAL_PREFETCH2=4` was not started. There is no decode rate, no ledger, and no token line for this step. The setting is not adopted. The serve-path rows match the unmodified binary, and the speed comparison against 24.92 tok/s was not measured. A third 12 GB process was not launched.

## Cause check — depth 0 on the cooled machine

Measurement only. No source change. The question is whether the two depth-10 speed kills were the prefetch or the machine.

Before any process, `pgrep -fl strata` found nothing, so nothing was killed. Page size 16384. `vm_stat`: free 967507, active 486085, inactive 475006, wired down 234448, occupied by the compressor 66806. That compressor is 66806 × 16384 = 1094549504 bytes (1.095 GB), under 2 GB. Free plus inactive is 1442513 pages, 23634132992 bytes (23.634 GB), over 20 GB. `memory_pressure` printed a system-wide free percentage of 85%. `kern.memorystatus_vm_pressure_level` was 1 and `kern.memorystatus_level` was 85. `vm.swapusage` used 1548.44M. The five largest `top -l 1 -o mem -n 5` entries were WindowServer 1314M, grok 525M, node 480M, Stats 354M, and TGOnDeviceInference 274M. The gate allowed a launch.

One control, `STRATA_METAL_PREFETCH2=0`, `STRATA_METAL_ARENA_GB=12`, the step 0 speed command. It exited 0. Pressure level stayed 1. Swap used stayed 1615268413 bytes (the gate's reading of the 1540.44M print at process start). Swapouts stayed 1785829. jetsam 37–86. Compressor pages base 66628, peak 712349 (11671126016 bytes, 11.671 GB), end 106239. Compressions rose by 924773.

The after-fill line: arena 12.00 GB, 8616 slots, filled in 2.3 s at 5.12 GB/s, available 18.8 GB, "the process holds 18.2 GB (10.3 GB resident), planned 21.7 GB". `time -l` peak footprint for the whole process was 18249125352 bytes, max RSS 15540731904. Decode 200 tokens in 6587.0 ms, 30.36 tok/s. The `output  :` line matches the step 0 reference (sha256 `d6d12f1383826252ac6500020a2aef6a918fba6b4700d1a18e72ab9c3eeadd11`). Expert stream over 200 windows: hits 0.9509, misses 19.3 per window, host 8.81 ms, waiting 8.13 ms. T=2: 199 windows, 32.70 ms each, GPU timestamp 31.50 ms.

That run is past the stop line (fill was not under 5 GB/s, resident was not under 10 GB) and short of step 0 (fill about 7 GB/s, resident 15 GB or more, decode about 24.9 tok/s). Depth 4 was not started, and depth 10 was not repeated. The step 1 kills remain the depth-10 rows below.

| Depth | Arena | Fill | Resident after fill | Held / planned after fill | Decode tok/s | Pressure peak |
| --- | ---: | --- | ---: | --- | ---: | --- |
| 0, this control | 12 GB | 2.3 s, 5.12 GB/s | 10.3 GB | holds 18.2 GB, planned 21.7 GB | 30.36 | level 1, jetsam 37, compressor 712349 pages |
| 10, step 1 first kill | 12 GB | 2.9 s, 4.04 GB/s | 2.9 GB | holds 18.2 GB, planned 21.7 GB | none, killed before prefill | level 2 on two samples, jetsam 43 then 37 |
| 10, step 1 second kill | 12 GB | 3.0 s, 3.97 GB/s | 3.1 GB | holds 18.2 GB, planned 21.7 GB | none, killed before prefill | level 2 on two samples, jetsam 40 then 36 |
| 4 | not run |  |  |  |  |  |

### What depth 4 and depth 10 allocate

`STRATA_METAL_PREFETCH2` is read once, at `metal/src/engine/stream.cpp` line 158: `pre_k2_ = env_int("STRATA_METAL_PREFETCH2", 0)`. The default stays 0. At a decode cut, lines 708–710 call `ahead(l + 1, T, pred, pre_k_)` and, when `pre_k2_ > 0`, `ahead(l + 2, T, pred2, pre_k2_)`.

`ahead` (lines 572–612) caps `K` at 32 (line 573). For each of the `T` rows it keeps that row's top `K` expert ids, then deduplicates them into `pw[8 * 32]` (line 574), at most 256 ids. Each id that is not already in the arena takes one existing slot from `victim` and `submit` (lines 601–603). `submit` (lines 332–342) points that slot at the expert and queues a `pread` into `host_[slot]`. It does not allocate the slot. The slot payload on this pack is one expert: `lay_` stores `g->bytes / NE`, `u->bytes / NE`, and `d->bytes / NE` (line 72), and the contiguous reader uses `gs + us + ds` (line 275). The experts file header has one size, 460800 + 460800 + 460800 = 1382400 bytes. `PAGE` is 16384 (line 25), so a slot is 85 pages. The `Slot` struct (stream.h line 77) is the metadata.

The only buffer created because prefetch depth is non-zero is `predL2_`. `window.cpp` line 52 sets `MAXT = 8`. Lines 545–546:

```
if (st_pred_) predL_ = A(4 * (size_t) 48 * MAXT * 512);
if (st_pred_ && st_->ahead2() > 0) predL2_ = A(4 * (size_t) 48 * MAXT * 512);
```

`A` is `cx_->alloc` (line 448), a device buffer, not an arena slot. The byte count is `4 * 48 * 8 * 512 = 786432`. The formula does not contain the depth, so depth 4 and depth 10 allocate the same 786432 bytes. The extra router GEMV (lines 772–775) writes into that buffer. 786432 bytes is under 2 GB, so a depth-10 run would have stayed at arena 12 GB. It was not launched, because the depth-0 control did not reproduce the step 0 fill.

On a steady T=2 window the cap in `ahead` is `2 * K` distinct experts when the two rows do not overlap: 8 at depth 4, 20 at depth 10. Each one occupies an existing 1382400-byte slot. That is 11059200 bytes of slot payload in play at depth 4 and 27648000 bytes at depth 10, inside the arena that was already filled, not on top of it.

## Alternating depth, arena 12 GB

Measurement only, on the unmodified binary. The step 0 fill rate and the after-fill resident size are not stop lines. A run is killed if pressure level 2 shows up on 3 samples in a row, sampled every 2 seconds, or if swap used grows by more than 256 MiB (268435456 bytes). Nothing ran between these five processes. `pgrep -fl strata` was empty before each one. All five exited 0. Every sample was pressure level 1. Swap used stayed 1615268413 bytes (`vm.swapusage` 1540.44M) from before A through after E. Swapouts stayed 1785829. Each `output  :` line matches the step 0 reference.

The before-run columns are compressor pages, free plus inactive pages, and the pressure level. Held after the fill is 18.2 GB and the planned footprint is 21.7 GB on every run. Arena 12.00 GB, 8616 slots.

| Run | Depth | Before, comp / free+inactive / level | Fill | Resident after fill | Compressor peak, pages | Level counts | Swap before → after | Decode tok/s |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: |
| A | 0 | 73507 / 1473455 / 1 | 2.1 s, 5.65 GB/s | 12.3 GB | 467639 | 8 at level 1 | 1615268413 → 1615268413 | 30.33 |
| B | 10 | 74697 / 1563936 / 1 | 1.7 s, 7.13 GB/s | 17.4 GB | 97278 | 7 at level 1 | 1615268413 → 1615268413 | 30.12 |
| C | 0 | 96267 / 1563501 / 1 | 1.6 s, 7.31 GB/s | 17.4 GB | 107035 | 7 at level 1 | 1615268413 → 1615268413 | 30.55 |
| D | 4 | 100082 / 1586718 / 1 | 1.7 s, 6.89 GB/s | 17.1 GB | 113754 | 7 at level 1 | 1615268413 → 1615268413 | 30.71 |
| E | 0 | 107981 / 1580724 / 1 | 1.7 s, 7.18 GB/s | 17.4 GB | 111864 | 7 at level 1 | 1615268413 → 1615268413 | 30.65 |

Decode times: A 6594.8 ms, B 6640.2 ms, C 6545.9 ms, D 6512.8 ms, E 6524.8 ms.

The controls are A, C, and E: 30.33, 30.55, and 30.65 tok/s. None was killed. None is more than 10% under 30.36 (the floor is 27.324). Their spread is 30.65 − 30.33 = 0.32 tok/s. A depth counts as faster only when it beats both neighbors by more than that spread.

Depth 10 is 30.12 tok/s, below A and below C. Depth 4 is 30.71 tok/s, 0.16 above C and 0.06 above E. Neither clears 0.32. No depth is faster. The LOGPOS check was not run.

The cold first control's compressor peaked at 467639 pages. The next four runs, depths 10, 0, 4, and 0, peaked at 97278, 107035, 113754, and 111864 pages. The peak does not separate by depth.

## Fix 4, ahead() bound, arena 12 GB, cold

Fix 4 is `81b66ba`. It stays. `STRATA_METAL_TIME_AHEAD` defaults to 0. Set to 1, `ahead()` adds its own wall time and prints one line. `host_ms_` is the same counter as before and still includes that work. One 200-token decode, arena 12 GB, the same command as the controls. `STRATA_METAL_PREFETCH2` was unset. `pgrep -fl strata` was empty before it. It exited 0. Every sample was pressure level 1. Swap used stayed 1590102589 bytes (`vm.swapusage` 1516.44M). Swapouts stayed 1785829. This run is cold. No warm-up was discarded. It overstates `wait_ms_`. It is not the bound used for the later fixes.

Before the run: compressor 54408 pages, free plus inactive 1349861 pages, pressure level 1. Nine samples, all level 1. Compressor peak 888577 pages. Fill 2.7 s at 4.39 GB/s. After the fill the process holds 18.2 GB (4.2 GB resident), planned 21.7 GB.

The stream line: host 8.68 ms per window, of it waiting 7.73 ms. The new line: `expert stream ahead-time 144.847 ms total over 200 decode windows (0.724 ms per window)`. The decode line: 200 tokens in 8502.0 ms, 23.52 tok/s. There are 200 decode windows and 200 decode tokens, so a per-window figure is a per-token figure.

| Quantity | Value |
| --- | ---: |
| `ahead()` total | 144.847 ms |
| `ahead()` per token | 0.724 ms |
| `host_ms_` per token | 8.68 ms |
| `wait_ms_` per token | 7.73 ms |
| Decode per token | 42.51 ms |

144.847 / 8502.0 = 1.70% of this run's per-token time. That is 1% or more, so the alternating sequence below ran. The 23.52 tok/s figure is this timing run (4.2 GB resident after a cold fill). It is the denominator of the 1% test.

## Alternating fix 4, arena 12 GB

The treatment binary is fix 4 plus the default-off timer. `STRATA_METAL_TIME_AHEAD` was unset, so the new counter did not run. The control is the saved unmodified binary. Nothing else ran between these five processes. Compressor pages before each run were under 150000, so none of them waited. A run is killed if pressure level 2 shows up on 3 samples in a row, sampled every 2 seconds, or if swap used grows by more than 256 MiB. None of those fired. `pgrep -fl strata` was empty before each one. All five exited 0. Every sample was pressure level 1. Swap used stayed 1590102589 bytes from before A through after E. Swapouts stayed 1785829. Each `output  :` line matches the step 0 reference.

Held after the fill is 18.2 GB and the planned footprint is 21.7 GB on every run. Arena 12.00 GB, 8616 slots. The before-run columns are compressor pages, free plus inactive pages, and the pressure level.

| Run | Binary | Before, comp / free+inactive / level | Fill | Resident after fill | Compressor peak, pages | Level counts | Swap before → after | Decode tok/s |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: |
| A | control | 107538 / 1441297 / 1 | 2.8 s, 4.19 GB/s | 3.4 GB, cold | 986519 | 8 at level 1 | 1590102589 → 1590102589 | 30.49 |
| B | fix 4 | 112052 / 1535152 / 1 | 1.7 s, 7.22 GB/s | 17.5 GB, warm | 126498 | 7 at level 1 | 1590102589 → 1590102589 | 30.78 |
| C | control | 124596 / 1545949 / 1 | 1.8 s, 6.68 GB/s | 15.8 GB, warm | 131478 | 7 at level 1 | 1590102589 → 1590102589 | 30.34 |
| D | fix 4 | 124323 / 1552379 / 1 | 1.7 s, 7.00 GB/s | 17.3 GB, warm | 136375 | 7 at level 1 | 1590102589 → 1590102589 | 30.94 |
| E | control | 133360 / 1555100 / 1 | 1.7 s, 7.03 GB/s | 17.4 GB, warm | 146681 | 7 at level 1 | 1590102589 → 1590102589 | 30.84 |

Decode times: A 6560.5 ms, B 6497.1 ms, C 6592.9 ms, D 6463.6 ms, E 6486.0 ms.

The controls are A, C, and E: 30.49, 30.34, and 30.84 tok/s. None was killed. None is more than 10% under 30.36 (the floor is 27.324). Their spread is 30.84 − 30.34 = 0.50 tok/s. A treatment counts as faster only when it beats both neighbors by more than that spread.

B is 30.78 tok/s, 0.29 above A and 0.44 above C. D is 30.94 tok/s, 0.60 above C and 0.10 above E. Fix 4: correct, likely about +1% (both treatments above the control mean, 30.86 vs 30.56), below the 0.50 noise floor, kept. No more runs on it.

A's compressor peaked at 986519 pages on the cold fill (3.4 GB resident). B, C, D, and E peaked at 126498, 131478, 136375, and 146681 pages. A is cold. B, C, D, and E are warm. The peak differs, and the first control is the high one.

## Warm split, arena 12 GB

Timing runs from here on discard one plain 200-token decode with `STRATA_METAL_TIME_AHEAD` unset, then measure. Each run's table says cold or warm. The cold ahead() table above had no discarded warm-up.

The discarded warm-up exited 0. Before it: compressor 95410 pages, free plus inactive 1421853 pages, pressure level 1. Eight samples, all level 1. Compressor peak 973330 pages. Fill 2.9 s at 4.09 GB/s. After the fill the process holds 18.2 GB (2.8 GB resident). That fill is cold, so the decode is discarded. Its decode line was 200 tokens in 6484.6 ms, 30.84 tok/s. Swap used stayed 1564936765 bytes (`vm.swapusage` 1492.44M). Swapouts stayed 1785829.

The timed run is the next process, `STRATA_METAL_TIME_AHEAD=1`, the same command, arena 12 GB. `STRATA_METAL_PREFETCH2` was unset. It exited 0. `pgrep -fl strata` was empty before the pair. Every sample was pressure level 1. Swap used stayed 1564936765 bytes. Swapouts stayed 1785829.

Before the timed run: compressor 124886 pages, free plus inactive 1541235 pages, pressure level 1. Seven samples, all level 1. Compressor peak 135971 pages. Fill 1.9 s at 6.34 GB/s. After the fill the process holds 18.2 GB (14.0 GB resident), planned 21.7 GB. Arena 12.00 GB, 8616 slots, read-ahead top 10. This run is warm. The warm speed runs above sat near 15–17 GB resident and about 7 GB/s. This fill is above the cold fills and below those.

Decode: 200 tokens in 6899.4 ms, 28.99 tok/s. Per token, 6899.4 / 200 = 34.497 ms. This is a timing run, not a speed verdict. 28.99 is above the floor of 27.324.

The stock stream line: 200 windows, 174585 distinct lookups (872.9 per window), hits 0.9508, read-ahead hits 0.0271 (2265 late), misses 0.0221 (19.3 per window), read-aheads 6914 (34.6 per window, 9900 dropped), 185.5 MB per window (26841 reads), host 8.84 ms per window, of it waiting 7.94 ms.

The ahead-time line: `expert stream ahead-time 133.945 ms total over 200 decode windows (0.670 ms per window); late-only wait 136.111 ms; misses 3861 hits 165994 read-ahead hits 4730 late 2265`. 3861 + 165994 + 4730 = 174585.

The encode line: `stream encode-before-commit 121.923 ms total over 200 decode windows (0.610 ms per window)`.

Windows: T=1, 1 decode, 74.76 ms (GPU 71.02 ms). T=2, 199 decode, 34.26 ms (GPU 33.08 ms).

GPU per token is `win_gpu_`, the sum of `run_graph_seg`'s return (`GPUEndTime − GPUStartTime`). (1 × 71.02 + 199 × 33.08) / 200 = 33.270 ms.

After fix 4 the GPU waits only until `release_seg()`, which runs after `smap` is published and before `ahead()`. The pre-release stall (the lookup, `wait_ms_`, and the publish) sits inside that GPU timestamp, because the GPU is blocked on the event. `ahead()` also sits inside `[GPUStart, GPUEnd]`: it runs while the next encoder is in flight, and an overshoot that delays the next release is still inside the timestamp. `host_ms_` not overlapped is 0. Remainder = decode wall per token − GPU per token − 0 = 34.497 − 33.270 = 1.227 ms.

`host_ms_ − ahead()` = 8.84 − 0.670 = 8.170 ms. That stall is already inside the GPU timestamp (`wait_ms_` 7.94 ms, plus about 0.23 ms of lookup and publish). It is not subtracted again. GPU per token minus that stall is 33.270 − 8.170 = 25.100 ms, kernels and the encoder boundary together.

Window wall is (74.76 + 199 × 34.26) / 200 = 34.463 ms. Decode wall minus window wall is 0.034 ms outside the window. On T=2, window minus GPU is 34.26 − 33.08 = 1.18 ms of CPU outside the GPU span inside the window. The encode counter is 0.610 ms of that. The rest of the 1.18 ms is embeddings, the C++ dispatch build, and the tail after `GPUEndTime`.

| Quantity | Per token | Where it comes from | Cold or warm |
| --- | ---: | --- | --- |
| Decode wall | 34.497 ms | decode line, 6899.4 / 200 | warm |
| `host_ms_` | 8.84 ms | stock stream line | warm |
| `wait_ms_` | 7.94 ms | stock stream line | warm |
| `ahead()` | 0.670 ms | ahead-time, 133.945 / 200 | warm |
| GPU | 33.270 ms | `win_gpu_` | warm |
| `host_ms_` not overlapped | 0 ms | pre-release stall and `ahead()` are both inside `win_gpu_` | warm |
| Remainder | 1.227 ms | wall − GPU − 0 | warm |

Raw decode counts: misses 3861, hits 165994, read-ahead hits 4730, late 2265. Hit rate 0.9508. Read-ahead hit rate 0.0271. Miss rate 0.0221.

### Bounds on this warm token

One percent of 34.497 ms is 0.345 ms.

Fix 3 targets late-only `done_cv_` waits: decode layers that submitted no miss, so every waited slot was already in flight. The counter is `late_only_ms_`, 136.111 / 200 = 0.681 ms, 1.97% of the wall. A miss and a late on the same layer share one condvar wait. That mixed wait stays in `wait_ms_` and is not in `late_only_ms_`, so the bound leaves out any extra on mixed layers beyond the miss. `wait_ms_` is 7.94 ms and is not this bound.

Fix 5 targets the pre-commit encode inside `run_graph_seg` on decode windows (`Tn < 8`). The counter is encode-before-commit, 121.923 / 200 = 0.610 ms, 1.77% of the wall. The 1.227 ms remainder also includes embeddings, dispatch construction, and CPU between windows. It is not fix 5's bound.

Fix 1 has no separable counter on this run. The mixed pool above is 25.100 ms. There is no fix-1 millisecond bound from this split.

Fix 4 is already kept. Warm `ahead()` is 0.670 ms, 1.94% of this token (133.945 / 6899.4). The alternating sequence already ran. No more runs on it.

### Rank

Fix 3 is 0.681 ms, 1.97%. Fix 5 is 0.610 ms, 1.77%. Neither is under 1%. Fix 3 is the higher bound, so it is the next change. Fix 5 is deferred because it is lower, not because it is under 1%. Fix 1 is not started.
