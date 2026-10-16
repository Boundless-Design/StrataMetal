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
