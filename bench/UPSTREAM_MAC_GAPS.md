# Upstream Mac inter-encoder gaps

Measurement only. No change is committed in Strata or in StrataMetal's engine. The numbers below come from one untraced process of upstream `296b59d2f73c8f04a805df29c2fa97655adace78` (0.1.44.1-metal) on this Mac, with a local debug patch that was reverted after the capture. Shader Timeline could not be turned on. Per-kernel times are `STRATA_METAL_KPROF` running in streamed mode.

## Setup

This Mac is the Apple M5 Max used for `bench/UPSTREAM_MAC.md` (36 GiB unified memory, `hw.memsize` 38654705664). The file cache was not purged. One GPU process at a time. Q2_0 weights already on disk, pack `~/Developer/Strata-data/packs/q2_0`.

Command, from `/Users/kamp/Developer/Strata-upstream`. `--max-new 40` on the 307-id p5 prompt. Env: `STRATA_METAL_STREAM=1`, `STRATA_METAL_GAPLOG=/tmp/upstream-mac/gap.log`, `STRATA_METAL_KPROF=2`.

```sh
/usr/bin/time -l -o /tmp/upstream-mac/gap.time \
  env STRATA_METAL_STREAM=1 STRATA_METAL_GAPLOG=/tmp/upstream-mac/gap.log STRATA_METAL_KPROF=2 \
  ./build-metal/strata \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile "$HOME/Developer/Strata-upstream/data/expert-profile.bin" \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --max-context 32768 --mmap-experts --kv fp16 \
  --tokens "$(cat /tmp/upstream-mac/p5.prompt)" --max-new 40
```

Exit 0. Wall 9.66 s. Max RSS 13289668608 (12.377 GiB). Peak footprint 15918349816 (15.918 GB). Page faults 139988. Block input operations 0. Arena 9.61 GB, 6901 slots, filled in 2.3 s at 4.17 GB/s. Planned footprint 19.3 GB against the 30.2 GB cap. Prefill 306 tokens in 5099.9 ms (60.0 tok/s). The warm speed runs in `bench/UPSTREAM_MAC.md` prefilled at a median 82.33 tok/s and the cold run at 26.00, so this process sits between them. The token timed below matches the warm T=2 GPU timestamps, which were 31.37, 35.79, and 37.03 ms.

The debug patch, uncommitted, did three things. `window.cpp` calls `kprof` on the 20th cached T=2 head window even when streaming. `stream.cpp` appends one `LAY` line per decode layer. `mtl_rt.mm` appends `ENC`, `SEG`, and `DONE` lines around `run_graph_seg`. The `LAY` counters follow the same branches as the engine's own hit, prefetch-hit, late, and miss counters. After the capture the three files were restored and `build-metal/strata` was rebuilt from `296b59d`.

`kprof` re-runs the window's dispatches after that window's streamed command buffer has returned, with those experts already in the arena. It uses `run_graph` (one compute encoder, no host handshake). The full graph is the median of 7. Each class is then run alone, median of 7, with a barrier before every dispatch. Class percentages in the engine output are shares of the summed class medians (15.705 ms), which is a serialized attribution. The overlapped full-graph median is 16.071 ms. The class sum is 0.366 ms under that median, so isolating the classes did not inflate them. Summing the class times is a fair split of the resident kernel time. It is not a split of the 35.5 ms streamed timestamp.

## Shader Timeline

`xcrun xctrace list instruments` shows GPU, Metal Application, Metal GPU Counters, Metal Performance Overview, and Metal Resource Events. There is no Shader Timeline instrument. `xcrun xctrace record --template "Metal System Trace" --show-recording-options` prints only Hangs and Time Profiler. Metal GPU Counters, Metal Application, and GPU print an empty option list. The template at `GPU.instrdst/Contents/Templates/Metal System Trace.tracetemplate` has shader-profiler keys inside nested binaries and no user-facing switch that turns the timeline on. The trace already stored for `bench/UPSTREAM_MAC.md` was recorded with Shader Timeline disabled, and its shader duration tables are empty. Per-kernel time is the kprof table below.

## The traced 32 ms

`bench/UPSTREAM_MAC.md` measured one token under Instruments (p1, position 110): 17.582 ms of compute-encoder GPU execution, 32.027 ms of gaps between those encoders, GPU timestamp 49.61 ms. That process's expert-stream line was hits 0.8816 and 21.40 ms of host SSD wait per window, and the decoder ran at 19.59 tok/s. The tracer was inflating the stall. Encoder execution on that trace stayed in a tight band (median 17.316 ms across 27 command buffers) while the span moved from 36.871 to 58.186 ms.

Untraced, on the warm token below, the gap between encoders is 13.179 ms inside a 35.519 ms GPU timestamp. Kernel time with the experts resident is 16.071 ms. The 32 ms figure is the traced view of this same handshake, with a larger SSD wait and with the tracer's scheduling bubbles counted as gaps.

## Kernels

`strata metal kprof: window T2, 1189 dispatches, GPU 16.071 ms; classes alone (serialized) sum 15.705 ms.`

Top 15 classes by isolated GPU time. The share column is the engine's, out of 15.705 ms. Together these 15 are 11.634 ms and 638 dispatches.

| Rank | Isolated ms | Share | Dispatches | Class |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.695 | 10.8% | 48 | moe_gu_42_32_1_4 |
| 2 | 1.324 | 8.4% | 72 | gdn_step |
| 3 | 1.099 | 7.0% | 48 | moe_down_142_8_1_4 |
| 4 | 1.088 | 6.9% | 1 | gemv_13_16_1_4_1 output.weight |
| 5 | 0.735 | 4.7% | 18 | gemv_11_16_1_4_1 attn_qkv.weight |
| 6 | 0.677 | 4.3% | 25 | gemv_11_16_1_4_1 attn_gate.weight |
| 7 | 0.637 | 4.1% | 48 | gr_down_q8_8_8 hc_attn_down.weight |
| 8 | 0.636 | 4.1% | 48 | gr_down_q8_8_8 hc_ffn_down.weight |
| 9 | 0.604 | 3.8% | 17 | gemv_23_32_1_8_2 ssm_out.weight |
| 10 | 0.593 | 3.8% | 48 | gr_up_q8 hc_ffn_up.weight |
| 11 | 0.593 | 3.8% | 48 | gr_up_q8 hc_attn_up.weight |
| 12 | 0.575 | 3.7% | 95 | gemv_30_32_1_4_1 ffn_gate_inp.weight |
| 13 | 0.557 | 3.5% | 13 | gemv_23_16_1_4_1 attn_qkv.weight |
| 14 | 0.484 | 3.1% | 98 | gr_norm |
| 15 | 0.337 | 2.1% | 11 | gemv_12_32_1_8_2 ssm_out.weight |

Dispatches per kernel type, summing every class in the kprof list (78 classes, 1189 dispatches, 15.702 ms after rounding to the printed milliseconds). `gemv` is every `gemv_*` pipeline. Each row's milliseconds are the sum of that type's isolated class medians.

| Type | Dispatches | Isolated ms |
| --- | ---: | ---: |
| gemv | 542 | 7.699 |
| moe_gu | 48 | 1.695 |
| gdn_step | 72 | 1.324 |
| gr_down_q8_8_8 | 97 | 1.285 |
| gr_up_q8 | 97 | 1.196 |
| moe_down | 48 | 1.099 |
| gr_norm | 98 | 0.484 |
| qsa_attn | 12 | 0.319 |
| moe_route | 48 | 0.266 |
| moe_combine_sh | 48 | 0.148 |
| swiglu_prep | 48 | 0.081 |
| qsa_merge | 12 | 0.042 |
| qsa_prep | 12 | 0.029 |
| ple_gnorm | 2 | 0.011 |
| ple_gate_norm | 1 | 0.008 |
| argmax_part | 1 | 0.006 |
| xprep | 1 | 0.005 |
| ple_conv_add | 1 | 0.003 |
| argmax_final | 1 | 0.002 |

The routed expert kernels (`moe_gu` + `moe_down`) are 2.794 ms isolated, 96 dispatches, one per layer each. `moe_route` is 0.266 ms. The resident graph is a wide spread of small GEMVs, not one expert kernel.

## One warm token

Decode window 40, the last of the 40 generated rounds. It is a T=2 head window: 1189 dispatches, 49 compute encoders, 48 host cuts. Its GPU timestamp is 35.519 ms, which is the median of the last 15 T=2 windows (range 25.365–44.040). Its SSD wait is 12.119 ms, the median of those same windows (range 2.787–20.776). The first decode window was T=1 (GPU timestamp 77.87 ms). The second had 1153 dispatches and a 102.70 ms timestamp. Both are outside this table. The engine's "stream windows T=2 39 (decode): 51.51 ms" mixes those early windows with the wall clock of the window that ran kprof. The layer log does not. kprof runs after `run_graph_seg` returns.

How a cut sits in the graph. `window.cpp` pushes a cut immediately before `moe_gu`. Encoder 0 runs the prologue and layer 0 through its router and `swiglu_prep`, then signals. The host loads layer 0's experts. Encoder 1 runs layer 0's expert GEMVs and combine, then layer 1's attention and router, then signals. The host loads layer 1. The last encoder, after the layer-47 release, runs layer 47's experts and the output head. The "encoder" column is the host clock from the previous release (from commit, for encoder 0) until the GPU signals that encoder done. It is an upper bound on that encoder's GPU execution: it includes the delay from the release until the GPU reaches the signal. The "gap" column is the host clock from that signal through `at_layer` and the store that releases the next encoder. On a single command buffer Metal reports one `GPUStartTime` and one `GPUEndTime`, so there is no second shader clock per encoder. The two host clocks add up to that GPU timestamp.

Ledger for this token:

| Piece | ms | Where it sits |
| --- | ---: | --- |
| Overlapped kernels, one encoder, experts already resident (kprof) | 16.071 | inside the encoders |
| Extra host-observed encoder time above that kprof median | 6.310 | inside the 49 encoder intervals (spin 21.019 + tail 1.362 − 16.071) |
| SSD wait inside `at_layer` | 12.119 | inside the gaps |
| Arena lookup, pin, LRU, submit | 0.105 | inside the gaps |
| Read-ahead scan and submit (`ahead`) | 0.601 | inside the gaps |
| Debug `fprintf` before the release | 0.354 | inside the gaps |
| Routed-id loads from the shared buffer | 0.000 | inside the gaps, below 0.001 per layer |
| Sum | 35.560 | GPU timestamp 35.519, residual 0.041 |

The identity that confirms the clocks: spin 21.019 + gap 13.179 + tail 1.362 = 35.560, and `GPUEndTime − GPUStartTime` is 35.519. The host spin is the encoder interval, overlapping GPU execution. Adding it on top of the gap would double-count. Command-buffer encoding is 0.577 ms and finishes before commit, so it is outside the GPU timestamp and outside the gaps. Across the last 15 T=2 windows, encoding is median 0.614 ms (0.577–0.668).

Gap split for the categories asked:

| Category | This token | What the clock is |
| --- | ---: | --- |
| (a) waiting for GPU completion, or readback of the route | 21.019 ms spin, plus the id loads | The spin waits until encoder N signals. It overlaps that encoder. The id load is the readback: `ids_` is a shared buffer, and the printed sum is 0.000 ms. |
| (b) arena lookups and bookkeeping | 0.105 ms | Id scan is already in (a). This is the locked lookup, pin, LRU, victim, and submit. |
| (c) SSD reads | 12.119 ms, 32 misses, 44.24 MB | `done_cv_` inside `at_layer`. Each expert is 1,382,400 bytes (gate + up + down in the contiguous experts file). |
| (d) encoding | 0.577 ms | The CPU encode loop, before commit. |

`read_delta` over the 48 waits is 84.33 MB. That counter moves for every pread that finishes during a wait, including the 32 misses (44.24 MB), the 24 late experts (33.18 MB), and prefetch of later layers that happens to complete in the same wait. It is not an extra 84 MB on top of the misses.

Per layer. "Encoder" is the interval that ends at this layer's cut. "Gap" is `at_layer` for this layer plus the release store. Miss MB is `miss * 1.3824`. Read MB is `bytes_read_` during the wait.

| Layer | Encoder ms | Gap ms | ids | book | wait | after | hit | pre | late | miss | Miss MB | Read MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.010 | 0.026 | 0.000 | 0.001 | 0.000 | 0.012 | 20 | 0 | 0 | 0 | 0.00 | 0.00 |
| 1 | 0.421 | 0.408 | 0.000 | 0.004 | 0.385 | 0.010 | 17 | 0 | 0 | 1 | 1.38 | 2.76 |
| 2 | 0.402 | 0.696 | 0.000 | 0.003 | 0.671 | 0.015 | 18 | 0 | 0 | 2 | 2.76 | 2.76 |
| 3 | 0.414 | 0.831 | 0.000 | 0.005 | 0.802 | 0.015 | 13 | 0 | 0 | 3 | 4.15 | 4.15 |
| 4 | 0.468 | 0.675 | 0.000 | 0.003 | 0.650 | 0.015 | 15 | 1 | 2 | 2 | 2.76 | 6.91 |
| 5 | 0.410 | 0.393 | 0.000 | 0.004 | 0.364 | 0.014 | 18 | 1 | 0 | 1 | 1.38 | 1.38 |
| 6 | 0.408 | 0.403 | 0.000 | 0.005 | 0.383 | 0.010 | 17 | 0 | 0 | 1 | 1.38 | 1.38 |
| 7 | 0.399 | 0.439 | 0.000 | 0.003 | 0.413 | 0.015 | 19 | 0 | 0 | 1 | 1.38 | 1.38 |
| 8 | 0.411 | 0.016 | 0.000 | 0.000 | 0.000 | 0.010 | 16 | 1 | 0 | 0 | 0.00 | 0.00 |
| 9 | 0.388 | 0.517 | 0.000 | 0.003 | 0.492 | 0.015 | 18 | 0 | 0 | 1 | 1.38 | 1.38 |
| 10 | 0.398 | 0.439 | 0.000 | 0.004 | 0.412 | 0.014 | 15 | 1 | 0 | 1 | 1.38 | 1.38 |
| 11 | 0.424 | 0.032 | 0.000 | 0.000 | 0.000 | 0.023 | 15 | 0 | 0 | 0 | 0.00 | 0.00 |
| 12 | 0.436 | 0.929 | 0.000 | 0.005 | 0.907 | 0.010 | 14 | 0 | 0 | 4 | 5.53 | 5.53 |
| 13 | 0.400 | 0.458 | 0.000 | 0.003 | 0.431 | 0.015 | 16 | 1 | 0 | 1 | 1.38 | 1.38 |
| 14 | 0.404 | 0.018 | 0.000 | 0.000 | 0.000 | 0.010 | 19 | 0 | 0 | 0 | 0.00 | 0.00 |
| 15 | 0.432 | 0.017 | 0.000 | 0.000 | 0.000 | 0.010 | 15 | 0 | 0 | 0 | 0.00 | 0.00 |
| 16 | 0.444 | 0.016 | 0.000 | 0.000 | 0.000 | 0.009 | 16 | 0 | 0 | 0 | 0.00 | 0.00 |
| 17 | 0.405 | 0.016 | 0.000 | 0.000 | 0.000 | 0.012 | 16 | 0 | 0 | 0 | 0.00 | 0.00 |
| 18 | 0.418 | 0.631 | 0.000 | 0.004 | 0.604 | 0.014 | 16 | 1 | 1 | 2 | 2.76 | 4.15 |
| 19 | 0.421 | 0.149 | 0.000 | 0.004 | 0.129 | 0.010 | 15 | 0 | 1 | 0 | 0.00 | 2.76 |
| 20 | 0.393 | 0.022 | 0.000 | 0.000 | 0.000 | 0.012 | 18 | 0 | 0 | 0 | 0.00 | 0.00 |
| 21 | 0.399 | 0.459 | 0.000 | 0.003 | 0.434 | 0.015 | 17 | 0 | 3 | 0 | 0.00 | 4.15 |
| 22 | 0.406 | 0.022 | 0.000 | 0.000 | 0.000 | 0.016 | 20 | 0 | 0 | 0 | 0.00 | 0.00 |
| 23 | 0.422 | 0.018 | 0.000 | 0.000 | 0.000 | 0.010 | 19 | 0 | 0 | 0 | 0.00 | 0.00 |
| 24 | 0.387 | 0.439 | 0.000 | 0.003 | 0.421 | 0.010 | 15 | 0 | 0 | 1 | 1.38 | 1.38 |
| 25 | 0.392 | 0.018 | 0.000 | 0.000 | 0.000 | 0.012 | 19 | 0 | 0 | 0 | 0.00 | 0.00 |
| 26 | 0.437 | 0.631 | 0.000 | 0.004 | 0.601 | 0.015 | 15 | 1 | 1 | 2 | 2.76 | 5.53 |
| 27 | 0.439 | 0.421 | 0.000 | 0.005 | 0.394 | 0.014 | 17 | 0 | 0 | 1 | 1.38 | 1.38 |
| 28 | 0.408 | 0.715 | 0.000 | 0.005 | 0.687 | 0.015 | 15 | 0 | 0 | 3 | 4.15 | 4.15 |
| 29 | 0.409 | 0.017 | 0.000 | 0.000 | 0.000 | 0.010 | 18 | 1 | 0 | 0 | 0.00 | 0.00 |
| 30 | 0.503 | 0.016 | 0.000 | 0.000 | 0.000 | 0.010 | 19 | 0 | 0 | 0 | 0.00 | 0.00 |
| 31 | 0.465 | 0.016 | 0.000 | 0.000 | 0.000 | 0.010 | 18 | 0 | 0 | 0 | 0.00 | 0.00 |
| 32 | 0.467 | 0.023 | 0.000 | 0.000 | 0.000 | 0.019 | 17 | 0 | 0 | 0 | 0.00 | 0.00 |
| 33 | 0.405 | 0.053 | 0.000 | 0.006 | 0.021 | 0.016 | 18 | 0 | 1 | 0 | 0.00 | 1.38 |
| 34 | 0.398 | 0.442 | 0.000 | 0.004 | 0.417 | 0.014 | 17 | 0 | 2 | 1 | 1.38 | 4.15 |
| 35 | 0.439 | 0.408 | 0.000 | 0.004 | 0.379 | 0.014 | 15 | 0 | 2 | 1 | 1.38 | 4.15 |
| 36 | 0.487 | 0.062 | 0.000 | 0.005 | 0.035 | 0.015 | 16 | 1 | 1 | 0 | 0.00 | 1.38 |
| 37 | 0.495 | 0.557 | 0.000 | 0.002 | 0.532 | 0.015 | 16 | 0 | 4 | 0 | 0.00 | 6.91 |
| 38 | 0.487 | 0.573 | 0.000 | 0.003 | 0.549 | 0.015 | 15 | 1 | 3 | 1 | 1.38 | 5.53 |
| 39 | 0.444 | 0.126 | 0.000 | 0.003 | 0.106 | 0.010 | 17 | 0 | 1 | 0 | 0.00 | 1.38 |
| 40 | 0.397 | 0.015 | 0.000 | 0.000 | 0.000 | 0.009 | 17 | 0 | 0 | 0 | 0.00 | 0.00 |
| 41 | 0.407 | 0.018 | 0.000 | 0.000 | 0.000 | 0.014 | 19 | 0 | 0 | 0 | 0.00 | 0.00 |
| 42 | 0.427 | 0.016 | 0.000 | 0.000 | 0.000 | 0.010 | 20 | 0 | 0 | 0 | 0.00 | 0.00 |
| 43 | 0.450 | 0.016 | 0.000 | 0.000 | 0.000 | 0.010 | 16 | 1 | 0 | 0 | 0.00 | 0.00 |
| 44 | 0.419 | 0.018 | 0.000 | 0.000 | 0.000 | 0.014 | 20 | 0 | 0 | 0 | 0.00 | 0.00 |
| 45 | 0.488 | 0.253 | 0.000 | 0.004 | 0.229 | 0.010 | 16 | 0 | 2 | 0 | 0.00 | 2.76 |
| 46 | 0.401 | 0.016 | 0.000 | 0.000 | 0.000 | 0.009 | 18 | 0 | 0 | 0 | 0.00 | 0.00 |
| 47 | 0.435 | 0.680 | 0.000 | 0.003 | 0.671 | 0.000 | 15 | 0 | 0 | 2 | 2.76 | 2.76 |
| tail | 1.362 |  |  |  |  |  |  |  |  |  |  |  |

Twenty-one layers wait 0 ms. Their gaps sum to 0.392 ms. Seven layers have only in-flight read-ahead (late, no miss); those waits sum to 1.486 ms. Twenty layers have at least one miss; those waits sum to 10.633 ms (min 0.364, median 0.462, max 0.907). A miss and a late on the same layer share one condvar wait, so the 10.633 ms includes any late experts on those miss layers.

Hit rate. The engine's "hits" count is idle, non-prefetch residents divided by lookups. Lookups equal hit + pre + late + miss.

| Sample | Lookups | Hits | Definition used |
| --- | ---: | ---: | --- |
| This token | 877 | 0.9236 | 810/877 idle hits. Prefetch-resident 11, late 24, miss 32. |
| This token, resident and not blocking | 877 | 0.9361 | (810+11)/877. Late and miss still enter the condvar. |
| Last 15 T=2 windows | 12899 | 0.9250 | Same engine definition. Misses 26.60 per window. Resident and not blocking 0.9447. |
| All 40 decode windows | 33764 | 0.8830 | Matches the process's expert-stream line. Early windows are in this average. |
| Warm speed runs, 200 windows each | 174585 | 0.9224, 0.9502, 0.9121 | From `bench/UPSTREAM_MAC.md`, same flags, arena 9.44 / 11.92 / 8.73 GB. |

This token found 845 of 877 experts already resident or already in flight (0.9635). The blocking set is 32 misses plus 24 late.

## CPU-GPU synchronization per token

A steady T=2 streamed window is one command buffer, 49 compute encoders, 48 cuts (`stream.h`: after each layer's router the GPU signals and waits; `at_layer` runs on the host). Counted on this token:

| Sync | Count | What it waits for | This token |
| --- | ---: | --- | --- |
| `encodeSignalEvent` value `base+2*s+1` | 48 | GPU publishes "encoder s finished". Router ids and the predicted logits for the next layer are then visible in shared memory. | overlapped with the encoder; the host observes it as spin |
| Host spin on `signaledValue` | 48 | That signal. This is the wait for GPU completion of the segment that produced the route. | 21.019 ms sum, overlapping GPU execution |
| `encodeWaitForEvent` value `base+2*s+2` | 48 | Host has returned from `at_layer(s)` and stored the release. Next encoder's `moe_gu` may read `smap`. | the 13.179 ms of gaps |
| Host store of `signaledValue` | 48 | The release itself. | included in the gap column |
| `waitUntilCompleted` | 1 | The last encoder, layer 47's experts through argmax, so the host can read the token. | tail 1.362 ms |
| Metal blit / `getBytes` readback | 0 | None on this path. Buffers are `MTLResourceStorageModeShared`. | id loads print as 0.000 ms |
| `done_cv_` (host thread, I/O threads) | 27 | Missing or in-flight expert preads. CPU to CPU, included because it is the stall the GPU event is holding for. | 12.119 ms across the 27 layers that wait |

`run`, `run_graph`, `run_wall`, and `run_icb` also call `waitUntilCompleted`. The streamed decode window does not. `run_graph`'s wait runs only in kprof, after the timed window. No other event wait was added by the debug patch. The patch only read clocks around the waits that were already there.

## Why each sync exists

The layer dependency is real. Layer l+1's attention consumes layer l's mixture output, and layer l's `moe_gu` reads expert bytes through `smap`. An expert that is not in a slot when the router names it has to arrive before that GEMV. Later layers cannot cover that read, because they need this layer's output. On this token each miss-bearing layer waited a median 0.462 ms, and twenty such layers in a row are 10.633 ms. That part is inherent for any expert the host could not have named earlier.

The one-layer read-ahead is the mechanism that is supposed to name the expert earlier. `at_layer` calls `ahead(l+1)` from this layer's predicted router logits (`predL_`), top 10 (`STRATA_METAL_PREFETCH` default 10 on this machine; the log says "read-ahead top 10"). `STRATA_METAL_PREFETCH2` defaults to 0, so layer l+2 is not prefetched. `ahead` runs inside the stall, and the GPU work that follows it is one encoder of about 0.4 ms. A 1.38 MB pread does not reliably finish in that window. The seven late-only layers (1.486 ms) are experts the predictor had already named. Waiting on those is design: the pread was started one encoder too late, and the release is held while `ahead` scans.

Hit layers show the rest of the design. Twenty-one layers on this token needed no byte. `smap` already held their slot codes. The host still cuts the command buffer, spins, walks the id list, touches the LRU, and releases. Those gaps sum to 0.392 ms. The larger design cost of the same cut is the encoder boundary itself: 49 encoders took 22.381 ms of host-observed encoder time where one encoder runs the same dispatches in 16.071 ms. That 6.310 ms is paid on hit layers and miss layers alike.

The final `waitUntilCompleted` is inherent. The host samples a token id that the last encoder writes.

Encoding all 49 encoders before commit is design and small. It is 0.577 ms, outside the gap. The host could encode the next token's buffer while this one runs. That does not shrink the 13 ms gap.

Shared-memory loads of `ids_` and `predL_` are not a separate sync. The signal is what makes the GPU's stores visible. There is no readback command.

## Ranked fixes

Expected milliseconds are for a warm T=2 token like window 40 (GPU timestamp 35.519 ms, resident kernel floor 16.071 ms). The pieces were measured separately and add up to the timestamp, so the savings below do not overlap each other. A surprise expert that is unnamed until the router signal still costs about 0.46 ms on the critical path; nothing in this list removes that.

**1. Drop the per-layer event when `smap` is already published. Expected 6.3 ms per token.**

The same 1189 dispatches, experts already in the arena, complete in one compute encoder in 16.071 ms. The streamed window spends 22.381 ms of host-observed encoder time (21.019 of spin plus 1.362 of tail) to do that work as 49 encoders. `smap` is the indirection `moe_gu` already uses, and the I/O thread writes a slot's code when the pread finishes. Encode the router and the expert GEMVs in one encoder, and let the expert kernel proceed when the slot code is present. Keep the host LRU and prefetch off the GPU's wait. The 6.3 ms is the boundary tax. A slot that is still empty stalls inside the expert kernel for the same pread the condvar waits for today, so this change leaves the 12.1 ms SSD wait in place until the bytes are resident.

**2. Raise the arena budget toward the 11.92 GB warm run. Expected about 4 ms per token.**

This capture waited 12.119 ms with a 9.61 GB arena. The three earlier warm speed runs, same binary flags and the same Mac, were not repeated here: 11.92 GB waited 8.20 ms (T=2 GPU timestamp 31.37 ms, 19.5 misses per window), 9.44 GB waited 12.22 ms (35.79 ms, 30.2 misses), 8.73 GB waited 13.41 ms (37.03 ms, 34.0 misses). The wait difference shows up in the GPU timestamp. This run planned a 19.3 GB footprint against a 30.2 GB cap, so the allocator stopped with room left under the cap. Growing the arena to the working set the 11.92 GB run already held is the change those three runs support. They still missed about 20 experts per window, and the layer dependency keeps each of those on the critical path.

**3. Submit the predicted experts more than one encoder ahead. Expected 1.5 ms per token.**

On this token, seven layers waited only for read-ahead that was already queued, and those waits sum to 1.486 ms. `ahead(l+1)` is called during layer l's stall, and the next encoder is the only GPU work that overlaps the pread. `STRATA_METAL_PREFETCH2` is 0, so the l+2 logits this layer already computed are ignored. Queue layer l+1, and the l+2 prediction, at the previous cut, and store the event release before the top-k scan. The twenty miss layers are not in the 1.5 ms. Starting the same predictor earlier still fails to name an expert it does not predict. A better predictor, or re-arming the previous token's routed set before encoder 0, would eat into the 10.633 ms of miss-layer wait. That fraction was not measured.

**4. Store the release before `ahead()`. Expected 0.60 ms per token.**

`after` is 0.601 ms on this token: the predicted-logit top-k and the prefetch queue. The GPU sits in `encodeWaitForEvent` for that scan. The scan only enqueues work for the six I/O threads. Release the shared event, then scan. The locked lookup is 0.105 ms and can stay where it is.

**5. Encode the next window while this one is on the GPU. Expected 0.58 ms per token.**

Building the 49 encoders took 0.577 ms and ended before commit. It is outside the 35.519 ms timestamp and outside the inter-encoder gaps. Overlapping that encode with the previous token removes it from the wall clock. The gap between encoders would stay the same size.
