# Kernel bandwidth

The warm split leaves 25.100 ms per token of kernels and encoder boundaries together. This note re-measures that window on the current build, puts the one-encoder GPU time next to the 25.100 ms, and compares each kernel class with a measured read rate.

## Capture

The source is `84c9cd7`. `kprof` runs only when `!stream_`. For this capture that guard was taken out locally, the binary was rebuilt, the capture was taken, and the guard was put back. The line is not in the tree.

Arena 12 GB, `STRATA_METAL_STREAM=1`. `STRATA_METAL_PREFETCH2`, `STRATA_METAL_PREFETCH_EARLY`, and `STRATA_METAL_TIME_AHEAD` were unset. A 200-token warm-up was discarded with `STRATA_METAL_KPROF` unset. The capture is 40 tokens with `STRATA_METAL_KPROF=2`, the 20th cached T=2 head window.

The discarded fill is cold. 2.6 s at 4.50 GB/s, the process holds 18.2 GB (5.6 GB resident). Compressor pages were 69,937 before it. It exited 0. Pressure level was 1 on all 8 samples, peak compressor 786,028. Swap used went from 1,514,605,117 to 1,580,990,464 bytes, a rise of 66,385,347, under the 256 MiB kill. Decode 200 tokens in 6898.5 ms, 28.99 tok/s. The warm-up is discarded. Not a speed row.

The capture fill is warm. 1.8 s at 6.79 GB/s, the process holds 18.2 GB (15.5 GB resident). Compressor pages were 135,397 before it, under 150,000, so there was no wait. It exited 0. Pressure level was 1 on all 5 samples, peak compressor 143,822. Swap used stayed 1,580,990,464. Arena 12.00 GB, 8616 slots, read-ahead top 10.

`kprof` printed: window T2, 1189 dispatches, GPU 16.041 ms; classes alone (serialized) sum 15.755 ms. The 78 printed lines add to 15.754 ms. The header is the engine's own sum. Decode 40 tokens in 1913.0 ms, 20.91 tok/s. That run includes the kprof re-runs. Not a speed row.

1189 dispatches is the folded T=2 head graph. `ffn_gate_inp.weight` is x95, the 48 routers plus the 47 layer-l+1 predictors. The layer-l+2 predictor is absent. That is the default schedule, `STRATA_METAL_PREFETCH_EARLY` unset.

## The boundary tax

25.100 ms is the warm-split pool. GPU per token is 33.270 ms. `host_ms_` is 8.84 ms and ahead is 0.670 ms, so the pre-release stall is 8.170 ms. 33.270 − 8.170 = 25.100. That stall is the lookup, the SSD wait, and the publish. It is already inside the GPU timestamp, and it is subtracted here.

The one-encoder replay of this window is 16.041 ms. 25.100 − 16.041 = 9.059 ms. That difference is the measured boundary tax.

The replay is `run_graph`: one concurrent compute encoder, the window's own barriers, no host cut. Each class then runs alone, median of 7, with a barrier before every dispatch. The one-encoder figure is the median of 7 over the whole list. The isolated sum is 0.286 ms under that median, so isolating the classes did not inflate them.

## Read peak

A Metal streaming-read kernel over a shared buffer of 2,147,483,648 bytes (2 × 2^30). 262,144 threads, each reading 512 `uint4` values strided by the thread count. The sink is one `uint` per thread, about 1 MB. One untimed warmup touches every page. Then 20 GPU times, `GPUEndTime − GPUStartTime`.

The program printed `median 5.134 ms  418.32 GB/s  bytes 2147483648  device Apple M5 Max`. The two central printed samples are 5.117 ms and 5.150 ms. GB/s is decimal, bytes / seconds / 1e9, the same unit as the rest of this campaign. 70% of the printed peak is 292.82 GB/s.

Each thread's loop is a chain of dependent loads. A class whose working set is smaller than 2 GB can come out above 418.32 GB/s. The table reports that as more than 100% of this probe. The probe was not retuned, and 418.32 GB/s is this kernel's rate, not a published peak for the chip.

## Bytes

Bytes are the traffic of this T=2 window. One such window is one generated token. A weight is counted once per dispatch. The GEMV decodes a unit once and applies it to both tokens, and the other kernels are counted the same way. An activation element is counted once, not once per output row that reads it again. Where `X<>` loads the per-16 sums (Q2_0, Q3_K, Q4_K, Q5_K, Q6_K), those sums are included. `T_Q2_0R` (the `moe_down` type 142) does not load them. A prepared-activation write is the four planes and the sums of the tokens that are written. `gr_up`, the GDN step, and `swiglu_prep` use `MAXT` as the plane stride and write two rows.

The HC matrices are BF16 in the file. The q8 kernels read the load-time Q8_0 copy: a down matrix is 3,481,600 bytes, an inject matrix is 43,520, an up matrix is 3,481,600. For `output_hc` the inject buffer is the down tensor, and the kernel loads the first four down rows again. Those bytes are counted.

`gemv_12_16_1_4_1 output_hc_up.weight` x1 matches no tensor of that name on that tile. The file's `output_hc_up` is BF16 and is the `gr_up_q8` class. The Q4_K tensors on this tile are four `attn_qkv` and three `attn_gate` (blk.0, blk.45, blk.46). Each gate is shape (2560, 6144), 8,847,360 weight bytes. kprof named two of the gates `attn_gate.weight` and the third `output_hc_up.weight`. The table keeps that name. Its 8,918,272 bytes are that gate plus the activations of the one dispatch.

`ffn_gate_inp` weights are the 48 tensors once, plus layers 1..47 again for the l+1 predictor. The second read is traffic the predictor issues. If it hits the cache, the DRAM traffic is lower than the table, and the GB/s is then high for that reason.

Routed experts: the kernel reads each distinct expert once and writes one `h` or `ye` row per routed slot. The slots are T × top-k = 20 even when ids repeat. This window's distinct count was not logged. Weight bytes use 20 experts per layer, an upper bound. The warm-split mean is 18.19 distinct (174,585 lookups / 200 / 48). At that mean, `moe_gu` is 808,394,496 bytes and 464.86 GB/s, still above the probe. The table keeps the upper bound, so its 510.90 GB/s is an upper bound on that class.

`qsa_attn`'s position was not printed. The 20th cached T=2 head hit is derived as pos0 329. Prefill ends at 306. Then the T=1 window, the T=2 window with no fold, and one miss on each history parity. Hit 20 is the 24th decode window, 306 + 23. The widths are 330 and 331. Each token's KV read is counted, so the shared prefix is counted twice. The rope table is the two cells.

GDN state is 3,145,728 bytes per layer, read by the fold and by the step, written by the fold only. q and k channels are shared by three value heads and counted once. `ssm_norm` is 512 bytes, one vector of 128 floats, not one vector per head.

## Table

Sorted by milliseconds saved if the class moved its bytes at 70% of 418.32 GB/s.

`ms saved = isolated ms − bytes / (0.70 × 418.32e9) × 1000`

A negative value means the class is already past that rate. Achieved GB/s is bytes / isolated seconds / 1e9.

The positive gaps sum to 2.427 ms before each row is rounded to 0.001 ms. Adding the printed positive cells gives 2.428. Classes under 2 MB are 0.567 ms of the unrounded sum. For those, 70% of a 2 GB read sits under the launch cost, so the column is the gap to the probe rate. The classes at 2 MB and up are the other 1.860 ms. This is headroom in the isolated times. It is not a measured fused graph. It sits next to the 9.059 ms boundary tax.

## Consecutive classes

`Window::layer` puts these producer-consumer chains in order:

- `gr_norm`, then `gr_down`, then `gr_up`. `gr_norm` writes `xn`. `gr_down` reads `xn` and writes `lo`. `gr_up` reads `lo` and `xn`. `gr_norm` and `gr_up` are not neighbors. The three dispatches are one `gr_read`.
- On a full-attention layer, `qsa_attn` then `qsa_merge`.
- The router GEMV (`ffn_gate_inp`) is followed by the l+1 predictor GEMV, then the shared-expert GEMVs, then `moe_route`. The router and `moe_route` have those dispatches between them. The predictor sits beside the router.

`STRATA_METAL_ROUTE=1` fuses the router GEMV and `moe_route`. The comment in `window.cpp` records that measurement: T2 GPU 13.035 ms against 13.027 ms, the fused kernel 0.615 ms against 0.566 ms for the two, and 0.3–0.5% slower end to end on the M5 Max. That run is not this capture. The default is off.

`moe_gu`, then the shared-expert down GEMV, then `moe_down`. `moe_down` reads `h` from `moe_gu`. The shared-expert GEMV sits between the two expert kernels.

No fusion is proposed. The largest positive gaps are `gr_norm` at 0.388 ms, `qsa_attn` at 0.276 ms, and `moe_route` at 0.265 ms. `moe_gu` is 122.1% of the probe and the router is 104.7%. `output.weight` is 98.3% and `moe_down` is 94.9%. Both of those are already past 70% of the probe, which is why their saved column is negative.

Fix 5 stays deferred. This note changes no engine source. The `kprof` guard is `&& !stream_` again.

## Table of classes


| class | dispatches | isolated ms | bytes | GB/s | % of peak | ms saved |
|---|---:|---:|---:|---:|---:|---:|
| `gr_norm` | 98 | 0.490 | 29,903,872 | 61.03 | 14.6 | +0.388 |
| `qsa_attn` | 12 | 0.319 | 12,688,752 | 39.78 | 9.5 | +0.276 |
| `moe_route` | 48 | 0.267 | 457,728 | 1.71 | 0.4 | +0.265 |
| `gemv_30_32_1_4_1 ffn_gate_inp_shexp.weight` | 48 | 0.138 | 1,229,184 | 8.91 | 2.1 | +0.134 |
| `gemv_23_32_1_8_2 ssm_out.weight` | 17 | 0.606 | 143,233,024 | 236.36 | 56.5 | +0.117 |
| `moe_combine_sh` | 48 | 0.148 | 11,923,584 | 80.56 | 19.3 | +0.107 |
| `gemv_11_16_1_4_1 attn_gate.weight` | 25 | 0.678 | 170,732,800 | 251.82 | 60.2 | +0.095 |
| `gemv_30_32_1_4_1 ssm_alpha.weight` | 36 | 0.112 | 9,598,464 | 85.70 | 20.5 | +0.079 |
| `gemv_30_32_1_4_1 ssm_beta.weight` | 36 | 0.112 | 9,598,464 | 85.70 | 20.5 | +0.079 |
| `swiglu_prep` | 48 | 0.084 | 1,489,920 | 17.74 | 4.2 | +0.079 |
| `gemv_11_32_1_4_1 ffn_up_shexp.weight` | 21 | 0.124 | 15,348,480 | 123.78 | 29.6 | +0.072 |
| `gemv_42_32_1_4_1 ffn_gate_shexp.weight` | 21 | 0.103 | 10,241,280 | 99.43 | 23.8 | +0.068 |
| `gr_down_q8_8_8 hc_ffn_down.weight` | 48 | 0.638 | 173,262,336 | 271.57 | 64.9 | +0.046 |
| `gemv_11_32_1_4_1 ffn_gate_shexp.weight` | 13 | 0.078 | 9,501,440 | 121.81 | 29.1 | +0.046 |
| `gr_down_q8_8_8 hc_attn_down.weight` | 48 | 0.637 | 173,262,336 | 272.00 | 65.0 | +0.045 |
| `gemv_42_32_1_4_1 ffn_up_shexp.weight` | 13 | 0.064 | 6,339,840 | 99.06 | 23.7 | +0.042 |
| `gemv_11_16_1_4_1 attn_qkv.weight` | 18 | 0.735 | 204,618,240 | 278.39 | 66.6 | +0.036 |
| `qsa_merge` | 12 | 0.042 | 3,027,456 | 72.08 | 17.2 | +0.032 |
| `gemv_42_32_1_4_1 attn_gate.weight` | 4 | 0.092 | 17,978,368 | 195.42 | 46.7 | +0.031 |
| `gemv_42_8_1_4_1 ffn_down_shexp.weight` | 16 | 0.056 | 7,787,520 | 139.06 | 33.2 | +0.029 |
| `gemv_42_32_1_4_1 attn_q.weight` | 2 | 0.087 | 17,934,848 | 206.15 | 49.3 | +0.026 |
| `gdn_step` | 72 | 1.323 | 380,063,232 | 287.27 | 68.7 | +0.025 |
| `qsa_prep` | 12 | 0.028 | 1,469,184 | 52.47 | 12.5 | +0.023 |
| `gemv_11_32_1_8_2 ssm_out.weight` | 3 | 0.092 | 20,493,312 | 222.75 | 53.2 | +0.022 |
| `gemv_2_8_1_4_1 ffn_down_shexp.weight` | 16 | 0.072 | 15,155,200 | 210.49 | 50.3 | +0.020 |
| `gemv_23_32_1_4_1 ffn_gate_shexp.weight` | 6 | 0.038 | 5,376,000 | 141.47 | 33.8 | +0.020 |
| `gemv_23_32_1_4_1 ffn_up_shexp.weight` | 5 | 0.032 | 4,480,000 | 140.00 | 33.5 | +0.017 |
| `gemv_12_32_1_4_1 ffn_gate_shexp.weight` | 6 | 0.035 | 5,690,880 | 162.60 | 38.9 | +0.016 |
| `gemv_12_32_1_4_1 ffn_up_shexp.weight` | 6 | 0.035 | 5,690,880 | 162.60 | 38.9 | +0.016 |
| `gemv_30_32_1_4_1 indexer.k_proj.weight` | 12 | 0.043 | 8,122,368 | 188.89 | 45.2 | +0.015 |
| `gemv_23_32_1_4_1 attn_k.weight` | 4 | 0.024 | 2,883,584 | 120.15 | 28.7 | +0.014 |
| `gemv_20_8_1_4_1 ffn_down_shexp.weight` | 7 | 0.035 | 6,630,400 | 189.44 | 45.3 | +0.012 |
| `gemv_42_32_1_4_1 attn_qkv.weight` | 1 | 0.037 | 7,476,480 | 202.07 | 48.3 | +0.011 |
| `gemv_42_32_1_4_1 ple_key.weight` | 1 | 0.037 | 7,476,480 | 202.07 | 48.3 | +0.011 |
| `gemv_11_32_1_4_1 attn_k.weight` | 3 | 0.017 | 1,767,168 | 103.95 | 24.8 | +0.011 |
| `gemv_23_32_1_4_1 attn_v.weight` | 3 | 0.018 | 2,162,688 | 120.15 | 28.7 | +0.011 |
| `ple_gnorm` | 2 | 0.011 | 409,600 | 37.24 | 8.9 | +0.010 |
| `gemv_11_16_1_4_1 attn_q.weight` | 6 | 0.289 | 81,821,184 | 283.12 | 67.7 | +0.010 |
| `gemv_6_8_1_4_1 ffn_down_shexp.weight` | 7 | 0.036 | 8,064,000 | 224.00 | 53.5 | +0.008 |
| `gemv_13_32_1_4_1 attn_v.weight` | 3 | 0.017 | 2,780,928 | 163.58 | 39.1 | +0.008 |
| `ple_gate_norm` | 1 | 0.008 | 389,120 | 48.64 | 11.6 | +0.007 |
| `gemv_12_32_1_4_1 attn_k.weight` | 2 | 0.011 | 1,526,272 | 138.75 | 33.2 | +0.006 |
| `gemv_12_32_1_4_1 attn_v.weight` | 2 | 0.011 | 1,526,272 | 138.75 | 33.2 | +0.006 |
| `gemv_14_32_1_4_1 attn_v.weight` | 3 | 0.017 | 3,303,168 | 194.30 | 46.4 | +0.006 |
| `gemv_13_32_1_4_1 attn_k.weight` | 2 | 0.012 | 1,853,952 | 154.50 | 36.9 | +0.006 |
| `gemv_13_32_1_4_1 ffn_up_shexp.weight` | 2 | 0.013 | 2,306,560 | 177.43 | 42.4 | +0.005 |
| `gemv_11_32_1_4_1 attn_v.weight` | 1 | 0.006 | 589,056 | 98.18 | 23.5 | +0.004 |
| `xprep` | 1 | 0.005 | 414,720 | 82.94 | 19.8 | +0.004 |
| `gemv_14_32_1_4_1 attn_k.weight` | 1 | 0.007 | 1,101,056 | 157.29 | 37.6 | +0.003 |
| `gemv_13_32_1_4_1 ffn_gate_shexp.weight` | 1 | 0.007 | 1,153,280 | 164.75 | 39.4 | +0.003 |
| `gemv_14_32_1_4_1 ffn_gate_shexp.weight` | 1 | 0.007 | 1,370,880 | 195.84 | 46.8 | +0.002 |
| `gemv_14_32_1_4_1 ffn_up_shexp.weight` | 1 | 0.007 | 1,370,880 | 195.84 | 46.8 | +0.002 |
| `argmax_final` | 1 | 0.002 | 1,928 | 0.96 | 0.2 | +0.002 |
| `ple_conv_add` | 1 | 0.003 | 737,280 | 245.76 | 58.7 | +0.000 |
| `gemv_12_32_1_8_2 ssm_out.weight` | 11 | 0.335 | 98,120,704 | 292.90 | 70.0 | -0.000 |
| `gr_down_q8_8_8 output_hc_down.weight` | 1 | 0.012 | 3,609,632 | 300.80 | 71.9 | -0.000 |
| `argmax_part` | 1 | 0.006 | 1,988,480 | 331.41 | 79.2 | -0.001 |
| `gemv_8_8_1_4_1 ffn_down_shexp.weight` | 2 | 0.011 | 3,532,800 | 321.16 | 76.8 | -0.001 |
| `gr_up_q8 output_hc_up.weight` | 1 | 0.010 | 3,649,280 | 364.93 | 87.2 | -0.002 |
| `gemv_12_32_1_8_2 attn_output.weight` | 4 | 0.117 | 35,680,256 | 304.96 | 72.9 | -0.005 |
| `gemv_23_16_1_4_1 attn_gate.weight` | 4 | 0.110 | 33,701,888 | 306.38 | 73.2 | -0.005 |
| `gemv_12_16_1_4_1 output_hc_up.weight` | 1 | 0.024 | 8,918,272 | 371.59 | 88.8 | -0.006 |
| `gr_up_q8 hc_attn_up.weight` | 48 | 0.588 | 175,165,440 | 297.90 | 71.2 | -0.010 |
| `gr_up_q8 hc_ffn_up.weight` | 48 | 0.587 | 175,165,440 | 298.41 | 71.3 | -0.011 |
| `gemv_12_16_1_4_1 attn_gate.weight` | 2 | 0.046 | 17,836,544 | 387.75 | 92.7 | -0.015 |
| `gemv_13_32_1_8_2 ssm_out.weight` | 3 | 0.092 | 32,658,432 | 354.98 | 84.9 | -0.020 |
| `gemv_13_32_1_8_2 attn_output.weight` | 4 | 0.126 | 43,544,576 | 345.59 | 82.6 | -0.023 |
| `gemv_14_32_1_8_2 ssm_out.weight` | 2 | 0.065 | 25,950,208 | 399.23 | 95.4 | -0.024 |
| `gemv_30_32_1_4_1 ple_value.weight` | 1 | 0.020 | 13,148,160 | 657.41 | 157.2 | -0.025 |
| `gemv_23_16_1_4_1 attn_q.weight` | 4 | 0.203 | 67,321,856 | 331.63 | 79.3 | -0.027 |
| `gemv_12_16_1_4_1 attn_qkv.weight` | 4 | 0.168 | 59,397,120 | 353.55 | 84.5 | -0.035 |
| `gemv_14_32_1_8_2 attn_output.weight` | 4 | 0.137 | 51,900,416 | 378.84 | 90.6 | -0.040 |
| `gemv_30_32_1_4_1 indexer.q_proj.weight` | 12 | 0.067 | 31,752,192 | 473.91 | 113.3 | -0.041 |
| `gemv_23_16_1_4_1 attn_qkv.weight` | 13 | 0.558 | 182,374,400 | 326.84 | 78.1 | -0.065 |
| `gemv_30_32_1_4_1 ffn_gate_inp.weight` | 95 | 0.574 | 251,371,520 | 437.93 | 104.7 | -0.284 |
| `moe_down_142_8_1_4` | 48 | 1.146 | 454,881,024 | 396.93 | 94.9 | -0.407 |
| `gemv_13_16_1_4_1 output.weight` | 1 | 1.068 | 439,051,520 | 411.10 | 98.3 | -0.431 |
| `moe_gu_42_32_1_4` | 48 | 1.739 | 888,463,104 | 510.90 | 122.1 | -1.295 |
