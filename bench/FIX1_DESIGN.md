# Fix 1 designs

Encoder boundaries measured free (0.1 µs/cut); the 9.06 ms tax is 48 host round trips at ~139 µs, of which ~38 µs is host work.

Design only. No engine change in this note. The measurements section replays one decode window from a private binary. The engine tree was not modified. The host-side fixes are in. Fix 4 stays. Fix 3 (`54168c5`) is correct and neutral, and `84c9cd7` leaves it behind `STRATA_METAL_PREFETCH_EARLY`, default off. Fix 5 stays deferred.

## The pool

The warm split leaves 25.100 ms per token of kernels and encoder boundaries together. GPU per token is 33.270 ms. The pre-release stall is `host_ms_` 8.84 ms minus ahead 0.670 ms, which is 8.170 ms, and 33.270 − 8.170 = 25.100. The one-encoder replay of that window is 16.041 ms, in the kernel bandwidth note (`docs/KERNEL_BANDWIDTH.md` on this branch, `bench/KERNEL_BANDWIDTH.md` in StrataMetal). 25.100 − 16.041 = 9.059 ms. That is the measured boundary tax.

The tax is separate from the SSD wait. On the warm split, `wait_ms_` is 7.94 ms per token and sits inside the 8.170 ms stall, which is already removed to reach 25.100. Shortening the SSD wait does not spend the 9.059 ms.

The 19.3 misses per token are the warm-split control figure, 3,861 misses over 200 windows. They still have to be loaded. They are not a count of layers. Several misses can share a layer, and the warm split did not break them out by layer.

The cut today is `run_graph_seg`. One command buffer. One concurrent compute encoder per segment. Between segments the buffer encodes a shared-event signal and then a wait. The host spins until the signal, `at_layer` publishes `smap`, and fix 4 calls `release_seg` before `ahead()`. The GPU is blocked on the event from the signal until that store. There are 48 cuts, one immediately before each layer's `moe_gu`.

## The constraint

No GPU wait inside an encoder.

`smap` and the expert bytes are shared `MTLBuffer`s. Apple's page for `MTLStorageMode.shared` says: "When either the CPU or GPU changes the contents of the resource, you're responsible for synchronizing access to the texture from the other participant. Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory complete before accessing that resource on the other processor." The same page says this storage mode is the default for `MTLBuffer` on integrated GPUs and on Apple silicon. A kernel that reads `smap` while the host is still storing it is the case that sentence leaves with the caller.

A shared event does not close the race on its own. On Apple Developer Forums thread 701231, in February 2022, an Apple engineer wrote: "The shared event signal handler does not block kernelB from running during the execution of the CPU side handler. It only synchronizes the two batches of GPU work." If both kernels are already in committed command buffers, "kernelB will also be immediately allowed to proceed. The CPU write will be "racily" visible to kernelB." The way they give to make the CPU write visible is to set `signaledValue` from the CPU handler after the write, or to schedule the next kernel from that handler. `run_graph_seg` does the first of those. The next encoder waits on a value the host stores after `smap` is published. A spin in the encoder that already contains the expert kernel is the racy pattern.

A long spin is also a watchdog problem. WWDC 2020 session 10616, "Debug GPU-side errors in Metal," says in the session transcript: "You may have a timeout, which can be caused by long-running or infinite loops." `MTLCommandBufferError.timeout` means the system interrupted and terminated the command buffer before it finished running. The listed causes are that the commands took more time than the system allows, and that the command buffer timed out waiting for another workload to signal an `MTLEvent`. `MTLCommandBufferError.accessRevoked` follows when the device is responsible for too many timeouts or hangs. These pages do not publish a duration. This note does not invent one, and it does not come from a hang on this machine.

The event cause matters for a fallback that parks the GPU across a large load. Today's miss wait is 7.94 ms per token and completes. A design that holds one event for much longer is the wait the timeout page names. The designs below do the load on the host and start the next encoder after the publish, which is the current pattern. They do not spin in the kernel.

## Designs

A design stays bit-identical when a layer commits only if its expert bytes are the ones the route selected. GDN state, the conv history, and the KV append commit only for those layers. A flagged miss re-runs from the first flagged layer after the load. The finished logits are the same bytes as today.

### a. One encoder per token

The token is one encoder. There is no host cut. The expert kernel reads `smap`. A missing expert is recorded as a (layer, expert) pair, and that layer writes no outputs. The host loads the recorded experts, publishes `smap`, and re-runs from the first flagged layer. Layers before that layer keep their result.

On a miss the token pays the kernels up to the flag, the load, and a resubmit that starts at the flagged layer. Layers before the flag keep their results, so the resubmit is the rest of the token. A later miss resubmits again from that layer. An earlier draft of this note priced the resubmit as another full 16.041 ms of kernels. The kernels before the flag had already run, so that full second pass counted them twice. The measurements section prices the suffix.

The 19.3 misses per token were not resident at the cut, and that cut already had mid-token loads. This design does not load during the token, so those 19.3 are missing when the encoder starts. A token with every routed expert already resident is the case that keeps the boundary tax. The warm split's miss count says that case is not the common one. How many layers the 19.3 misses occupy was not recorded. The measurements section carries the bound.

The main risk is a layer after the flag committing GDN state or KV from a slot `smap` did not name. The encoder is concurrent. A flag written by `moe_gu` is visible to a later dispatch only if a buffer barrier is encoded between them. The re-run starts at the first flag, and the first pass commits nothing past it.

### b. Fewer cuts

Cut only where a miss is likely. The two schedules are the layers that missed on the previous token, and a fixed cut every k layers. Elsewhere, rely on read-ahead. If a layer that was not cut routes an expert that is not resident, that is the fallback: record it, commit nothing from that layer, load, and re-run from there.

A layer that was cut is today's path. A layer that was not cut, and whose experts are resident, uses the bytes already published. A cut on a layer that then hits still pays the handshake.

The worst case is a miss on every layer the schedule skipped. Each one re-runs the tail of its chunk, and the cuts that were kept are still paid. An every-k schedule pays 48/k handshakes on layers that would have hit. The previous token's miss set was not compared with the next token's route. This note does not treat it as a predictor of the 19.3.

If the 9.059 ms were spread evenly across the 48 cuts, an every-4 schedule would leave 12 cuts and 2.265 ms of tax, a gap of 6.794 ms, before any fallback re-run. The per-cut cost was not measured when this paragraph was written, so the even split was an assumption. The measurements section is that split. A fallback spends kernel time that the tax figure does not include. At 19.3 misses per token, a skipped layer will miss, and the fallback is part of the ordinary token.

The main risk is a stale `smap` read on a layer the schedule did not cut. A read-ahead that has been issued and not published is not resident. Using it without the event the host signals after the store is the race from thread 701231.

### c. Route one layer ahead

`predL_` is written by the layer-l+1 predictor in the encoder that ends at the cut before layer l's experts. `ahead(l+1)` already starts at that cut, after `release_seg`, and the load overlaps the next encoder. This design keeps that load and changes which cuts remain.

When the router encoder completes, the actual ids are in memory and the host reads them. It cuts only when that set contains an expert that is neither resident nor in flight. Resident means `smap` already names a published slot. In flight means the load was issued from `predL_` and the encoder that reads the expert waits on the event the host sets after the store, the same `signaledValue` rule as today. If the event is already signaled when the GPU reaches it, the wait adds no bubble. An in-flight expert does not skip that event. Skipping the event is how the kernel sees a stale map.

An expert that is neither resident nor in flight is a miss. The host loads it, publishes `smap`, signals, and the expert kernel runs after that signal. That is one cut, for that layer. A layer whose actual set is already published is encoded with no host round trip between the router and the experts.

The expert kernel still reads `smap`. If the slot is missing it records the (layer, expert) pair and writes no outputs for that layer. The host loads and re-runs from the first flagged layer. A prediction that names the wrong expert takes this path. The re-run uses the loaded bytes, and the first pass does not commit the bad layer, so the logits match.

The worst case is a miss that is not in flight on every layer. That is 48 cuts, today's schedule, plus the check. The 19.3 misses per token are an upper bound on the layers that still cut, if each miss falls on its own layer. They are not a measured layer count.

If the 9.059 ms is spread evenly, 19.3/48 of it is 3.642 ms still paid, and 28.7/48 of it is 5.417 ms of handshake on layers with no miss. Misses that share a layer leave less than 3.642 ms. The per-cut split was not measured when this paragraph was written, so 5.417 ms is a bound under that assumption. The measurements section replaces it. The 5.417 ms figure stays as that assumption's arithmetic. It is not a speed claim. The 7.94 ms SSD wait on a true miss stays, and it is outside the 9.059 ms.

The main risk is treating an in-flight expert as readable before the host has stored `smap` and signaled the event. The kernel then uses a stale map, or a slot that still holds the previous expert. The concurrent encoder does not order that store with a kernel committed before the store.

## The test

Each design gets the same two checks. Describing them is not an engine change.

The LOGPOS byte check, the one already used for fix 4 and fix 3. The logits file matches the reference byte for byte. A stale expert or a stale `smap` entry changes those bytes. The score is `score_logpos.py` with the existing rules, PICK_GAP 0.5, PICK_RANK 3, NLL_BAND 0.01. One matching greedy string does not replace the byte file.

A forced-miss test. After the route is known, evict one expert the route selected: mark its `smap` entry missing and drop the slot. The design loads it and re-runs from that layer, and the LOGPOS file matches the run that did not evict it. Do this on layer 0 and on a late layer. For (b), evict an expert on a layer the schedule did not cut. For (c), also hold back the publish of one expert that `predL_` had in flight until after the expert kernel would have run, and require the flag and the re-run. A pass that matches because the kernel read the old slot is a failure. The byte file is what shows it.

## Measurements

Private replay. Arena 12 GB, `STRATA_METAL_STREAM=1`, and the other stream env vars unset. A 200-token discard warmed the file cache (23.77 tok/s on that discard; not a speed row). The timed window is the 20th cached T=2 head hit of the next process, the window `kprof` replays. The live decode had just placed that window's experts. 1189 dispatches, 48 cuts. Seven reps. The median is sorted index 3, the same rule as `kprof`.

### Per cut

| run | what | median GPU ms |
| --- | --- | --- |
| a | 1 encoder, no events | 15.988667 |
| b | 49 encoders, every wait pre-signaled, no host round trip | 15.993583 |
| c | 48 host round trips | 22.659333 |

(b − a) / 48 = 0.000102 ms, 0.102 µs. That is the encoder-boundary cost per cut from the two medians. (c − b) / 48 = 0.138870 ms, 138.87 µs. That is the host-round-trip cost per cut on the GPU clock.

(a)'s seven GPU times, ms: 16.326, 16.195, 16.008, 15.987, 15.937, 15.989, 15.961. The spread is 0.389 ms. (b − a) is 0.005 ms, inside that spread. (b)'s seven: 15.993, 15.983, 16.030, 16.005, 15.957, 15.994, 16.087.

(c), GPU ms and the host clock around `at_cut`:

| rep | GPU ms | host ms |
| --- | --- | --- |
| 0 | 48.138 | 27.855 |
| 1 | 23.229 | 2.334 |
| 2 | 22.487 | 1.348 |
| 3 | 22.993 | 1.799 |
| 4 | 22.387 | 1.301 |
| 5 | 22.374 | 1.333 |
| 6 | 22.659 | 1.825 |

Rep 0 is a miss replay. The host spent 27.855 ms in `at_cut`. Reps 1–6 spent 1.301–2.334 ms across all 48 cuts, against the warm-split `wait_ms_` of 7.94 ms per token. The median of the seven GPU times is 22.659, rep 6, inside that resident set.

On those six reps the host clock is 1.3–2.3 ms and the GPU timestamp is 6.666 ms above (b). The GPU-visible round trip is larger than the host's own `at_cut` time. The 0.392 ms hit-layer host gaps in the earlier trace are a host-clock sum. They do not price this bubble. (b) puts the 49 encoder boundaries on the GPU with the waits already signaled, and that adds 0.005 ms. The cost (c) keeps is the round trip.

The published one-encoder replay remains 16.041 ms. This replay's (a) is 15.988667 ms. The pool section's 9.059 ms stays 25.100 − 16.041. It is not (c − a) here. This replay's (c − a) is 6.671 ms, of which (b − a) is 0.005 ms.

### Early-out

One encoder, 1,000 dispatches. The kernel's first instruction reads a device flag that is 1 and returns. No buffer barrier between the dispatches. Three uncounted warmups, then seven reps, median at index 3. The figure is GPU time / 1000, in µs.

`gr_norm`, all 98 dispatches, grid (4, 2, 1), threadgroup 256: 0.133 µs.

`moe_gu` is an indirect dispatch in the graph. This launch is a direct dispatch of the same grid, threadgroup 128, `gx_gu` 160. The plan buffer after the live window holds the last layer: `og` = 12. Grid (160, 12, 1) is 1.313 µs. Grid (160, 18, 1), with 18 the published warm-split mean distinct count and not this window's `og`, is 2.232 µs.

`gemv`, 542 dispatches, 38 grids. The count-weighted mean is 0.426 µs (unrounded 0.425789). One grid, (31040, 1, 1), is a single vocabulary dispatch at 29.728 µs. The other 37 grids run from 0.125 µs to 2.059 µs. The count-18 grid (1280, 1, 1) is the noisy one: its seven reps were 2.790, 2.791, 2.263, 1.261, 1.202, 1.201, 1.268 µs, and the median 1.268 µs is in the lower cluster. The same grid shape from another pipeline, count 4, measured 0.896 µs on its own median. The weight uses each row's count.

| class | count | grid | threadgroup x | µs |
| --- | --- | --- | --- | --- |
| gemv | 18 | 1280 1 1 | 128 | 1.268 |
| gemv | 6 | 1536 1 1 | 128 | 1.060 |
| gemv | 25 | 768 1 1 | 128 | 0.619 |
| gemv | 4 | 128 1 1 | 128 | 0.233 |
| gemv | 34 | 160 1 1 | 128 | 0.268 |
| gemv | 3 | 640 1 1 | 256 | 0.725 |
| gemv | 4 | 1280 1 1 | 128 | 0.896 |
| gemv | 3 | 768 1 1 | 128 | 0.619 |
| gemv | 4 | 128 1 1 | 128 | 0.231 |
| gemv | 12 | 160 1 1 | 128 | 0.267 |
| gemv | 15 | 640 1 1 | 256 | 0.726 |
| gemv | 1 | 31040 1 1 | 128 | 29.728 |
| gemv | 5 | 128 1 1 | 128 | 0.235 |
| gemv | 3 | 160 1 1 | 128 | 0.268 |
| gemv | 7 | 640 1 1 | 256 | 0.726 |
| gemv | 4 | 128 1 1 | 128 | 0.238 |
| gemv | 2 | 160 1 1 | 128 | 0.268 |
| gemv | 6 | 640 1 1 | 256 | 0.726 |
| gemv | 7 | 160 1 1 | 128 | 0.267 |
| gemv | 13 | 1280 1 1 | 128 | 0.896 |
| gemv | 4 | 1536 1 1 | 128 | 1.059 |
| gemv | 4 | 768 1 1 | 128 | 0.619 |
| gemv | 7 | 128 1 1 | 128 | 0.238 |
| gemv | 11 | 160 1 1 | 128 | 0.267 |
| gemv | 17 | 640 1 1 | 256 | 0.726 |
| gemv | 16 | 160 1 1 | 128 | 0.264 |
| gemv | 48 | 1 1 1 | 128 | 0.127 |
| gemv | 72 | 12 1 1 | 128 | 0.125 |
| gemv | 107 | 128 1 1 | 128 | 0.231 |
| gemv | 12 | 32 1 1 | 128 | 0.133 |
| gemv | 1 | 640 1 1 | 128 | 0.585 |
| gemv | 4 | 1536 1 1 | 128 | 1.060 |
| gemv | 34 | 160 1 1 | 128 | 0.264 |
| gemv | 2 | 2560 1 1 | 128 | 1.723 |
| gemv | 2 | 3072 1 1 | 128 | 2.059 |
| gemv | 16 | 160 1 1 | 128 | 0.267 |
| gemv | 7 | 160 1 1 | 128 | 0.268 |
| gemv | 2 | 160 1 1 | 128 | 0.268 |
| gr_norm | 98 | 4 2 1 | 256 | 0.133 |
| moe_gu | 48 | 160 12 1 | 128 | 1.313 |
| moe_gu, mean 18 | 48 | 160 18 1 | 128 | 2.232 |

501 dispatches are in none of these three classes. Their early-out cost was not measured. The estimate below prices them at 0.133 µs, the `gr_norm` launch, and at 2.232 µs, the dearer `moe_gu` launch.

### Resubmit

An empty command buffer: one encoder, no dispatches, commit, wait for completion. Twenty uncounted warmups, then 200. The median is sorted index 100: 122.042 µs. Min 77.792, p10 102.625, p90 162.792, max 648.875.

### Distinct miss layers

The warm split recorded 3,861 misses over 200 windows, 19.3 per token. It did not record how many layers those misses occupied, and the stream counters do not keep a per-layer total that would re-derive it. A T=2 window names at most 20 experts on one layer, so the mean number of miss layers M is in [1, 19.3]. Late-only waits were 2,265 / 200 = 11.325 per token. Those waits are hits that arrived late. They are not miss layers. The 40-window gap capture is a different run, 38.55 misses per window, and its layer counts are not this 19.3.

### Per-token estimate

Productive kernel time is (a), 15.988667 ms, once. The passes split that work. Early-out is the aborted suffix. Each command buffer adds one resubmit of 0.122042 ms. The warm-split SSD wait, 7.94 ms, is added to today and to every design, because the same 19.3 misses are still loaded. It cancels in the delta. A resubmit loads after the flag, so that load does not overlap the kernels that need the bytes. The table does not give a design a shorter load than 7.94 ms, and it does not charge a longer one.

A miss on layer L aborts from the segment whose first dispatch is that layer's `moe_gu` through the end of what was encoded. That segment also holds the next layer's pre-expert dispatches, so the counted suffix runs a few dispatches into the next layer. There are 49 segments. Segment 0 is the work before the first `moe_gu`.

resume-(a) encodes the whole token in one buffer, with a stop flag on every dispatch, and resubmits at each miss layer. Buffers = M + 1. The aborted suffix is the rest of the token.

Chunked resume encodes k layers at a time, k = 4, 6, 8, 12, which is 12, 8, 6, and 4 chunks. A stop flag sits inside the chunk. The host resubmits at a stop or at a chunk end. Buffers = (number of chunks) + M. The aborted suffix is the rest of the chunk.

today, resident = 22.659333 ms. today, with the warm-split SSD wait = 22.659333 + 7.94 = 30.599 ms. The warm-split GPU timestamp was 33.270 ms on a different window. 30.599 is this replay plus that wait. It is not 33.270.

The delta is resident today minus (15.988667 + early-out + buffers × 0.122042), in ms. Positive means the design's scheduling is ahead of the resident replay. "Late" puts the M misses on the last layers. "Early" puts them on the first layers. For a chunk, "early" puts each miss at the start of a chunk and stacks the extras on the largest chunk. The best chunk case aborts nothing, so it does not depend on the unmeasured dispatches.

| design | M | where | other µs | delta ms |
| --- | --- | --- | --- | --- |
| resume-(a) | 1 | late | 0.133 | +6.423 |
| resume-(a) | 1 | early | 2.232 | +5.105 |
| resume-(a) | 19.3 | late | 0.133 | +2.787 |
| resume-(a) | 19.3 | late | 2.232 | −1.010 |
| resume-(a) | 19.3 | early | 0.133 | −1.440 |
| resume-(a) | 19.3 | early | 2.232 | −16.288 |
| chunk k=4 | 1 | late in the chunk | either | +5.084 |
| chunk k=4 | 19.3 | late in the chunk | either | +2.851 |
| chunk k=4 | 19.3 | early in the chunk | 0.133 | +2.261 |
| chunk k=4 | 19.3 | early in the chunk | 2.232 | +0.658 |
| chunk k=6 | 19.3 | late in the chunk | either | +3.339 |
| chunk k=6 | 19.3 | early in the chunk | 0.133 | +2.455 |
| chunk k=6 | 19.3 | early in the chunk | 2.232 | +0.073 |
| chunk k=8 | 19.3 | late in the chunk | either | +3.583 |
| chunk k=8 | 19.3 | early in the chunk | 0.133 | +2.400 |
| chunk k=8 | 19.3 | early in the chunk | 2.232 | −0.793 |
| chunk k=12 | 19.3 | late in the chunk | either | +3.827 |
| chunk k=12 | 19.3 | early in the chunk | 0.133 | +2.053 |
| chunk k=12 | 19.3 | early in the chunk | 2.232 | −2.696 |

The early-out kernel has no memory barrier. A barrier between the flag's writer and the later readers was not measured. One would add cost on the abort path, which moves the deltas down.

## Recommendation

None beats today by more than 2 ms.

M was not recorded, and 501 dispatches have no early-out number. On that bound, resume-(a) is ahead by about 6 ms at M = 1 and behind at M = 19.3 when the misses sit on the early layers. Chunked k = 4 moves the least. Its worst point at M = 19.3 is +2.261 ms when the unmeasured dispatches cost a `gr_norm` launch, and +0.658 ms when they cost the dearer `moe_gu` launch. The second of those is the bound that has to clear 2 ms, and it does not.

Design (c) stays a description above. The resident replay spends 138.87 µs of GPU time on each of the 48 waits. Pre-signaling the waits, which is (b), removes that time because the GPU does not wait for the host. A cut that still has to hear the host's answer keeps the wait. The encode-time choice cannot drop a wait it has already encoded, and it cannot pre-signal a wait it still needs.

The bandwidth table's positive gaps still sum to 2.427 ms. That pool is smaller than this replay's 6.666 ms round-trip bubble, and it does not pick a schedule.

This note stops here for review. No design is implemented.
