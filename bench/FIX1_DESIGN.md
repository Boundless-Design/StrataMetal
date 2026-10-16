# Fix 1 designs

Design only. No engine change in this note. The host-side fixes are in. Fix 4 stays. Fix 3 (`54168c5`) is correct and neutral, and `84c9cd7` leaves it behind `STRATA_METAL_PREFETCH_EARLY`, default off. Fix 5 stays deferred.

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

On a miss the token pays the first pass up to the flag, the load, and a second pass from that layer. The worst case is a miss on layer 0. The second pass is the whole token, another 16.041 ms of kernels, against a tax of 9.059 ms.

The 19.3 misses per token were not resident at the cut, and that cut already had mid-token loads. This design does not load during the token, so those 19.3 are missing when the encoder starts. A token with every routed expert already resident is the case that keeps the 9.059 ms. The warm split's miss count says that case is not the common one. Where the first flagged layer falls was not measured, so the expected saving is not a number from this data. The measured quantities are the 9.059 ms tax and the 16.041 ms a full re-run adds.

The main risk is a layer after the flag committing GDN state or KV from a slot `smap` did not name. The encoder is concurrent. A flag written by `moe_gu` is visible to a later dispatch only if a buffer barrier is encoded between them. The re-run starts at the first flag, and the first pass commits nothing past it.

### b. Fewer cuts

Cut only where a miss is likely. The two schedules are the layers that missed on the previous token, and a fixed cut every k layers. Elsewhere, rely on read-ahead. If a layer that was not cut routes an expert that is not resident, that is the fallback: record it, commit nothing from that layer, load, and re-run from there.

A layer that was cut is today's path. A layer that was not cut, and whose experts are resident, uses the bytes already published. A cut on a layer that then hits still pays the handshake.

The worst case is a miss on every layer the schedule skipped. Each one re-runs the tail of its chunk, and the cuts that were kept are still paid. An every-k schedule pays 48/k handshakes on layers that would have hit. The previous token's miss set was not compared with the next token's route. This note does not treat it as a predictor of the 19.3.

If the 9.059 ms were spread evenly across the 48 cuts, an every-4 schedule would leave 12 cuts and 2.265 ms of tax, a gap of 6.794 ms, before any fallback re-run. The per-cut cost was not measured, so the even split is an assumption. A fallback spends kernel time that the tax figure does not include. At 19.3 misses per token, a skipped layer will miss, and the fallback is part of the ordinary token.

The main risk is a stale `smap` read on a layer the schedule did not cut. A read-ahead that has been issued and not published is not resident. Using it without the event the host signals after the store is the race from thread 701231.

### c. Route one layer ahead

`predL_` is written by the layer-l+1 predictor in the encoder that ends at the cut before layer l's experts. `ahead(l+1)` already starts at that cut, after `release_seg`, and the load overlaps the next encoder. This design keeps that load and changes which cuts remain.

When the router encoder completes, the actual ids are in memory and the host reads them. It cuts only when that set contains an expert that is neither resident nor in flight. Resident means `smap` already names a published slot. In flight means the load was issued from `predL_` and the encoder that reads the expert waits on the event the host sets after the store, the same `signaledValue` rule as today. If the event is already signaled when the GPU reaches it, the wait adds no bubble. An in-flight expert does not skip that event. Skipping the event is how the kernel sees a stale map.

An expert that is neither resident nor in flight is a miss. The host loads it, publishes `smap`, signals, and the expert kernel runs after that signal. That is one cut, for that layer. A layer whose actual set is already published is encoded with no host round trip between the router and the experts.

The expert kernel still reads `smap`. If the slot is missing it records the (layer, expert) pair and writes no outputs for that layer. The host loads and re-runs from the first flagged layer. A prediction that names the wrong expert takes this path. The re-run uses the loaded bytes, and the first pass does not commit the bad layer, so the logits match.

The worst case is a miss that is not in flight on every layer. That is 48 cuts, today's schedule, plus the check. The 19.3 misses per token are an upper bound on the layers that still cut, if each miss falls on its own layer. They are not a measured layer count.

If the 9.059 ms is spread evenly, 19.3/48 of it is 3.642 ms still paid, and 28.7/48 of it is 5.417 ms of handshake on layers with no miss. Misses that share a layer leave less than 3.642 ms. The per-cut split was not measured, so 5.417 ms is a bound under that assumption. It is not a speed claim. The 7.94 ms SSD wait on a true miss stays, and it is outside the 9.059 ms.

The main risk is treating an in-flight expert as readable before the host has stored `smap` and signaled the event. The kernel then uses a stale map, or a slot that still holds the previous expert. The concurrent encoder does not order that store with a kernel committed before the store.

## The test

Each design gets the same two checks. Describing them is not an engine change.

The LOGPOS byte check, the one already used for fix 4 and fix 3. The logits file matches the reference byte for byte. A stale expert or a stale `smap` entry changes those bytes. The score is `score_logpos.py` with the existing rules, PICK_GAP 0.5, PICK_RANK 3, NLL_BAND 0.01. One matching greedy string does not replace the byte file.

A forced-miss test. After the route is known, evict one expert the route selected: mark its `smap` entry missing and drop the slot. The design loads it and re-runs from that layer, and the LOGPOS file matches the run that did not evict it. Do this on layer 0 and on a late layer. For (b), evict an expert on a layer the schedule did not cut. For (c), also hold back the publish of one expert that `predL_` had in flight until after the expert kernel would have run, and require the flag and the re-run. A pass that matches because the kernel read the old slot is a failure. The byte file is what shows it.

## Recommendation

Design (c). This note stops here for review.

(a) does not load during the token. The 19.3 misses then wait until the encoder ends, and the re-run from the first of them can cost a large part of the 16.041 ms kernel floor. A miss on layer 0 costs more than the 9.059 ms the design is there to remove.

(b) cuts on a guess. The previous token's misses were not measured against the next token's route, and an every-k pattern pays a handshake on hits. At 19.3 misses per token the fallback re-run is the ordinary case, not only the worst one.

(c) keeps the load `predL_` already starts during the previous encoder. It spends a host round trip when the actual set has an expert that is not published and not in flight. The layers with no such expert are where the 9.059 ms can move. Under an even split the bound on that piece is 5.417 ms, and clustering the 19.3 misses makes it larger. The kernel check and the re-run are what keep a wrong prediction off a stale slot. The two tests above are the gate before any of this is written.

The bandwidth table does not change the choice. If every class under 70% of the read probe moved to that rate, the positive gaps sum to 2.427 ms. Of that, 0.567 ms is classes under 2 MB and 1.860 ms is the rest. That is smaller than the boundary tax. The boundary is the larger pool.
