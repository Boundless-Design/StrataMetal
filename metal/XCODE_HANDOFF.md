# Strata Metal handoff

Open this project and start here. The snapshot date is 2026-10-15.

This copy lives on the internal disk:

```
~/Developer/Strata-metal/xcode/strata_metal.xcodeproj
```

The canonical tree is `/Volumes/Code/Personal/Strata`. Edits you make here are not in that checkout. Copy `metal/` changes back there before committing. Refresh this copy from the checkout with:

```sh
/Volumes/Code/Personal/Strata/metal/xcode.sh --local
```

That overwrites `metal/`, `include/`, `docs/APPLE_SILICON.md`, and the reference sources listed below, then regenerates the Xcode project. Derived Data stays at Xcode's default, `~/Library/Developer/Xcode/DerivedData`, also on this disk.

## Run

The scheme menu has twenty-five tools. Frame capture and shader validation are on. Run uses the Release configuration.

| Scheme | What Run does |
| --- | --- |
| `strata-metal-device` | Names the GPU and prints which published model sizes fit the working set. |
| `strata-metal-bf16-gemv` | Passes `--selftest`. A BF16 matrix-vector product against a CPU reference, including one GGUF tensor. |
| `strata-metal-gdn` | Passes `--selftest`. The DeltaNet step, convolution, norms, and beta sigmoid. |
| `strata-metal-qsa` | Passes `--selftest`. Decode attention over a paged KV cache: fp16, int8, q4_0, and int8 keys with q4_0 values. |
| `strata-metal-iq` | Passes `--selftest`. The Coder expert product: IQ2_S, IQ3_XXS, IQ3_S or IQ4_XS gate and up, SiLU, then IQ4_NL or Q2_0 down. |
| `strata-metal-sgemv` | Passes `--selftest`. The S-form dense projection: IQ4_XS, Q4_K, Q5_K, and an Iq4Nl codebook, with Q8_K and Q8_0 activations. |
| `strata-metal-elementwise` | Passes `--selftest`. The GDN glue, one embedding row, the expert-half add, and the QSA weighted norm. |
| `strata-metal-gdn-layer` | Passes `--selftest`. One GDN layer: the projection, the conv, the glue, and the step, on synthetic weights. |
| `strata-metal-gr` | Passes `--selftest`. The gated residual: a per-stream norm, silu of the down projection over hc, the mean, and the 2*sigmoid write. |
| `strata-metal-router` | Passes `--selftest`. The MoE router: softmax over every expert, the top 10, the lower index on a tie, and the clamped renorm. |
| `strata-metal-sampler` | Passes `--selftest`. One token from f32 logits: penalties, top-k, top-p, min-p, temperature, then one Philox draw. |
| `strata-metal-moe-combine` | Passes `--selftest`. The routed expert rows, weighted, plus the shared output added plain. |
| `strata-metal-s2-gemv` | Passes `--selftest`. A 2-bit row, group 64, bias -1, dotted with a Q8_0 activation. |
| `strata-metal-shared-expert` | Passes `--selftest`. One shared expert: SiLU on the gate projection, then one scalar from the original hidden state. |
| `strata-metal-qsa-gate` | Passes `--selftest`. The QSA gate: sigmoid of the second half of each head, times the attention output. |
| `strata-metal-indexer` | Passes `--selftest`. One raw indexer key. A finished block is pooled, normed, and rotated at its first cell. |
| `strata-metal-qsa-index` | Passes `--selftest`. Indexer cell scores: ReLU per head, the block bias, then 1e9 on an incomplete tail. |
| `strata-metal-topk` | Passes `--selftest`. The largest cells, the lower index on a tie, ids ascending. |
| `strata-metal-kv` | Passes `--selftest`. One cell into a paged fp16 cache, and a gather back out. The page table is reversed. |
| `strata-metal-rope` | Passes `--selftest`. Partial NEOX rotation. Pair i with i + n_rot/2. The table is host float64. |
| `strata-metal-qsa-layer` | Passes `--selftest`. One QSA attention block. No new shader. Synthetic 4-bit weights, top-k 2. |
| `strata-metal-token` | Passes `--selftest`. One greedy token: embed, one block, the output projection, the sampler. One draft step from a caller residual and the next token. One greedy token at n_embd 2560, indexer dim 128, 64 rotated values, top-k 2. One greedy token at that width with residual rank 320. One greedy token there with routed hidden 640 and 8 experts. One greedy token there with 512 experts. One greedy token there with shared hidden 640. One greedy token at vocabulary 248320. Sixteen greedy cells at that vocabulary, top-k 2 against top-k 2048. `--serve` answers on the small fixture. |
| `strata-metal-stack` | Passes `--selftest`. Four layers on the pack schedule: three GDN, one QSA, one greedy sample. |
| `strata-metal-pack` | Passes `--selftest`. Every layer of the Q2_0 model, one token at a time, by name from the GGUF. |
| `strata-metal-layer-tensors` | Passes `--selftest`. The by-name tensor lookup on a written two-shard GGUF. Host code only. |

To capture a kernel: select a parity scheme, Run, then Debug > GPU Capture. The shaders are `metal/shaders/bf16_gemv.metal`, `metal/shaders/gdn.metal`, `metal/shaders/qsa.metal`, `metal/shaders/iq.metal`, `metal/shaders/sgemv.metal`, `metal/shaders/elementwise.metal`, `metal/shaders/gr.metal`, `metal/shaders/router.metal`, `metal/shaders/sampler.metal`, `metal/shaders/moe_combine.metal`, `metal/shaders/s2_gemv.metal`, `metal/shaders/shared_expert.metal`, `metal/shaders/qsa_gate.metal`, `metal/shaders/indexer.metal`, `metal/shaders/qsa_index.metal`, `metal/shaders/topk.metal`, `metal/shaders/kv.metal`, and `metal/shaders/rope.metal`. `strata-metal-qsa-layer`, `strata-metal-token`, `strata-metal-stack`, and `strata-metal-pack` add no shader. They are compiled when the process starts, from those files, as Metal 4.1 with safe math and the precise `exp` and `sqrt`. Editing a `.metal` file and running again is enough. Editing a `.mm` or `.cpp` file needs a rebuild.

`strata-metal-device` does not compile a shader. Add `--selftest` under Product > Scheme > Edit Scheme > Run > Arguments to repeat its 1 MiB blit.

## Measured on this Mac

Apple M5 Max, 36 GB, Mac Studio, Apple10, Metal 4, unified memory. The device
check's working set was 28.080 GiB. The blit passed. That check loads no model.

`strata-metal-pack --selftest` checks every layer of the Q2_0 model, one token
at a time. Layers 0 and 1 keep their own checks. The walk runs layers 2 through
47, the last layer (`n_layers - 1`). It passed with 0 failures. On this Mac the
run took 116.97 s.

Every `blk.*` tensor is in shard 1, the metadata shard. The token embedding and
the output weight are there too. Shard 2 holds one tensor,
`per_layer_token_embd.weight`. The walk does not read it.

`host_ref` covers Q2_0, IQ4_XS, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0,
and Q8_0. Q4_K and Q5_K use the scalar reference.

`gdn_iq_host` tries IQ4_XS, Q3_K, and Q2_0 on the query, the gate, and the
output. The gate and the output also try Q4_K and Q5_K through the scalar
reference. Anything else falls through to `host_ref`.

`qsa_iq_host` keeps `k_ref` for Q4_K and Q5_K on the key, the value, and the
output. It also has its own branches for Q3_K, IQ4_XS, Q2_0, and Q6_K. Anything
else falls through to `host_ref`. Those K-quants are not sent through the
scalar reference.

The QSA block projects from the GPU's gated vector, so a one-ulp `exp`
difference is not requantized. The gate kernel is Metal `exp`. Its f32 check
against the double reference passed at 3.912e-08. The bound is 1e-6.

The deepest layer is 47. All 48 layers pass.

## Forward pass

`strata-metal-run` takes token ids and writes one full logit row per position.
GDN state, QSA fp16 KV, and the PLE window carry across positions. The ten
routed experts and the PLE table stay on disk. Dense weights, the head, and
the small PLE projections are resident.

```sh
cmake --build build-metal --target strata-metal-run
build-metal/strata-metal-run --tokens 760,6511,314,9338,369 --max-new 0 \
  --dump-logits /tmp/p1.bin
python3 metal/tools/compare_fixtures.py /tmp/p1.bin p1
```

`--max-new 0` scores the prompt except its last token. A positive count also
scores that last prompt token, then that many new positions minus one.
`--teacher-force` replaces the greedy picks after the prompt. The lowest id
wins a tie. The dump is little-endian int32 vocab, int32 rows, then f32 rows.

`compare_fixtures.py --gate` scores every prompt and generation row of p1-p5
against a clean dump and a one-ulp dump. A pick matches CUDA, or it sits in
CUDA's top 3 and CUDA's gap from its #1 to that pick is at most 0.5. On p5's
prompt rows whose real next prompt token is in CUDA's top 64, the Mac mean
NLL has to be within 1% of CUDA's and not more than 1% worse. Over all 492
rows, the 95th percentile of |logprob delta| and of top-64 total variation
must each be at most twice the same percentile for the one-ulp run. The raw
top-64 logit gap is printed and is not a gate. The percentile is Hyndman-Fan
type 7.

On this Mac the resident set was 3.275 GiB of 28 GiB: 48 layer execs 2.856
GiB and the head 0.419 GiB. PLE projections were 0.019 GiB and session scratch
0.001 GiB. The 492-row gate passed. 478 picks matched. These 14 did not, and
each is inside the top-3 rule. The widest gap is 0.198.

- p3 pos 17: CUDA 460, Mac 745, rank 2, gap 0.060
- p5 pos 8: CUDA 198, Mac 271, rank 2, gap 0.057
- p5 pos 26: CUDA 39332, Mac 18237, rank 2, gap 0.083
- p5 pos 37: CUDA 13759, Mac 5157, rank 2, gap 0.142
- p5 pos 56: CUDA 321, Mac 1024, rank 2, gap 0.086
- p5 pos 59: CUDA 780, Mac 264, rank 2, gap 0.036
- p5 pos 73: CUDA 310, Mac 60999, rank 2, gap 0.114
- p5 pos 126: CUDA 7481, Mac 26164, rank 2, gap 0.198
- p5 pos 168: CUDA 995, Mac 557, rank 2, gap 0.026
- p5 pos 173: CUDA 8526, Mac 557, rank 3, gap 0.052
- p5 pos 207: CUDA 3043, Mac 539, rank 2, gap 0.034
- p5 pos 222: CUDA 3571, Mac 7481, rank 2, gap 0.106
- p5 pos 223: CUDA 13569, Mac 3162, rank 2, gap 0.115
- p5 pos 299: CUDA 5820, Mac 5924, rank 2, gap 0.002

p5 prompt NLL used 293 rows. CUDA's mean was 1.593 and the Mac's was 1.584.
The Mac was lower on 159 rows, CUDA on 134, with no ties. One ulp of layer-0
noise on the same 492 rows has |logprob delta| p95 0.107 and TV p95 0.054.
Mac against CUDA is 0.114 and 0.061. Both are under twice the noise.

Free-running greedy for 32 new tokens, no teacher force: p1 matched CUDA at
all 36 positions, 1.066 s/token. p3 first diverged at position 17, the same
accepted pick as the teacher-forced row (CUDA 460, Mac 745, gap 0.060). That
run was 1.118 s/token over 35 positions. Later p3 tokens leave the fixture
path because the fed token has changed. The rate is a measurement, not a
target. A long teacher-forced p5 run was 0.775 s/token over 338 positions.

### Diagnostics

`STRATA_PERTURB_ULP` and `STRATA_LOG_MARGINS` are off unless set. Fixture runs
leave both unset, except the noise dump, which sets only the first.

`STRATA_PERTURB_ULP=1` flips the low mantissa bit of every layer-0 residual
float after the embedding broadcast, at every position, including position 0.
Any other value, including unset, does not flip.

`STRATA_LOG_MARGINS` names a file. The run writes the router's 10th-minus-11th
probability, before the top-10 weights are divided by their sum, one row per
layer per position. Unset or empty writes nothing.

## What does not run

The layer walk is still one token at a time, against a host reference. The
forward pass is greedy. It does not sample. There is no tokenizer, no server,
and no batched prefill. Routed experts and the PLE table are not copied into
a GPU buffer.

The CUDA and HIP engines are not built. `python setup.py` on macOS stops and
points at `docs/APPLE_SILICON.md`.

## Next

The forward pass matches the CUDA fixtures under the gates in
`metal/tools/compare_fixtures.py`. Do not start resident experts, kernel
fusion, a speed pass, batched prefill, a server, or a tokenizer.

The Reference group is the CUDA source of the kernels already ported, except
the expert product, the S-form projection, and the layer glue. The `.cu` files
under `src/kernels/cuda` are not built here.

### How a layer is checked now

Layers 2 onward run through one loop, `run_layer_walk`, in
`metal/src/pack_parity.cpp`. Each layer is found by name in the GGUF
(`metal/include/strata/metal/layer_tensors.hpp`): its kind (`attn_q` is QSA,
`attn_qkv` is GDN), every tensor's offset and byte count, and every quant type.
The attention half is `walk_gdn`, or `walk_attn_products` plus `walk_qsa_block`
plus `walk_attn_write`. Then `walk_ffn` runs twice: the routed ten alone, then
routed plus shared. That residual feeds the next layer.

- The model comes from `STRATA_MODEL_GGUF` (any shard), or the Q2_0 shard 1
  under `$HOME`. No path is compiled in.
- The default last layer is `n_layers - 1`, from `qwen4exp.block_count`.
  `STRATA_WALK_LAST` stops the walk early. A new layer is a larger model, not a
  new function.
- Do not add a `run_layerN_*` function, an offset literal, a per-layer
  quant-type literal, or an `if (L == n)` special case. When a layer fails, fix
  the generic code: a quant type the host references do not know, a kernel that
  does not take that type, or the lookup.
- When adding to a type chain, keep every existing branch and add only the
  missing type. `qsa_iq_host` uses `k_ref` for K-quants. Do not reroute those
  through `host_ref`. `host_ref` is only the fallback before the throw.
- Do not loosen a tolerance, skip a verdict, or catch and ignore an error to
  get a pass.
- The QSA block projects from the GPU's gated vector when the walk has one.
  The gate kernel stays Metal `exp`. The gate's own check against the double
  reference stays at 1e-6.
- `strata-metal-layer-tensors --selftest` checks the lookup on a written
  two-shard GGUF. No check reads the model file by a hardcoded offset. Reads of
  `experts.bin` stay. That file is not the GGUF.
- `metal/tools/compare_pack_runs.sh` defaults to commit `128cc72` and diffs
  layers 2-18 number for number. Flip lines are compared on yes or no only.
  `--since <commit>` diffs every layer both runs printed, the same way.
- Layers 0 and 1, `run_routed_ten`, and `run_carried_pair` resolve tensors by
  name and keep their own checks. The walk starts at layer 2. Those checks
  print a different line than the walk, so folding them in would change the
  verdicts.
- The walk does not draw a token. The forward pass is `metal/src/forward.cpp`,
  and it calls the same execs. `STRATA_PERTURB_ULP` and `STRATA_LOG_MARGINS`
  are diagnostics. Both are off unless set. See Forward pass.
