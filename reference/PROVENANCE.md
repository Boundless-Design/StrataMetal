# Provenance of the CUDA reference fixtures

These files are greedy-decode logits from upstream Strata's CUDA engine, for checking the Metal port token by token. Nothing in either engine's math was changed. No file under `reference/` is a model weight.

## Upstream

- Repository: https://github.com/Niko1221/Strata
- Commit: `296b59d2f73c8f04a805df29c2fa97655adace78`
- Tag: `v0.1.44.1` (the engine prints `session is up (engine 0.1.44.1)`)
- Commit subject: Merge f-1441-perf (Responses/Codex honours reasoning_effort; setup keeps the installed size on a re-run) into f-1441
- Commit date: 2026-10-14 01:51:12 +0200
- Clone used to build: `/mnt/scratch/Strata` (outside this repo). `git status` there was clean `main` at that commit. No source patch was applied.

StrataMetal's LICENSE credits Niko1221 and the Strata contributors. `include/strata/artifact/gguf_split.hpp` cites eddoursul/Strata commit `8029fa9` for the GGUF split helper. eddoursul/Strata is a fork. The engine built here is the canonical Niko1221/Strata repository at the commit above, not that fork and not pinned to `8029fa9`.

llama.cpp was fetched by upstream's CMake FetchContent at git tag `3cf03257f219afbe7334045ff7c6a06ac68c627d` (configure logged ggml 0.24.0, commit `3cf03257f`).

## Logit dump: no source patch

Upstream already writes per-step logits. `--dump-logits PATH` copies each position's full vocabulary row to a file after the output projection and before `sample_tokens`. The file starts with two little-endian int32 values, `{n_vocab, n_rows}`, written with the first row, then `n_rows` rows of `n_vocab` float32 values. `n_rows` is `n_prompt - 1 + max_new`. The comment in `src/program/generate.cpp` says a native IQ pack (`native_experts.txt`) never reaches this site. This run used the canonical AVX-512 pack, which has no `native_experts.txt`, with `--spec 0`, so the per-token loop does reach it.

That was checked, not assumed. Each dump's header is vocab 248320 and `prompt_len - 1 + 32` rows, and the engine's chosen token at every recorded position equals `argmax` of that dumped row (lowest index on a tie). The dump is therefore the raw pre-sample f32 logits.

There is no `reference/upstream-logit-dump.patch` because no upstream source change was required. The raw dumps stay on the scratch disk and are not committed (a 307-token prompt's dump is about 320 MB).

## Hardware and software

- OS: Ubuntu 22.04.5 LTS, kernel 5.15.0-185-generic
- CPU: Intel Xeon Platinum 8468, 20 cores, 1 thread per core. Flags include AVX, AVX2, and AVX-512 (F, DQ, IFMA, CD, BW, VL, VBMI, VBMI2, VNNI, BF16, FP16, VPOPCNTDQ, BITALG).
- RAM: 235 GiB, no swap
- GPU: NVIDIA H100 80GB HBM3, 81559 MiB, compute capability 9.0, PCI bus 00000000:00:09.0. The engine printed `GPU 0: NVIDIA H100 80GB HBM3, compute capability 9.0`.
- Driver: 580.173.02 (`nvidia-smi` CUDA Version 13.0)
- CUDA toolkit: `/usr/local/cuda`, nvcc 13.1.115 (built Dec 16 2025), SDK 13.1.2, cudart 13.1.80. The driver and toolkit were already installed.
- Host compiler: gcc/g++ 11.4.0
- Python: 3.10.12. Build tools came from a venv at `/mnt/scratch/venv` (`requirements.txt`: cmake 4.4.3, ninja 1.13.2). Apt cmake 3.22.1 is older than the project's `cmake_minimum_required` 3.24 and was not used for the successful configure.
- Disks: boot `/dev/vda1` ext4, 698G, about 682G free (not tight). Scratch `/dev/vdc1` LABEL `DOSCRATCH` UUID `f0928360-6688-454a-afda-aeb0c7a4466c`, ext4, 5.0T, mounted at `/mnt/scratch` for this work (it was unmounted at the start). Both virtio disks report rotational (`queue/rotational` = 1).

Large artifacts were placed on `/mnt/scratch` even though the boot disk had room:

- upstream source and build: `/mnt/scratch/Strata`, `/mnt/scratch/Strata/build`
- model shards: `/mnt/scratch/Strata-data/models/Q2_0/`
- pack: `/mnt/scratch/Strata-data/packs/q2_0`
- raw logit dumps: `/mnt/scratch/runs/`

## Model

The installer on this machine would not have picked Q2_0. `docs/AI_SETUP.md` selects `--family qwen --model IQ3_S` at 96 GB of RAM and above. Q2_0 was forced by downloading the two files `setup.py` would fetch for `--family qwen --model Q2_0`.

`setup.py` pins `HF_REVISIONS["ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"]` to `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (2026-09-29). The family URL is `https://huggingface.co/{repo}/resolve/{revision}/Q2_0/`. The files:

- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf

Sizes on disk: 37623740192 and 28800138432 bytes. `sha256sum`:

```
69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
```

Both match the expected hashes. Shard 2 is the PLE / n-gram table (`per_layer_token_embd.weight`, 320001536 rows). It is passed as `--ple-gguf` and is not copied into the pack.

## Pack

This CPU has AVX-512, so `setup.py`'s Q2_0 branch for family `qwen` runs `tools/strata_pack.py`, not `tools/iq_pack.py`. That pack does not write `native_experts.txt`. The packer states its rule as `decode(pack) == dequant_ggml(source)` bit for bit. `--skip-hash` is what setup passes (the shards were hashed separately, above). `--force` is what setup passes when rebuilding.

```
STRATA_GGUF_PY=/mnt/scratch/Strata/build/_deps/strata_llamacpp-src/gguf-py \
  /mnt/scratch/venv/bin/python /mnt/scratch/Strata/tools/strata_pack.py build \
  --gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --out /mnt/scratch/Strata-data/packs/q2_0 \
  --skip-hash --force
/mnt/scratch/venv/bin/python /mnt/scratch/Strata/tools/pack_index.py --pack /mnt/scratch/Strata-data/packs/q2_0
/mnt/scratch/venv/bin/python /mnt/scratch/Strata/tools/strata_tokenizer.py \
  --gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --out /mnt/scratch/Strata-data/packs/q2_0
```

The pack log: `experts.bin` is 48 layers × 512 experts; `dense.bin` 1078 tensors, 5.44 GiB; `embd.bin` 1 tensor, 0.44 GiB; `index.txt` 1079 tensors. Tokenizer: vocab 248320, merges 247587, pre `qwen35`, `add_bos_token` false. Confirmed no `native_experts.txt`.

`setup.sh` itself was not used to launch the engine. Its generated arguments are `--expert-profile`, `--expert-cache auto`, `--prefill auto`, `--spec 4`, `--spec-min-p 0.5`, `--mtp`, and `--max-context` from VRAM (131072 on an 80 GB card), plus `--kv` when the context is above 8192. Those turn on speculative decoding and the MTP draft. The MTP layer was not downloaded or built. The download URL, the pinned revision, and the AVX-512 pack recipe are the installer's. The launch flags below are the deterministic ones, not setup's server command.

## Build

`CMakeLists.txt` defaults `CMAKE_CUDA_ARCHITECTURES` to 120 when it is unset. sm_90 is accepted when it is passed. Only the cache flags were changed. No source file was edited.

```
/mnt/scratch/venv/bin/cmake -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/mnt/scratch/venv/bin/ninja \
  -S /mnt/scratch/Strata -B /mnt/scratch/Strata/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
/mnt/scratch/venv/bin/cmake --build /mnt/scratch/Strata/build --target strata -j 10
```

`-j 10` is setup's `max(2, cpu_count // 2)` on 20 cores. Configure logged `Strata: CUDA enabled, arch 90` and ggml `-march=native` (upstream's default `GGML_NATIVE`; `STRATA_ISA_FLOOR` was not set). `STRATA_EXPERIMENTAL_SM60` was not set. The build finished `[153/153] Linking CXX executable strata` on the first attempt (`BUILD_EXIT:0`). Binary: `/mnt/scratch/Strata/build/strata`.

## Runtime settings

Environment: `STRATA_ROUTE_TAIL_SKIP=0`. The CLI already leaves the rank at 0 (`tail_skip_rank` in `src/core/expert_source.cpp`). The rank-7 default that changes answers is only in the CUDA `--serve` path when not every expert is VRAM-resident. It was set explicitly anyway. `STRATA_ARGMAX_MULTI`, `STRATA_LOGPOS`, and `STRATA_DUMP_FIRST_LOGITS` were unset, so the default sm_90 cluster greedy kernel is the argmax.

Every prompt used the same flags. Two processes per prompt (`-a` then `-b`), each a cold start.

Sampling:

- `--greedy` (also the default). `sample_tokens` takes the argmax branch when `greedy` is set, before temperature, top-k, top-p, and min-p.
- No `--seed`. Passing `--seed` would turn greedy off.
- No `--temperature`, `--top-k`, `--top-p`, or `--min-p`. The struct defaults (`temperature` 1, `top_k` 20, `top_p` 0.95, `min_p` 0) are not applied on the greedy branch. The log line is `sampling greedy`.
- Repetition penalties stay at the sampler defaults: `penalty_last_n` 0 (off), `penalty_repeat` 1, `penalty_freq` 0, `penalty_present` 0. No logit bias.
- Argmax ties keep the lowest token id. Every recorded `next_token` matched that argmax on the dumped row.
- `--stop-eos` was not passed (`stop_eos` defaults false), so an end token does not cut the 32. Several prompts do emit `<|im_end|>` inside the 32.

Decoding path:

- `--spec 0`. CUDA's default is already 0 (Metal's default is 2). No speculative window, so every logit is the main model's own forward. The spec loop, which is where MTP and suffix drafts run, was not entered.
- `--suffix-draft 0`. The default is 3, and it only lengthens a window when `spec >= 2`. Passed as 0 so a later reader does not have to notice that.
- No `--mtp`. The draft layer does not exist on this machine.
- No `--prefill`. `prefill_chunk` stays 0, so positions `0 .. prompt_len-2` are teacher-forced through the same per-token forward as decode, not the batched prefill kernels. The log's prefill line counts `prompt_len - 1` tokens. Setup's `--prefill auto` was left off on purpose: it is a different kernel path.
- `--kv fp16`, which is the default and the most precise of `fp16`, `int8`, `q4_0`, and `k8v4`. `--kv-resident` was not passed, so it stays 0 and the whole KV cache is in VRAM. `--max-context 4096` (the default) is the allocation size. The longest prompt plus 32 new tokens is 339, under that cap. KV streaming was not used.

Experts and the PLE table:

- `--expert-cache auto` and `--expert-profile /mnt/scratch/Strata/data/expert-profile.bin`. That shipped profile ranks all 24576 pairs (48 layers × 512 experts).
- `--adapt-every 0`. The default is 4, and the swap loop lives in the speculative path, which was not entered. Passed as 0 so the VRAM tier cannot move. `--adapt-async` was not passed (default off, and it requires `--serve`).
- No `--mmap-experts`, `--resident-experts`, `--resident-budget-gib`, or `--resident-cpu-experts`. Those are the low-RAM / SSD modes. This machine is not in that regime: Q2_0's `arena_gb` is 34.0 and `ram_gb` is 48, against 235 GiB.
- `--ple-gguf` is shard 2. Without it, layer 1's PLE is absent and every later number changes. The engine refuses `--native` without it.
- `--ple-io ram`. The scratch disk reports rotational, and setup adds `--ple-io ram` when that is true and RAM holds the model plus the table (`ram >= 48 + table_gb + 4`). 235 GiB does. The mode locks the table in RAM (`mlock`). It is the same bytes as a disk read. It was passed explicitly so a slow direct-I/O probe could not change the mode between runs. The log says `PLE table locked in RAM (--ple-io ram)`.

`--native` on shard 1 is what setup launches for this model. It is not the IQ `native_experts.txt` pack. The flag turns on, in `generate.cpp`: `stream_token`, `gr_native_mmvf`, `native_bf16`, `native_bf16_extra`, `native_ple_key`, `native_moe_combine`, `native_gdn`, `native_router`, `native_qsa`, `native_qsa_indexer`, `native_rope`, `native_ple_postops`, the native head from shard 1, and native dense matrices from the shards. It does not turn on `--native-flash-attn-short`. The log confirms it: `301 canonical tensors skipped: served natively`, `301 native projection matrices, 1376.20 MiB`, and `experimental native Q5_K head, 437043200 bytes`.

Two numeric facts that were not overridden, because overriding them would mean editing upstream:

- With `--native`, the QSA scorer is `native_qsa_score.cu`. Its MMA is `mma.sync` `tf32` and the comment says it deliberately does not round with `cvt.rn.tf32`; raw f32 bits are passed through. That is the kernel this flag selects.
- Batched prefill GEMM (`src/prefill/gemm.cu`) sets `CUBLAS_DEFAULT_MATH` and `CUBLAS_COMPUTE_32F` on fp16/bf16 inputs. This run did not use `--prefill`, so the prompt went through the captured per-token graph (`token graph captured (48 layers, one launch per token)`), not that GEMM.

Every startup prints a PCIe bandwidth probe (about 55 GB/s on this host; the printed figure moved between 55.3 and 55.4 across the ten processes) and `the canonical pack starts at pcie_frac 0.20`. That fraction splits CPU and GPU expert work inside a verify window when experts miss the cache. Every expert was in the cache (below), and `--spec 0` did not open a verify window, so expert math did not take that split. The probe's small run-to-run change did not change any logit (the dumps are bit-identical).

## Weight placement

The same placement was printed on all 10 processes (two runs of each of the five prompts). From the first p1 run, with the slot counts unchanged on the other nine:

```
expert arena read through the file cache (buffered (not Windows))
expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable (needed 16201 2 MiB pages, vm.nr_hugepages=0); transparent huge pages requested (MADV_HUGEPAGE)
loaded 31.64 GiB at 3.74 GiB/s
expert cache auto: 74.53 GiB free, 700 MiB reserved (+0 MiB for the draft head) -> 24576 slots
expert cache 24576 slots, 31.64 GiB of VRAM; policy is
PROFILE, ranked by routing frequency, no eviction.
pre-filled 24576 of 24576 slots from the profile in 1.0 s; slot 0 verified
R4 hit path ON - resident experts are computed on the GPU
token graph hit path: 24576 resident experts, decided on the device
```

The arena load rate varied from 3.64 to 3.76 GiB/s. Free VRAM, slot count, and the 24576/24576 prefill did not vary.

What that means on this machine:

- All 48 × 512 experts are in the RAM arena (31.64 GiB, pinned with `cudaHostRegister`). They are not read from SSD during the forward, and `--mmap-experts` was not used.
- All 24576 experts are also prefilled into VRAM before the first token. The GPU computes every one of them (`R4 hit path ON`). None are computed on the CPU. The engine's own note, printed every run, is that a GPU expert and the CPU expert round differently; that split does not occur here.
- Dense pack weights: 1879 MiB loaded, of which 301 tensors are skipped because `--native` serves them from the GGUF. Native projections: 1376.20 MiB. Native Q5_K output head: 437043200 bytes.
- PLE table: shard 2, 320001536 rows, locked in RAM.
- KV: fp16, resident in VRAM, allocated for `--max-context 4096`.

## Tokenizer and prompts

The CLI has no tokenizer (`--tokens` is required). It does not apply the GGUF chat template. The template is present (`tokenizer.chat_template`, a Qwen-style Jinja template that can open an assistant turn with `<think>`). It was not applied. `chat_template` in the meta files is null and `bos` is false.

Token ids come from `tools/strata_tokenizer.py` `Tokenizer.encode(text)` with `parse_special` left false. That does not prepend a BOS token (`tokenizer.ggml.add_bos_token` is false; the BOS id 248044 is not inserted). Decoding the ids reproduces the prompt string exactly. Prompt lengths: p1 5, p2 11, p3 4, p4 10, p5 307.

The strings are the ones in the meta files. p1 through p4 are the four required sentences. p5 is a 307-token prose passage (about 300), stored in full in `p5.meta.json`.

## What each JSONL row is

Generating 32 tokens runs the forward at positions `prompt_len - 1` through `prompt_len - 1 + 31` and then stops. That is 32 rows. `pos` is that index. `input_token` is the id fed at `pos` (a prompt id for the last prompt position, otherwise the previously chosen id). `next_token` is the greedy choice, and the 32 `next_token` values are the generated ids in order. The last generated token is not fed back, so there is no 33rd row.

The per-token loop also computes teacher-forced logits at positions `0 .. prompt_len - 2`. Those rows are in `<prompt-id>.prompt.jsonl`, described in the next section. The generation file's positions are the last prompt position and the 31 following positions that the generated tokens were fed into, which together produce the 32 new tokens.

`top` is 64 `[id, logit]` pairs, highest raw logit first, and the lower id first when logits tie. Logits are the dumped f32 values before softmax and before temperature. JSON numbers are the exact float32 value promoted to float64 (Python's `json` encoding of that promotion). Casting the parsed number back to float32 restores the dump bits.

`max` is the first (lowest-index) maximum, as a float32. `lse` is the log-sum-exp over all 248320 rows, in float64: `max + log(sum(exp(logit - max)))`, the same reduction as `Verifier::window_logprobs` in `src/core/verify.cpp`.

`output_text` is the decode of the 32 new tokens only, special tokens included, for a person reading the file.

## Teacher-forced prompt rows

`reference/fixtures/<prompt-id>.prompt.jsonl` holds positions `0 .. prompt_len - 2` from the same `-a` raw dumps (`/mnt/scratch/runs/<id>-a.bin`). The engine was not run again. The row format matches `<prompt-id>.jsonl`. `input_token` is `prompt_tokens[pos]`. `next_token` is the argmax of that row, lowest id on a tie: the model's prediction at that position, not `prompt_tokens[pos + 1]`. `top`, `max`, and `lse` use the same rules as the generation files.

Row counts: p1 4 (positions 0..3), p2 10 (0..9), p3 3 (0..2), p4 9 (0..8), p5 306 (0..305).

The `-b` dumps were compared with the `-a` dumps on these positions, over the full 248320-wide row. All five are bit-identical, so the argmax, the top 64, `max`, and `lse` match to the last bit.

## Commands

### p1

```
STRATA_ROUTE_TAIL_SKIP=0 /mnt/scratch/Strata/build/strata \
  --pack /mnt/scratch/Strata-data/packs/q2_0 \
  --native /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --ple-io ram \
  --expert-profile /mnt/scratch/Strata/data/expert-profile.bin \
  --expert-cache auto \
  --adapt-every 0 \
  --spec 0 \
  --suffix-draft 0 \
  --greedy \
  --kv fp16 \
  --max-context 4096 \
  --max-new 32 \
  --dump-logits /mnt/scratch/runs/p1-a.bin \
  --tokens 760,6511,314,9338,369
```

The second run is the same command with the dump path `p1-b.bin`. The token list is exactly `prompt_tokens` in `reference/fixtures/p1.meta.json`.

### p2

```
STRATA_ROUTE_TAIL_SKIP=0 /mnt/scratch/Strata/build/strata \
  --pack /mnt/scratch/Strata-data/packs/q2_0 \
  --native /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --ple-io ram \
  --expert-profile /mnt/scratch/Strata/data/expert-profile.bin \
  --expert-cache auto \
  --adapt-every 0 \
  --spec 0 \
  --suffix-draft 0 \
  --greedy \
  --kv fp16 \
  --max-context 4096 \
  --max-new 32 \
  --dump-logits /mnt/scratch/runs/p2-a.bin \
  --tokens 814,20139,303,1330,22157,3069,279,12515,369,6105,13
```

The second run is the same command with the dump path `p2-b.bin`. The token list is exactly `prompt_tokens` in `reference/fixtures/p2.meta.json`.

### p3

```
STRATA_ROUTE_TAIL_SKIP=0 /mnt/scratch/Strata/build/strata \
  --pack /mnt/scratch/Strata-data/packs/q2_0 \
  --native /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --ple-io ram \
  --expert-profile /mnt/scratch/Strata/data/expert-profile.bin \
  --expert-cache auto \
  --adapt-every 0 \
  --spec 0 \
  --suffix-draft 0 \
  --greedy \
  --kv fp16 \
  --max-context 4096 \
  --max-new 32 \
  --dump-logits /mnt/scratch/runs/p3-a.bin \
  --tokens 727,73111,1393,1590
```

The second run is the same command with the dump path `p3-b.bin`. The token list is exactly `prompt_tokens` in `reference/fixtures/p3.meta.json`.

### p4

```
STRATA_ROUTE_TAIL_SKIP=0 /mnt/scratch/Strata/build/strata \
  --pack /mnt/scratch/Strata-data/packs/q2_0 \
  --native /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --ple-io ram \
  --expert-profile /mnt/scratch/Strata/data/expert-profile.bin \
  --expert-cache auto \
  --adapt-every 0 \
  --spec 0 \
  --suffix-draft 0 \
  --greedy \
  --kv fp16 \
  --max-context 4096 \
  --max-new 32 \
  --dump-logits /mnt/scratch/runs/p4-a.bin \
  --tokens 7734,264,6185,36974,883,264,14367,303,40297,13
```

The second run is the same command with the dump path `p4-b.bin`. The token list is exactly `prompt_tokens` in `reference/fixtures/p4.meta.json`.

### p5

```
STRATA_ROUTE_TAIL_SKIP=0 /mnt/scratch/Strata/build/strata \
  --pack /mnt/scratch/Strata-data/packs/q2_0 \
  --native /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /mnt/scratch/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --ple-io ram \
  --expert-profile /mnt/scratch/Strata/data/expert-profile.bin \
  --expert-cache auto \
  --adapt-every 0 \
  --spec 0 \
  --suffix-draft 0 \
  --greedy \
  --kv fp16 \
  --max-context 4096 \
  --max-new 32 \
  --dump-logits /mnt/scratch/runs/p5-a.bin \
  --tokens 760,14367,6092,37444,1518,279,18866,1467,13,2332,279,809,314,279,9506,13759,264,5054,303,264,19177,22162,716,264,21664,17189,314,39332,383,279,6784,321,14936,279,2919,3166,1172,279,5157,287,13,2838,995,14428,411,1560,1396,6353,364,1578,13,357,7914,383,264,33828,5642,1024,11,64779,780,27491,10598,11,321,279,4950,67441,974,279,79338,321,3571,1142,62086,13,469,68042,14217,383,279,11130,793,430,413,781,995,978,8194,1017,310,1698,279,30592,13,271,1311,279,854,1292,8379,279,2981,9057,279,3000,995,42942,279,73685,13,561,73237,557,2582,32713,88562,8366,279,5373,11,321,279,21388,314,7948,15530,2197,279,72555,21388,314,279,14367,13,8771,70704,17736,11,1973,18654,11,883,3315,279,40554,1000,2404,1518,35915,13,561,5054,10471,264,73541,321,264,5392,314,10482,13,2838,1467,524,45659,13,561,1834,8047,557,18541,11,864,557,1092,1292,4674,494,424,25,11592,310,4087,11,264,50083,14795,11,264,3814,310,1024,10326,383,279,23198,13,271,7691,7338,364,264,1345,383,279,3238,6784,28428,279,33619,3626,321,28838,264,6365,314,279,15530,13,357,292,2657,7481,40352,748,13569,421,279,14714,9047,310,1683,1518,424,8597,13,11721,17957,557,12125,13,561,2919,557,6121,11,23271,11,321,2400,314,2526,60637,314,10708,13,3095,1292,14217,685,11,93240,10729,8366,279,25374,321,279,338,68042,3571,1441,506,2957,13,2838,45465,279,5392,2094,279,2645,314,279,73541,321,3804,383,8574,279,23198,11,1332,279,5459,70454,321,279,6092,11639,10729,3023,4661,1024,13,198
```

The second run is the same command with the dump path `p5-b.bin`. The token list is exactly `prompt_tokens` in `reference/fixtures/p5.meta.json`.

## Determinism

Each prompt was run twice from a cold process, same binary, same flags, same pack. The two raw dumps were compared over every recorded row and the whole 248320-wide vocabulary, not only the top 64.

| prompt | chosen tokens | raw dump |
| --- | --- | --- |
| p1 | identical | bit-identical |
| p2 | identical | bit-identical |
| p3 | identical | bit-identical |
| p4 | identical | bit-identical |
| p5 | identical | bit-identical |

Max absolute difference and max relative difference on the full rows, on `max`, and on `lse` were all 0. The top-64 ids matched. Nothing in the reference moved between the two runs, so no setting had to be blamed for drift.

## Coherence

Before the other prompts, p1 was decoded. The first generated tokens are ` Paris.` (ids 11751, 13), which is a coherent completion of "The capital of France is". p3 continues `def fibonacci(n):` with a recursive body. The model then often emits chat-template control tokens (`<|im_end|>`, `<|im_start|>assistant`, `<think>`) because the strings were raw completions and the checkpoint was trained with that template. That text is what this configuration actually produced.
