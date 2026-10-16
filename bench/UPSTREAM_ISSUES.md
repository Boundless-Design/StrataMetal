# Upstream issues from the Mac native-pack runs

Collected from the runs in `bench/UPSTREAM_MAC.md`. No new engine change. Checkout `/Users/kamp/Developer/Strata-upstream`, version string `0.1.44.1-metal`.

| | |
| --- | --- |
| Upstream commit | `296b59d2f73c8f04a805df29c2fa97655adace78` ("Merge f-1441-perf (Responses/Codex honours reasoning_effort; setup keeps the installed size on a re-run) into f-1441", 2026-10-14 01:51:12 +0200) |
| macOS | 27.0.1, build 26A434 |
| Mac | Mac Studio, model identifier Mac17,14 |
| Chip | Apple M5 Max |
| RAM | 36 GB unified (`hw.memsize` 38654705664). Setup prints "36 GB unified memory". The engine prints RAM 38.7 GB, the same byte count in decimal GB. |

The pack is the native IQ pack at `/Users/kamp/Developer/Strata-data/packs/q2_0` (`native_experts.txt` present). Both commands ran from `/Users/kamp/Developer/Strata-upstream` on 2026-10-15.

## 1. `--dump-logits` writes an empty file

Command, from `/tmp/upstream-mac/correctness.sh`. The prompt file is the five p1 ids `760,6511,314,9338,369`.

```sh
env STRATA_METAL_STREAM=1 STRATA_ROUTE_TAIL_SKIP=0 STRATA_HC_Q8=0 STRATA_PREFILL_CPU_SHARE=0 \
  STRATA_METAL_PROMPT=window \
  ./build-metal/strata \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile "$HOME/Developer/Strata-upstream/data/expert-profile.bin" \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --max-context 4096 --mmap-experts --kv fp16 --adapt-every 0 \
  --tokens "$(cat /tmp/upstream-mac/p1.prompt)" --max-new 1 \
  --dump-logits /tmp/upstream-mac/p1-dump.bin
```

Exit 0. `wc -c /tmp/upstream-mac/p1-dump.bin` printed `0 /tmp/upstream-mac/p1-dump.bin`. The file is 0 bytes. Stdout printed:

```
logits dumped            /tmp/upstream-mac/p1-dump.bin
```

`metal/tools/compare_fixtures.py` then printed `/tmp/upstream-mac/p1-dump.bin: shorter than a header`.

The per-token loop that writes the dump breaks on a native pack before any row. At `296b59d`, `src/program/generate.cpp` line 13286:

```cpp
// plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
if (native_pack) { spec_pos = pos; break; }
```

The fwrite of the header and the vocabulary row is later in that same loop (`generate.cpp` lines 13422 and 13428). The header is written with the first row, so a run that never enters the loop leaves the file `fopen` created at line 6083 empty. The "logits dumped" line is printed whenever that `FILE*` is open (`generate.cpp` line 14171), including when no row was written.

The comment on the native-pack path, `src/program/generate.cpp` lines 13848-13850:

```cpp
// Diagnostics: the first window runs the prompt's last token over the state the prompt path left
// (keys, values, recurrent state), so its logits carry whatever that path did. A native pack never
// runs the per-token loop --dump-logits reads; this is where prompt paths can be compared by output.
```

The open of the dump file carries the same point, lines 6073-6074:

```cpp
// The logits header is written WITH THE FIRST ROW, not at open: a native pack never reaches the
// per-token dump site, and a header promising rows that were never written is worse than no file.
```

## 2. `--kv fp16` is printed, and K/V are stored as int8

Command, the serve that follows the dump in the same script. This is the process whose banner names the KV type.

```sh
env STRATA_METAL_STREAM=1 STRATA_ROUTE_TAIL_SKIP=0 STRATA_HC_Q8=0 STRATA_PREFILL_CPU_SHARE=0 \
  STRATA_METAL_PROMPT=window \
  STRATA_LOGPOS=/tmp/upstream-mac/logpos.tsv STRATA_LOGPOS_TOPK=256 \
  /usr/bin/time -l -o /tmp/upstream-mac/correct.time \
  ./build-metal/strata --serve \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile "$HOME/Developer/Strata-upstream/data/expert-profile.bin" \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --max-context 4096 --mmap-experts --kv fp16 --adapt-every 0 \
  --short-read 8192 --prompt-cache 0 \
  < /tmp/upstream-mac/requests.txt
```

The serve banner (`/tmp/upstream-mac/correct.out`) printed:

```
INFO logit_bias=1 context=4096 kv=fp16 kv_resident=0 expert_slots=64 expert_cache_mib=84 expert_slots_primary=64 expert_cache_primary_mib=84 spec=2 mtp_max=0 lookup=0 vram_free_mib=21277 cvec=0 arena_mib=0 pool_workers=17 pcie_frac=0.55 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 conversation_cache_min_tokens=0 tail_role_token=-1 vram_elastic=0 engine=0.1.44.1-metal
```

`kv=fp16` is the flag string (`generate.cpp` line 9596 prints `o.kv`). The Metal QSA kernel stores K and V as int8 with an fp16 scale per 64 values on every append. At `296b59d`, `metal/kernels/qsa.metal` lines 135-147, inside `qsa_prep`. `kq` and `vq` are `device char*`. `ksc` and `vsc` are `device half*`. There is no read of `--kv` in this kernel:

```metal
// int8 with an fp16 scale per 64 values: groups = simdgroup pairs (64 threads)
float m = simd_max(fabs(x));
if (lane == 0) amax[sg] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
const int g = tid >> 6;
m = max(amax[2 * g], amax[2 * g + 1]);
const half sh = half(m / 127.0f);
const float sf = float(sh);
int q = 0;
if (sf > 0.0f) q = clamp(int(rint(x / sf)), -127, 127);
const uint rowi = cell * QKV + uint(h);
(isv ? vq : kq)[rowi * QD + tid] = char(q);
if ((tid & 63) == 0) (isv ? vsc : ksc)[rowi * 4 + g] = sh;
```

The file's contract comment states the same store, lines 6-7: `k rms_norm * k_norm, RoPE -> int8 KV append (fp16 scale per 64 = fp16(amax/127), rint, clamp);` and `v -> int8 append`. The cache layout comment at line 48 is `kq/vq [cell][kv head][256] int8, ks/vs [cell][kv head][4] half`.
