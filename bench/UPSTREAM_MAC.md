# Upstream Strata Metal on this Mac

Measured 2026-10-15. No StrataMetal engine change. The run used the experimental native Metal engine in upstream Strata, commit `296b59d2f73c8f04a805df29c2fa97655adace78` ("Merge f-1441-perf (Responses/Codex honours reasoning_effort; setup keeps the installed size on a re-run) into f-1441", 2026-10-14 01:51:12 +0200). Checkout: `/Users/kamp/Developer/Strata-upstream`. Version string: `0.1.44.1-metal`.

This machine is an Apple M5 Max, 18 CPUs, `hw.memsize` 38654705664 bytes (36 GiB). Setup prints "36 GB unified memory, macOS 27.0.1". The engine prints RAM 38.7 GB, which is the same byte count in decimal GB (38654705664 / 1e9 = 38.65). macOS 27.0.1, build 26A434. Instruments names the GPU M5 Max. The file cache was not purged.

## Install

Command, from `/Users/kamp/Developer/Strata-upstream`:

```sh
./setup.sh --yes --model Q2_0 --gguf-dir "$HOME/Developer/Strata-data/models/Q2_0" --no-start --no-browser
```

Exit 0. Log: `/tmp/upstream-mac/setup.log`. The server script `run-q2_0.sh` was written and not started.

Shards were already present. Setup printed "already downloaded" for both `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` and `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` and did not fetch another copy. The pack `/Users/kamp/Developer/Strata-data/packs/q2_0` (including `native_experts.txt` and `tokenizer/vocab.json`) is dated 2026-10-14 22:48:27. The experts file `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf.experts` is 33973878784 bytes, mtime 2026-10-14 22:49:02, which is before this setup (log mtime 2026-10-15 17:42:57). Setup did not print a rewrite of that file. No MTP draft layer was downloaded. The download step names only those two shards.

What setup chose:

| Setting | Value |
| --- | --- |
| Memory mode | Streamed. "36 GB of memory: no pack fits whole, so the engine streams the experts from the SSD." "Q2_0 needs ~46 GB of the 29 GB Metal keeps resident on this Mac." |
| Context | 32768 |
| MTP | Off. "no MTP draft layer (one token per step is faster there)." `strata-q2_0.json` has no `--mtp`. Serve info later printed `mtp_max=0`. |
| Experts file | Reused, not rebuilt. Streaming copy is the `.experts` file next to shard 1. |
| Arena | Not a single size. It follows free RAM. Cold speed run: 10.74 GB, 7715 slots. Warm speed runs: 9.44, 11.92, 8.73 GB. Correctness serve: 12.63 GB, 9070 slots. Dump attempt: 9.66 GB, 6934 slots. |
| Expert cache | 64 slots, about 0.08 GiB, plus the streaming arena. |
| I/O | 6 threads, read-ahead top 10, 2 buffers. |
| PLE | On. The engine reserves 0.5 GB for the n-gram table. It is not a resident copy of shard 2 (28800138432 bytes). |
| Vision | Not installed. |
| Config | `/Users/kamp/Developer/Strata-upstream/strata-q2_0.json`, `open_browser` false, env `STRATA_METAL_STREAM=1`. Args: `--pack packs/q2_0 --native <shard1> --ple-gguf <shard2> --expert-profile data/expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts`. |

The Metal binary was already built. Setup printed "built for this source" and did not rebuild. `build-metal/strata` mtime 2026-10-14 21:06:22. `build-metal/BUILD.json` says version 0.1.44.1, backend metal, source `7ecedefd170999c46d9a6776114f31b58021b94f0c956f766e857470fdd6d895`.

Every engine run also printed "PCIe probe failed -> pcie_frac default 0.55" and "this CPU has no AVX-512: the expert kernels run on ggml-cpu vec_dot (no AVX2: the older-CPU build)". The serve banner's pool is 17 workers. The generate summary's CPU-expert counters stayed at 0.

## Settings that could not be matched

The CUDA fixtures are greedy, `--spec 0`, fp16 KV, and a per-token prompt, with `STRATA_ROUTE_TAIL_SKIP=0` (`reference/PROVENANCE.md`). The native Metal pack cannot do several of those.

`--spec 0` is illegal. `src/program/generate.cpp` returns 2 for a native pack when `--spec` is below 2, when `--native` is missing, when the canonical path is kept, or when `--prefill` is off and the prompt has more than one token. The message is "is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK". The Metal default before any flag is `int spec = 2` (`generate.cpp`). `--spec 0` was not timed.

`--dump-logits` writes no rows for a native pack. The per-token loop breaks immediately (`generate.cpp`: `if (native_pack) { spec_pos = pos; break; }`). A p1 generate with `--dump-logits /tmp/upstream-mac/p1-dump.bin` exited 0, printed "logits dumped", and left a 0-byte file. `metal/tools/compare_fixtures.py` printed `/tmp/upstream-mac/p1-dump.bin: shorter than a header`.

The prompt is not one token at a time. Speed prefill is one matrix chunk. The engine printed "prompts on the matrix units in chunks of 1024" and then "prefill 306 tokens in 1 chunks". Correctness scoring used verify windows of width `--spec 2`, not width 1. `--short-read 8192` kept the 339-token p5 serve read on that window path. `STRATA_LOGPOS` is what produced a row per position.

`--kv fp16` was passed, and the serve banner printed `kv=fp16`. The Metal QSA kernel still appends int8 KV with an fp16 scale per 64 (`metal/kernels/qsa.metal`). The flag does not select an fp16 KV store.

Speed left the default Q8_0 hyper-connection path on ("HC read in Q8_0") and left prefill CPU share at its default ON. The same prefill stats line then said "on the CPU 0 (share 0.00)", so the flag was on and this prompt's CPU-expert counter stayed 0. Correctness set `STRATA_HC_Q8=0` and `STRATA_PREFILL_CPU_SHARE=0`. The correctness log has no Q8_0 line and no CPU-share line.

Route-tail skip is off in these generate-mode speed runs. The serve default that can turn rank 7 on was not measured. Correctness set `STRATA_ROUTE_TAIL_SKIP=0`.

Context was 32768 for speed and 4096 for the correctness serve. MTP stayed off, which is the setup default. With MTP off, `--spec 2` still runs a window of 2. The speed runs accepted 0 of 199 drafts and emitted 1.00 tokens per round.

## Correctness

`compare_fixtures.py` could not score the empty dump. The rows below come from the serve path's `STRATA_LOGPOS` file, scored with the same pick rule and the same p5 prompt-NLL rule. Spread was not run, because there is no one-ulp dump. Raw top-64 `|d|` and `|lse|` are absent, because LOGPOS writes log-probabilities (`row[id] - lse` over the full vocab) and not raw logits. Total variation was computed: every CUDA top-64 id landed in the logged top 256.

Command, from `/Users/kamp/Developer/Strata-upstream`. The p1 dump attempt ran first, then this serve. Common flags: `--pack <q2_0> --native <shard1> --ple-gguf <shard2> --expert-profile <repo>/data/expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 4096 --mmap-experts --kv fp16 --adapt-every 0`.

```sh
env STRATA_METAL_STREAM=1 STRATA_ROUTE_TAIL_SKIP=0 STRATA_HC_Q8=0 STRATA_PREFILL_CPU_SHARE=0 \
  STRATA_METAL_PROMPT=window \
  STRATA_LOGPOS=/tmp/upstream-mac/logpos.tsv STRATA_LOGPOS_TOPK=256 \
  /usr/bin/time -l -o /tmp/upstream-mac/correct.time \
  ./build-metal/strata --serve <common flags> --short-read 8192 --prompt-cache 0 \
  < /tmp/upstream-mac/requests.txt
```

`requests.txt` is five lines, `GEN 1 ckpt=0 <space-separated ids>`, then `QUIT`. Each id list is the fixture prompt inputs, the later generation inputs, and the final fixture `next_token`. Counts: p1 prompt 5, sequence 37, scored 36; p2 11 / 43 / 42; p3 4 / 36 / 35; p4 10 / 42 / 41; p5 307 / 339 / 338.

Serve banner: `context=4096 kv=fp16 spec=2 mtp_max=0 pool_workers=17 pcie_frac=0.55 engine=0.1.44.1-metal`. Arena 12.63 GB, 9070 slots. Planned footprint 21.4 GB = arena 12.6 + 8.3 held + 0.5 n-gram reserve, cap 30.2 GB. `time -l`: wall 18.39 s, max RSS 15.656 GiB, peak footprint 18.737 GB. Those RSS figures include loading and the LOGPOS path. They are not the speed numbers. The serve PP rates are the logpos path: p5 scored 338 tokens in 7216 ms (46.8 tok/s on the `PP` line, 47.0 on the serve line).

Pick rule, unchanged: exact argmax, or the Mac id is in CUDA's top 3 and CUDA's gap from its own #1 to that id is at most 0.5. p5 prompt NLL uses the 293 prompt rows whose next prompt token is in CUDA's top 64. The Mac mean must be within 1% of CUDA and not more than 1% worse.

492 positions. 479 exact. 13 accepted. 0 pick failures. Criterion 1 passes. p5 prompt NLL: CUDA mean 1.593251, Mac mean 1.590797, relative 0.001540. Mac lower on 143 rows, CUDA lower on 150, no ties. Both NLL checks pass. Spread was not checked.

| Prompt | Rows | Exact | Accepted | Pick failures | TV max | TV mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p1 | 36 | 36 | 0 | 0 | 0.040581 | 0.007212 |
| p2 | 42 | 41 | 1 | 0 | 0.091655 | 0.011917 |
| p3 | 35 | 34 | 1 | 0 | 0.039756 | 0.005906 |
| p4 | 41 | 40 | 1 | 0 | 0.028991 | 0.003960 |
| p5 | 338 | 328 | 10 | 0 | 0.153649 | 0.029032 |

Accepted mismatches. Rank is the Mac id's place in CUDA's top 64. Gap is CUDA's gap from its #1 to that id.

| Prompt | Pos | CUDA | Mac | Rank | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| p2 | 36 | 321 | 25899 | 2 | 0.192181 |
| p3 | 17 | 460 | 745 | 2 | 0.059782 |
| p4 | 7 | 40297 | 12050 | 2 | 0.047735 |
| p5 | 17 | 7338 | 5898 | 3 | 0.190884 |
| p5 | 26 | 39332 | 18237 | 2 | 0.083109 |
| p5 | 56 | 321 | 1024 | 2 | 0.086435 |
| p5 | 69 | 25374 | 9506 | 2 | 0.209074 |
| p5 | 168 | 995 | 557 | 2 | 0.025885 |
| p5 | 174 | 279 | 1141 | 2 | 0.027206 |
| p5 | 187 | 310 | 11 | 2 | 0.040298 |
| p5 | 207 | 3043 | 539 | 2 | 0.033726 |
| p5 | 299 | 5820 | 5924 | 2 | 0.002113 |
| p5 | 325 | 18641 | 51067 | 2 | 0.189552 |

The widest gap is 0.209 at p5 position 69. p3 position 17 is CUDA 460 against Mac 745, the same ids as StrataMetal's free-run divergence. The thresholds were not changed.

The per-row tables follow. `logprob` is the Mac log-probability of CUDA's next token. `tv` is total variation over CUDA's top 64, using the logged top 256. `gap` is 0 on an exact match.

```
spread: not checked, this measurement has no perturbed dump
max|d| and |lse|: not checked, STRATA_LOGPOS does not write raw logits

=== p1 ===
positions scored 36, fixture positions absent 0
  pos    input     cuda      mac  rank        gap    logprob         tv  note
    0      760      220      220     1     0.0000     0.0175     0.0076  match
    1     6511      314      314     1     0.0000    -0.0020     0.0123  match
    2      314      279      279     1     0.0000    -0.0426     0.0151  match
    3     9338      369      369     1     0.0000    -0.0163     0.0130  match
    4      369    11751    11751     1     0.0000     0.0375     0.0258  match
    5    11751       13       13     1     0.0000     0.0235     0.0162  match
    6       13   248046   248046     1     0.0000     0.0441     0.0181  match
    7   248046      198      198     1     0.0000     0.0000     0.0000  match
    8      198   248045   248045     1     0.0000     0.0000     0.0000  match
    9   248045    74455    74455     1     0.0000     0.0000     0.0000  match
   10    74455      198      198     1     0.0000    -0.0012     0.0012  match
   11      198   248068   248068     1     0.0000     0.0000     0.0000  match
   12   248068      198      198     1     0.0000     0.0144     0.0114  match
   13      198      760      760     1     0.0000     0.0000     0.0000  match
   14      760     1156     1156     1     0.0000     0.0000     0.0000  match
   15     1156      682      682     1     0.0000     0.0081     0.0078  match
   16      682    10661    10661     1     0.0000     0.0112     0.0088  match
   17    10661      264      264     1     0.0000    -0.0000     0.0000  match
   18      264     4145     4145     1     0.0000    -0.0308     0.0214  match
   19     4145    57879    57879     1     0.0000     0.0631     0.0406  match
   20    57879     4952     4952     1     0.0000     0.0029     0.0029  match
   21     4952       25       25     1     0.0000     0.0000     0.0000  match
   22       25      328      328     1     0.0000     0.0000     0.0000  match
   23      328      760      760     1     0.0000    -0.0000     0.0000  match
   24      760     6511     6511     1     0.0000    -0.0000     0.0000  match
   25     6511      314      314     1     0.0000     0.0000     0.0000  match
   26      314     9338     9338     1     0.0000    -0.0000     0.0000  match
   27     9338      369      369     1     0.0000    -0.0000     0.0000  match
   28      369    11751    11751     1     0.0000    -0.0000     0.0000  match
   29    11751     1149     1149     1     0.0000    -0.0000     0.0000  match
   30     1149     1061     1061     1     0.0000    -0.0000     0.0000  match
   31     1061      369      369     1     0.0000    -0.0000     0.0000  match
   32      369      264      264     1     0.0000     0.0276     0.0196  match
   33      264     4252     4252     1     0.0000     0.0033     0.0170  match
   34     4252     4952     4952     1     0.0000    -0.0206     0.0204  match
   35     4952       13       13     1     0.0000     0.0003     0.0004  match
summary: 36 positions, 36 exact, 0 accepted mismatches, 0 pick failures
info top-64 |d| and |lse|: not available (LOGPOS has log-probabilities, not raw logits)
info TV over CUDA top-64: 36 of 36 rows had every CUDA id in the logged top-k; max 0.040581 mean 0.007212
accepted mismatches: none
pick failures: none

=== p2 ===
positions scored 42, fixture positions absent 0
  pos    input     cuda      mac  rank        gap    logprob         tv  note
    0      814    20139    20139     1     0.0000    -0.0002     0.0007  match
    1    20139      279      279     1     0.0000    -0.0185     0.0094  match
    2      303     7472     7472     1     0.0000     0.0296     0.0173  match
    3     1330    22157    22157     1     0.0000    -0.0224     0.0139  match
    4    22157     3069     3069     1     0.0000     0.0308     0.0199  match
    5     3069      279      279     1     0.0000    -0.0006     0.0083  match
    6      279    12515    12515     1     0.0000     0.0716     0.0114  match
    7    12515      369      369     1     0.0000     0.0217     0.0168  match
    8      369     6105     6105     1     0.0000    -0.0003     0.0006  match
    9     6105       13       13     1     0.0000    -0.0140     0.0126  match
   10       13      271      271     1     0.0000    -0.0010     0.0020  match
   11      271   248068   248068     1     0.0000    -0.0090     0.0087  match
   12   248068      198      198     1     0.0000     0.0044     0.0039  match
   13      198      760      760     1     0.0000    -0.0000     0.0000  match
   14      760     1156     1156     1     0.0000    -0.0000     0.0000  match
   15     1156     6587     6587     1     0.0000     0.0360     0.0268  match
   16     6587      264      264     1     0.0000    -0.0000     0.0000  match
   17      264    61446    61446     1     0.0000     0.0482     0.0243  match
   18    61446     1330     1330     1     0.0000     0.0342     0.0240  match
   19     1330     1284     1284     1     0.0000    -0.0000     0.0000  match
   20     1284    17834    17834     1     0.0000     0.0000     0.0000  match
   21    17834    15673    15673     1     0.0000    -0.0000     0.0000  match
   22    15673      314      314     1     0.0000     0.0000     0.0000  match
   23      314     3069     3069     1     0.0000     0.0000     0.0000  match
   24     3069      279      279     1     0.0000     0.0000     0.0000  match
   25      279    12515    12515     1     0.0000    -0.0000     0.0000  match
   26    12515      369      369     1     0.0000    -0.0009     0.0009  match
   27      369     6105     6105     1     0.0000     0.0000     0.0000  match
   28     6105       13       13     1     0.0000     0.0000     0.0000  match
   29       13      353      353     1     0.0000    -0.0097     0.0094  match
   30      353     1144     1144     1     0.0000     0.0048     0.0046  match
   31     1144      310      310     1     0.0000     0.0000     0.0000  match
   32      310     3311     3311     1     0.0000     0.0056     0.0050  match
   33     3311    12870    12870     1     0.0000     0.0039     0.0021  match
   34    12870    60845    60845     1     0.0000    -0.0000     0.0000  match
   35    60845    69377    69377     1     0.0000    -0.0000     0.0000  match
   36    69377      321    25899     2     0.1922    -0.2358     0.0660  accepted
   37      321      279      279     1     0.0000     0.0244     0.0199  match
   38      279    43796    43796     1     0.0000    -0.0526     0.0357  match
   39    43796    41863    41863     1     0.0000    -0.0912     0.0467  match
   40    41863      303      303     1     0.0000    -0.0349     0.0177  match
   41      303     1066     1066     1     0.0000    -0.2084     0.0917  match
summary: 42 positions, 41 exact, 1 accepted mismatches, 0 pick failures
info top-64 |d| and |lse|: not available (LOGPOS has log-probabilities, not raw logits)
info TV over CUDA top-64: 42 of 42 rows had every CUDA id in the logged top-k; max 0.091655 mean 0.011917
accepted mismatches: 1
  p2 pos 36 cuda 321 mac 25899 rank 2 gap 0.192181
pick failures: none

=== p3 ===
positions scored 35, fixture positions absent 0
  pos    input     cuda      mac  rank        gap    logprob         tv  note
    0      727     6093     6093     1     0.0000    -0.0390     0.0107  match
    1    73111     1393     1393     1     0.0000    -0.0047     0.0064  match
    2     1393     1590     1590     1     0.0000    -0.0034     0.0032  match
    3     1590      198      198     1     0.0000     0.0002     0.0007  match
    4      198      262      262     1     0.0000    -0.0013     0.0012  match
    5      262      413      413     1     0.0000     0.0177     0.0120  match
    6      413      307      307     1     0.0000    -0.0015     0.0015  match
    7      307     2564     2564     1     0.0000     0.0015     0.0048  match
    8     2564      220      220     1     0.0000     0.0011     0.0014  match
    9      220       16       16     1     0.0000    -0.0365     0.0231  match
   10       16       25       25     1     0.0000    -0.0002     0.0003  match
   11       25      198      198     1     0.0000     0.0024     0.0027  match
   12      198      285      285     1     0.0000    -0.0000     0.0000  match
   13      285      460      460     1     0.0000     0.0000     0.0001  match
   14      460      307      307     1     0.0000    -0.0024     0.0023  match
   15      307      198      198     1     0.0000    -0.0082     0.0067  match
   16      198      262      262     1     0.0000    -0.0000     0.0000  match
   17      262      460      745     2     0.0598    -0.0726     0.0331  accepted
   18      460    73111    73111     1     0.0000    -0.0018     0.0018  match
   19    73111     1393     1393     1     0.0000    -0.0001     0.0001  match
   20     1393       12       12     1     0.0000     0.0065     0.0038  match
   21       12       16       16     1     0.0000    -0.0009     0.0009  match
   22       16        8        8     1     0.0000    -0.0019     0.0019  match
   23        8      478      478     1     0.0000     0.0001     0.0001  match
   24      478    73111    73111     1     0.0000     0.0002     0.0002  match
   25    73111     1393     1393     1     0.0000     0.0001     0.0001  match
   26     1393       12       12     1     0.0000     0.0006     0.0006  match
   27       12       17       17     1     0.0000     0.0000     0.0001  match
   28       17        8        8     1     0.0000    -0.0000     0.0000  match
   29        8      271      271     1     0.0000     0.0123     0.0096  match
   30      271      727      727     1     0.0000    -0.0741     0.0398  match
   31      727     1822     1822     1     0.0000     0.1025     0.0363  match
   32     1822     4406     4406     1     0.0000    -0.0008     0.0009  match
   33     4406      198      198     1     0.0000     0.0004     0.0004  match
   34      198      262      262     1     0.0000    -0.0000     0.0001  match
summary: 35 positions, 34 exact, 1 accepted mismatches, 0 pick failures
info top-64 |d| and |lse|: not available (LOGPOS has log-probabilities, not raw logits)
info TV over CUDA top-64: 35 of 35 rows had every CUDA id in the logged top-k; max 0.039756 mean 0.005906
accepted mismatches: 1
  p3 pos 17 cuda 460 mac 745 rank 2 gap 0.059782
pick failures: none

=== p4 ===
positions scored 41, fixture positions absent 0
  pos    input     cuda      mac  rank        gap    logprob         tv  note
    0     7734      264      264     1     0.0000    -0.0050     0.0043  match
    1      264     3255     3255     1     0.0000     0.0043     0.0112  match
    2     6185    36974    36974     1     0.0000     0.0004     0.0003  match
    3    36974      883      883     1     0.0000     0.0052     0.0110  match
    4      883      264      264     1     0.0000     0.0013     0.0194  match
    5      264     7993     7993     1     0.0000     0.1787     0.0290  match
    6    14367      271      271     1     0.0000    -0.0578     0.0271  match
    7      303    40297    12050     2     0.0477    -0.0291     0.0172  accepted
    8    40297       13       13     1     0.0000     0.0386     0.0224  match
    9       13      271      271     1     0.0000    -0.0035     0.0034  match
   10      271   248068   248068     1     0.0000     0.0010     0.0012  match
   11   248068      198      198     1     0.0000     0.0004     0.0004  match
   12      198      760      760     1     0.0000     0.0000     0.0000  match
   13      760     1156     1156     1     0.0000    -0.0000     0.0000  match
   14     1156     6587     6587     1     0.0000    -0.0009     0.0009  match
   15     6587      264      264     1     0.0000    -0.0000     0.0000  match
   16      264     6185     6185     1     0.0000    -0.0000     0.0000  match
   17     6185    36974    36974     1     0.0000    -0.0000     0.0000  match
   18    36974      883      883     1     0.0000    -0.0001     0.0001  match
   19      883      264      264     1     0.0000     0.0000     0.0000  match
   20      264    14367    14367     1     0.0000    -0.0000     0.0000  match
   21    14367      303      303     1     0.0000     0.0000     0.0000  match
   22      303    40297    40297     1     0.0000     0.0000     0.0000  match
   23    40297       13       13     1     0.0000    -0.0000     0.0000  match
   24       13      357      357     1     0.0000    -0.0011     0.0011  match
   25      357     6185     6185     1     0.0000     0.0000     0.0000  match
   26     6185    36974    36974     1     0.0000    -0.0000     0.0000  match
   27    36974    10695    10695     1     0.0000     0.0015     0.0017  match
   28    10695      279      279     1     0.0000    -0.0000     0.0000  match
   29      279      220      220     1     0.0000    -0.0010     0.0011  match
   30      220       20       20     1     0.0000    -0.0000     0.0000  match
   31       20       12       12     1     0.0000    -0.0000     0.0000  match
   32       12       22       22     1     0.0000     0.0000     0.0000  match
   33       22       12       12     1     0.0000     0.0000     0.0000  match
   34       12       20       20     1     0.0000    -0.0000     0.0000  match
   35       20    46018    46018     1     0.0000     0.0000     0.0000  match
   36    46018      470      470     1     0.0000    -0.0000     0.0000  match
   37      470     5759     5759     1     0.0000    -0.0107     0.0101  match
   38     5759       13       13     1     0.0000    -0.0002     0.0002  match
   39       13     6558     6558     1     0.0000     0.0003     0.0003  match
   40     6558      728      728     1     0.0000     0.0000     0.0000  match
summary: 41 positions, 40 exact, 1 accepted mismatches, 0 pick failures
info top-64 |d| and |lse|: not available (LOGPOS has log-probabilities, not raw logits)
info TV over CUDA top-64: 41 of 41 rows had every CUDA id in the logged top-k; max 0.028991 mean 0.003960
accepted mismatches: 1
  p4 pos 7 cuda 40297 mac 12050 rank 2 gap 0.047735
pick failures: none

=== p5 ===
positions scored 338, fixture positions absent 0
  pos    input     cuda      mac  rank        gap    logprob         tv  note
    0      760      220      220     1     0.0000     0.0175     0.0076  match
    1    14367      369      369     1     0.0000     0.0507     0.0186  match
    2     6092      314      314     1     0.0000    -0.0191     0.0124  match
    3    37444      685      685     1     0.0000    -0.0402     0.0319  match
    4     1518    37119    37119     1     0.0000    -0.0549     0.0355  match
    5      279     6797     6797     1     0.0000     0.0601     0.0584  match
    6    18866     1467     1467     1     0.0000     0.0269     0.0391  match
    7     1467       13       13     1     0.0000     0.0112     0.0105  match
    8       13      198      198     1     0.0000     0.0023     0.0341  match
    9     2332    37119    37119     1     0.0000     0.0183     0.0219  match
   10      279     6607     6607     1     0.0000    -0.0189     0.0372  match
   11      809      314      314     1     0.0000     0.0008     0.0008  match
   12      314      279      279     1     0.0000     0.0404     0.0365  match
   13      279    21268    21268     1     0.0000    -0.1532     0.0508  match
   14     9506    21268    21268     1     0.0000    -0.1335     0.0708  match
   15    13759       11       11     1     0.0000    -0.0508     0.0420  match
   16      264     5054     5054     1     0.0000     0.1975     0.0384  match
   17     5054     7338     5898     3     0.1909    -0.0855     0.0593  accepted
   18      303      264      264     1     0.0000    -0.0067     0.0092  match
   19      264     6105     6105     1     0.0000    -0.0879     0.0390  match
   20    19177    22162    22162     1     0.0000    -0.0763     0.0354  match
   21    22162    14217    14217     1     0.0000    -0.0159     0.0356  match
   22      716      680      680     1     0.0000     0.0337     0.0530  match
   23      264    43254    43254     1     0.0000    -0.1872     0.0530  match
   24    21664    17189    17189     1     0.0000     0.0099     0.0406  match
   25    17189      383      383     1     0.0000    -0.0265     0.0197  match
   26      314    39332    18237     2     0.0831    -0.1592     0.0386  accepted
   27    39332      383      383     1     0.0000    -0.0144     0.0165  match
   28      383      279      279     1     0.0000    -0.0131     0.0115  match
   29      279    32709    32709     1     0.0000    -0.0966     0.0387  match
   30     6784       11       11     1     0.0000    -0.0337     0.0215  match
   31      321    28260    28260     1     0.0000    -0.0961     0.0621  match
   32    14936      279      279     1     0.0000    -0.0188     0.0159  match
   33      279     3000     3000     1     0.0000    -0.1593     0.0677  match
   34     2919     2404     2404     1     0.0000     0.0678     0.0459  match
   35     3166       13       13     1     0.0000    -0.0721     0.0447  match
   36     1172      279      279     1     0.0000    -0.0261     0.0244  match
   37      279    13759    13759     1     0.0000     0.1290     0.0556  match
   38     5157       13       13     1     0.0000     0.0058     0.0164  match
   39      287       13       13     1     0.0000    -0.0050     0.0107  match
   40       13      357      357     1     0.0000     0.2581     0.0975  match
   41     2838      995      995     1     0.0000    -0.1665     0.0684  match
   42      995     2445     2445     1     0.0000    -0.0134     0.0624  match
   43    14428      411      411     1     0.0000    -0.0555     0.0464  match
   44      411     5832     5832     1     0.0000     0.0006     0.0315  match
   45     1560     1396     1396     1     0.0000    -0.0115     0.0264  match
   46     1396     6353     6353     1     0.0000    -0.0187     0.0160  match
   47     6353      364      364     1     0.0000     0.0085     0.0076  match
   48      364    25292    25292     1     0.0000    -0.1231     0.0474  match
   49     1578       11       11     1     0.0000    -0.0251     0.0159  match
   50       13      561      561     1     0.0000    -0.0132     0.0455  match
   51      357     7914     7914     1     0.0000     0.1278     0.0518  match
   52     7914      440      440     1     0.0000     0.0680     0.0458  match
   53      383      264      264     1     0.0000    -0.0232     0.0208  match
   54      264    33828    33828     1     0.0000     0.0111     0.0123  match
   55    33828     5642     5642     1     0.0000     0.1070     0.0484  match
   56     5642      321     1024     2     0.0864    -0.1242     0.0668  accepted
   57     1024       11       11     1     0.0000    -0.0980     0.0447  match
   58       11     1179     1179     1     0.0000     0.0140     0.0206  match
   59    64779      780      780     1     0.0000     0.0291     0.0105  match
   60      780    27491    27491     1     0.0000     0.0000     0.0004  match
   61    27491    10598    10598     1     0.0000     0.1375     0.0630  match
   62    10598       11       11     1     0.0000     0.0065     0.0121  match
   63       11      321      321     1     0.0000     0.0120     0.0125  match
   64      321     1292     1292     1     0.0000    -0.0455     0.0343  match
   65      279    39332    39332     1     0.0000     0.1436     0.0566  match
   66     4950    11354    11354     1     0.0000     0.0826     0.0313  match
   67    67441      974      974     1     0.0000     0.0130     0.0232  match
   68      974      279      279     1     0.0000     0.0020     0.0021  match
   69      279    25374     9506     2     0.2091    -0.3472     0.1124  accepted
   70    79338       13       13     1     0.0000     0.0435     0.0373  match
   71      321     3571     3571     1     0.0000     0.0067     0.0188  match
   72     3571     1142     1142     1     0.0000    -0.0001     0.0002  match
   73     1142      310      310     1     0.0000    -0.0113     0.0724  match
   74    62086     1056     1056     1     0.0000    -0.0718     0.0471  match
   75       13      271      271     1     0.0000     0.0248     0.0431  match
   76      469    68042    68042     1     0.0000    -0.1038     0.0746  match
   77    68042     4086     4086     1     0.0000    -0.0808     0.0480  match
   78    14217      383      383     1     0.0000     0.0471     0.0314  match
   79      383      279      279     1     0.0000     0.0007     0.0045  match
   80      279    13176    13176     1     0.0000    -0.0120     0.0425  match
   81    11130      793      793     1     0.0000    -0.0000     0.0000  match
   82      793       11       11     1     0.0000     0.0063     0.0328  match
   83      430      413      413     1     0.0000    -0.0319     0.0279  match
   84      413      781      781     1     0.0000    -0.0219     0.0147  match
   85      781      995      995     1     0.0000    -0.0574     0.0363  match
   86      995      978      978     1     0.0000    -0.0414     0.0450  match
   87      978     8318     8318     1     0.0000    -0.0502     0.0532  match
   88     8194     1017     1017     1     0.0000    -0.0002     0.0003  match
   89     1017       13       13     1     0.0000     0.1380     0.0655  match
   90      310     2426     2426     1     0.0000     0.0568     0.0436  match
   91     1698      279      279     1     0.0000     0.0031     0.0189  match
   92      279    30592    30592     1     0.0000    -0.0335     0.0476  match
   93    30592       13       13     1     0.0000     0.0040     0.0046  match
   94       13      271      271     1     0.0000     0.0965     0.0452  match
   95      271      760      760     1     0.0000     0.0289     0.0334  match
   96     1311      279      279     1     0.0000     0.0942     0.0519  match
   97      279      854      854     1     0.0000     0.0058     0.0090  match
   98      854      279      279     1     0.0000    -0.0008     0.0029  match
   99     1292     8379     8379     1     0.0000     0.0002     0.0083  match
  100     8379      279      279     1     0.0000     0.0100     0.0098  match
  101      279     2981     2981     1     0.0000     0.0310     0.0418  match
  102     2981       11       11     1     0.0000    -0.0577     0.0310  match
  103     9057       11       11     1     0.0000    -0.1111     0.0667  match
  104      279     3000     3000     1     0.0000    -0.0341     0.0377  match
  105     3000      995      995     1     0.0000    -0.0128     0.0103  match
  106      995     5324     5324     1     0.0000    -0.0092     0.0288  match
  107    42942      279      279     1     0.0000     0.0880     0.0468  match
  108      279     8551     8551     1     0.0000     0.0774     0.0606  match
  109    73685       13       13     1     0.0000    -0.0038     0.0135  match
  110       13      561      561     1     0.0000     0.1520     0.0647  match
  111      561     9057     9057     1     0.0000    -0.0178     0.0431  match
  112    73237      725      725     1     0.0000     0.0567     0.0316  match
  113      557     2582     2582     1     0.0000    -0.0047     0.0171  match
  114     2582    21575    21575     1     0.0000    -0.0220     0.0334  match
  115    32713    88562    88562     1     0.0000     0.0576     0.0312  match
  116    88562      314      314     1     0.0000    -0.0309     0.0183  match
  117     8366      279      279     1     0.0000     0.0422     0.0300  match
  118      279     5373     5373     1     0.0000    -0.0010     0.0080  match
  119     5373       11       11     1     0.0000    -0.1056     0.0559  match
  120       11      321      321     1     0.0000    -0.0137     0.0126  match
  121      321      279      279     1     0.0000    -0.0010     0.0045  match
  122      279    21388    21388     1     0.0000    -0.0045     0.0051  match
  123    21388      314      314     1     0.0000    -0.0270     0.0237  match
  124      314    15530    15530     1     0.0000     0.1130     0.0566  match
  125     7948    15530    15530     1     0.0000    -0.0046     0.0058  match
  126    15530     7481     7481     1     0.0000    -0.0528     0.0312  match
  127     2197      279      279     1     0.0000     0.0028     0.0033  match
  128      279    21388    21388     1     0.0000     0.0087     0.0097  match
  129    72555    21388    21388     1     0.0000    -0.0331     0.0248  match
  130    21388      314      314     1     0.0000    -0.0274     0.0250  match
  131      314      279      279     1     0.0000     0.0074     0.0160  match
  132      279    14367    14367     1     0.0000    -0.0022     0.0039  match
  133    14367       13       13     1     0.0000    -0.0018     0.0055  match
  134       13      357      357     1     0.0000    -0.0219     0.0325  match
  135     8771     2859     2859     1     0.0000     0.0057     0.0226  match
  136    70704    17736    17736     1     0.0000    -0.0759     0.0536  match
  137    17736      883      883     1     0.0000     0.0090     0.0499  match
  138       11      524      524     1     0.0000    -0.0879     0.0544  match
  139     1973    17987    17987     1     0.0000     0.0103     0.0259  match
  140    18654       11       11     1     0.0000    -0.0002     0.0005  match
  141       11      883      883     1     0.0000     0.0185     0.0149  match
  142      883     3315     3315     1     0.0000     0.0055     0.0109  match
  143     3315      279      279     1     0.0000    -0.0243     0.0219  match
  144      279    40554    40554     1     0.0000    -0.0259     0.0194  match
  145    40554      995      995     1     0.0000     0.0210     0.0351  match
  146     1000     2404     2404     1     0.0000    -0.1250     0.0689  match
  147     2404     1518     1518     1     0.0000    -0.0986     0.0602  match
  148     1518    35915    35915     1     0.0000    -0.0514     0.0440  match
  149    35915       13       13     1     0.0000    -0.0006     0.0008  match
  150       13      357      357     1     0.0000     0.0114     0.0144  match
  151      561     5054     5054     1     0.0000    -0.1498     0.0762  match
  152     5054    10471    10471     1     0.0000    -0.0217     0.0404  match
  153    10471      264      264     1     0.0000    -0.0555     0.0468  match
  154      264    73541    73541     1     0.0000     0.0166     0.0166  match
  155    73541       11       11     1     0.0000    -0.0270     0.0196  match
  156      321    59918    59918     1     0.0000     0.1552     0.0608  match
  157      264    28628    28628     1     0.0000    -0.0272     0.0407  match
  158     5392      314      314     1     0.0000     0.0043     0.0075  match
  159      314    11679    11679     1     0.0000     0.2042     0.0837  match
  160    10482       11       11     1     0.0000    -0.0670     0.0567  match
  161       13     2838     2838     1     0.0000     0.0324     0.0253  match
  162     2838     1467     1467     1     0.0000     0.0545     0.0523  match
  163     1467      524      524     1     0.0000    -0.0002     0.0003  match
  164      524    45659    45659     1     0.0000    -0.1373     0.0700  match
  165    45659       13       13     1     0.0000     0.0097     0.0110  match
  166       13      561      561     1     0.0000     0.0346     0.0419  match
  167      561     6092     6092     1     0.0000    -0.0401     0.0407  match
  168     1834      995      557     2     0.0259    -0.0695     0.0406  accepted
  169     8047      557      557     1     0.0000    -0.0954     0.0552  match
  170      557     1248     1248     1     0.0000     0.0029     0.0225  match
  171    18541       11       11     1     0.0000    -0.0355     0.0257  match
  172       11      321      321     1     0.0000     0.0030     0.0088  match
  173      864     8526     8526     1     0.0000     0.0840     0.0255  match
  174      557      279     1141     2     0.0272    -0.0657     0.0504  accepted
  175     1092     1292     1292     1     0.0000    -0.0331     0.0269  match
  176     1292     4674     4674     1     0.0000     0.0255     0.0219  match
  177     4674       13       13     1     0.0000     0.0101     0.0102  match
  178      494      424      424     1     0.0000     0.0027     0.0033  match
  179      424       13       13     1     0.0000     0.0011     0.0084  match
  180       25      264      264     1     0.0000     0.0166     0.0835  match
  181    11592       11       11     1     0.0000     0.1047     0.0619  match
  182      310     3165     3165     1     0.0000     0.0322     0.0577  match
  183     4087       11       11     1     0.0000    -0.0004     0.0007  match
  184       11      264      264     1     0.0000    -0.0236     0.0315  match
  185      264     3136     3136     1     0.0000    -0.1004     0.0445  match
  186    50083    14795    14795     1     0.0000     0.1042     0.0428  match
  187    14795      310       11     2     0.0403    -0.0264     0.0144  accepted
  188       11      264      264     1     0.0000    -0.0264     0.0234  match
  189      264     4099     4099     1     0.0000    -0.1647     0.0538  match
  190     3814      494      494     1     0.0000    -0.0893     0.0538  match
  191      310      279      279     1     0.0000     0.0049     0.0102  match
  192     1024    12553    12553     1     0.0000    -0.0177     0.0152  match
  193    10326      303      303     1     0.0000     0.0422     0.0266  match
  194      383      279      279     1     0.0000    -0.0782     0.0663  match
  195      279      975      975     1     0.0000    -0.0172     0.0170  match
  196    23198       13       13     1     0.0000    -0.0594     0.0320  match
  197       13      271      271     1     0.0000    -0.0165     0.0272  match
  198      271     1597     1597     1     0.0000    -0.0634     0.0407  match
  199     7691     7338     7338     1     0.0000    -0.0178     0.0348  match
  200     7338      383      383     1     0.0000    -0.0049     0.0236  match
  201      364      264      264     1     0.0000    -0.0057     0.0056  match
  202      264     4299     4299     1     0.0000    -0.0411     0.0281  match
  203     1345      383      383     1     0.0000     0.0153     0.0180  match
  204      383      279      279     1     0.0000    -0.0303     0.0226  match
  205      279    13038    13038     1     0.0000     0.0178     0.0160  match
  206     3238     6784     6784     1     0.0000    -0.0469     0.0288  match
  207     6784     3043      539     2     0.0337    -0.0581     0.0241  accepted
  208    28428      279      279     1     0.0000     0.0030     0.0030  match
  209      279    54718    54718     1     0.0000    -0.0237     0.0604  match
  210    33619     3626     3626     1     0.0000     0.0598     0.0383  match
  211     3626       13       13     1     0.0000    -0.0118     0.0139  match
  212      321    14936    14936     1     0.0000     0.0112     0.0152  match
  213    28838     4139     4139     1     0.0000     0.0404     0.0384  match
  214      264    15511    15511     1     0.0000    -0.0092     0.0105  match
  215     6365      314      314     1     0.0000     0.0000     0.0000  match
  216      314    22544    22544     1     0.0000     0.0586     0.0378  match
  217      279    15530    15530     1     0.0000     0.0414     0.0320  match
  218    15530       13       13     1     0.0000    -0.0027     0.0061  match
  219       13      357      357     1     0.0000    -0.0276     0.0294  match
  220      357      855      855     1     0.0000     0.1241     0.0409  match
  221      292     2657     2657     1     0.0000    -0.0002     0.0002  match
  222     2657     3571     3571     1     0.0000     0.0867     0.0333  match
  223     7481    13569    13569     1     0.0000    -0.0667     0.0311  match
  224    40352       11       11     1     0.0000    -0.0112     0.0110  match
  225      748    13569    13569     1     0.0000     0.0002     0.0003  match
  226    13569      421      421     1     0.0000    -0.0080     0.0071  match
  227      421      424      424     1     0.0000     0.0125     0.0226  match
  228      279     2919     2919     1     0.0000    -0.0466     0.0630  match
  229    14714     9047     9047     1     0.0000    -0.0288     0.0399  match
  230     9047      310      310     1     0.0000     0.0210     0.0312  match
  231      310      381      381     1     0.0000    -0.0625     0.0695  match
  232     1683      883      883     1     0.0000    -0.0503     0.0399  match
  233     1518      424      424     1     0.0000    -0.0028     0.0066  match
  234      424     8597     8597     1     0.0000    -0.0923     0.0765  match
  235     8597       13       13     1     0.0000    -0.0015     0.0019  match
  236       13      357      357     1     0.0000     0.0013     0.0215  match
  237    11721      303      303     1     0.0000    -0.0914     0.0568  match
  238    17957     6716     6716     1     0.0000    -0.0029     0.0064  match
  239      557    12125    12125     1     0.0000     0.0124     0.0117  match
  240    12125       11       11     1     0.0000    -0.0155     0.0145  match
  241       13     2844     2844     1     0.0000    -0.0058     0.0206  match
  242      561     6092     6092     1     0.0000    -0.2722     0.1536  match
  243     2919      557      557     1     0.0000     0.0428     0.0464  match
  244      557    19177    19177     1     0.0000     0.0300     0.0209  match
  245     6121      321      321     1     0.0000     0.0076     0.0198  match
  246       11      279      279     1     0.0000    -0.0062     0.0203  match
  247    23271       11       11     1     0.0000    -0.0056     0.0055  match
  248       11      321      321     1     0.0000    -0.0350     0.0326  match
  249      321     2400     2400     1     0.0000    -0.0539     0.0308  match
  250     2400      314      314     1     0.0000    -0.0032     0.0051  match
  251      314     2526     2526     1     0.0000    -0.0382     0.0326  match
  252     2526     2433     2433     1     0.0000    -0.0905     0.0527  match
  253    60637       13       13     1     0.0000    -0.0147     0.0174  match
  254      314      279      279     1     0.0000    -0.0645     0.0355  match
  255    10708       13       13     1     0.0000    -0.0311     0.0239  match
  256       13      271      271     1     0.0000    -0.0502     0.0338  match
  257     3095     1292     1292     1     0.0000    -0.0025     0.0036  match
  258     1292    14217    14217     1     0.0000     0.0251     0.0212  match
  259    14217      685      685     1     0.0000     0.0298     0.0186  match
  260      685       11       11     1     0.0000    -0.0633     0.0414  match
  261       11      279      279     1     0.0000    -0.0421     0.0294  match
  262    93240    10729    10729     1     0.0000    -0.0128     0.0203  match
  263    10729      494      494     1     0.0000     0.0196     0.0256  match
  264     8366      279      279     1     0.0000     0.0060     0.0058  match
  265      279    25374    25374     1     0.0000     0.0002     0.0159  match
  266    25374       11       11     1     0.0000    -0.0843     0.0373  match
  267      321      279      279     1     0.0000     0.0401     0.0431  match
  268      279      338      338     1     0.0000     0.0704     0.0544  match
  269      338    68042    68042     1     0.0000    -0.0001     0.0001  match
  270    68042     3738     3738     1     0.0000    -0.0643     0.0475  match
  271     3571    11784    11784     1     0.0000    -0.0666     0.0327  match
  272     1441      310      310     1     0.0000     0.0109     0.0141  match
  273      506     2957     2957     1     0.0000    -0.0021     0.0024  match
  274     2957       11       11     1     0.0000    -0.0061     0.0249  match
  275       13   248046   248046     1     0.0000    -0.1160     0.0432  match
  276     2838    59462    59462     1     0.0000     0.0077     0.0216  match
  277    45465      279      279     1     0.0000    -0.0014     0.0017  match
  278      279     5392     5392     1     0.0000     0.0041     0.0102  match
  279     5392       11       11     1     0.0000    -0.0092     0.0175  match
  280     2094      279      279     1     0.0000     0.0005     0.0043  match
  281      279     2645     2645     1     0.0000     0.0745     0.0470  match
  282     2645      314      314     1     0.0000    -0.0212     0.0169  match
  283      314      279      279     1     0.0000     0.0002     0.0003  match
  284      279    73541    73541     1     0.0000     0.0014     0.0011  match
  285    73541      321      321     1     0.0000    -0.0161     0.0097  match
  286      321    14428    14428     1     0.0000    -0.0283     0.0217  match
  287     3804     2046     2046     1     0.0000     0.0789     0.0416  match
  288      383       13       13     1     0.0000     0.0292     0.0215  match
  289     8574      279      279     1     0.0000     0.0032     0.0105  match
  290      279    23198    23198     1     0.0000    -0.0225     0.0650  match
  291    23198       11       11     1     0.0000    -0.0085     0.0071  match
  292       11     1332     1332     1     0.0000    -0.0565     0.0365  match
  293     1332      279      279     1     0.0000     0.0026     0.0056  match
  294      279    14533    14533     1     0.0000    -0.0463     0.0634  match
  295     5459      557      557     1     0.0000    -0.0053     0.0226  match
  296    70454      321      321     1     0.0000     0.0037     0.0091  match
  297      321      279      279     1     0.0000     0.0031     0.0037  match
  298      279    14533    14533     1     0.0000     0.0216     0.0265  match
  299     6092     5820     5924     2     0.0021    -0.1387     0.0332  accepted
  300    11639      270      270     1     0.0000     0.0568     0.0315  match
  301    10729     4661     4661     1     0.0000    -0.0424     0.0309  match
  302     3023       13       13     1     0.0000    -0.0347     0.0228  match
  303     4661     1024     1024     1     0.0000     0.0043     0.0040  match
  304     1024       13       13     1     0.0000    -0.0093     0.0111  match
  305       13   248046   248046     1     0.0000    -0.0432     0.0366  match
  306      198   248046   248046     1     0.0000    -0.1901     0.0511  match
  307   248046      198      198     1     0.0000     0.0000     0.0000  match
  308      198   248045   248045     1     0.0000     0.0000     0.0000  match
  309   248045    74455    74455     1     0.0000     0.0000     0.0000  match
  310    74455      198      198     1     0.0000     0.0041     0.0040  match
  311      198   248068   248068     1     0.0000     0.0000     0.0000  match
  312   248068      198      198     1     0.0000    -0.0213     0.0197  match
  313      198      760      760     1     0.0000     0.0000     0.0000  match
  314      760     1156     1156     1     0.0000     0.0000     0.0000  match
  315     1156      682      682     1     0.0000    -0.0000     0.0000  match
  316      682     5904     5904     1     0.0000     0.0004     0.0004  match
  317     5904      264      264     1     0.0000     0.0017     0.0017  match
  318      264     6365     6365     1     0.0000    -0.0623     0.0430  match
  319     6365      314      314     1     0.0000    -0.0000     0.0000  match
  320      314    30333    30333     1     0.0000     0.0326     0.0247  match
  321    30333    58655    58655     1     0.0000    -0.0016     0.0089  match
  322    58655      471      471     1     0.0000     0.1071     0.0449  match
  323      471      264      264     1     0.0000     0.0001     0.0001  match
  324      264     2716     2716     1     0.0000     0.0443     0.0323  match
  325     2716    18641    51067     2     0.1896    -0.1131     0.0527  accepted
  326    18641    20438    20438     1     0.0000    -0.0233     0.0172  match
  327    20438      883      883     1     0.0000    -0.0162     0.0159  match
  328      883      264      264     1     0.0000     0.0001     0.0001  match
  329      264     5054     5054     1     0.0000    -0.0144     0.0138  match
  330     5054      303      303     1     0.0000     0.1294     0.0837  match
  331      303      264      264     1     0.0000     0.0000     0.0000  match
  332      264    14367    14367     1     0.0000     0.0001     0.0001  match
  333    14367     6092     6092     1     0.0000    -0.0000     0.0000  match
  334     6092       13       13     1     0.0000     0.0981     0.0667  match
  335       13     2302     2302     1     0.0000     0.0053     0.0035  match
  336     2302     8719     8719     1     0.0000    -0.0000     0.0000  match
  337     8719      914      914     1     0.0000     0.0000     0.0000  match
summary: 338 positions, 328 exact, 10 accepted mismatches, 0 pick failures
info top-64 |d| and |lse|: not available (LOGPOS has log-probabilities, not raw logits)
info TV over CUDA top-64: 338 of 338 rows had every CUDA id in the logged top-k; max 0.153649 mean 0.029032
accepted mismatches: 10
  p5 pos 17 cuda 7338 mac 5898 rank 3 gap 0.190884
  p5 pos 26 cuda 39332 mac 18237 rank 2 gap 0.083109
  p5 pos 56 cuda 321 mac 1024 rank 2 gap 0.086435
  p5 pos 69 cuda 25374 mac 9506 rank 2 gap 0.209074
  p5 pos 168 cuda 995 mac 557 rank 2 gap 0.025885
  p5 pos 174 cuda 279 mac 1141 rank 2 gap 0.027206
  p5 pos 187 cuda 310 mac 11 rank 2 gap 0.040298
  p5 pos 207 cuda 3043 mac 539 rank 2 gap 0.033726
  p5 pos 299 cuda 5820 mac 5924 rank 2 gap 0.002113
  p5 pos 325 cuda 18641 mac 51067 rank 2 gap 0.189552
pick failures: none

=== p5 prompt NLL (293 rows whose next prompt token is in CUDA's top 64) ===
CUDA mean NLL 1.593251
Mac  mean NLL 1.590797
relative |Mac - CUDA| / CUDA = 0.001540 (limit 0.01)
Mac lower NLL 143, CUDA lower 150, ties 0
NLL within 1%: PASS
NLL not more than 1% worse: PASS

=== gate (picks and p5 NLL only; spread not run) ===
criterion 1 picks: PASS
criterion 2 p5 prompt NLL: PASS
criterion 3 spread: not checked
SCORE_EXIT:0
```

## Speed

Default Mac settings, which are already MTP off and `--spec 2`. The second requested point, `--spec 0` and no MTP, was not run: `--spec 0` is the refusal above, and MTP was already off.

One cold process, then three warm processes. Cold is the first process after setup. Nothing was purged, so the file cache may already have been warm from the 2026-10-14 pack build. Warm is the next three processes. The median is over those three.

Command, from `/Users/kamp/Developer/Strata-upstream`. The cold run and each warm run used this command. Env is only `STRATA_METAL_STREAM=1`. No `STRATA_HC_Q8`, no `STRATA_ROUTE_TAIL_SKIP`, no `STRATA_PREFILL_CPU_SHARE`.

```sh
/usr/bin/time -l -o TIMEFILE env STRATA_METAL_STREAM=1 ./build-metal/strata \
  --pack "$HOME/Developer/Strata-data/packs/q2_0" \
  --native "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --expert-profile "$HOME/Developer/Strata-upstream/data/expert-profile.bin" \
  --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
  --max-context 32768 --mmap-experts --kv fp16 \
  --tokens "$(cat /tmp/upstream-mac/p5.prompt)" --max-new 200
```

The prompt file is the 307 fixture prompt ids. Prefill reports 306 tokens (positions 0 through n-2). Decode reports the 200 generated tokens. All four processes exited 0. Drafts accepted 0 of 199, 1.00 tokens per round. GPU working set printed 30.2 GB on every run. That number is the engine's cap, not a measured allocation. Peak memory is `time -l`: maximum resident set size, and peak memory footprint. On this unified-memory Mac the footprint is the figure that matches the engine's "process holds" line. There is no separate GPU-only sampler.

| Run | Prompt tok/s | Prompt ms | Decode tok/s | Decode ms | TTFT ms | Wall s | Max RSS | Peak footprint | Arena |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 26.00 | 11768.4 | 21.46 | 9318.3 | 17535.0 | 28.24 | 10.948 GiB | 17.019 GB (15.850 GiB) | 10.74 GB, 7715 slots |
| Warm 1 | 82.33 | 3716.6 | 26.88 | 7440.9 | 4080.4 | 12.72 | 14.273 GiB | 16.615 GB | 9.44 GB, 6778 slots |
| Warm 2 | 77.47 | 3949.8 | 30.52 | 6553.2 | 4323.1 | 11.82 | 14.597 GiB | 18.168 GB | 11.92 GB, 8558 slots |
| Warm 3 | 102.87 | 2974.5 | 25.98 | 7699.2 | 3372.3 | 11.91 | 14.584 GiB | 17.313 GB | 8.73 GB, 6271 slots |
| Warm median | 82.33 | 3716.6 | 26.88 | 7440.9 | 4080.4 | 11.91 | 14.584 GiB | 17.313 GB | 9.44 GB |

`time -l` bytes: cold RSS 11754897408, footprint 17018653248. Warm RSS 15325167616, 15673507840, 15659794432. Warm footprint 16614555608, 18167860352, 17313317568. Page faults: cold 51602, warm 1885 / 18 / 16. Block input operations stayed 0, because the reads are mmap faults rather than the block-I/O counter.

The printed time to first token is the engine clock started in `generate.cpp` before the prompt walk. It is 5766.6 ms above the cold prefill line, and 363.8, 373.3, and 397.8 ms above the three warm prefill lines. Arena fill is outside that gap: warm 2 filled an 11.92 GB arena in 1.9 s while its TTFT sat only 0.37 s above prefill. Cold arena fill was 2.1 s at 5.20 GB/s. Cold planned footprint 17.3 GB = arena 10.7 + 6.1 held + 0.5 reserve. After the fill the engine said the process holds 17.0 GB, and `time -l` peak footprint is 17.019 GB.

Cold expert stream, 200 decode windows: hits 0.9388, read-ahead hits 0.0338 (2475 late), misses 0.0274 (23.9 per window), 200.4 MB read per window (28995 reads), host 10.56 ms per window of which waiting 9.85 ms. Stream windows T=2: 199 decode windows, 43.98 ms each, GPU timestamp 41.09 ms. The one T=1 window was 558.70 ms (GPU timestamp 82.03 ms). Prompt chunk: 9579 lookups, host 1154.1 ms of which waiting 1132.9 ms, while the generate prefill counter said "experts streamed 0 ... on the CPU 0".

Warm expert-stream host wait per decode window: 12.22, 8.20, 13.41 ms. Warm T=2 window means: 36.95, 32.54, 38.20 ms.

HC Q8_0 was built on each speed run (290 matrices, 680 MB). Startup also said prefill CPU share is on by default for chunks under 1024 tokens.

## Profile

One Instruments attach, plus the engine's own counters. This run is slower than the speed table because Instruments was attached. Do not read its tok/s as the speed result. Decode under the tracer: 120 tokens in 6126.2 ms, 19.59 tok/s. The prompt was p1 (5 ids) and `--max-new 120`. Drafts accepted 1 of 118, 1.01 tokens per round. The steady lines below are the T=2 windows, which the engine counted as 118.

Command:

```sh
xcrun xctrace record --template "Metal System Trace" \
  --output /tmp/upstream-mac/decode.trace --window 3s --no-prompt --time-limit 120s \
  --target-stdout /tmp/upstream-mac/xctrace-child.out \
  --env STRATA_METAL_STREAM=1 --env STRATA_METAL_PROFILE=1 \
  --env STRATA_METAL_TRACE=1 --env STRATA_METAL_KPROF=2 \
  --launch -- /tmp/upstream-mac/profile-run.sh
```

`profile-run.sh` execs `./build-metal/strata` with the same pack flags as the speed command, `--max-context 32768 --mmap-experts --kv fp16`, tokens from `p1.prompt`, `--max-new 120`. Trace exit 0. The saved trace spans 6.000 s. The template's recorded option is "Shader Timeline: Disabled". `--show-recording-options` for that template lists Hangs and Time Profiler only.

`STRATA_METAL_KPROF=2` printed nothing. `window.cpp` calls `kprof` only when `!stream_`. This Mac is streaming, so the engine's own per-kernel timer does not run.

Exported tables, all for this trace:

| Table | Rows | What it holds |
| --- | ---: | --- |
| `metal-shader-profiler-intervals` | 0 | Shader Timeline samples. Empty. |
| `gpu-shader-profiler-sample` | 0 | PC samples. Empty. |
| `gpu-shader-profiler-interval` | 0 | Schema only. |
| `metal-shader-profiler-shader-list` | names only | Compiled shaders and PC ranges, no durations. |
| `metal-gpu-intervals` | 1360 for pid 54527 | Compute-channel GPU Execution. No kernel name. Bytes column 0. |
| `metal-application-encoders-list` | encoders | Labels are "Compute Command N". CPU encode time, not GPU time. |

Per-kernel GPU time was not captured. There is no top-10 kernel table, and none is inferred from shader names. The shader list for pid 54527 does show the engine's compiled functions, untimed: `gemv_*`, `gdn_step`, `gr_up_q8`, `gr_down_q8_8_8`, `gr_norm`, `moe_route`, `moe_combine_sh`, `moe_down_*`, `moe_gu_*`, `qsa_attn`, `qsa_score`, `qsa_merge`, `qsa_topk2`, `ple_gnorm`, `ple_conv_add`, `ple_gate_norm`, `swiglu_prep`, `coop_layout_16_32_32`, `argmax_part`, `argmax_final`.

The GPU timestamp is not kernel time. `run_graph_seg` (`metal/src/mtl_rt.mm`) puts the layer encoders in one command buffer and, between segments, makes the GPU wait on a shared event while the host runs `ExpertStream::at_layer`. That wait includes the SSD read. `GPUEndTime - GPUStartTime` therefore contains the stalls. The profile line's "GPU" column is that timestamp.

Engine profile, 118 steady T=2 windows, mean 51.028 ms:

| Part | ms | Share of the 51.028 ms window |
| --- | ---: | ---: |
| GPU timestamp, kernels plus the event-wait gaps | 49.125 | 96.3% |
| encode / submit / wait outside that timestamp | 1.419 | 2.8% |
| host before the graph | 0.460 | 0.9% |
| patch + finish | 0.024 | 0.05% |

Commit is a separate 0.008 ms mean over 119 calls. The T=1 window was 80.024 ms and the T=4 prompt window was 3088.985 ms. Both are under Instruments. The T=4 `patch+finish` of 2050.850 ms is that first window, not a decode token.

Instruments, pid 54527, Compute channel only. No blit channel. 27 complete command buffers have 49 compute encoders each (Compute Command 0 through 48). Encoder busy time is stable: min 16.766 ms, median 17.116 ms, max 17.736 ms, mean 17.316 ms. The span is not: min 36.871 ms, median 44.353 ms, max 58.186 ms. The variation is the gaps between encoders. Median gap inside a buffer is 0.545 ms. The largest single gap in the set is 2.158 ms, so the idle is many short waits, not one stall.

One steady token, matched to the engine line `window pos 110 token 314 -> 9338 ... 51.48 ms (GPU 49.61 ms, 1189 dispatches)`. The command buffer whose GPU span is 49.610 ms is that timestamp.

| Part of the 51.48 ms wall | ms | Share |
| --- | ---: | ---: |
| Compute-encoder GPU busy (49 encoders, sum of GPU Execution) | 17.582 | 34.2% |
| Gaps on the GPU timeline between those encoders | 32.027 | 62.2% |
| Outside the GPU timestamp (host before, encode/submit, patch/finish) | 1.870 | 3.6% |

Of the 49.61 ms GPU timestamp, 35.4% is encoder busy and 64.6% is those gaps. The busy time is the whole encoder, and each encoder holds many dispatches. It is not a per-kernel split.

Kernel dispatches per token: 1189, from the engine line on every traced T=2 head window, including position 110. Those 1189 dispatches are packed into 49 compute encoders. One profile round accepted a draft, so that round emitted two tokens. Position 110 did not. The speed runs accepted no drafts, so each T=2 window there is one accepted token and, on the traced shape, 1189 dispatches.

SSD reads and the stall. On the traced run, 120 windows: hits 0.8816, read-ahead hits 0.0661 (3513 late), misses 0.0523 (45.9 per window), 243.8 MB read per window (21162 reads, about 176 reads per window), host 22.24 ms per window of which waiting 21.40 ms. That wait is the compute thread blocked in `at_layer` until the I/O threads finish the missing experts. It sits inside the GPU timestamp, in the gaps above, so it is not an extra 21 ms on top of the 49 ms. It is also larger than the untraced cold run (9.85 ms waiting inside a 41.09 ms GPU timestamp, host 10.56 ms). The Instruments attach inflated the stall. The cold speed-run figure is the one to use for the default configuration. Mean idle across the 27 traced buffers is 28.025 ms, against 21.40 ms of SSD wait, so the gaps are the SSD wait plus the host handshake between the 49 segments.

CPU-side work outside the GPU timestamp is the 0.460 + 1.419 + 0.024 ms split above, 3.7% of the average traced window. The expert-stream host time (22.24 ms traced, 10.56 ms cold) is the handshake inside the GPU timestamp, and almost all of it is the wait for experts.

Copies. The 1360 GPU intervals for this process are Compute, and the bytes column is 0. There is no blit encoder in the trace for pid 54527. Host state copies after the graph are inside patch+finish, 0.024 ms on the T=2 mean (0.05% of the window). Expert bytes move by the streaming reads above, not by a Metal blit.

## What this leaves unmeasured

Per-kernel GPU percentages. `--spec 0`. An fp16 KV store. A per-token prompt. Spread against a one-ulp dump. Raw logit `|d|` and `|lse|`. A serve-mode speed run, where the tail-skip default can differ. A purged-cache cold start.
