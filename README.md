# StrataMetal

StrataMetal is an Apple Silicon Metal port of upstream Strata. It is MIT
licensed. Copyright (c) 2026 Niko1221 and the Strata contributors.

## Build

From the repository root, on an Apple Silicon Mac:

```sh
cmake -S metal -B build-metal && cmake --build build-metal
```

The tools land in `build-metal/`. A `.metal` shader is compiled when a tool
starts, so editing a shader does not need a rebuild. Editing a `.cpp` or `.mm`
file does.

## The model

`STRATA_MODEL_GGUF` may name any shard of a split GGUF. When it is unset, the
tools open the Q2_0 shard 1 under `$HOME`:

`$HOME/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`

The second shard is found from that name. `STRATA_WALK_LAST` stops the layer
walk early. The default walks every layer the model has.

## Selftests

```sh
./build-metal/strata-metal-layer-tensors --selftest
./build-metal/strata-metal-pack --selftest
./metal/tools/compare_pack_runs.sh
```

Each `strata-metal-*` tool takes `--selftest`. The pack check is the layer
walk. `compare_pack_runs.sh` diffs layers 2-18 against commit `128cc72`.
`--since <commit>` diffs every layer both runs printed.

## Status

Every layer of the Q2_0 model is verified against a host reference, one token
at a time. `strata-metal-run` is a separate forward pass: token ids in, a full
logit row per position, greedy argmax, state carried. Experts and the PLE
table stay on disk. `STRATA_PERTURB_ULP` and `STRATA_LOG_MARGINS` are off
unless set. `metal/XCODE_HANDOFF.md` describes both.

`metal/tools/compare_fixtures.py` checks those rows against the CUDA fixtures.
Teacher-forced p1 through p5 passed that gate (478 exact picks, 14 accepted
top-3 mismatches, none outside the rule). Free-running greedy for 32 new
tokens matched p1 at every position, 1.066 s/token over 36 positions. p3 first
diverged at position 17, an accepted gap of 0.060, 1.118 s/token over 35
positions. The rate is a measurement, not a target.
