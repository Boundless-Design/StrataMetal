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
at a time. There is no sampling. There is no token rate.
