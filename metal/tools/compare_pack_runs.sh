#!/bin/sh
# Run strata-metal-pack from an older commit and from this tree, and diff the verdicts.
#
#   ./metal/tools/compare_pack_runs.sh [BASE]
#   ./metal/tools/compare_pack_runs.sh --since <commit>
#
# The default BASE is 128cc72, the commit with the hand-written layer functions. That mode diffs
# layers 2-18 only. Its success line stays:
#   layers 2-18: the walk matches the hand-written functions
#
# --since <commit> diffs every layer that both runs printed. Flip lines are yes/no only in both
# modes: the walk searches for a moving byte, so the byte and the differ count may change.
# Everything else must match exactly, numbers included.
#
# Both builds read the same model (STRATA_MODEL_GGUF, or the Q2_0 shard 1 under $HOME).
set -eu
SINCE=0
if [ "${1:-}" = "--since" ]; then
    if [ -z "${2:-}" ]; then
        echo "usage: compare_pack_runs.sh [--since COMMIT | BASE]" >&2
        exit 2
    fi
    SINCE=1
    BASE=$2
else
    BASE="${1:-128cc72}"
fi
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK="${TMPDIR:-/tmp}/strata-compare"
rm -rf "$WORK"
mkdir -p "$WORK"

git -C "$ROOT" worktree add --detach "$WORK/base" "$BASE" >/dev/null
trap 'git -C "$ROOT" worktree remove --force "$WORK/base" >/dev/null 2>&1 || true' EXIT

cmake -S "$WORK/base/metal" -B "$WORK/base-build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WORK/base-build" --target strata-metal-pack -j >/dev/null
cmake -S "$ROOT/metal" -B "$WORK/new-build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WORK/new-build" --target strata-metal-pack strata-metal-layer-tensors -j >/dev/null

"$WORK/new-build/strata-metal-layer-tensors" --selftest

"$WORK/base-build/strata-metal-pack" --selftest > "$WORK/base.txt" 2>&1 || true
"$WORK/new-build/strata-metal-pack" --selftest > "$WORK/new.txt" 2>&1 || true

# Name, yes/no, and the numbers in the detail. Quant type names are dropped. Flip lines keep
# only name and yes/no. Default mode keeps layers 2-18. --since keeps every layer line, and the
# next step drops a layer that only one run printed.
filter() {
    python3 - "$1" "$2" <<'PY'
import re, sys
path, mode = sys.argv[1], sys.argv[2]
keep = re.compile(r"layer-([2-9]|1[0-8])(?![0-9])")
any_layer = re.compile(r"layer-(\d+)(?!\d)")
for line in open(path, encoding="utf-8", errors="replace"):
    m = re.match(r"^  (.*?)\s+(yes|\*\*\* NO \*\*\*)(?:\s+(.*))?$", line.rstrip("\n"))
    if not m:
        continue
    name = m.group(1)
    if mode == "since":
        if not any_layer.search(name):
            continue
    elif not keep.search(name):
        continue
    status, detail = m.group(2), m.group(3) or ""
    if re.match(r"a changed .* byte moves", name):
        print(name, "|", status)
        continue
    detail = re.sub(r"\b(?:I?Q\d\w*|BF16|F16|F32)\b", "", detail)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", detail)
    print(name, "|", status, "|", " ".join(nums))
PY
}
MODE=default
if [ "$SINCE" = 1 ]; then MODE=since; fi
filter "$WORK/base.txt" "$MODE" > "$WORK/base.verdicts"
filter "$WORK/new.txt" "$MODE" > "$WORK/new.verdicts"
if [ "$SINCE" = 1 ]; then
    python3 - "$WORK/base.verdicts" "$WORK/new.verdicts" <<'PY'
import re, sys
base_path, new_path = sys.argv[1], sys.argv[2]
def layers(path):
    found = set()
    for line in open(path, encoding="utf-8"):
        m = re.search(r"layer-(\d+)(?!\d)", line)
        if m:
            found.add(int(m.group(1)))
    return found
both = layers(base_path) & layers(new_path)
def keep(path):
    lines = []
    for line in open(path, encoding="utf-8"):
        m = re.search(r"layer-(\d+)(?!\d)", line)
        if m and int(m.group(1)) in both:
            lines.append(line)
    open(path, "w", encoding="utf-8").writelines(lines)
keep(base_path)
keep(new_path)
PY
fi

echo "base: $(wc -l < "$WORK/base.verdicts") verdict lines, $(grep -c 'NO \*\*\*' "$WORK/base.verdicts" || true) failing"
echo "walk: $(wc -l < "$WORK/new.verdicts") verdict lines, $(grep -c 'NO \*\*\*' "$WORK/new.verdicts" || true) failing"
if diff -u "$WORK/base.verdicts" "$WORK/new.verdicts" > "$WORK/verdicts.diff"; then
    if [ "$SINCE" = 1 ]; then
        echo "layers in both runs match"
    else
        echo "layers 2-18: the walk matches the hand-written functions"
    fi
else
    if [ "$SINCE" = 1 ]; then
        echo "layers in both runs differ; see $WORK/verdicts.diff (full runs in $WORK/base.txt and $WORK/new.txt)"
    else
        echo "layers 2-18 differ; see $WORK/verdicts.diff (full runs in $WORK/base.txt and $WORK/new.txt)"
    fi
    head -60 "$WORK/verdicts.diff"
    exit 1
fi
