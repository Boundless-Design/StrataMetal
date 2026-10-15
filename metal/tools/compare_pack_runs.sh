#!/bin/sh
# Run strata-metal-pack from before the layer walk and from this tree, and diff the layer 2-18 verdicts.
#
#   ./metal/tools/compare_pack_runs.sh [BASE]
#
# BASE is the commit with the hand-written layer functions (default 128cc72). Both builds read the same
# model (STRATA_MODEL_GGUF, or the Q2_0 shard 1 under $HOME). The walk must print the same numbers for
# every layer the hand-written functions reached: same errors, same expert ids, same differ counts.
#
# The one expected difference is the flip checks: the walk searches for a moving byte instead of trying a
# hand-picked list, so "byte N" and the differ count on lines that start "a changed ... byte" may change.
# Those lines are compared on yes/no only. Everything else must match exactly.
set -eu
BASE="${1:-128cc72}"
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

# Keep the verdict lines of layers 2-18 as: name, yes/no, and the numbers in the detail. Wording and quant
# type names are dropped (the hand-written details named types on some layers and not others). Flip lines keep
# only name and yes/no.
filter() {
    python3 - "$1" <<'PY'
import re, sys
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.match(r"^  (.*?)\s+(yes|\*\*\* NO \*\*\*)(?:\s+(.*))?$", line.rstrip("\n"))
    if not m or not re.search(r"layer-([2-9]|1[0-8])(?![0-9])", m.group(1)):
        continue
    name, status, detail = m.group(1), m.group(2), m.group(3) or ""
    if re.match(r"a changed .* byte moves", name):
        print(name, "|", status)
        continue
    detail = re.sub(r"\b(?:I?Q\d\w*|BF16|F16|F32)\b", "", detail)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", detail)
    print(name, "|", status, "|", " ".join(nums))
PY
}
filter "$WORK/base.txt" > "$WORK/base.verdicts"
filter "$WORK/new.txt" > "$WORK/new.verdicts"

echo "base: $(wc -l < "$WORK/base.verdicts") verdict lines, $(grep -c 'NO \*\*\*' "$WORK/base.verdicts" || true) failing"
echo "walk: $(wc -l < "$WORK/new.verdicts") verdict lines, $(grep -c 'NO \*\*\*' "$WORK/new.verdicts" || true) failing"
if diff -u "$WORK/base.verdicts" "$WORK/new.verdicts" > "$WORK/verdicts.diff"; then
    echo "layers 2-18: the walk matches the hand-written functions"
else
    echo "layers 2-18 differ; see $WORK/verdicts.diff (full runs in $WORK/base.txt and $WORK/new.txt)"
    head -60 "$WORK/verdicts.diff"
    exit 1
fi
