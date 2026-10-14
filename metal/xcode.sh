#!/bin/sh
# Generate an Xcode project from metal/CMakeLists.txt.
# CMake stays the file list.
#
#   ./metal/xcode.sh
#       Project next to the checkout: build-metal-xcode/strata_metal.xcodeproj
#
#   ./metal/xcode.sh --local
#       Copy the Metal tree onto the internal disk and generate the project there:
#       ~/Developer/Strata-metal/xcode/strata_metal.xcodeproj
#       Read metal/XCODE_HANDOFF.md in the Handoff group.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCAL="${HOME}/Developer/Strata-metal"
if command -v cmake >/dev/null 2>&1; then
  CMAKE=cmake
elif [ -x /opt/homebrew/bin/cmake ]; then
  CMAKE=/opt/homebrew/bin/cmake
else
  echo "cmake not found" >&2
  exit 1
fi

local_copy() {
  mkdir -p "$LOCAL/docs" "$LOCAL/src/kernels/cuda" "$LOCAL/include"
  rsync -a --delete "$ROOT/metal/" "$LOCAL/metal/"
  rsync -a --delete "$ROOT/include/" "$LOCAL/include/"
  cp "$ROOT/docs/APPLE_SILICON.md" "$LOCAL/docs/APPLE_SILICON.md"
  cp "$ROOT/src/kernels/cuda/bf16_gemv.cu" "$LOCAL/src/kernels/cuda/bf16_gemv.cu"
  cp "$ROOT/src/kernels/cuda/gdn.cu" "$LOCAL/src/kernels/cuda/gdn.cu"
  cp "$ROOT/src/kernels/cuda/qsa_decode_attn.cu" "$LOCAL/src/kernels/cuda/qsa_decode_attn.cu"
  cp "$ROOT/src/kernels/bf16_gemv_parity.cpp" "$LOCAL/src/kernels/bf16_gemv_parity.cpp"
  cp "$ROOT/src/kernels/gdn_parity.cpp" "$LOCAL/src/kernels/gdn_parity.cpp"
  cp "$ROOT/src/kernels/qsa_parity.cpp" "$LOCAL/src/kernels/qsa_parity.cpp"
  cat > "$LOCAL/Open Strata Metal.command" << EOF
#!/bin/sh
open "\$HOME/Developer/Strata-metal/xcode/strata_metal.xcodeproj"
EOF
  chmod 755 "$LOCAL/Open Strata Metal.command"
}

if [ "${1:-}" = "--local" ]; then
  local_copy
  "$CMAKE" -S "$LOCAL/metal" -B "$LOCAL/xcode" -G Xcode -DCMAKE_XCODE_GENERATE_SCHEME=ON
  echo "open $LOCAL/xcode/strata_metal.xcodeproj"
  echo "Read the Handoff group: XCODE_HANDOFF.md"
else
  if [ -n "${1:-}" ]; then
    echo "usage: metal/xcode.sh [--local]" >&2
    exit 2
  fi
  cd "$ROOT"
  "$CMAKE" -S metal -B build-metal-xcode -G Xcode -DCMAKE_XCODE_GENERATE_SCHEME=ON
  echo "open build-metal-xcode/strata_metal.xcodeproj"
fi
echo "Schemes: strata-metal-device, strata-metal-bf16-gemv, strata-metal-gdn, strata-metal-qsa, strata-metal-iq, strata-metal-sgemv, strata-metal-elementwise, strata-metal-gdn-layer, strata-metal-gr, strata-metal-router, strata-metal-sampler, strata-metal-moe-combine, strata-metal-s2-gemv, strata-metal-shared-expert, strata-metal-qsa-gate, strata-metal-indexer, strata-metal-qsa-index, strata-metal-topk, strata-metal-kv, strata-metal-rope, strata-metal-qsa-layer, strata-metal-token, strata-metal-stack, strata-metal-pack"
echo "Derived Data stays on the internal disk (Xcode's default)."
