# Strata on Apple Silicon: the Metal build (docs/APPLE_SILICON.md).
#
# A separate project on purpose, same as sycl/. The CUDA sources in src/kernels/cuda do not compile to Metal,
# and the AVX2/AVX-512 expert kernels do not run on this CPU. Nothing under src/ is built here. The planner
# header (include/strata/plan/plan.hpp) is host arithmetic and is included as it stands.
#
# CMake is the file list, the same as the CUDA and HIP builds. metal/xcode.sh generates an Xcode project
# from this file so the tools can be captured on the GPU. Do not keep a second list of sources by hand.
#
#   cmake -S metal -B build-metal && cmake --build build-metal
#   ./build-metal/strata-metal-device --selftest
#   ./build-metal/strata-metal-bf16-gemv --selftest
#   ./build-metal/strata-metal-gdn --selftest
#   ./build-metal/strata-metal-qsa --selftest
#   ./build-metal/strata-metal-iq --selftest
#   ./build-metal/strata-metal-sgemv --selftest
#   ./build-metal/strata-metal-elementwise --selftest
#   ./build-metal/strata-metal-gdn-layer --selftest
#   ./build-metal/strata-metal-gr --selftest
#   ./build-metal/strata-metal-router --selftest
#   ./build-metal/strata-metal-sampler --selftest
#   ./build-metal/strata-metal-moe-combine --selftest
#   ./build-metal/strata-metal-s2-gemv --selftest
#   ./build-metal/strata-metal-shared-expert --selftest
#   ./build-metal/strata-metal-qsa-gate --selftest
#   ./build-metal/strata-metal-indexer --selftest
#   ./build-metal/strata-metal-qsa-index --selftest
#   ./build-metal/strata-metal-topk --selftest
#   ./build-metal/strata-metal-kv --selftest
#   ./build-metal/strata-metal-rope --selftest
#   ./build-metal/strata-metal-qsa-layer --selftest
#   ./build-metal/strata-metal-token --selftest
#   ./build-metal/strata-metal-stack --selftest
#   ./build-metal/strata-metal-pack --selftest
#   ./build-metal/strata-metal-layer-tensors --selftest
#
# The device binary names the GPU and says which published model sizes fit its working set.
# The others check a BF16 matrix-vector product, the DeltaNet kernels, paged QSA attention,
# the Coder expert product, the S-form dense projection, the GDN layer glue, one GDN
# layer built from those kernels, the gated residual, the MoE router, the sampler, the
# MoE combine, the 2-bit product over a Q8_0 activation, one shared expert, the
# QSA gate, the indexer append, the indexer cell scores, the top-k selection, the
# paged KV append and gather, the partial NEOX rotation, one QSA attention
# block, one synthetic token through that block, four layers on the pack
# schedule, and the pack index loaded into those kernels. None of them runs the model.
# strata-metal-pack also walks layers 2 onward from the GGUF (STRATA_MODEL_GGUF, STRATA_WALK_LAST).
# strata-metal-layer-tensors checks the by-name tensor lookup that walk uses; it is host code only.
cmake_minimum_required(VERSION 3.24)
project(strata_metal VERSION 0.1.39 LANGUAGES CXX OBJCXX)
if(NOT APPLE)
  message(FATAL_ERROR "The Metal engine builds on macOS only (docs/APPLE_SILICON.md)")
endif()
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Read when each target is created. Ignored by the Makefile generator.
set(CMAKE_XCODE_GENERATE_SCHEME ON)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}-metal")

function(strata_metal_scheme target)
  set_target_properties(${target} PROPERTIES
    XCODE_GENERATE_SCHEME ON
    XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE "Metal"
    XCODE_SCHEME_ENABLE_GPU_API_VALIDATION ON
    XCODE_SCHEME_ENABLE_GPU_SHADER_VALIDATION ON
    XCODE_SCHEME_LAUNCH_CONFIGURATION "Release")
endfunction()

add_library(strata_metal_runtime STATIC
  src/runtime.mm
  src/bf16_gemv.mm
  src/gdn.mm
  src/qsa.mm
  src/iq.mm
  src/sgemv.mm
  src/elementwise.mm
  src/gdn_layer.cpp
  src/gr.mm
  src/router.mm
  src/sampler.mm
  src/moe_combine.mm
  src/s2_gemv.mm
  src/shared_expert.mm
  src/qsa_gate.mm
  src/indexer.mm
  src/qsa_index.mm
  src/topk.mm
  src/kv.mm
  src/rope.mm
  src/qsa_layer.cpp
  src/token.cpp
  src/stack.cpp
  src/pack.cpp)
target_include_directories(strata_metal_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include)
target_compile_definitions(strata_metal_runtime PRIVATE
  "STRATA_METAL_SHADER_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\"")
target_compile_options(strata_metal_runtime PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(strata_metal_runtime PUBLIC
  "-framework Metal"
  "-framework Foundation")

add_executable(strata-metal-device
  src/device_main.cpp
  src/device.mm
  src/fit.cpp)
target_include_directories(strata-metal-device PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include)
target_compile_options(strata-metal-device PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(strata-metal-device PRIVATE
  "-framework Metal"
  "-framework Foundation")
strata_metal_scheme(strata-metal-device)

add_executable(strata-metal-bf16-gemv src/bf16_gemv_parity.cpp)
target_link_libraries(strata-metal-bf16-gemv PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-bf16-gemv)
set_property(TARGET strata-metal-bf16-gemv PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-gdn src/gdn_parity.cpp)
target_link_libraries(strata-metal-gdn PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-gdn)
set_property(TARGET strata-metal-gdn PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-qsa src/qsa_parity.cpp)
target_link_libraries(strata-metal-qsa PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-qsa)
set_property(TARGET strata-metal-qsa PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-iq src/iq_parity.cpp)
target_link_libraries(strata-metal-iq PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-iq)
set_property(TARGET strata-metal-iq PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-sgemv src/sgemv_parity.cpp)
target_link_libraries(strata-metal-sgemv PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-sgemv)
set_property(TARGET strata-metal-sgemv PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-elementwise src/elementwise_parity.cpp)
target_link_libraries(strata-metal-elementwise PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-elementwise)
set_property(TARGET strata-metal-elementwise PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-gdn-layer src/gdn_layer_parity.cpp)
target_link_libraries(strata-metal-gdn-layer PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-gdn-layer)
set_property(TARGET strata-metal-gdn-layer PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-gr src/gr_parity.cpp)
target_link_libraries(strata-metal-gr PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-gr)
set_property(TARGET strata-metal-gr PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-router src/router_parity.cpp)
target_link_libraries(strata-metal-router PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-router)
set_property(TARGET strata-metal-router PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-sampler src/sampler_parity.cpp)
target_link_libraries(strata-metal-sampler PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-sampler)
set_property(TARGET strata-metal-sampler PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-moe-combine src/moe_combine_parity.cpp)
target_link_libraries(strata-metal-moe-combine PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-moe-combine)
set_property(TARGET strata-metal-moe-combine PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-s2-gemv src/s2_gemv_parity.cpp)
target_link_libraries(strata-metal-s2-gemv PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-s2-gemv)
set_property(TARGET strata-metal-s2-gemv PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-shared-expert src/shared_expert_parity.cpp)
target_link_libraries(strata-metal-shared-expert PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-shared-expert)
set_property(TARGET strata-metal-shared-expert PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-qsa-gate src/qsa_gate_parity.cpp)
target_link_libraries(strata-metal-qsa-gate PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-qsa-gate)
set_property(TARGET strata-metal-qsa-gate PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-indexer src/indexer_parity.cpp)
target_link_libraries(strata-metal-indexer PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-indexer)
set_property(TARGET strata-metal-indexer PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-qsa-index src/qsa_index_parity.cpp)
target_link_libraries(strata-metal-qsa-index PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-qsa-index)
set_property(TARGET strata-metal-qsa-index PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-topk src/topk_parity.cpp)
target_link_libraries(strata-metal-topk PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-topk)
set_property(TARGET strata-metal-topk PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-kv src/kv_parity.cpp)
target_link_libraries(strata-metal-kv PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-kv)
set_property(TARGET strata-metal-kv PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-rope src/rope_parity.cpp)
target_link_libraries(strata-metal-rope PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-rope)
set_property(TARGET strata-metal-rope PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-qsa-layer src/qsa_layer_parity.cpp)
target_link_libraries(strata-metal-qsa-layer PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-qsa-layer)
set_property(TARGET strata-metal-qsa-layer PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-token src/token_parity.cpp)
target_link_libraries(strata-metal-token PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-token)
set_property(TARGET strata-metal-token PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-stack src/stack_parity.cpp)
target_link_libraries(strata-metal-stack PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-stack)
set_property(TARGET strata-metal-stack PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

add_executable(strata-metal-pack src/pack_parity.cpp)
target_link_libraries(strata-metal-pack PRIVATE strata_metal_runtime)
strata_metal_scheme(strata-metal-pack)
set_property(TARGET strata-metal-pack PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

# The by-name tensor lookup the layer walk uses. Host code only: no Metal, no runtime.
add_executable(strata-metal-layer-tensors src/layer_tensors_parity.cpp)
target_include_directories(strata-metal-layer-tensors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include)
strata_metal_scheme(strata-metal-layer-tensors)
set_property(TARGET strata-metal-layer-tensors PROPERTY XCODE_SCHEME_ARGUMENTS "--selftest")

# Files to read in the generated Xcode project. Nothing here is compiled.
add_custom_target(Handoff SOURCES
  XCODE_HANDOFF.md
  ${STRATA_ROOT}/docs/APPLE_SILICON.md)
add_custom_target(Reference SOURCES
  ${STRATA_ROOT}/src/kernels/cuda/bf16_gemv.cu
  ${STRATA_ROOT}/src/kernels/bf16_gemv_parity.cpp
  ${STRATA_ROOT}/src/kernels/cuda/gdn.cu
  ${STRATA_ROOT}/src/kernels/gdn_parity.cpp
  ${STRATA_ROOT}/src/kernels/cuda/qsa_decode_attn.cu
  ${STRATA_ROOT}/src/kernels/qsa_parity.cpp
  ${STRATA_ROOT}/include/strata/kernels/qsa_decode_attn.hpp)

# So the shaders are files in the generated Xcode project. The runtime compiles them from source.
add_custom_target(strata_metal_shaders SOURCES
  shaders/bf16_gemv.metal
  shaders/gdn.metal
  shaders/qsa.metal
  shaders/iq.metal
  shaders/sgemv.metal
  shaders/elementwise.metal
  shaders/gr.metal
  shaders/router.metal
  shaders/sampler.metal
  shaders/moe_combine.metal
  shaders/s2_gemv.metal
  shaders/shared_expert.metal
  shaders/qsa_gate.metal
  shaders/indexer.metal
  shaders/qsa_index.metal
  shaders/topk.metal
  shaders/kv.metal
  shaders/rope.metal)
