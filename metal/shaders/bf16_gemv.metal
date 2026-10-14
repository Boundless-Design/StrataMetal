// BF16 GEMV. Same contract as src/kernels/cuda/bf16_gemv.cu:
//   y[o] = sum_i f32(x[i]) * f32(w[o * n_in + i])
// Weight row o is a contiguous run of n_in bf16 values (ne0 = n_in).
// Compile with fast math OFF. The parity test bounds summation order, and a reassociated
// multiply would no longer be the reduction the test is measuring.
#include <metal_stdlib>
using namespace metal;

inline float f32_from_bf16(ushort h) {
    return as_type<float>(uint(h) << 16u);
}

struct GemvDim { uint n_in; uint n_out; };
struct SplitDim { uint n_in; uint n_out; uint tpr; };

kernel void bf16_gemv_naive(device const ushort* x [[buffer(0)]],
                            device const ushort* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant GemvDim& dim [[buffer(3)]],
                            uint o [[thread_position_in_grid]]) {
    if (o >= dim.n_out) return;
    device const ushort* row = w + o * dim.n_in;
    float acc = 0.0f;
    for (uint i = 0; i < dim.n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

// One simdgroup (32 lanes) per output row. Lanes stride the row, so a simdgroup reads
// consecutive bf16 values. The tree is shuffle-down, lane 0 keeps the sum: the same
// association as the CUDA warp kernel.
kernel void bf16_gemv_warp(device const ushort* x [[buffer(0)]],
                           device const ushort* w [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           constant GemvDim& dim [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]]) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint o = group * 8u + warp;
    if (o >= dim.n_out) return;
    device const ushort* row = w + o * dim.n_in;
    float acc = 0.0f;
    for (uint i = lane; i < dim.n_in; i += 32u)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    for (ushort off = 16; off > 0; off >>= 1)
        acc += simd_shuffle_down(acc, off);
    if (lane == 0u) y[o] = acc;
}

kernel void bf16_gemv_split(device const ushort* x [[buffer(0)]],
                            device const ushort* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant SplitDim& dim [[buffer(3)]],
                            uint t [[thread_index_in_threadgroup]],
                            uint o [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    if (o >= dim.n_out) return;
    device const ushort* row = w + o * dim.n_in;
    float acc = 0.0f;
    for (uint i = t; i < dim.n_in; i += dim.tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    scratch[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = dim.tpr >> 1u; off > 0u; off >>= 1u) {
        if (t < off) scratch[t] += scratch[t + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) y[o] = scratch[0];
}
