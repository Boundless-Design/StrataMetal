// The gated residual. One token.
// The norm is one simdgroup per stream. The f32-activation GEMV is the bf16 product with an f32 x.
// Weights stay bf16. lo = silu(dot / hc). The write centres the gate with 2*sigmoid.
#include <metal_stdlib>
using namespace metal;

inline float gr_bf16(ushort h) { return as_type<float>(uint(h) << 16u); }

struct GrNorm {
    int n_embd;
    int hc;
    float eps;
};

kernel void gr_norm(const device float* R [[buffer(0)]], const device float* w [[buffer(1)]],
                    device float* xn [[buffer(2)]], constant GrNorm& dim [[buffer(3)]],
                    uint lane [[thread_index_in_simdgroup]], uint c [[threadgroup_position_in_grid]]) {
    if ((int) c >= dim.hc) return;
    const device float* r = R + (size_t) c * (size_t) dim.n_embd;
    device float* y = xn + (size_t) c * (size_t) dim.n_embd;
    const device float* g = w + (size_t) c * (size_t) dim.n_embd;
    float acc = 0.0f;
    for (int d = (int) lane; d < dim.n_embd; d += 32) {
        float v = r[d];
        acc += v * v;
    }
    for (ushort off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    float inv = 0.0f;
    if (lane == 0u) inv = rsqrt(acc / (float) dim.n_embd + dim.eps);
    inv = simd_shuffle(inv, (ushort) 0);
    for (int d = (int) lane; d < dim.n_embd; d += 32) y[d] = r[d] * inv * g[d];
}

struct GrGemv {
    int n_in;
    int n_out;
};

kernel void gr_f32_gemv_naive(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant GrGemv& dim [[buffer(3)]],
                              uint o [[thread_position_in_grid]]) {
    if ((int) o >= dim.n_out) return;
    const device ushort* row = w + (size_t) o * (size_t) dim.n_in;
    float acc = 0.0f;
    for (int i = 0; i < dim.n_in; ++i) acc += x[i] * gr_bf16(row[i]);
    y[o] = acc;
}

kernel void gr_f32_gemv_warp(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                             device float* y [[buffer(2)]], constant GrGemv& dim [[buffer(3)]],
                             uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint lane = tid & 31u;
    uint warp = tid >> 5u;
    uint o = group * 8u + warp;
    if ((int) o >= dim.n_out) return;
    const device ushort* row = w + (size_t) o * (size_t) dim.n_in;
    float acc = 0.0f;
    for (int i = (int) lane; i < dim.n_in; i += 32) acc += x[i] * gr_bf16(row[i]);
    for (ushort off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    if (lane == 0u) y[o] = acc;
}

// silu(dot / hc). The division is inside the silu.
struct GrSilu {
    int n;
    int hc;
};

kernel void gr_silu_hc(device float* x [[buffer(0)]], constant GrSilu& dim [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if ((int) i >= dim.n) return;
    float z = x[i] / (float) dim.hc;
    x[i] = z / (1.0f + exp(-z));
}

struct GrN {
    int n;
};

kernel void gr_sigmoid_mul(const device float* xn [[buffer(0)]], device float* gate [[buffer(1)]],
                           constant GrN& dim [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int) i >= dim.n) return;
    float g = gate[i];
    float s = 1.0f / (1.0f + exp(-g));
    gate[i] = xn[i] * s;
}

struct GrMean {
    int n_embd;
    int hc;
};

kernel void gr_mean(const device float* gated [[buffer(0)]], device float* mixed [[buffer(1)]],
                    constant GrMean& dim [[buffer(2)]], uint d [[thread_position_in_grid]]) {
    if ((int) d >= dim.n_embd) return;
    float m = 0.0f;
    for (int c = 0; c < dim.hc; ++c) m += gated[(size_t) c * (size_t) dim.n_embd + d];
    mixed[d] = m / (float) dim.hc;
}

struct GrWrite {
    int n_embd;
    int hc;
};

kernel void gr_write(const device float* R [[buffer(0)]], const device float* block_out [[buffer(1)]],
                     const device float* inject [[buffer(2)]], device float* out [[buffer(3)]],
                     constant GrWrite& dim [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    int n = dim.hc * dim.n_embd;
    if ((int) i >= n) return;
    int c = (int) i / dim.n_embd;
    int d = (int) i - c * dim.n_embd;
    float z = inject[c] / (float) dim.hc;
    float s = 1.0f / (1.0f + exp(-z));
    out[i] = R[i] + block_out[d] * (2.0f * s);
}
