// Gated delta-net, the non-projection half. Same arithmetic as src/kernels/cuda/gdn.cu.
// State is (S, h_v, S) with j fastest. One thread owns one (h, j) column.
// Apple GPUs have no fp64. l2, the output norm, and the step's two dots accumulate as a
// double-single so the reduction stays next to the f64 reference the CUDA kernel uses.
// A float sum of the step dot is a few ulps off that reference, and the output q8 row
// rounds the other way. The products are exact: fma recovers the low half.
#include <metal_stdlib>
using namespace metal;

struct GdnDim { int S; int h_k; int h_v; };
struct ConvDim { int C; int dc; };
struct L2Dim { int cols; float eps; };
struct OutDim { int S; float eps; };
struct BetaDim { int n; };

struct DS { float hi; float lo; };

inline DS ds_add(DS a, float b) {
    float s = a.hi + b;
    float z = s - a.hi;
    float e = (a.hi - (s - z)) + (b - z) + a.lo;
    float z2 = s + e;
    DS r;
    r.hi = z2;
    r.lo = e - (z2 - s);
    return r;
}

inline DS ds_add2(DS a, DS b) {
    return ds_add(ds_add(a, b.hi), b.lo);
}

inline DS warp_sum_ds(DS acc, uint lane) {
    for (ushort off = 16; off > 0; off >>= 1) {
        DS other;
        other.hi = simd_shuffle_down(acc.hi, off);
        other.lo = simd_shuffle_down(acc.lo, off);
        if ((uint)lane + (uint)off < 32u) acc = ds_add2(acc, other);
    }
    return acc;
}

inline DS ds_zero() {
    DS r;
    r.hi = 0.0f;
    r.lo = 0.0f;
    return r;
}

// Exact product, then a double-single add. fma returns the part the float product rounded off.
inline DS ds_add_prod(DS acc, float a, float b) {
    const float p = a * b;
    const float e = fma(a, b, -p);
    return ds_add(ds_add(acc, p), e);
}

inline float ds_float(DS a) { return a.hi + a.lo; }

kernel void gdn_step(device float* state [[buffer(0)]],
                     device const float* q [[buffer(1)]],
                     device const float* k [[buffer(2)]],
                     device const float* v [[buffer(3)]],
                     device const float* gate [[buffer(4)]],
                     device const float* beta [[buffer(5)]],
                     device float* o [[buffer(6)]],
                     constant GdnDim& dim [[buffer(7)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float ks[128];
    threadgroup float qs[128];
    threadgroup float dec_s;
    threadgroup float beta_s;

    const int S = dim.S;
    const int h = (int)group.y;
    const int j = (int)group.x * 32 + (int)tid;
    const int src = h % dim.h_k;
    for (int i = (int)tid; i < S; i += 32) {
        ks[i] = k[src * S + i];
        qs[i] = q[src * S + i];
    }
    if (tid == 0u) {
        dec_s = exp(gate[h]);
        beta_s = beta[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j >= S) return;

    const float dec = dec_s;
    const float b = beta_s;
    device float* col = state + (size_t)h * (size_t)S + (size_t)j;
    const size_t stride = (size_t)dim.h_v * (size_t)S;

    DS sk = ds_zero();
    for (int i = 0; i < S; ++i) {
        const float s = col[(size_t)i * stride] * dec;
        col[(size_t)i * stride] = s;
        sk = ds_add_prod(sk, s, ks[i]);
    }
    const float d = (v[(size_t)h * (size_t)S + (size_t)j] - ds_float(sk)) * b;
    DS dot = ds_zero();
    for (int i = 0; i < S; ++i) {
        const float s = col[(size_t)i * stride] + ks[i] * d;
        col[(size_t)i * stride] = s;
        dot = ds_add_prod(dot, s, qs[i]);
    }
    o[(size_t)h * (size_t)S + (size_t)j] = ds_float(dot);
}

kernel void gdn_conv(device float* cs [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device const float* kW [[buffer(2)]],
                     device float* out [[buffer(3)]],
                     constant ConvDim& dim [[buffer(4)]],
                     uint c [[thread_position_in_grid]]) {
    if ((int)c >= dim.C) return;
    const int dc = dim.dc;
    device float* st = cs + (size_t)c * (size_t)(dc - 1);
    device const float* w = kW + (size_t)c * (size_t)dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
    acc += x[c] * w[dc - 1];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

kernel void gdn_l2(device float* x [[buffer(0)]],
                   constant L2Dim& dim [[buffer(1)]],
                   uint lane [[thread_index_in_threadgroup]],
                   uint row [[threadgroup_position_in_grid]]) {
    device float* p = x + (size_t)row * (size_t)dim.cols;
    DS acc;
    acc.hi = 0.0f;
    acc.lo = 0.0f;
    for (int i = (int)lane; i < dim.cols; i += 32)
        acc = ds_add(acc, p[i] * p[i]);
    acc = warp_sum_ds(acc, lane);
    const float inv = simd_shuffle(1.0f / sqrt(acc.hi + acc.lo + dim.eps), (ushort)0);
    for (int i = (int)lane; i < dim.cols; i += 32) p[i] *= inv;
}

kernel void gdn_beta(device float* beta [[buffer(0)]],
                     constant BetaDim& dim [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if ((int)i < dim.n) {
        const float x = beta[i];
        beta[i] = 1.0f / (1.0f + exp(-x));
    }
}

kernel void gdn_out_norm(device const float* o [[buffer(0)]],
                         device const float* z [[buffer(1)]],
                         device const float* ssm_norm [[buffer(2)]],
                         device float* y [[buffer(3)]],
                         constant OutDim& dim [[buffer(4)]],
                         uint lane [[thread_index_in_threadgroup]],
                         uint h [[threadgroup_position_in_grid]]) {
    device const float* po = o + (size_t)h * (size_t)dim.S;
    device const float* pz = z + (size_t)h * (size_t)dim.S;
    device float* py = y + (size_t)h * (size_t)dim.S;
    DS acc;
    acc.hi = 0.0f;
    acc.lo = 0.0f;
    for (int i = (int)lane; i < dim.S; i += 32)
        acc = ds_add(acc, po[i] * po[i]);
    acc = warp_sum_ds(acc, lane);
    const float mean = (acc.hi + acc.lo) / (float)dim.S;
    const float inv = simd_shuffle(1.0f / sqrt(mean + dim.eps), (ushort)0);
    for (int i = (int)lane; i < dim.S; i += 32) {
        const float zz = pz[i];
        py[i] = po[i] * inv * ssm_norm[i] * (1.0f / (1.0f + exp(-zz)));
    }
}
