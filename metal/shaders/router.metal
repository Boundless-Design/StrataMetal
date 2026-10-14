// MoE router. One threadgroup per token.
// Softmax over every expert, top k by a strict argmax (the lower index wins a tie),
// then divide those k weights by their sum. The sum is clamped at 2**-14.
// This GPU has no fp64. The exponential is the precise f32 exp. The sum of those
// values is a pair of floats, added in ascending expert order, so the weights stay
// next to the f64 reference. The BF16 product that produces the logits is not here.
#include <metal_stdlib>
using namespace metal;

struct RouterP {
    int n_tokens;
    int n_expert;
    int k;
    int nt;
};

struct RtDS { float hi; float lo; };

inline RtDS rt_add(RtDS a, float b) {
    float s = a.hi + b;
    float z = s - a.hi;
    float e = (a.hi - (s - z)) + (b - z) + a.lo;
    float z2 = s + e;
    RtDS r;
    r.hi = z2;
    r.lo = e - (z2 - s);
    return r;
}

kernel void router_top10(device const float* logits [[buffer(0)]],
                         device int* ids [[buffer(1)]],
                         device float* weights [[buffer(2)]],
                         constant RouterP& dim [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint t [[threadgroup_position_in_grid]]) {
    threadgroup float ex[1024];
    threadgroup float prob[1024];
    threadgroup uchar taken[1024];
    threadgroup float red_v[16];
    threadgroup int red_i[16];
    threadgroup float sum_hi;
    threadgroup float sum_lo;

    if (t >= (uint) dim.n_tokens) return;
    const int n_expert = dim.n_expert;
    const int k = dim.k;
    const int nt = dim.nt;
    const int lane = (int) tid;
    device const float* l = logits + (size_t) t * (size_t) n_expert;

    float mx = -INFINITY;
    for (int e = lane; e < n_expert; e += nt) mx = fmax(mx, l[e]);
    for (ushort off = 16; off > 0; off >>= 1) mx = fmax(mx, simd_shuffle_down(mx, off));
    if ((lane & 31) == 0) red_v[lane >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        const int nw = (nt + 31) >> 5;
        float v = (lane < nw) ? red_v[lane] : -INFINITY;
        for (ushort off = 16; off > 0; off >>= 1) v = fmax(v, simd_shuffle_down(v, off));
        if (lane == 0) red_v[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red_v[0];

    for (int e = lane; e < n_expert; e += nt) ex[e] = exp(l[e] - mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        RtDS sum;
        sum.hi = 0.0f;
        sum.lo = 0.0f;
        for (int e = 0; e < n_expert; ++e) sum = rt_add(sum, ex[e]);
        sum_hi = sum.hi;
        sum_lo = sum.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.0f / sum_hi;
    const float corr = sum_lo * inv;
    inv = inv - corr * inv;
    for (int e = lane; e < n_expert; e += nt) prob[e] = ex[e] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = lane; e < n_expert; e += nt) taken[e] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
        for (int e = lane; e < n_expert; e += nt) {
            if (taken[e]) continue;
            const float pe = prob[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((lane & 31) == 0) {
            red_v[lane >> 5] = bv;
            red_i[lane >> 5] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (lane < nw) ? red_v[lane] : -INFINITY;
            int ix = (lane < nw) ? red_i[lane] : n_expert;
            for (ushort off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_down(v, off);
                const int oi = simd_shuffle_down(ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (lane == 0 && ix < n_expert) {
                ids[(size_t) t * (size_t) k + (size_t) i] = ix;
                weights[(size_t) t * (size_t) k + (size_t) i] = v;
                taken[ix] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        RtDS s;
        s.hi = 0.0f;
        s.lo = 0.0f;
        device float* w = weights + (size_t) t * (size_t) k;
        for (int i = 0; i < k; ++i) s = rt_add(s, w[i]);
        float sc = s.hi + s.lo;
        sc = fmax(sc, 6.103515625e-05f);
        for (int i = 0; i < k; ++i) w[i] = w[i] / sc;
    }
}
