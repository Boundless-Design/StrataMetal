// The largest `width` cells. One threadgroup, so the ascending-index tie is one walk.
// A float is ordered as an unsigned key. -0 matches +0. A NaN sorts below every real.
#include <metal_stdlib>
using namespace metal;

struct TkP {
    int n_kv;
    int width;
    int pad0;
    int pad1;
};

inline uint tk_order(float s) {
    const float v = s + 0.0f;
    if (v != v) return 0u;
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void topk_cells(const device float* scores [[buffer(0)]], device int* ids [[buffer(1)]],
                       constant TkP& dim [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int s_a[256];
    threadgroup int s_b[256];
    threadgroup int s_cgt;
    const int t = (int) tid;
    const int n_kv = dim.n_kv;
    const int width = dim.width;
    const int chunk = (n_kv + 255) / 256;
    const int lo = t * chunk;
    int hi = lo + chunk;
    if (hi > n_kv) hi = n_kv;

    uint v = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const uint cand = v | (1u << bit);
        int c = 0;
        for (int j = lo; j < hi; ++j)
            if (tk_order(scores[j]) >= cand) ++c;
        s_a[t] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) s_a[t] += s_a[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int tot = s_a[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tot >= width) v = cand;
    }
    const uint thr = v;

    int gt = 0;
    int eq = 0;
    for (int j = lo; j < hi; ++j) {
        const uint k = tk_order(scores[j]);
        if (k > thr) ++gt;
        else if (k == thr) ++eq;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int ag = 0;
        int ae = 0;
        for (int i = 0; i < 256; ++i) {
            const int g = s_a[i];
            const int e = s_b[i];
            s_a[i] = ag;
            s_b[i] = ae;
            ag += g;
            ae += e;
        }
        s_cgt = ag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int off_eq = s_b[t];
    const int eq_budget = width - s_cgt;

    int sel = 0;
    {
        int e = off_eq;
        for (int j = lo; j < hi; ++j) {
            const uint k = tk_order(scores[j]);
            if (k > thr) ++sel;
            else if (k == thr && e < eq_budget) {
                ++sel;
                ++e;
            }
        }
    }
    s_a[t] = sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < 256; ++i) {
            const int c = s_a[i];
            s_a[i] = a;
            a += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int w = s_a[t];
    int e = off_eq;
    for (int j = lo; j < hi; ++j) {
        const uint k = tk_order(scores[j]);
        if (k > thr) ids[w++] = j;
        else if (k == thr && e < eq_budget) {
            ++e;
            ids[w++] = j;
        }
    }
}
