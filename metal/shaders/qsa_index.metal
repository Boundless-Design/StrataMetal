// Per-cell indexer scores. One simdgroup per indexer head, one threadgroup per pooled row.
// The dot is a pair of floats. This GPU has no fp64. ReLU is applied to each head, then the
// heads are summed. The incomplete tail gets +1e9 after that sum is rounded to f32.
#include <metal_stdlib>
using namespace metal;

struct QiP {
    int idx_n_head;
    int idx_dim;
    int r;
    int n_bid;
    int n_kv;
    int has_bias;
    int pad0;
    int pad1;
};

struct QiDS {
    float hi;
    float lo;
};

inline QiDS qi_add(QiDS a, float b) {
    float s = a.hi + b;
    float z = s - a.hi;
    float e = (a.hi - (s - z)) + (b - z) + a.lo;
    float z2 = s + e;
    QiDS r;
    r.hi = z2;
    r.lo = e - (z2 - s);
    return r;
}

kernel void qsa_index_cells(const device float* pooled [[buffer(0)]], const device float* q_idx [[buffer(1)]],
                            const device float* bias [[buffer(2)]], device float* cell_scores [[buffer(3)]],
                            constant QiP& dim [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                            uint b [[threadgroup_position_in_grid]]) {
    if ((int) b > dim.n_bid) return;
    const uint lane = tid & 31u;
    const uint wid = tid >> 5;
    threadgroup float dot_hi[32];
    threadgroup float dot_lo[32];

    QiDS acc;
    acc.hi = 0.0f;
    acc.lo = 0.0f;
    if ((int) wid < dim.idx_n_head) {
        const device float* row = pooled + (size_t) b * (size_t) dim.idx_dim;
        const device float* q = q_idx + (size_t) wid * (size_t) dim.idx_dim;
        for (int d = (int) lane; d < dim.idx_dim; d += 32) {
            const float p = row[d];
            const float x = q[d];
            const float prod = p * x;
            const float err = fma(p, x, -prod);
            acc = qi_add(acc, prod);
            acc = qi_add(acc, err);
        }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        QiDS other;
        other.hi = simd_shuffle_down(acc.hi, off);
        other.lo = simd_shuffle_down(acc.lo, off);
        if (lane + (uint) off < 32u) {
            acc = qi_add(acc, other.hi);
            acc = qi_add(acc, other.lo);
        }
    }
    if (lane == 0u && (int) wid < dim.idx_n_head) {
        dot_hi[wid] = acc.hi;
        dot_lo[wid] = acc.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) return;
    QiDS score;
    score.hi = 0.0f;
    score.lo = 0.0f;
    for (int h = 0; h < dim.idx_n_head; ++h) {
        const float hi = dot_hi[h];
        const float lo = dot_lo[h];
        if (hi > 0.0f || (hi == 0.0f && lo > 0.0f)) {
            score = qi_add(score, hi);
            score = qi_add(score, lo);
        }
    }
    if (dim.has_bias != 0) score = qi_add(score, bias[b]);
    float sc = score.hi + score.lo;
    if ((int) b == dim.n_bid && (dim.n_kv % dim.r) != 0) sc += 1e9f;
    int lo = (int) b * dim.r;
    int hi = lo + dim.r;
    if ((int) b == dim.n_bid) hi = dim.n_kv;
    if (hi > dim.n_kv) hi = dim.n_kv;
    for (int j = lo; j < hi; ++j) cell_scores[j] = sc;
}
