// Sampler. One threadgroup per row.
// penalties, then top-k, then top-p, then min-p, then temperature, then one Philox draw.
// The argmax is a strict greater-than, so a tie keeps the lower index. NaN and -inf never win.
// A round that finds nothing stores id 0.
// This GPU has no fp64. The tail's exp is the precise f32 exp. The draw is Philox 4x32-10.
// The penalty bitmap covers 248320 logits. A wider penalised row counts the window instead.
#include <metal_stdlib>
using namespace metal;

struct SamplerP {
    int n_vocab;
    int k;
    int min_keep;
    int hlen;
    int history_len;
    int greedy;
    int bits_words;
    float top_p;
    float min_p;
    float inv_t;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    uint seed_lo;
    uint seed_hi;
    uint counter_lo;
    uint counter_hi;
};

inline float sm_apply(float logit, int count, constant SamplerP& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

inline int sm_count(device const int* hrow, int hlen, int v, threadgroup atomic_uint* bits, int bits_words) {
    if (hlen <= 0) return 0;
    if (bits_words > 0) {
        uint word = atomic_load_explicit(bits + (v >> 5), memory_order_relaxed);
        if ((word & (1u << (v & 31))) == 0u) return 0;
    }
    int c = 0;
    for (int i = 0; i < hlen; ++i)
        if (hrow[i] == v) ++c;
    return c;
}

inline float sm_philox(constant SamplerP& p, uint row) {
    ulong counter = ((ulong) p.counter_hi << 32) | (ulong) p.counter_lo;
    counter += (ulong) row;
    uint c0 = (uint) counter;
    uint c1 = (uint) (counter >> 32);
    uint c2 = p.seed_lo;
    uint c3 = p.seed_hi;
    for (uint i = 0u; i < 10u; ++i) {
        uint hi0 = (uint) (((ulong) 0x9E3779B9u * (ulong) c0) >> 32);
        uint hi1 = (uint) (((ulong) 0xBB67AE85u * (ulong) c2) >> 32);
        uint lo0 = 0x9E3779B9u * c0;
        uint lo1 = 0xBB67AE85u * c2;
        uint n0 = hi1 ^ c1 ^ i;
        uint n1 = lo1;
        uint n2 = hi0 ^ c3;
        uint n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return float(c0 >> 8) * (1.0f / 16777216.0f);
}

inline void sm_take(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
    }
}

// bits_words > 0 indexes `bits` for the first bits_words words. 0 counts the window instead.
void sm_sample(device const float* logits, device const int* history, device int* out, constant SamplerP& p,
               threadgroup atomic_uint* bits, threadgroup int* sel_ids, threadgroup float* sel_logit,
               threadgroup float* red_v, threadgroup int* red_i, uint tid, uint t) {
    const int n_vocab = p.n_vocab;
    const int nt = 512;
    const int lane = (int) tid;
    device const float* l = logits + (size_t) t * (size_t) n_vocab;
    device const int* hrow = history;
    const int hlen = p.hlen;
    if (hlen > 0) hrow = history + (size_t) t * (size_t) p.history_len + (size_t) (p.history_len - hlen);

    if (p.bits_words > 0) {
        for (int w = lane; w < p.bits_words; w += nt)
            atomic_store_explicit(bits + w, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = lane; i < hlen; i += nt) {
            const int id = hrow[i];
            if (id >= 0 && id < n_vocab)
                atomic_fetch_or_explicit(bits + (id >> 5), 1u << (uint) (id & 31), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int rounds = p.greedy ? 1 : p.k;
    for (int i = 0; i < rounds; ++i) {
        float bv = -INFINITY;
        int best = n_vocab;
        for (int v = lane; v < n_vocab; v += nt) {
            bool taken = false;
            if (!p.greedy) {
                for (int j = 0; j < i; ++j)
                    if (sel_ids[j] == v) {
                        taken = true;
                        break;
                    }
            }
            if (taken) continue;
            const float s = sm_apply(l[v], sm_count(hrow, hlen, v, bits, p.bits_words), p);
            if (s > bv) {
                bv = s;
                best = v;
            }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(best, off);
            sm_take(bv, best, ov, oi);
        }
        if ((lane & 31) == 0) {
            red_v[lane >> 5] = bv;
            red_i[lane >> 5] = best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (lane < nw) ? red_v[lane] : -INFINITY;
            int ix = (lane < nw) ? red_i[lane] : n_vocab;
            for (ushort off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_down(v, off);
                const int oi = simd_shuffle_down(ix, off);
                sm_take(v, ix, ov, oi);
            }
            if (lane == 0) {
                const int id = (ix < n_vocab) ? ix : 0;
                if (p.greedy) {
                    out[t] = id;
                } else {
                    sel_ids[i] = id;
                    sel_logit[i] = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (p.greedy || lane != 0) return;

    const int k = p.k;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        float ex[64];
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            ex[i] = exp(sel_logit[i] - mx);
            sum += ex[i];
        }
        float cum = 0.0f;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += ex[i] / sum;
            if (cum >= p.top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) {
                n_keep = i;
                break;
            }
    }
    float ex[64];
    float smx = sel_logit[0] * p.inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmax(smx, sel_logit[i] * p.inv_t);
    float sum = 0.0f;
    for (int i = 0; i < n_keep; ++i) {
        float z = sel_logit[i] * p.inv_t;
        z = z - smx;
        ex[i] = exp(z);
        sum += ex[i];
    }
    const float u = sm_philox(p, t);
    float cum = 0.0f;
    int pick = sel_ids[n_keep > 0 ? n_keep - 1 : 0];
    for (int i = 0; i < n_keep; ++i) {
        cum += ex[i] / sum;
        if (u < cum) {
            pick = sel_ids[i];
            break;
        }
    }
    out[t] = pick;
}

kernel void sampler_bits(device const float* logits [[buffer(0)]], device const int* history [[buffer(1)]],
                         device int* out [[buffer(2)]], constant SamplerP& p [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]], uint t [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint bits[7760];
    threadgroup int sel_ids[64];
    threadgroup float sel_logit[64];
    threadgroup float red_v[16];
    threadgroup int red_i[16];
    sm_sample(logits, history, out, p, bits, sel_ids, sel_logit, red_v, red_i, tid, t);
}

kernel void sampler_scan(device const float* logits [[buffer(0)]], device const int* history [[buffer(1)]],
                         device int* out [[buffer(2)]], constant SamplerP& p [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]], uint t [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint bits[1];
    threadgroup int sel_ids[64];
    threadgroup float sel_logit[64];
    threadgroup float red_v[16];
    threadgroup int red_i[16];
    sm_sample(logits, history, out, p, bits, sel_ids, sel_logit, red_v, red_i, tid, t);
}
