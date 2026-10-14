// Paged split-K decode attention. Same arithmetic as src/kernels/cuda/qsa_decode_attn.cu
// (the kernel the engine ships; not the experimental pre-sm_75 variant).
//
// One threadgroup owns one chunk of 64 selected cells and one KV head, and the 12 query
// heads that share it. Head dim is 256: lane L holds dimensions [8L, 8L+8) of the key,
// and thread t holds value dimension t. A simdgroup is 32 lanes, and the dot reduction
// is the xor butterfly, so the sum is the CUDA warp sum.
//
// KV_MODE 0 fp16, 1 int8, 2 q4_0, 3 hybrid (int8 keys, q4_0 values). A page-table entry
// of -1 is a cell the cache could not keep resident: its score is -FLT_MAX and its
// weight is 0. The grid is sized by the capacity. The live count is step[3].
#include <metal_stdlib>
using namespace metal;

constant int kHd = 256;
constant int kG = 12;
constant int kChunk = 64;
constant int kThreads = 256;
constant int kWarps = kThreads / 32;
constant int kQ4Bytes = (kHd / 32) * 18;

struct QsaChunkDim {
    int n_kv_heads;
    int page_size;
    int n_chunks;
    int cap;
    int stride;
    float scale;
};
struct QsaMergeDim {
    int n_chunks;
    int n_head;
    int stride;
};

inline float neg_flt_max() { return as_type<float>(0xFF7FFFFFu); }

// Exact fp16 bits to f32, including subnormals. The same mapping as f32_from_f16.
inline float f32_from_f16(ushort h) {
    uint sign = uint(h & 0x8000u) << 16u;
    uint ex = (uint(h) >> 10u) & 0x1Fu;
    uint man = uint(h) & 0x3FFu;
    uint out;
    if (ex == 0u) {
        if (man == 0u) {
            out = sign;
        } else {
            uint m = man;
            int s = 0;
            while ((m & 0x400u) == 0u) { m <<= 1u; ++s; }
            out = sign | (uint(113 - s) << 23u) | ((m & 0x3FFu) << 13u);
        }
    } else if (ex == 31u) {
        out = sign | 0x7F800000u | (man << 13u);
    } else {
        out = sign | ((ex - 15u + 127u) << 23u) | (man << 13u);
    }
    return as_type<float>(out);
}

inline float warp_sum(float v) {
    v += simd_shuffle_xor(v, ushort(16));
    v += simd_shuffle_xor(v, ushort(8));
    v += simd_shuffle_xor(v, ushort(4));
    v += simd_shuffle_xor(v, ushort(2));
    v += simd_shuffle_xor(v, ushort(1));
    return v;
}

inline float warp_max(float v) {
    v = max(v, simd_shuffle_xor(v, ushort(16)));
    v = max(v, simd_shuffle_xor(v, ushort(8)));
    v = max(v, simd_shuffle_xor(v, ushort(4)));
    v = max(v, simd_shuffle_xor(v, ushort(2)));
    v = max(v, simd_shuffle_xor(v, ushort(1)));
    return v;
}

inline int sext8(uchar u) { return (u & 0x80u) != 0u ? int(u) - 256 : int(u); }

inline float f16_bytes(device const uchar* p, long index) {
    device const uchar* b = p + index * 2;
    ushort bits = ushort(b[0]) | (ushort(b[1]) << 8u);
    return f32_from_f16(bits);
}

inline void load8_f16(device const uchar* pool, long row, int d0, thread float* out) {
    long base = (row * kHd + d0) * 2;
    for (int j = 0; j < 8; ++j) {
        ushort bits = ushort(pool[base + 2 * j]) | (ushort(pool[base + 2 * j + 1]) << 8u);
        out[j] = f32_from_f16(bits);
    }
}

inline void load8_q8(device const uchar* codes, device const uchar* scales, long row, int d0, thread float* out) {
    float sc = f16_bytes(scales, row * (kHd / 64) + d0 / 64);
    for (int j = 0; j < 8; ++j) out[j] = float(sext8(codes[row * kHd + d0 + j])) * sc;
}

inline float q4_scale(device const uchar* blk) {
    ushort bits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
    return f32_from_f16(bits);
}

inline void load8_q4(device const uchar* pool, long row, int d0, thread float* out) {
    int b = d0 / 32;
    int rem = d0 - b * 32;
    device const uchar* blk = pool + row * kQ4Bytes + b * 18;
    float d = q4_scale(blk);
    int j = (rem == 0 || rem == 16) ? 0 : 8;
    device const uchar* bytes = blk + 2 + j;
    if (rem < 16) {
        for (int k = 0; k < 8; ++k) out[k] = float(int(bytes[k] & 0x0Fu) - 8) * d;
    } else {
        for (int k = 0; k < 8; ++k) out[k] = float(int(bytes[k] >> 4u) - 8) * d;
    }
}

inline float load_v_f16(device const uchar* pool, long row, int dim) {
    return f16_bytes(pool, row * kHd + dim);
}

inline float load_v_q8(device const uchar* codes, device const uchar* scales, long row, int dim) {
    float sc = f16_bytes(scales, row * (kHd / 64) + dim / 64);
    return float(sext8(codes[row * kHd + dim])) * sc;
}

inline float load_v_q4(device const uchar* pool, long row, int dim) {
    int b = dim / 32;
    int rem = dim - b * 32;
    device const uchar* blk = pool + row * kQ4Bytes + b * 18;
    float d = q4_scale(blk);
    int j = rem < 16 ? rem : (rem - 16);
    uchar byte = blk[2 + j];
    int nibble = rem < 16 ? (int(byte & 0x0Fu) - 8) : (int(byte >> 4u) - 8);
    return float(nibble) * d;
}

// MODE 0 fp16, 1 int8, 2 q4_0, 3 int8 keys and q4_0 values.
template <int MODE>
inline void load_k8(device const uchar* k, device const uchar* k_extra, long row, int d0, thread float* out) {
    if (MODE == 0) load8_f16(k, row, d0, out);
    else if (MODE == 2) load8_q4(k, row, d0, out);
    else load8_q8(k, k_extra, row, d0, out);
}

template <int MODE>
inline float load_v(device const uchar* v, device const uchar* v_extra, long row, int dim) {
    if (MODE == 0) return load_v_f16(v, row, dim);
    if (MODE == 1) return load_v_q8(v, v_extra, row, dim);
    return load_v_q4(v, row, dim);
}

template <int MODE>
void chunk_body(device const float* q, device const int* ids, device const int* step, device const int* page_table,
                device float* scratch, device const uchar* k_bytes, device const uchar* v_bytes,
                device const uchar* k_extra, device const uchar* v_extra, constant QsaChunkDim& dim,
                threadgroup float* sq, threadgroup float* sp, threadgroup long* srow, uint t, uint3 gpos) {
    const int chunk = int(gpos.x);
    const int kvh = int(gpos.y);
    const int z = int(gpos.z);
    const int lane = int(t) & 31;
    const int warp = int(t) >> 5;
    const int n_head = dim.n_kv_heads * kG;
    const long acc_n = (long) dim.n_chunks * n_head * kHd;
    const long ml_n = (long) dim.n_chunks * n_head;
    const long base = (long) z * dim.stride;
    q += (long) z * dim.n_kv_heads * kG * kHd;
    ids += (long) z * dim.cap;
    step += (long) z * 4;
    device float* part_acc = scratch + base;
    device float* part_m = scratch + acc_n + base;
    device float* part_l = scratch + acc_n + ml_n + base;

    const int n_ids = step[3];
    const int c0 = chunk * kChunk;
    const int n_here = min(kChunk, n_ids - c0);
    const int slot = kvh * dim.n_chunks + chunk;
    if (n_here <= 0) {
        if (int(t) < kG) {
            part_m[slot * kG + int(t)] = neg_flt_max();
            part_l[slot * kG + int(t)] = 0.0f;
        }
        return;
    }

    for (int i = int(t); i < kG * kHd; i += kThreads) sq[i] = q[(long) kvh * kG * kHd + i];
    if (int(t) < kChunk) {
        long r = -1;
        if (int(t) < n_here) {
            const int cell = ids[c0 + int(t)];
            const int page = page_table[cell / dim.page_size];
            if (page >= 0) r = ((long) page * dim.n_kv_heads + kvh) * dim.page_size + (cell % dim.page_size);
        }
        srow[t] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = warp; c < kChunk; c += kWarps) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < kG) sp[lane * kChunk + c] = neg_flt_max();
            continue;
        }
        float k8[8];
        load_k8<MODE>(k_bytes, k_extra, srow[c], lane * 8, k8);
        for (int h = 0; h < kG; ++h) {
            float s = 0.0f;
            for (int j = 0; j < 8; ++j) s += k8[j] * sq[h * kHd + lane * 8 + j];
            s = warp_sum(s);
            if (lane == 0) sp[h * kChunk + c] = s * dim.scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = warp; h < kG; h += kWarps) {
        const float a = sp[h * kChunk + lane];
        const float b = sp[h * kChunk + lane + 32];
        const float m = warp_max(max(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? exp(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? exp(b - m) : 0.0f;
        sp[h * kChunk + lane] = ea;
        sp[h * kChunk + lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) {
            part_m[slot * kG + h] = m;
            part_l[slot * kG + h] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[kG];
    for (int h = 0; h < kG; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        const float v = load_v<MODE>(v_bytes, v_extra, srow[c], int(t));
        for (int h = 0; h < kG; ++h) acc[h] = fma(sp[h * kChunk + c], v, acc[h]);
    }
    for (int h = 0; h < kG; ++h) part_acc[((long) slot * kG + h) * kHd + t] = acc[h];
}

// Four names, one body. A mode the kernel does not read still has a buffer bound, so the
// signature stays one function and the unused pointer is the pool itself.
kernel void qsa_chunk_f16(device const float* q [[buffer(0)]], device const int* ids [[buffer(1)]],
                          device const int* step [[buffer(2)]], device const int* page_table [[buffer(3)]],
                          device float* scratch [[buffer(4)]], device const uchar* k_bytes [[buffer(5)]],
                          device const uchar* v_bytes [[buffer(6)]], constant QsaChunkDim& dim [[buffer(7)]],
                          uint t [[thread_index_in_threadgroup]], uint3 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float sq[kG * kHd];
    threadgroup float sp[kG * kChunk];
    threadgroup long srow[kChunk];
    chunk_body<0>(q, ids, step, page_table, scratch, k_bytes, v_bytes, k_bytes, v_bytes, dim, sq, sp, srow, t, gpos);
}

kernel void qsa_chunk_q8(device const float* q [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device const int* step [[buffer(2)]], device const int* page_table [[buffer(3)]],
                         device float* scratch [[buffer(4)]], device const uchar* k_bytes [[buffer(5)]],
                         device const uchar* v_bytes [[buffer(6)]], device const uchar* k_scale [[buffer(7)]],
                         device const uchar* v_scale [[buffer(8)]], constant QsaChunkDim& dim [[buffer(9)]],
                         uint t [[thread_index_in_threadgroup]], uint3 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float sq[kG * kHd];
    threadgroup float sp[kG * kChunk];
    threadgroup long srow[kChunk];
    chunk_body<1>(q, ids, step, page_table, scratch, k_bytes, v_bytes, k_scale, v_scale, dim, sq, sp, srow, t, gpos);
}

kernel void qsa_chunk_q4(device const float* q [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device const int* step [[buffer(2)]], device const int* page_table [[buffer(3)]],
                         device float* scratch [[buffer(4)]], device const uchar* k_bytes [[buffer(5)]],
                         device const uchar* v_bytes [[buffer(6)]], constant QsaChunkDim& dim [[buffer(7)]],
                         uint t [[thread_index_in_threadgroup]], uint3 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float sq[kG * kHd];
    threadgroup float sp[kG * kChunk];
    threadgroup long srow[kChunk];
    chunk_body<2>(q, ids, step, page_table, scratch, k_bytes, v_bytes, k_bytes, v_bytes, dim, sq, sp, srow, t, gpos);
}

kernel void qsa_chunk_k8v4(device const float* q [[buffer(0)]], device const int* ids [[buffer(1)]],
                           device const int* step [[buffer(2)]], device const int* page_table [[buffer(3)]],
                           device float* scratch [[buffer(4)]], device const uchar* k_bytes [[buffer(5)]],
                           device const uchar* k_scale [[buffer(6)]], device const uchar* v_bytes [[buffer(7)]],
                           constant QsaChunkDim& dim [[buffer(8)]], uint t [[thread_index_in_threadgroup]],
                           uint3 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float sq[kG * kHd];
    threadgroup float sp[kG * kChunk];
    threadgroup long srow[kChunk];
    chunk_body<3>(q, ids, step, page_table, scratch, k_bytes, v_bytes, k_scale, v_bytes, dim, sq, sp, srow, t, gpos);
}

kernel void qsa_attn_merge(device const float* scratch [[buffer(0)]], device float* attn [[buffer(1)]],
                           constant QsaMergeDim& dim [[buffer(2)]], uint d [[thread_index_in_threadgroup]],
                           uint3 gpos [[threadgroup_position_in_grid]]) {
    const int h = int(gpos.x);
    const int z = int(gpos.y);
    const int kvh = h / kG;
    const int hl = h - kvh * kG;
    const long acc_n = (long) dim.n_chunks * dim.n_head * kHd;
    const long ml_n = (long) dim.n_chunks * dim.n_head;
    const long base = (long) z * dim.stride;
    device const float* part_acc = scratch + base;
    device const float* part_m = scratch + acc_n + base;
    device const float* part_l = scratch + acc_n + ml_n + base;
    device float* out = attn + (long) z * dim.n_head * kHd;

    float M = neg_flt_max();
    for (int c = 0; c < dim.n_chunks; ++c) M = max(M, part_m[(kvh * dim.n_chunks + c) * kG + hl]);
    float L = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < dim.n_chunks; ++c) {
        const int slot = kvh * dim.n_chunks + c;
        const float m = part_m[slot * kG + hl];
        if (m == neg_flt_max()) continue;
        const float w = exp(m - M);
        L = fma(part_l[slot * kG + hl], w, L);
        acc = fma(part_acc[((long) slot * kG + hl) * kHd + d], w, acc);
    }
    out[(long) h * kHd + d] = L > 0.0f ? acc / L : 0.0f;
}

kernel void qsa_simd_width(device uint* out [[buffer(0)]], uint simd_w [[threads_per_simdgroup]],
                           uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0u) out[0] = simd_w;
}
