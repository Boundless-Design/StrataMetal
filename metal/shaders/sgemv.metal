// Dense projections. The weight is the canonical S form: code, scale, optional offset.
// Q8_K is 292 bytes per 256 values. Q8_0 is 34 bytes per 32. A simdgroup of 32 owns one
// output row in the split kernels and adds sixteen partials, then a down-shuffle.
#include <metal_stdlib>
using namespace metal;

struct SgDim {
    int n_in;
    int n_out;
    int bias;
    int codebook;
    int group_elems;
    int group_shift;
    int has_offset;
    int code_bits;
};

constant int sg_iq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float sg_f32_from_f16(ushort h) {
    uint sign = (uint) (h & 0x8000u) << 16;
    uint ex = (h >> 10) & 0x1Fu;
    uint man = h & 0x3FFu;
    uint outb;
    if (ex == 0) {
        if (man == 0) outb = sign;
        else {
            uint s = 0;
            uint m = man;
            while ((m & 0x400u) == 0) { m <<= 1; s++; }
            m &= 0x3FFu;
            outb = sign | ((113u - s) << 23) | (m << 13);
        }
    } else if (ex == 31) {
        outb = sign | 0x7F800000u | (man << 13);
    } else {
        outb = sign | ((ex + 112u) << 23) | (man << 13);
    }
    return as_type<float>(outb);
}

inline ushort sg_u16(const device uint8_t* p) { return (ushort) p[0] | ((ushort) p[1] << 8); }

inline float sg_f32_load(const device uint8_t* p) {
    uint u = (uint) p[0] | ((uint) p[1] << 8) | ((uint) p[2] << 16) | ((uint) p[3] << 24);
    return as_type<float>(u);
}

inline void sg_f32_store(device uint8_t* p, float f) {
    uint u = as_type<uint>(f);
    p[0] = (uint8_t) u;
    p[1] = (uint8_t) (u >> 8);
    p[2] = (uint8_t) (u >> 16);
    p[3] = (uint8_t) (u >> 24);
}

inline int sg_i8(uint b) { return b < 128u ? (int) b : (int) b - 256; }

inline float sg_decode(int code, int bias, int codebook) {
    if (codebook == 1) return (float) sg_iq4[code & 0x0F];
    return (float) (code + bias);
}

// ggml nearest_int: add 1.5 * 2^23, then the low mantissa bits are the rounded integer.
// The product has to be rounded to f32 before this add. A fused multiply-add picks different ties.
inline int sg_nearest_int(float fval) {
    float val = fval + 12582912.0f;
    int i = as_type<int>(val);
    return (i & 0x007fffff) - 0x00400000;
}

inline float sg_q8k_at(const device uint8_t* x, int i) {
    const device uint8_t* blk = x + (i / 256) * 292;
    return sg_f32_load(blk) * (float) sg_i8(blk[4 + (i % 256)]);
}

inline float sg_q8_0_at(const device uint8_t* x, int i) {
    const device uint8_t* blk = x + (i / 32) * 34;
    return sg_f32_from_f16(sg_u16(blk)) * (float) sg_i8(blk[2 + (i % 32)]);
}

inline uint sg_word(const device uint8_t* p, int qw) {
    uint v = p[0];
    if (qw > 1) v |= (uint) p[1] << 8;
    if (qw > 2) v |= ((uint) p[2] << 16) | ((uint) p[3] << 24);
    return v;
}

template <int CODE_BITS, int Q8K>
inline void sg_split(const device uint8_t* x, const device uint8_t* codes, const device float* scales,
                     const device float* offset, device float* y, constant SgDim& dim, uint lane, uint row) {
    if (row >= (uint) dim.n_out) return;
    constexpr int PER_BYTE = 8 / CODE_BITS;
    constexpr int QW = 4 * CODE_BITS / 8;
    constexpr int QE = 16;
    constexpr uint MASK = (1u << CODE_BITS) - 1u;
    int n_groups = dim.n_in >> dim.group_shift;
    const device uint8_t* c = codes + (long) row * (dim.n_in / PER_BYTE);
    const device float* s = scales + (long) row * n_groups;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    float a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0;
    for (int i = (int) lane * QE; i + QE <= dim.n_in; i += 32 * QE) {
        int g = i >> dim.group_shift;
        float d = s[g];
        float b = 0.0f;
        if (dim.has_offset) b = offset[(long) row * n_groups + g];
        const device uint8_t* cp = c + i / PER_BYTE;
        uint v0 = sg_word(cp, QW);
        uint v1 = sg_word(cp + QW, QW);
        uint v2 = sg_word(cp + 2 * QW, QW);
        uint v3 = sg_word(cp + 3 * QW, QW);
        constexpr int BLK = Q8K ? 256 : 32;
        constexpr int BBYTES = Q8K ? 292 : 34;
        constexpr int QOFF = Q8K ? 4 : 2;
        const device uint8_t* xb = x + (i / BLK) * BBYTES;
        int xi = i % BLK;
        float xd = Q8K ? sg_f32_load(xb) : sg_f32_from_f16(sg_u16(xb));
        const device uint8_t* xq = xb + QOFF + xi;
        float w0 = sg_decode((int) (v0 & MASK), dim.bias, dim.codebook) * d + b;
        float w1 = sg_decode((int) ((v0 >> CODE_BITS) & MASK), dim.bias, dim.codebook) * d + b;
        float w2 = sg_decode((int) ((v0 >> (2 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w3 = sg_decode((int) ((v0 >> (3 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w4 = sg_decode((int) (v1 & MASK), dim.bias, dim.codebook) * d + b;
        float w5 = sg_decode((int) ((v1 >> CODE_BITS) & MASK), dim.bias, dim.codebook) * d + b;
        float w6 = sg_decode((int) ((v1 >> (2 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w7 = sg_decode((int) ((v1 >> (3 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        a0 += w0 * (xd * (float) sg_i8(xq[0]));
        a1 += w1 * (xd * (float) sg_i8(xq[1]));
        a2 += w2 * (xd * (float) sg_i8(xq[2]));
        a3 += w3 * (xd * (float) sg_i8(xq[3]));
        a4 += w4 * (xd * (float) sg_i8(xq[4]));
        a5 += w5 * (xd * (float) sg_i8(xq[5]));
        a6 += w6 * (xd * (float) sg_i8(xq[6]));
        a7 += w7 * (xd * (float) sg_i8(xq[7]));
        float w8 = sg_decode((int) (v2 & MASK), dim.bias, dim.codebook) * d + b;
        float w9 = sg_decode((int) ((v2 >> CODE_BITS) & MASK), dim.bias, dim.codebook) * d + b;
        float w10 = sg_decode((int) ((v2 >> (2 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w11 = sg_decode((int) ((v2 >> (3 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w12 = sg_decode((int) (v3 & MASK), dim.bias, dim.codebook) * d + b;
        float w13 = sg_decode((int) ((v3 >> CODE_BITS) & MASK), dim.bias, dim.codebook) * d + b;
        float w14 = sg_decode((int) ((v3 >> (2 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        float w15 = sg_decode((int) ((v3 >> (3 * CODE_BITS)) & MASK), dim.bias, dim.codebook) * d + b;
        a8 += w8 * (xd * (float) sg_i8(xq[8]));
        a9 += w9 * (xd * (float) sg_i8(xq[9]));
        a10 += w10 * (xd * (float) sg_i8(xq[10]));
        a11 += w11 * (xd * (float) sg_i8(xq[11]));
        a12 += w12 * (xd * (float) sg_i8(xq[12]));
        a13 += w13 * (xd * (float) sg_i8(xq[13]));
        a14 += w14 * (xd * (float) sg_i8(xq[14]));
        a15 += w15 * (xd * (float) sg_i8(xq[15]));
    }
    float acc = (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
                (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)));
    for (uint step = 16u; step > 0u; step >>= 1) acc += simd_shuffle_down(acc, (ushort) step);
    if (lane == 0) y[row] = acc;
}

template <int CODE_BITS>
inline void sg_naive_q8k(const device uint8_t* x, const device uint8_t* codes, const device float* scales,
                         const device float* offset, device float* y, constant SgDim& dim, uint row) {
    if (row >= (uint) dim.n_out) return;
    constexpr int PER_BYTE = 8 / CODE_BITS;
    int n_groups = dim.n_in / dim.group_elems;
    const device uint8_t* c = codes + (long) row * (dim.n_in / PER_BYTE);
    const device float* s = scales + (long) row * n_groups;
    float acc = 0;
    for (int g = 0; g < n_groups; g++) {
        float d = s[g];
        float b = 0.0f;
        if (dim.has_offset) b = offset[(long) row * n_groups + g];
        int base = g * dim.group_elems;
        for (int j = 0; j < dim.group_elems; j++) {
            int i = base + j;
            int code = (c[i / PER_BYTE] >> ((i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            float w = sg_decode(code, dim.bias, dim.codebook) * d + b;
            acc += w * sg_q8k_at(x, i);
        }
    }
    y[row] = acc;
}

template <int CODE_BITS>
inline void sg_naive_f16(const device ushort* x, const device uint8_t* codes, const device float* scales,
                         const device float* offset, device float* y, constant SgDim& dim, uint row) {
    if (row >= (uint) dim.n_out) return;
    constexpr int PER_BYTE = 8 / CODE_BITS;
    int n_groups = dim.n_in / dim.group_elems;
    const device uint8_t* c = codes + (long) row * (dim.n_in / PER_BYTE);
    const device float* s = scales + (long) row * n_groups;
    float acc = 0;
    for (int g = 0; g < n_groups; g++) {
        float d = s[g];
        float b = 0.0f;
        if (dim.has_offset) b = offset[(long) row * n_groups + g];
        int base = g * dim.group_elems;
        for (int j = 0; j < dim.group_elems; j++) {
            int i = base + j;
            int code = (c[i / PER_BYTE] >> ((i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            float w = sg_decode(code, dim.bias, dim.codebook) * d + b;
            acc += w * sg_f32_from_f16(x[i]);
        }
    }
    y[row] = acc;
}

inline void sg_dispatch_split(const device uint8_t* x, const device uint8_t* codes, const device float* scales,
                              const device float* offset, device float* y, constant SgDim& dim, uint lane,
                              uint row, int q8k) {
    if (q8k) {
        if (dim.code_bits == 8) sg_split<8, 1>(x, codes, scales, offset, y, dim, lane, row);
        else sg_split<4, 1>(x, codes, scales, offset, y, dim, lane, row);
    } else {
        if (dim.code_bits == 8) sg_split<8, 0>(x, codes, scales, offset, y, dim, lane, row);
        else sg_split<4, 0>(x, codes, scales, offset, y, dim, lane, row);
    }
}

kernel void sg_gemv_q8k_split(const device uint8_t* x [[buffer(0)]], const device uint8_t* codes [[buffer(1)]],
                              const device float* scales [[buffer(2)]], const device float* offset [[buffer(3)]],
                              device float* y [[buffer(4)]], constant SgDim& dim [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint tg [[threadgroup_position_in_grid]]) {
    uint row = tg * 8u + simd;
    sg_dispatch_split(x, codes, scales, offset, y, dim, lane, row, 1);
}

kernel void sg_gemv_q8_0_split(const device uint8_t* x [[buffer(0)]], const device uint8_t* codes [[buffer(1)]],
                               const device float* scales [[buffer(2)]], const device float* offset [[buffer(3)]],
                               device float* y [[buffer(4)]], constant SgDim& dim [[buffer(5)]],
                               uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint tg [[threadgroup_position_in_grid]]) {
    uint row = tg * 8u + simd;
    sg_dispatch_split(x, codes, scales, offset, y, dim, lane, row, 0);
}

kernel void sg_gemv_q8k(const device uint8_t* x [[buffer(0)]], const device uint8_t* codes [[buffer(1)]],
                        const device float* scales [[buffer(2)]], const device float* offset [[buffer(3)]],
                        device float* y [[buffer(4)]], constant SgDim& dim [[buffer(5)]],
                        uint row [[thread_position_in_grid]]) {
    if (dim.code_bits == 8) sg_naive_q8k<8>(x, codes, scales, offset, y, dim, row);
    else sg_naive_q8k<4>(x, codes, scales, offset, y, dim, row);
}

kernel void sg_gemv_f16(const device ushort* x [[buffer(0)]], const device uint8_t* codes [[buffer(1)]],
                        const device float* scales [[buffer(2)]], const device float* offset [[buffer(3)]],
                        device float* y [[buffer(4)]], constant SgDim& dim [[buffer(5)]],
                        uint row [[thread_position_in_grid]]) {
    if (dim.code_bits == 8) sg_naive_f16<8>(x, codes, scales, offset, y, dim, row);
    else if (dim.code_bits == 2) sg_naive_f16<2>(x, codes, scales, offset, y, dim, row);
    else sg_naive_f16<4>(x, codes, scales, offset, y, dim, row);
}

kernel void sg_q8k(const device float* x [[buffer(0)]], device uint8_t* blocks [[buffer(1)]],
                   constant int& n_blocks [[buffer(2)]], uint blk [[thread_position_in_grid]]) {
    if (blk >= (uint) n_blocks) return;
    const device float* xb = x + (long) blk * 256;
    device uint8_t* out = blocks + (long) blk * 292;
    float maxv = 0.0f, amax = 0.0f;
    for (int j = 0; j < 256; j++) {
        float xj = xb[j];
        float ax = fabs(xj);
        if (ax > amax) {
            amax = ax;
            maxv = xj;
        }
    }
    if (amax == 0.0f) {
        for (int j = 0; j < 292; j++) out[j] = 0;
        return;
    }
    float iscale = -127.0f / maxv;
    for (int j = 0; j < 256; j++) {
        float prod = iscale * xb[j];
        int v = sg_nearest_int(prod);
        if (v > 127) v = 127;
        out[4 + j] = (uint8_t) v;
    }
    for (int j = 0; j < 16; j++) {
        int sum = 0;
        for (int ii = 0; ii < 16; ii++) sum += sg_i8(out[4 + j * 16 + ii]);
        int s = sum;
        out[260 + 2 * j] = (uint8_t) s;
        out[260 + 2 * j + 1] = (uint8_t) (s >> 8);
    }
    sg_f32_store(out, 1.0f / iscale);
}

kernel void sg_dequant_q8k(const device uint8_t* blocks [[buffer(0)]], device float* x [[buffer(1)]],
                           constant int& n_blocks [[buffer(2)]], uint blk [[thread_position_in_grid]]) {
    if (blk >= (uint) n_blocks) return;
    const device uint8_t* b = blocks + (long) blk * 292;
    float d = sg_f32_load(b);
    device float* out = x + (long) blk * 256;
    for (int j = 0; j < 256; j++) out[j] = (float) sg_i8(b[4 + j]) * d;
}
