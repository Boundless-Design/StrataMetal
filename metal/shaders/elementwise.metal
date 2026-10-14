// Glue between the projections. softplus returns x above 20, before exp overflows.
// SiLU uses the precise f32 exp: a shader on this GPU has no fp64.
#include <metal_stdlib>
using namespace metal;

struct EwGate {
    int n_tokens;
    int h_v;
};
struct EwScale {
    int n;
    float s;
};
struct EwN {
    int n;
};

inline ushort ew_f16_from_f32(float f) {
    uint x = as_type<uint>(f);
    uint sign = (x >> 16) & 0x8000u;
    uint rawexp = (x >> 23) & 0xFFu;
    int exp = (int) rawexp - 127 + 15;
    uint man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (ushort) (sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (ushort) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (ushort) sign;
        man |= 0x800000u;
        uint sh = (uint) (14 - exp);
        uint h = (man >> sh) & 0x3FFu;
        uint rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) h++;
        return (ushort) (sign | h);
    }
    ushort h = (ushort) (sign | ((uint) exp << 10) | (man >> 13));
    uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return h;
}

// log(1 + y) without losing a small y. Metal has no log1p. When 1 + y rounds
// back to 1, the result is y itself.
inline float ew_log1p(float y) {
    float u = 1.0f + y;
    if (u == 1.0f) return y;
    return log(u) * (y / (u - 1.0f));
}

// Bit 16 is the lowest bit that survives. That is the even tie, not bit 15.
// A NaN is quieted. The rounding add would turn 0x7FFFFFFF into -0.
inline ushort ew_bf16_from_f32(float f) {
    uint i = as_type<uint>(f);
    if ((i & 0x7fffffffu) > 0x7f800000u) return (ushort) ((i >> 16) | 64u);
    i = (i + ((i >> 16) & 1u) + 0x7fffu) & 0xffff0000u;
    return (ushort) (i >> 16);
}

kernel void ew_gate(const device float* alpha [[buffer(0)]], const device float* dt [[buffer(1)]],
                    const device float* ssm_a [[buffer(2)]], device float* gate [[buffer(3)]],
                    constant EwGate& dim [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    int n = dim.n_tokens * dim.h_v;
    if ((int) i >= n) return;
    int h = (int) i % dim.h_v;
    float x = alpha[i] + dt[h];
    float sp = x;
    if (x <= 20.0f) sp = ew_log1p(exp(x));
    gate[i] = sp * ssm_a[h];
}

kernel void ew_scale(device float* x [[buffer(0)]], constant EwScale& dim [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if ((int) i < dim.n) x[i] *= dim.s;
}

kernel void ew_silu(device float* x [[buffer(0)]], constant EwN& dim [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if ((int) i >= dim.n) return;
    float v = x[i];
    x[i] = v / (1.0f + exp(-v));
}

kernel void ew_f32_to_f16(const device float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                          constant EwN& dim [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < dim.n) y[i] = ew_f16_from_f32(x[i]);
}

kernel void ew_f32_to_bf16(const device float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                           constant EwN& dim [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < dim.n) y[i] = ew_bf16_from_f32(x[i]);
}

// One embedding row. Codes are packed low bits first. The multiply and the add are
// separate statements: an fma rounds differently, and the check is bitwise.
struct EwEmbed {
    int n;
    int code_bits;
    int code_bias;
    int group_elems;
    int has_offset;
    int row;
    int out_base;
};

kernel void ew_embed(const device uchar* codes [[buffer(0)]], const device float* scales [[buffer(1)]],
                     const device float* offsets [[buffer(2)]], device float* out [[buffer(3)]],
                     constant EwEmbed& dim [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if ((int) i >= dim.n) return;
    int per_byte = 8 / dim.code_bits;
    int row_bytes = dim.n / per_byte;
    int row_groups = dim.n / dim.group_elems;
    const device uchar* c = codes + dim.row * row_bytes;
    const device float* s = scales + dim.row * row_groups;
    uint mask = (1u << dim.code_bits) - 1u;
    int code = (int) ((c[i / per_byte] >> ((i % per_byte) * dim.code_bits)) & mask);
    float value = (float) (code + dim.code_bias);
    float product = value * s[i / dim.group_elems];
    float add = 0.0f;
    if (dim.has_offset != 0) {
        const device float* o = offsets + dim.row * row_groups;
        add = o[i / dim.group_elems];
    }
    out[dim.out_base + i] = product + add;
}

kernel void ew_add(device float* dst [[buffer(0)]], const device float* src [[buffer(1)]],
                   constant EwN& dim [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < dim.n) dst[i] += src[i];
}

// QSA's norm: divide by sqrt(mean of squares + eps), then multiply by w. A null w is legal.
// This is not the DeltaNet L2, which divides by sqrt(sum of squares) and has no weight.
// One simdgroup per row. A rounded-up grid must not write the next row.
struct EwRms {
    int rows;
    int cols;
    float eps;
    int has_w;
};

kernel void ew_rms(device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                   constant EwRms& dim [[buffer(2)]], uint lane [[thread_index_in_simdgroup]],
                   uint row [[threadgroup_position_in_grid]]) {
    if ((int) row >= dim.rows) return;
    device float* r = x + (size_t) row * (size_t) dim.cols;
    float acc = 0.0f;
    for (int c = (int) lane; c < dim.cols; c += 32) acc += r[c] * r[c];
    for (ushort off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    float inv = 0.0f;
    if (lane == 0u) inv = rsqrt(acc / (float) dim.cols + dim.eps);
    inv = simd_shuffle(inv, (ushort) 0);
    for (int c = (int) lane; c < dim.cols; c += 32) {
        float v = r[c];
        if (dim.has_w != 0) v = v * w[c];
        r[c] = v * inv;
    }
}
