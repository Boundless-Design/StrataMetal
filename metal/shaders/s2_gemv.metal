// 2-bit GEMV over a Q8_0 activation. One threadgroup per output row.
// A code is (byte >> shift) & 3, and the weight is (code - 1) * scale. The scale
// covers 64 values. Four int8 of one Q8_0 block share that block's fp16 scale.
// The four partials are added, then the threadgroup reduces them in index order.
#include <metal_stdlib>
using namespace metal;

struct S2P {
    int n_in;
    int n_out;
    int tpr;
    int pad;
};

inline float s2_f32_from_f16(ushort h) {
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

inline int s2_i8(uint b) { return b < 128u ? (int) b : (int) b - 256; }

kernel void s2_gemv_q8(device const uint8_t* act [[buffer(0)]],
                       device const uint8_t* codes [[buffer(1)]],
                       device const float* scales [[buffer(2)]],
                       device float* y [[buffer(3)]],
                       constant S2P& dim [[buffer(4)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint o [[threadgroup_position_in_grid]]) {
    threadgroup float partial[1024];
    if (o >= (uint) dim.n_out) return;
    const int tpr = dim.tpr;
    const int n_in = dim.n_in;
    const long n_quads = (long) n_in / 4;
    device const uint8_t* c = codes + (long) o * n_quads;
    device const float* s = scales + (long) o * (n_in / 64);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long q = (long) tid; q < n_quads; q += tpr) {
        const uint byte = c[q];
        const float d = s[q >> 4];
        const long ablk = (q * 4) / 32;
        device const uint8_t* blk = act + ablk * 34;
        const ushort dbits = (ushort) blk[0] | ((ushort) blk[1] << 8);
        const float dx = s2_f32_from_f16(dbits);
        const int off = (int) ((q * 4) % 32);
        const float w0 = (float) ((int) (byte & 3u) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3u) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3u) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3u) - 1) * d;
        a0 += w0 * ((float) s2_i8(blk[2 + off + 0]) * dx);
        a1 += w1 * ((float) s2_i8(blk[2 + off + 1]) * dx);
        a2 += w2 * ((float) s2_i8(blk[2 + off + 2]) * dx);
        a3 += w3 * ((float) s2_i8(blk[2 + off + 3]) * dx);
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = tpr >> 1; step > 0; step >>= 1) {
        if ((int) tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[o] = partial[0];
}
