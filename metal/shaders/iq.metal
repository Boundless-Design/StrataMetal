// Coder expert product. The integer dots are the ones in src/kernels/cuda/iq_kernels.cu
// (llama.cpp vec_dot_*_q8_1): IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS for gate and up, IQ4_NL and Q2_0
// for down. Q4_K and Q5_K are the dense gate and the dense output. Q3_K and Q5_0 are the
// shared-expert up and down. A simdgroup of 32 owns one weight row. Lane L adds partials
// L, L+32, L+64, ... and the xor butterfly is the warp sum. q8_1 is 36 bytes: fp16 scale,
// fp16 sum, 32 int8. This port's sum is the sum of the floats, so Q5_0 recomputes the
// int8 sum instead of reading that word.
#include <metal_stdlib>
using namespace metal;

struct IqDim {
    int n_in;
    int n_out;
    int n_col;
    int row_bytes;
    int byte_off;
};

constant int kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float iq_f32_from_f16(ushort h) {
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

inline ushort iq_f16_from_f32(float f) {
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

inline int iq_ld(const device uint8_t* p) {
    return (int) p[0] | ((int) p[1] << 8) | ((int) p[2] << 16) | ((int) p[3] << 24);
}

inline int iq_b2(const device uint8_t* p, int i) { return iq_ld(p + 4 * i); }

inline ushort iq_u16(const device uint8_t* p) { return (ushort) p[0] | ((ushort) p[1] << 8); }

inline int iq_dp4a(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; i++) {
        int sa = (a >> (8 * i)) & 0xFF;
        int sb = (b >> (8 * i)) & 0xFF;
        if (sa >= 128) sa -= 256;
        if (sb >= 128) sb -= 256;
        s += sa * sb;
    }
    return s;
}

inline int iq_vcmpne4(uint a) {
    int r = 0;
    for (int i = 0; i < 4; i++)
        if (((a >> (8 * i)) & 0xFFu) != 0) r |= 0xFF << (8 * i);
    return r;
}

inline int iq_vsub4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; i++) {
        int ba = (a >> (8 * i)) & 0xFF;
        int bb = (b >> (8 * i)) & 0xFF;
        r |= ((ba - bb) & 0xFF) << (8 * i);
    }
    return r;
}

inline uint iq_unpack_ksigns(uint v) {
    v &= 0xFFu;
    uint p = popcount(v) & 1u;
    uint s = v ^ (p << 7);
    return s * 0x01010101u;
}

inline int iq_sign_pair(int grid, uint sb, bool high) {
    uint bits = high ? (((sb & 0x30u) << 3) | ((sb & 0xC0u) << 17))
                     : (((sb & 0x03u) << 7) | ((sb & 0x0Cu) << 21));
    int mask = iq_vcmpne4(bits);
    return iq_vsub4(grid ^ mask, mask);
}

inline float iq_warp_add(float v) {
    v += simd_shuffle_xor(v, 16u);
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

inline int iq_q8_word(const device uint8_t* block, int i) { return iq_ld(block + 4 + 4 * i); }

inline float iq_q8_d(const device uint8_t* block) { return iq_f32_from_f16(iq_u16(block)); }

inline void iq_nibbles(int aux, int& vx, int& vy) {
    vx = 0;
    vy = 0;
    for (int b = 0; b < 4; b++) {
        uint byte = (uint) (aux >> (8 * b)) & 0xFFu;
        vx |= (int) (uint) (uchar) kIq4[byte & 0xFu] << (8 * b);
        vy |= (int) (uint) (uchar) kIq4[byte >> 4] << (8 * b);
    }
}

inline int iq_q2_pack(uint byte) {
    int r = 0;
    for (int c = 0; c < 4; c++) {
        int sym = (int) ((byte >> (2 * c)) & 3u) - 1;
        r |= (sym & 0xFF) << (8 * c);
    }
    return r;
}

inline float iq_dot_iq2_s(const device uint8_t* blk, const device uint8_t* x0, int iqs, const device ulong* grid) {
    int packed = iq_b2(blk + 2, iqs / 2);
    int qh = (blk + 66)[iqs / 2];
    int signs = iq_b2(blk + 2, 8 + iqs / 2);
    int sc = (blk + 74)[iqs / 2];
    int ls0 = sc & 0x0F, ls1 = sc >> 4;
    int sum0 = 0, sum1 = 0;
    const device uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint qb = (uint) (packed >> (8 * (l0 / 2))) & 0xFFu;
        uint idx = qb | (uint) ((qh << (8 - l0)) & 0x300);
        ulong g = grid[idx];
        uint sb = (uint) (signs >> (8 * (l0 / 2))) & 0xFFu;
        int gl = iq_sign_pair((int) (g & 0xFFFFFFFFul), sb, false);
        int gh = iq_sign_pair((int) (g >> 32), sb, true);
        int u0 = iq_q8_word(x, l0);
        int u1 = iq_q8_word(x, l0 + 1);
        if (l0 < 4) {
            sum0 = iq_dp4a(gl, u0, sum0);
            sum0 = iq_dp4a(gh, u1, sum0);
        } else {
            sum1 = iq_dp4a(gl, u0, sum1);
            sum1 = iq_dp4a(gh, u1, sum1);
        }
    }
    int sumi = (sum0 * ls0 + sum1 * ls1 + (sum0 + sum1) / 2) / 4;
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x);
    return d * (float) sumi;
}

inline float iq_dot_iq3_xxs(const device uint8_t* blk, const device uint8_t* x0, int iqs, const device uint* grid) {
    const device uint8_t* qs = blk + 2;
    int aux = iq_b2(qs, 16 + iqs / 2);
    int sumi = 0;
    const device uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint i0 = qs[4 * iqs + l0];
        uint i1 = qs[4 * iqs + l0 + 1];
        uint signs = iq_unpack_ksigns((uint) aux >> (uint) (7 * l0 / 2));
        int gl = iq_vsub4((int) grid[i0] ^ iq_vcmpne4(signs & 0x08040201u), iq_vcmpne4(signs & 0x08040201u));
        int gh = iq_vsub4((int) grid[i1] ^ iq_vcmpne4(signs & 0x80402010u), iq_vcmpne4(signs & 0x80402010u));
        sumi = iq_dp4a(gl, iq_q8_word(x, l0), sumi);
        sumi = iq_dp4a(gh, iq_q8_word(x, l0 + 1), sumi);
    }
    // aux32 is unsigned in the CUDA dot. A signed shift would sign-extend when bit 31 is set.
    int ls = (int) ((uint) aux >> 28);
    sumi = (ls * sumi + sumi / 2) / 2;
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x);
    return d * (float) sumi;
}

inline float iq_dot_iq3_s(const device uint8_t* blk, const device uint8_t* x0, int iqs, const device uint* grid) {
    const device uint8_t* qs = blk + 2;
    int qh = (blk + 66)[iqs / 2];
    int signs = iq_b2(blk + 74, iqs / 2);
    int sumi = 0;
    const device uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint i0 = qs[4 * iqs + l0] | (uint) ((qh << (8 - l0)) & 0x100);
        uint i1 = qs[4 * iqs + l0 + 1] | (uint) ((qh << (7 - l0)) & 0x100);
        uint sb = (uint) (signs >> (8 * (l0 / 2))) & 0xFFu;
        int gl = iq_sign_pair((int) grid[i0], sb, false);
        int gh = iq_sign_pair((int) grid[i1], sb, true);
        sumi = iq_dp4a(gl, iq_q8_word(x, l0), sumi);
        sumi = iq_dp4a(gh, iq_q8_word(x, l0 + 1), sumi);
    }
    int nib = ((blk + 106)[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F;
    sumi *= 1 + 2 * nib;
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x);
    return d * (float) sumi;
}

inline float iq_dot_iq4_nl(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    int sumi = 0;
    for (int l = 0; l < 2; l++) {
        int vx, vy;
        iq_nibbles(iq_b2(blk + 2, iqs + l), vx, vy);
        sumi = iq_dp4a(vx, iq_q8_word(x0, iqs + l), sumi);
        sumi = iq_dp4a(vy, iq_q8_word(x0, iqs + l + 4), sumi);
    }
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x0);
    return d * (float) sumi;
}

inline float iq_dot_iq4_xs(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    const device uint8_t* qs = blk + 8;
    const device uint8_t* x = x0 + (iqs / 4) * 36;
    int sumi = 0;
    for (int j = 0; j < 4; j++) {
        int vx, vy;
        iq_nibbles(iq_ld(qs + 4 * (iqs + j)), vx, vy);
        sumi = iq_dp4a(vx, iq_q8_word(x, j), sumi);
        sumi = iq_dp4a(vy, iq_q8_word(x, j + 4), sumi);
    }
    ushort sh = iq_u16(blk + 2);
    int ls = (((blk + 4)[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((sh >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x);
    return d * (float) sumi;
}

inline void iq_k_scale_min(const device uint8_t* scales8, int bq8_offset, ushort aux[2]) {
    int j = bq8_offset / 2;
    int jm = j & 1;
    uint s0 = (uint) iq_u16(scales8 + 2 * (jm + 0));
    uint s2 = (uint) iq_u16(scales8 + 2 * (jm + 2));
    uint s4 = (uint) iq_u16(scales8 + 2 * (jm + 4));
    uint hi = j >= 2 ? 0xFFFFFFFFu : 0u;
    aux[0] = (ushort) (((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi));
    aux[1] = (ushort) (((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi));
}

inline float iq_dot_q4_k(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    int bq8_offset = 2 * ((iqs / 2) / 4);
    const device uint8_t* qs = blk + 16;
    int byte = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    int v0 = iq_ld(qs + byte);
    int v1 = iq_ld(qs + byte + 16);
    ushort aux[2];
    iq_k_scale_min(blk + 4, bq8_offset, aux);
    uint packed = (uint) aux[0] | ((uint) aux[1] << 16);
    float sum_d = 0;
    float sum_m = 0;
    for (int i = 0; i < 2; i++) {
        const device uint8_t* bq8 = x0 + (long) (bq8_offset + i) * 36;
        float d8 = iq_q8_d(bq8);
        int qoff = (iqs / 2) % 4;
        int u0 = iq_ld(bq8 + 4 + 4 * qoff);
        int u1 = iq_ld(bq8 + 4 + 4 * (qoff + 4));
        int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
        int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        int dot1 = iq_dp4a(v1i, u1, iq_dp4a(v0i, u0, 0));
        int dot2 = iq_dp4a(0x01010101, u1, iq_dp4a(0x01010101, u0, 0));
        int sc = (int) ((packed >> (8 * i)) & 0xFFu);
        int mn = (int) ((packed >> (16 + 8 * i)) & 0xFFu);
        sum_d += d8 * (float) (dot1 * sc);
        sum_m += d8 * (float) (dot2 * mn);
    }
    float d = iq_f32_from_f16(iq_u16(blk));
    float dmin = iq_f32_from_f16(iq_u16(blk + 2));
    return d * sum_d - dmin * sum_m;
}

inline float iq_dot_q5_k(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    int bq8_offset = 2 * ((iqs / 2) / 4);
    const device uint8_t* qh = blk + 16;
    const device uint8_t* ql = blk + 48;
    int byte = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    int vl0 = iq_ld(ql + byte);
    int vl1 = iq_ld(ql + byte + 16);
    int qhb = 4 * ((iqs / 2) % 4);
    int vh0 = iq_ld(qh + qhb) >> bq8_offset;
    int vh1 = iq_ld(qh + qhb + 16) >> bq8_offset;
    ushort aux[2];
    iq_k_scale_min(blk + 4, bq8_offset, aux);
    uint packed = (uint) aux[0] | ((uint) aux[1] << 16);
    float sum_d = 0;
    float sum_m = 0;
    for (int i = 0; i < 2; i++) {
        const device uint8_t* bq8 = x0 + (long) (bq8_offset + i) * 36;
        float d8 = iq_q8_d(bq8);
        int qoff = (iqs / 2) % 4;
        int u0 = iq_ld(bq8 + 4 + 4 * qoff);
        int u1 = iq_ld(bq8 + 4 + 4 * (qoff + 4));
        int vl0i = (vl0 >> (4 * i)) & 0x0F0F0F0F;
        int vl1i = (vl1 >> (4 * i)) & 0x0F0F0F0F;
        int vh0i = ((vh0 >> i) << 4) & 0x10101010;
        int vh1i = ((vh1 >> i) << 4) & 0x10101010;
        int dot1 = iq_dp4a(vl0i | vh0i, u0, iq_dp4a(vl1i | vh1i, u1, 0));
        int dot2 = iq_dp4a(0x01010101, u0, iq_dp4a(0x01010101, u1, 0));
        int sc = (int) ((packed >> (8 * i)) & 0xFFu);
        int mn = (int) ((packed >> (16 + 8 * i)) & 0xFFu);
        sum_d += d8 * (float) (dot1 * sc);
        sum_m += d8 * (float) (dot2 * mn);
    }
    float d = iq_f32_from_f16(iq_u16(blk));
    float dmin = iq_f32_from_f16(iq_u16(blk + 2));
    return d * sum_d - dmin * sum_m;
}

// Q5_0 is 22 bytes and 32 values: fp16 d, 4-byte high bits, 16 nibble bytes. One call covers
// iqs and iqs+1 (vdr 2). Two calls, iqs 0 and 2, cover the block. The stored q8 sum is not
// the int8 sum, so the -16 is 16 times the int8 values this call dotted.
inline float iq_dot_q5_0(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    const device uint8_t* qs = blk + 6;
    int sumi = 0;
    int suma = 0;
    for (int i = 0; i < 2; i++) {
        int vl = iq_b2(qs, iqs + i);
        uint vh = (uint) iq_b2(blk + 2, 0) >> (4 * (iqs + i));
        int u0 = iq_q8_word(x0, iqs + i);
        int u1 = iq_q8_word(x0, iqs + i + 4);
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (int) ((vh << 4) & 0x00000010u);
        vi0 |= (int) ((vh << 11) & 0x00001000u);
        vi0 |= (int) ((vh << 18) & 0x00100000u);
        vi0 |= (int) ((vh << 25) & 0x10000000u);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (int) ((vh >> 12) & 0x00000010u);
        vi1 |= (int) ((vh >> 5) & 0x00001000u);
        vi1 |= (int) ((vh << 2) & 0x00100000u);
        vi1 |= (int) ((vh << 9) & 0x10000000u);
        sumi = iq_dp4a(vi0, u0, sumi);
        sumi = iq_dp4a(vi1, u1, sumi);
        suma = iq_dp4a(0x01010101, u0, suma);
        suma = iq_dp4a(0x01010101, u1, suma);
    }
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x0);
    return d * (float) (sumi - 16 * suma);
}

// Q3_K is 110 bytes and 256 values: hmask[32], qs[64], scales[12], fp16 d last. iqs is 0..15.
// A clear high bit subtracts 4. The 16 scales are 6-bit, bias -32, packed in the 12 scale bytes.
inline float iq_dot_q3_k(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    int bq8_offset = 4 * (iqs / 8);
    int scale_offset = iqs - (iqs % 8) + (iqs % 8) / 4;
    const device uint8_t* scales = blk + 96;
    int vl = iq_b2(blk + 32, iqs);
    uint vh = (~(uint) iq_b2(blk, iqs % 8)) >> bq8_offset;
    float sumf = 0;
    for (int i = 0; i < 4; i++) {
        int isc = scale_offset + 2 * i;
        int sc_low = (scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        int sc_high = ((scales[8 + (isc % 4)] >> (2 * (isc / 4))) & 3) << 4;
        int sc = (sc_low | sc_high) - 32;
        int vil = (vl >> (2 * i)) & 0x03030303;
        int vih = (int) (((vh >> i) << 2) & 0x04040404u);
        int vi = iq_vsub4(vil, vih);
        const device uint8_t* bq8 = x0 + (long) (bq8_offset + i) * 36;
        sumf += iq_q8_d(bq8) * (float) (iq_dp4a(vi, iq_q8_word(bq8, iqs % 8), 0) * sc);
    }
    return iq_f32_from_f16(iq_u16(blk + 108)) * sumf;
}

inline float iq_dot_q2_0(const device uint8_t* blk, const device uint8_t* x0, int iqs) {
    const device uint8_t* qs = blk + 2;
    const device uint8_t* x = x0 + iqs * 36;
    int sumi = 0;
    for (int j = 0; j < 4; j++) {
        uint b0 = qs[8 * iqs + 2 * j];
        uint b1 = qs[8 * iqs + 2 * j + 1];
        sumi = iq_dp4a(iq_q2_pack(b0), iq_q8_word(x, j * 2), sumi);
        sumi = iq_dp4a(iq_q2_pack(b1), iq_q8_word(x, j * 2 + 1), sumi);
    }
    float d = iq_f32_from_f16(iq_u16(blk)) * iq_q8_d(x);
    return d * (float) sumi;
}

template <int QK, int IPB, int STEP, int BLOCK>
inline void iq_row(const device uint8_t* row, const device uint8_t* col, int n_in, uint lane, thread float& acc,
                   int which, const device ulong* g2, const device uint* g3) {
    int nb = n_in / QK;
    int npart = nb * IPB;
    int steps = (npart + 31) >> 5;
    for (int s = 0; s < steps; s++) {
        int k = (int) lane + s * 32;
        if (k < npart) {
            int kbx = k / IPB;
            int iqs = STEP * (k % IPB);
            const device uint8_t* blk = row + (long) kbx * BLOCK;
            const device uint8_t* x0 = col + (long) kbx * (QK / 32) * 36;
            float part = 0;
            if (which == 22) part = iq_dot_iq2_s(blk, x0, iqs, g2);
            else if (which == 18) part = iq_dot_iq3_xxs(blk, x0, iqs, g3);
            else if (which == 21) part = iq_dot_iq3_s(blk, x0, iqs, g3);
            else if (which == 20) part = iq_dot_iq4_nl(blk, x0, iqs);
            else if (which == 23) part = iq_dot_iq4_xs(blk, x0, iqs);
            else if (which == 12) part = iq_dot_q4_k(blk, x0, iqs);
            else if (which == 13) part = iq_dot_q5_k(blk, x0, iqs);
            else if (which == 11) part = iq_dot_q3_k(blk, x0, iqs);
            else if (which == 6) part = iq_dot_q5_0(blk, x0, iqs);
            else part = iq_dot_q2_0(blk, x0, iqs);
            acc += part;
        }
    }
}

inline void iq_mmvq_body(const device uint8_t* w, const device uint8_t* x, device float* y, constant IqDim& dim,
                         uint lane, uint row, int which, const device ulong* g2, const device uint* g3) {
    if (row >= (uint) dim.n_out) return;
    const device uint8_t* wr = w + dim.byte_off + (long) row * dim.row_bytes;
    int xblocks = dim.n_in / 32;
    for (int c = 0; c < dim.n_col; c++) {
        float acc = 0;
        const device uint8_t* col = x + (long) c * xblocks * 36;
        if (which == 22) iq_row<256, 8, 2, 82>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 18) iq_row<256, 8, 2, 98>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 21) iq_row<256, 8, 2, 110>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 20) iq_row<32, 2, 2, 18>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 23) iq_row<256, 8, 4, 136>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 12) iq_row<256, 16, 2, 144>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 13) iq_row<256, 16, 2, 176>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 11) iq_row<256, 16, 1, 110>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else if (which == 6) iq_row<32, 2, 2, 22>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        else iq_row<64, 2, 1, 18>(wr, col, dim.n_in, lane, acc, which, g2, g3);
        acc = iq_warp_add(acc);
        if (lane == 0) y[(long) c * dim.n_out + row] = acc;
    }
}

kernel void iq_mmvq_iq2_s(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                          device float* y [[buffer(2)]], const device ulong* grid [[buffer(3)]],
                          constant IqDim& dim [[buffer(4)]], uint lane [[thread_index_in_threadgroup]],
                          uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 22, grid, nullptr);
}

kernel void iq_mmvq_iq3_xxs(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                            device float* y [[buffer(2)]], const device uint* grid [[buffer(3)]],
                            constant IqDim& dim [[buffer(4)]], uint lane [[thread_index_in_threadgroup]],
                            uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 18, nullptr, grid);
}

kernel void iq_mmvq_iq3_s(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                          device float* y [[buffer(2)]], const device uint* grid [[buffer(3)]],
                          constant IqDim& dim [[buffer(4)]], uint lane [[thread_index_in_threadgroup]],
                          uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 21, nullptr, grid);
}

kernel void iq_mmvq_iq4_nl(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                           device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                           uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 20, nullptr, nullptr);
}

kernel void iq_mmvq_iq4_xs(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                           device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                           uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 23, nullptr, nullptr);
}

kernel void iq_mmvq_q2_0(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                         device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 42, nullptr, nullptr);
}

kernel void iq_mmvq_q4_k(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                         device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 12, nullptr, nullptr);
}

kernel void iq_mmvq_q5_k(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                         device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 13, nullptr, nullptr);
}

kernel void iq_mmvq_q3_k(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                         device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 11, nullptr, nullptr);
}

kernel void iq_mmvq_q5_0(const device uint8_t* w [[buffer(0)]], const device uint8_t* x [[buffer(1)]],
                         device float* y [[buffer(2)]], constant IqDim& dim [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    iq_mmvq_body(w, x, y, dim, lane, row, 6, nullptr, nullptr);
}

kernel void iq_q8_1(const device float* x [[buffer(0)]], device uint8_t* y [[buffer(1)]], constant int& n [[buffer(2)]],
                    uint lane [[thread_index_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]) {
    long i = (long) blk * 32 + lane;
    float xi = i < n ? x[i] : 0;
    float amax = fabs(xi);
    float sum = xi;
    for (uint o = 16; o > 0; o >>= 1) {
        amax = max(amax, simd_shuffle_xor(amax, o));
        sum += simd_shuffle_xor(sum, o);
    }
    float d = amax / 127.0f;
    if (fabs(d) > 65504.0f) d = copysign(65504.0f, d);
    int q = 0;
    if (amax != 0.0f) {
        float r = round(xi / d);
        if (r > 127.0f) r = 127.0f;
        if (r < -127.0f) r = -127.0f;
        q = (int) r;
    }
    device uint8_t* b = y + (long) blk * 36;
    b[4 + lane] = (uint8_t) q;
    if (lane == 0) {
        float sc = fabs(sum) > 65504.0f ? copysign(65504.0f, sum) : sum;
        ushort dh = iq_f16_from_f32(d);
        ushort sh = iq_f16_from_f32(sc);
        b[0] = (uint8_t) dh;
        b[1] = (uint8_t) (dh >> 8);
        b[2] = (uint8_t) sh;
        b[3] = (uint8_t) (sh >> 8);
    }
}

// SiLU(gate) * up, then the same q8_1 store. `n` is a multiple of 32.
kernel void iq_swiglu_q8(const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                         device uint8_t* y [[buffer(2)]], constant int& n [[buffer(3)]],
                         uint lane [[thread_index_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]) {
    long i = (long) blk * 32 + lane;
    float g = i < n ? gate[i] : 0;
    float u = i < n ? up[i] : 0;
    float xi = (g / (1.0f + exp(-g))) * u;
    float amax = fabs(xi);
    float sum = xi;
    for (uint o = 16; o > 0; o >>= 1) {
        amax = max(amax, simd_shuffle_xor(amax, o));
        sum += simd_shuffle_xor(sum, o);
    }
    float d = amax / 127.0f;
    if (fabs(d) > 65504.0f) d = copysign(65504.0f, d);
    int q = 0;
    if (amax != 0.0f) {
        float r = round(xi / d);
        if (r > 127.0f) r = 127.0f;
        if (r < -127.0f) r = -127.0f;
        q = (int) r;
    }
    device uint8_t* b = y + (long) blk * 36;
    b[4 + lane] = (uint8_t) q;
    if (lane == 0) {
        float sc = fabs(sum) > 65504.0f ? copysign(65504.0f, sum) : sum;
        ushort dh = iq_f16_from_f32(d);
        ushort sh = iq_f16_from_f32(sc);
        b[0] = (uint8_t) dh;
        b[1] = (uint8_t) (dh >> 8);
        b[2] = (uint8_t) sh;
        b[3] = (uint8_t) (sh >> 8);
    }
}
