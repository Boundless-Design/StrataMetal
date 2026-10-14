// The Coder expert product against the integer dots it loads. The bound is 1e-5.
// A wrong scale, a Q2_0 code that forgets the -1, and a swapped gate/up have to land more than 5% away.
#include "strata/metal/iq.hpp"
#include "strata/metal/iq_grids.hpp"
#include "strata/metal/device.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;
namespace bits = strata::kernels;

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

void report(const char* name, double rel, double bound) {
    const bool ok = rel <= bound && std::isfinite(rel);
    std::printf("  %-52s rel %.3e%s\n", name, rel, ok ? "" : "   *** NO ***");
    if (!ok) ++g_fail;
}

std::string pct(double rel) {
    char b[64];
    std::snprintf(b, sizeof b, "(%.2f%% apart)", rel * 100.0);
    return b;
}

Bytes upload(const void* p, size_t n) {
    Bytes b(n ? n : 1);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}
template <class T>
Bytes upload(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
}
template <class T>
std::vector<T> download(const Bytes& b, size_t n) {
    std::vector<T> v(n);
    std::memcpy(v.data(), b.data(), n * sizeof(T));
    return v;
}

int ld(const uint8_t* p) {
    return (int) p[0] | ((int) p[1] << 8) | ((int) p[2] << 16) | ((int) p[3] << 24);
}
int b2(const uint8_t* p, int i) { return ld(p + 4 * i); }
uint16_t u16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

int dp4a(int a, int b, int c) {
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
int vcmpne4(uint32_t a) {
    int r = 0;
    for (int i = 0; i < 4; i++)
        if (((a >> (8 * i)) & 0xFFu) != 0) r |= 0xFF << (8 * i);
    return r;
}
int vsub4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; i++) {
        int ba = (a >> (8 * i)) & 0xFF;
        int bb = (b >> (8 * i)) & 0xFF;
        r |= ((ba - bb) & 0xFF) << (8 * i);
    }
    return r;
}
uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFFu;
    uint32_t p = (uint32_t) std::popcount(v) & 1u;
    uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}
int sign_pair(int grid, uint32_t sb, bool high) {
    uint32_t bits = high ? (((sb & 0x30u) << 3) | ((sb & 0xC0u) << 17))
                         : (((sb & 0x03u) << 7) | ((sb & 0x0Cu) << 21));
    int mask = vcmpne4(bits);
    return vsub4(grid ^ mask, mask);
}
int q8_word(const uint8_t* block, int i) { return ld(block + 4 + 4 * i); }
float q8_d(const uint8_t* block) { return bits::f32_from_f16(u16(block)); }

const int kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void nibbles(int aux, int& vx, int& vy) {
    vx = vy = 0;
    for (int b = 0; b < 4; b++) {
        uint32_t byte = (uint32_t) (aux >> (8 * b)) & 0xFFu;
        vx |= (int) (uint8_t) kIq4[byte & 0xFu] << (8 * b);
        vy |= (int) (uint8_t) kIq4[byte >> 4] << (8 * b);
    }
}
int q2_pack(uint32_t byte) {
    int r = 0;
    for (int c = 0; c < 4; c++) {
        int sym = (int) ((byte >> (2 * c)) & 3u) - 1;
        r |= (sym & 0xFF) << (8 * c);
    }
    return r;
}

float dot_iq2_s(const uint8_t* blk, const uint8_t* x0, int iqs) {
    int packed = b2(blk + 2, iqs / 2);
    int qh = (blk + 66)[iqs / 2];
    int signs = b2(blk + 2, 8 + iqs / 2);
    int sc = (blk + 74)[iqs / 2];
    int ls0 = sc & 0x0F, ls1 = sc >> 4;
    int sum0 = 0, sum1 = 0;
    const uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t qb = (uint32_t) (packed >> (8 * (l0 / 2))) & 0xFFu;
        uint32_t idx = qb | (uint32_t) ((qh << (8 - l0)) & 0x300);
        uint64_t g = iq2s_grid[idx];
        uint32_t sb = (uint32_t) (signs >> (8 * (l0 / 2))) & 0xFFu;
        int gl = sign_pair((int) (g & 0xFFFFFFFFull), sb, false);
        int gh = sign_pair((int) (g >> 32), sb, true);
        if (l0 < 4) {
            sum0 = dp4a(gl, q8_word(x, l0), sum0);
            sum0 = dp4a(gh, q8_word(x, l0 + 1), sum0);
        } else {
            sum1 = dp4a(gl, q8_word(x, l0), sum1);
            sum1 = dp4a(gh, q8_word(x, l0 + 1), sum1);
        }
    }
    int sumi = (sum0 * ls0 + sum1 * ls1 + (sum0 + sum1) / 2) / 4;
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x);
    return d * (float) sumi;
}

float dot_iq3_xxs(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 2;
    int aux = b2(qs, 16 + iqs / 2);
    int sumi = 0;
    const uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t i0 = qs[4 * iqs + l0];
        uint32_t i1 = qs[4 * iqs + l0 + 1];
        uint32_t signs = unpack_ksigns((uint32_t) aux >> (7 * l0 / 2));
        int m0 = vcmpne4(signs & 0x08040201u);
        int m1 = vcmpne4(signs & 0x80402010u);
        int gl = vsub4((int) iq3xxs_grid[i0] ^ m0, m0);
        int gh = vsub4((int) iq3xxs_grid[i1] ^ m1, m1);
        sumi = dp4a(gl, q8_word(x, l0), sumi);
        sumi = dp4a(gh, q8_word(x, l0 + 1), sumi);
    }
    int ls = (int) ((uint32_t) aux >> 28);
    sumi = (ls * sumi + sumi / 2) / 2;
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x);
    return d * (float) sumi;
}

float dot_iq3_s(const uint8_t* blk, const uint8_t* x0, int iqs, bool scale) {
    const uint8_t* qs = blk + 2;
    int qh = (blk + 66)[iqs / 2];
    int signs = b2(blk + 74, iqs / 2);
    int sumi = 0;
    const uint8_t* x = x0 + (iqs / 2) * 36;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t i0 = qs[4 * iqs + l0] | (uint32_t) ((qh << (8 - l0)) & 0x100);
        uint32_t i1 = qs[4 * iqs + l0 + 1] | (uint32_t) ((qh << (7 - l0)) & 0x100);
        uint32_t sb = (uint32_t) (signs >> (8 * (l0 / 2))) & 0xFFu;
        int gl = sign_pair((int) iq3s_grid[i0], sb, false);
        int gh = sign_pair((int) iq3s_grid[i1], sb, true);
        sumi = dp4a(gl, q8_word(x, l0), sumi);
        sumi = dp4a(gh, q8_word(x, l0 + 1), sumi);
    }
    int nib = scale ? (((blk + 106)[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F) : 0;
    sumi *= 1 + 2 * nib;
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x);
    return d * (float) sumi;
}

float dot_iq4_nl(const uint8_t* blk, const uint8_t* x0, int iqs) {
    int sumi = 0;
    for (int l = 0; l < 2; l++) {
        int vx, vy;
        nibbles(b2(blk + 2, iqs + l), vx, vy);
        sumi = dp4a(vx, q8_word(x0, iqs + l), sumi);
        sumi = dp4a(vy, q8_word(x0, iqs + l + 4), sumi);
    }
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x0);
    return d * (float) sumi;
}

float dot_iq4_xs(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 8;
    const uint8_t* x = x0 + (iqs / 4) * 36;
    int sumi = 0;
    for (int j = 0; j < 4; j++) {
        int vx, vy;
        nibbles(ld(qs + 4 * (iqs + j)), vx, vy);
        sumi = dp4a(vx, q8_word(x, j), sumi);
        sumi = dp4a(vy, q8_word(x, j + 4), sumi);
    }
    uint16_t sh = u16(blk + 2);
    int ls = (((blk + 4)[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((sh >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x);
    return d * (float) sumi;
}

float dot_q2_0(const uint8_t* blk, const uint8_t* x0, int iqs, bool minus_one) {
    const uint8_t* qs = blk + 2;
    const uint8_t* x = x0 + iqs * 36;
    int sumi = 0;
    for (int j = 0; j < 4; j++) {
        uint32_t b0 = qs[8 * iqs + 2 * j];
        uint32_t b1 = qs[8 * iqs + 2 * j + 1];
        int p0 = q2_pack(b0);
        int p1 = q2_pack(b1);
        if (!minus_one) {
            // code, not code - 1: add 1 to every symbol byte
            p0 = dp4a(0x01010101, 0x01010101, p0); // not quite: adding 1 per byte
            int a0 = 0, a1 = 0;
            for (int c = 0; c < 4; c++) {
                int s0 = (((b0 >> (2 * c)) & 3));
                int s1 = (((b1 >> (2 * c)) & 3));
                a0 |= (s0 & 0xFF) << (8 * c);
                a1 |= (s1 & 0xFF) << (8 * c);
            }
            p0 = a0;
            p1 = a1;
        }
        sumi = dp4a(p0, q8_word(x, j * 2), sumi);
        sumi = dp4a(p1, q8_word(x, j * 2 + 1), sumi);
    }
    float d = bits::f32_from_f16(u16(blk)) * q8_d(x);
    return d * (float) sumi;
}

struct Geom {
    int qk, ipb, step, block;
};
Geom geom_of(int t) {
    if (t == kIq4Nl) return {32, 2, 2, 18};
    if (t == kQ2_0) return {64, 2, 1, 18};
    if (t == kIq4Xs) return {256, 8, 4, 136};
    if (t == kIq3Xxs) return {256, 8, 2, 98};
    if (t == kIq3S) return {256, 8, 2, 110};
    return {256, 8, 2, 82};
}

float partial(int t, const uint8_t* blk, const uint8_t* x0, int iqs, bool alt) {
    if (t == kIq2S) return dot_iq2_s(blk, x0, iqs);
    if (t == kIq3Xxs) return dot_iq3_xxs(blk, x0, iqs);
    if (t == kIq3S) return dot_iq3_s(blk, x0, iqs, !alt);
    if (t == kIq4Nl) return dot_iq4_nl(blk, x0, iqs);
    if (t == kIq4Xs) return dot_iq4_xs(blk, x0, iqs);
    return dot_q2_0(blk, x0, iqs, !alt);
}

float row_dot(int t, const uint8_t* row, const uint8_t* col, int n_in, bool alt) {
    const Geom g = geom_of(t);
    float lane[32] = {};
    const int npart = (n_in / g.qk) * g.ipb;
    for (int k = 0; k < npart; k++) {
        int kbx = k / g.ipb;
        int iqs = g.step * (k % g.ipb);
        const uint8_t* blk = row + (size_t) kbx * g.block;
        const uint8_t* x0 = col + (size_t) kbx * (g.qk / 32) * 36;
        lane[k & 31] += partial(t, blk, x0, iqs, alt);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; i++) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

std::vector<float> ref_mmvq(int t, const std::vector<uint8_t>& w, const std::vector<uint8_t>& x, int n_in, int n_out,
                            int n_col, bool alt = false) {
    const size_t row = iq_row_bytes(t, n_in);
    std::vector<float> y((size_t) n_col * n_out);
    const int xblocks = n_in / 32;
    for (int c = 0; c < n_col; c++)
        for (int r = 0; r < n_out; r++)
            y[(size_t) c * n_out + r] =
                row_dot(t, w.data() + (size_t) r * row, x.data() + (size_t) c * xblocks * 36, n_in, alt);
    return y;
}

double rel_ff(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        n += std::fabs((double) a[i] - b[i]);
        d += std::fabs((double) b[i]);
    }
    return n / (d + 1e-30);
}

void put_f16(uint8_t* p, uint16_t h) {
    p[0] = (uint8_t) h;
    p[1] = (uint8_t) (h >> 8);
}

std::vector<uint8_t> random_rows(int t, int rows, int n_in, std::mt19937& rng) {
    const size_t rb = iq_row_bytes(t, n_in);
    const size_t bs = iq_block_bytes(t);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    std::uniform_int_distribution<int> ex(7, 11);
    std::uniform_int_distribution<int> man(0, 1023);
    std::uniform_int_distribution<int> sgn(0, 3);
    for (size_t o = 0; o < w.size(); o += bs) {
        uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        put_f16(w.data() + o, h);
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

std::vector<uint8_t> quant_gpu(const std::vector<float>& x) {
    Bytes dx = upload(x);
    Bytes dy(iq_q8_1_bytes((int64_t) x.size()));
    iq_quantize_q8_1(dx, (int64_t) x.size(), dy);
    return download<uint8_t>(dy, iq_q8_1_bytes((int64_t) x.size()));
}

void quant_host_block(const float* x, uint8_t* y) {
    float a[32], s[32];
    for (int i = 0; i < 32; i++) {
        a[i] = std::fabs(x[i]);
        s[i] = x[i];
    }
    for (int o = 16; o > 0; o >>= 1) {
        float na[32], ns[32];
        for (int i = 0; i < 32; i++) {
            na[i] = std::max(a[i], a[i ^ o]);
            ns[i] = s[i] + s[i ^ o];
        }
        std::memcpy(a, na, sizeof a);
        std::memcpy(s, ns, sizeof s);
    }
    float amax = a[0];
    float sum = s[0];
    float d = amax / 127.f;
    if (std::fabs(d) > 65504.f) d = std::copysign(65504.f, d);
    for (int i = 0; i < 32; i++) {
        int q = 0;
        if (amax != 0.f) {
            float r = std::round(x[i] / d);
            if (r > 127.f) r = 127.f;
            if (r < -127.f) r = -127.f;
            q = (int) r;
        }
        y[4 + i] = (uint8_t) q;
    }
    float sc = std::fabs(sum) > 65504.f ? std::copysign(65504.f, sum) : sum;
    put_f16(y, bits::f16_from_f32(d));
    put_f16(y + 2, bits::f16_from_f32(sc));
}

float silu(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }

std::vector<float> ref_expert(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& x, const IqExpert& e,
                              int n_tok) {
    const size_t gu = iq_row_bytes(e.gu_type, e.n_embd);
    const size_t up = (size_t) e.n_ff * gu;
    const size_t down = 2 * up;
    std::vector<float> h((size_t) n_tok * e.n_ff);
    const int xb = (int) (e.n_embd / 32);
    for (int t = 0; t < n_tok; t++) {
        const uint8_t* col = x.data() + (size_t) t * xb * 36;
        for (int r = 0; r < e.n_ff; r++) {
            float g = row_dot(e.gu_type, blob.data() + (size_t) r * gu, col, (int) e.n_embd, false);
            float u = row_dot(e.gu_type, blob.data() + up + (size_t) r * gu, col, (int) e.n_embd, false);
            h[(size_t) t * e.n_ff + r] = silu(g, u);
        }
    }
    std::vector<uint8_t> hq(iq_q8_1_bytes((int64_t) h.size()));
    for (size_t i = 0; i < h.size(); i += 32) quant_host_block(h.data() + i, hq.data() + i / 32 * 36);
    std::vector<float> y((size_t) n_tok * e.n_embd);
    const size_t drow = iq_row_bytes(e.d_type, e.n_ff);
    const int hb = (int) (e.n_ff / 32);
    for (int t = 0; t < n_tok; t++)
        for (int r = 0; r < e.n_embd; r++)
            y[(size_t) t * e.n_embd + r] = row_dot(e.d_type, blob.data() + down + (size_t) r * drow,
                                                   hq.data() + (size_t) t * hb * 36, (int) e.n_ff, false);
    return y;
}

const char* name_of(int t) {
    switch (t) {
        case kIq2S: return "IQ2_S";
        case kIq3Xxs: return "IQ3_XXS";
        case kIq3S: return "IQ3_S";
        case kIq4Xs: return "IQ4_XS";
        case kIq4Nl: return "IQ4_NL";
        case kQ2_0: return "Q2_0";
        default: return "?";
    }
}

void check_mmvq(int t, int n_in, int n_out, int n_col, std::mt19937& rng) {
    auto w = random_rows(t, n_out, n_in, rng);
    auto x = random_x((size_t) n_col * n_in, rng);
    auto xq = quant_gpu(x);
    Bytes dw = upload(w);
    Bytes dx = upload(xq);
    Bytes dy((size_t) n_col * n_out * 4);
    iq_mmvq(t, dw, 0, dx, n_in, n_out, n_col, dy);
    auto got = download<float>(dy, (size_t) n_col * n_out);
    auto want = ref_mmvq(t, w, xq, n_in, n_out, n_col);
    char label[80];
    std::snprintf(label, sizeof label, "%s %d x %d, %d col", name_of(t), n_out, n_in, n_col);
    bool finite = true;
    for (float v : got)
        if (!std::isfinite(v)) finite = false;
    if (!finite) verdict(label, false, "non-finite");
    else report(label, rel_ff(got, want), 1e-5);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-iq [--selftest]\n");
        return 2;
    }
    std::printf("coder expert: IQ2_S IQ3_XXS IQ3_S IQ4_XS gate/up, IQ4_NL Q2_0 down, q8_1 activations\n\n");
    try {
        std::mt19937 rng(20261014);

        {
            auto x = random_x(128, rng);
            x[3] = 1e10f;
            x[40] = -8e6f;
            auto gpu = quant_gpu(x);
            std::vector<uint8_t> host(gpu.size());
            for (int b = 0; b < 4; b++) quant_host_block(x.data() + b * 32, host.data() + b * 36);
            verdict("q8_1 matches the host block, including a clamped scale", gpu == host, gpu == host ? "bitwise" : "");
        }

        const int types[] = {kIq2S, kIq3Xxs, kIq3S, kIq4Xs, kIq4Nl, kQ2_0};
        for (int t : types) {
            const int n_in = (t == kIq4Nl || t == kQ2_0) ? 256 : 512;
            check_mmvq(t, n_in, 40, 3, rng);
            check_mmvq(t, iq_block_values(t), 1, 1, rng);
        }

        {
            const int n_in = 256, n_out = 4, n_col = 1;
            auto w = random_rows(kIq3S, n_out, n_in, rng);
            const size_t bs = iq_block_bytes(kIq3S);
            for (size_t o = 106; o < w.size(); o += bs) {
                w[o] = w[o + 1] = w[o + 2] = w[o + 3] = 0xFF;  // every nibble 15, scale 31
            }
            auto x = random_x(n_in, rng);
            auto xq = quant_gpu(x);
            Bytes dw = upload(w), dx = upload(xq), dy(n_out * 4);
            iq_mmvq(kIq3S, dw, 0, dx, n_in, n_out, n_col, dy);
            auto got = download<float>(dy, n_out);
            auto right = ref_mmvq(kIq3S, w, xq, n_in, n_out, n_col, false);
            auto bare = ref_mmvq(kIq3S, w, xq, n_in, n_out, n_col, true);
            report("IQ3_S scale 31 vs the reference", rel_ff(got, right), 1e-5);
            verdict("dropping the IQ3_S scale is observable", rel_ff(got, bare) > 0.05, pct(rel_ff(got, bare)));
        }
        {
            const int n_in = 256, n_out = 4;
            auto w = random_rows(kQ2_0, n_out, n_in, rng);
            auto x = random_x(n_in, rng);
            auto xq = quant_gpu(x);
            Bytes dw = upload(w), dx = upload(xq), dy(n_out * 4);
            iq_mmvq(kQ2_0, dw, 0, dx, n_in, n_out, 1, dy);
            auto got = download<float>(dy, n_out);
            auto right = ref_mmvq(kQ2_0, w, xq, n_in, n_out, 1, false);
            auto bare = ref_mmvq(kQ2_0, w, xq, n_in, n_out, 1, true);
            report("Q2_0 vs the reference", rel_ff(got, right), 1e-5);
            verdict("a Q2_0 code that forgets the -1 is observable", rel_ff(got, bare) > 0.05, pct(rel_ff(got, bare)));
        }

        const int pairs[][2] = {{kIq2S, kIq4Nl}, {kIq3Xxs, kQ2_0}, {kIq3S, kIq4Nl}, {kIq4Xs, kQ2_0}};
        for (auto p : pairs) {
            IqExpert e{p[0], p[1], 256, 64};
            const int n_tok = 2;
            auto blob = random_rows(e.gu_type, (int) (2 * e.n_ff), (int) e.n_embd, rng);
            auto down = random_rows(e.d_type, (int) e.n_embd, (int) e.n_ff, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            auto x = random_x((size_t) n_tok * e.n_embd, rng);
            for (int i = 0; i < e.n_embd; i++) x[(size_t) e.n_embd + i] = -0.5f * x[i] + 0.25f;
            auto xq = quant_gpu(x);
            Bytes db = upload(blob), dx = upload(xq), dy(iq_expert_bytes(e) > 0 ? (size_t) n_tok * e.n_embd * 4 : 4);
            iq_expert(db, dx, e, n_tok, dy);
            auto got = download<float>(dy, (size_t) n_tok * e.n_embd);
            auto want = ref_expert(blob, xq, e, n_tok);
            char label[80];
            std::snprintf(label, sizeof label, "expert %s / %s, 2 tokens", name_of(p[0]), name_of(p[1]));
            bool finite = true;
            for (float v : got)
                if (!std::isfinite(v)) finite = false;
            if (!finite) verdict(label, false, "non-finite");
            else report(label, rel_ff(got, want), 1e-5);
            if (p[0] == kIq3S) {
                std::vector<float> a(got.begin(), got.begin() + e.n_embd);
                std::vector<float> b(got.begin() + e.n_embd, got.end());
                verdict("the two tokens do not share an output", rel_ff(a, b) > 0.05, pct(rel_ff(a, b)));
                const size_t gu = iq_row_bytes(e.gu_type, e.n_embd);
                const size_t half = (size_t) e.n_ff * gu;
                std::vector<uint8_t> swapped = blob;
                for (size_t i = 0; i < half; i++) std::swap(swapped[i], swapped[half + i]);
                Bytes ds = upload(swapped), d2((size_t) n_tok * e.n_embd * 4);
                iq_expert(ds, dx, e, n_tok, d2);
                auto alt = download<float>(d2, (size_t) n_tok * e.n_embd);
                verdict("swapping gate and up is observable", rel_ff(got, alt) > 0.05, pct(rel_ff(got, alt)));
                report("swapped gate and up vs the reference", rel_ff(alt, ref_expert(swapped, xq, e, n_tok)), 1e-5);
            }
        }

        {
            IqExpert e{kIq3S, kIq4Nl, 1280, 128};
            const int n_tok = 2;
            auto blob = random_rows(e.gu_type, (int) (2 * e.n_ff), (int) e.n_embd, rng);
            auto down = random_rows(e.d_type, (int) e.n_embd, (int) e.n_ff, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            auto xq = quant_gpu(random_x((size_t) n_tok * e.n_embd, rng));
            Bytes db = upload(blob), dx = upload(xq), dy((size_t) n_tok * e.n_embd * 4);
            iq_expert(db, dx, e, n_tok, dy);
            auto got = download<float>(dy, (size_t) n_tok * e.n_embd);
            report("expert IQ3_S / IQ4_NL, 1280 x 128, two partials", rel_ff(got, ref_expert(blob, xq, e, n_tok)), 1e-5);
        }

        {
            IqExpert e{kIq3S, kQ2_0, 256, 64};
            Bytes blob(iq_expert_bytes(e)), x(iq_q8_1_bytes(e.n_embd)), y((size_t) e.n_embd * 4);
            std::memset(y.data(), 0, 4);
            float* yp = (float*) y.data();
            for (int i = 0; i < e.n_embd; i++) yp[i] = 7.f;
            iq_expert(blob, x, e, 0, y);
            bool same = true;
            for (int i = 0; i < e.n_embd; i++)
                if (yp[i] != 7.f) same = false;
            verdict("zero tokens leaves the output alone", same, "");
        }
        {
            bool threw = false;
            try {
                iq_row_bytes(29, 256);
            } catch (const MetalError&) { threw = true; }
            verdict("IQ1_M blocks are not this product", threw, "");
        }
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("\niq: %d failures\n", g_fail);
    if (g_fail) return 1;
    std::printf("iq_parity OK\n");
    return 0;
}
