// The 2-bit product over a Q8_0 activation. The host reference quantizes, then dots in f32.
// The same weights with an fp16 activation have to sit apart, or this check cannot see the format.
#include "strata/metal/s2_gemv.hpp"
#include "strata/metal/sgemv.hpp"
#include "strata/metal/device.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr double kTol = 1e-4;

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

Bytes upload(const void* p, size_t n) {
    Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}

template <class F>
bool throws_metal(F&& fn) {
    try {
        fn();
        return false;
    } catch (const MetalError&) {
        return true;
    }
}

float act_at(const std::vector<uint8_t>& act, long long i) {
    const uint8_t* blk = act.data() + (size_t) (i / 32) * 34;
    const uint16_t bits = (uint16_t) (blk[0] | (blk[1] << 8));
    return strata::kernels::f32_from_f16(bits) * (float) (int8_t) blk[2 + (i % 32)];
}

std::vector<float> reference(const std::vector<uint8_t>& act, const std::vector<uint8_t>& codes,
                             const std::vector<float>& scales, long long n_in, long long n_out) {
    std::vector<float> y((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        for (long long g = 0; g < n_in / 64; ++g) {
            const float d = scales[(size_t) (o * (n_in / 64) + g)];
            for (int j = 0; j < 64; ++j) {
                const long long i = g * 64 + j;
                const uint8_t byte = codes[(size_t) (o * (n_in / 4) + i / 4)];
                const int code = (byte >> ((i % 4) * 2)) & 3;
                acc += (float) (code - 1) * d * act_at(act, i);
            }
        }
        y[(size_t) o] = acc;
    }
    return y;
}

double worst_rel(const std::vector<float>& want, const std::vector<float>& got, double& mean) {
    double worst = 0, sum = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double a = want[i], b = got[i];
        sum += std::fabs(a);
        const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
        if (!(rel <= worst)) worst = rel;
    }
    mean = sum / (double) want.size();
    return worst;
}

struct Case {
    std::vector<float> x;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    long long n_in;
    long long n_out;
};

Case make_case(uint32_t seed, long long n_in, long long n_out) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    Case c;
    c.n_in = n_in;
    c.n_out = n_out;
    c.x.resize((size_t) n_in);
    for (auto& v : c.x) v = gauss(rng);
    c.codes.resize((size_t) n_out * n_in / 4);
    for (auto& v : c.codes) v = (uint8_t) (rng() & 0xFF);
    c.scales.resize((size_t) n_out * (n_in / 64));
    for (auto& v : c.scales) v = 0.001f + 0.0001f * (float) (rng() % 100);
    return c;
}

double run_case(const Case& c, int tpr, std::vector<float>& got, bool& guard_ok) {
    Bytes dx = upload(c.x.data(), c.x.size() * 4);
    Bytes act(q8_0_bytes(c.n_in));
    quantize_q8_0(dx, c.n_in, act);
    std::vector<uint8_t> hact(act.size());
    std::memcpy(hact.data(), act.data(), hact.size());
    const std::vector<float> want = reference(hact, c.codes, c.scales, c.n_in, c.n_out);

    Bytes codes = upload(c.codes.data(), c.codes.size());
    Bytes scales = upload(c.scales.data(), c.scales.size() * 4);
    Bytes y((uint64_t) (c.n_out + 16) * 4);
    float* tail = (float*) y.data();
    for (long long i = 0; i < c.n_out + 16; ++i) tail[i] = 7.0f;
    s2_gemv_q8(act, codes, scales, y, c.n_in, c.n_out, tpr);
    got.assign(tail, tail + c.n_out);
    guard_ok = true;
    for (int i = 0; i < 16; ++i)
        if (tail[c.n_out + i] != 7.0f) guard_ok = false;

    double mean = 0;
    return worst_rel(want, got, mean);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-s2-gemv [--selftest]\n");
        return 2;
    }
    std::printf("s2_gemv_q8: 2-bit weights, Q8_0 activation, group 64, bias -1\n\n");
    try {
        const Case wide = make_case(31, 2560, 64);
        std::vector<float> got;
        bool guard = false;
        const double rel = run_case(wide, 32, got, guard);
        char detail[80];
        std::snprintf(detail, sizeof detail, "worst rel %.3e", rel);
        verdict("q8 activation vs host reference", rel <= kTol, detail);
        verdict("16 words past the end stay put", guard, "");

        const Case down = make_case(64, 64, 8);
        const double rel_d = run_case(down, 32, got, guard);
        std::snprintf(detail, sizeof detail, "worst rel %.3e", rel_d);
        verdict("width 64, the small down shape", guard && rel_d <= kTol, detail);

        const Case mid = make_case(256, 256, 16);
        const double rel_m = run_case(mid, 64, got, guard);
        std::snprintf(detail, sizeof detail, "worst rel %.3e", rel_m);
        verdict("width 256, 64 threads", guard && rel_m <= kTol, detail);

        {
            Bytes dx = upload(wide.x.data(), wide.x.size() * 4);
            Bytes act(q8_0_bytes(wide.n_in));
            quantize_q8_0(dx, wide.n_in, act);
            Bytes codes = upload(wide.codes.data(), wide.codes.size());
            Bytes scales = upload(wide.scales.data(), wide.scales.size() * 4);
            Bytes yq(wide.n_out * 4), y16(wide.n_out * 4);
            s2_gemv_q8(act, codes, scales, yq, wide.n_in, wide.n_out, 32);
            std::vector<uint16_t> hx((size_t) wide.n_in);
            for (long long i = 0; i < wide.n_in; ++i) hx[(size_t) i] = strata::kernels::f16_from_f32(wide.x[(size_t) i]);
            Bytes x16 = upload(hx.data(), hx.size() * 2);
            const SForm form{2, -1, 64, Codebook::Affine, false};
            s_gemv_f16(x16, codes, scales, nullptr, y16, wide.n_in, wide.n_out, form);
            const float* aq = (const float*) yq.data();
            const float* af = (const float*) y16.data();
            double diff = 0, mass = 0;
            for (long long o = 0; o < wide.n_out; ++o) {
                diff += std::fabs((double) aq[o] - (double) af[o]);
                mass += std::fabs((double) aq[o]);
            }
            const double gap = diff / (mass > 1e-30 ? mass : 1e-30);
            std::snprintf(detail, sizeof detail, "(%.4f%% apart)", gap * 100.0);
            verdict("fp16 activation is a different product", gap >= 1e-4, detail);
        }

        verdict("a width that is not a multiple of 64 is refused",
                throws_metal([] {
                    Bytes a(64), c(64), s(64), y(64);
                    s2_gemv_q8(a, c, s, y, 32, 1, 32);
                }),
                "");
        verdict("a thread count that is not a power of two is refused",
                throws_metal([] {
                    Bytes a(q8_0_bytes(64)), c(64), s(16), y(16);
                    s2_gemv_q8(a, c, s, y, 64, 1, 24);
                }),
                "");
        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes a(16), c(16), s(16), y(16);
                    s2_gemv_q8(a, c, s, y, -64, 1, 32);
                }),
                "");
        {
            Bytes a(16), c(16), s(16), y(16);
            float* p = (float*) y.data();
            p[0] = 7.0f;
            s2_gemv_q8(a, c, s, y, 0, 64, 32);
            s2_gemv_q8(a, c, s, y, 64, 0, 32);
            verdict("zero length leaves the buffer", p[0] == 7.0f, "");
        }

        std::printf("\ns2_gemv_q8: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("s2_gemv_q8_parity OK\n");
    return 0;
}
