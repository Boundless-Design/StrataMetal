// The QSA gate. The value is the second half of each head, and the function is a sigmoid.
// Both wrong readings are required to sit more than 5% away before the kernel is judged.
#include "strata/metal/qsa_gate.hpp"
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

double rel_l1(const std::vector<double>& want, const std::vector<float>& got) {
    double n = 0, d = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        n += std::fabs(want[i] - (double) got[i]);
        d += std::fabs(want[i]);
    }
    return n / (d > 1e-30 ? d : 1e-30);
}

std::vector<double> reference(const std::vector<float>& attn, const std::vector<float>& q_full, int n_head,
                              int head_dim, bool second, bool sigmoid) {
    std::vector<double> out((size_t) n_head * head_dim, 0.0);
    for (int h = 0; h < n_head; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            const int gi = second ? h * 2 * head_dim + head_dim + d : h * 2 * head_dim + d;
            const double g = q_full[(size_t) gi];
            const double s = sigmoid ? 1.0 / (1.0 + std::exp(-g)) : g / (1.0 + std::exp(-g));
            out[(size_t) (h * head_dim + d)] = (double) attn[(size_t) (h * head_dim + d)] * s;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-qsa-gate [--selftest]\n");
        return 2;
    }
    std::printf("qsa_gate: sigmoid of the second half of each head\n\n");
    try {
        const int nh = 24, hd = 256;
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> u(-2.0f, 2.0f);
        std::vector<float> attn((size_t) nh * hd), q((size_t) nh * 2 * hd);
        for (auto& v : attn) v = u(rng);
        for (auto& v : q) v = u(rng);

        const std::vector<double> want = reference(attn, q, nh, hd, true, true);
        const std::vector<double> first = reference(attn, q, nh, hd, false, true);
        const std::vector<double> silu = reference(attn, q, nh, hd, true, false);
        auto apart = [&](const std::vector<double>& alt) {
            double n = 0, d = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                n += std::fabs(want[i] - alt[i]);
                d += std::fabs(want[i]);
            }
            return n / (d > 1e-30 ? d : 1e-30);
        };
        const double rel_first = apart(first);
        const double rel_silu = apart(silu);
        char detail[80];
        std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel_first * 100.0);
        verdict("the gate from the first half", rel_first > 0.05, detail);
        std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel_silu * 100.0);
        verdict("silu instead of sigmoid", rel_silu > 0.05, detail);

        Bytes ba = upload(attn.data(), attn.size() * 4);
        Bytes bq = upload(q.data(), q.size() * 4);
        Bytes y((uint64_t) (nh * hd + 16) * 4);
        float* tail = (float*) y.data();
        for (int i = 0; i < nh * hd + 16; ++i) tail[i] = 7.0f;
        qsa_gate_apply_f32(ba, bq, nh, hd, y);
        bool guard = true;
        for (int i = 0; i < 16; ++i)
            if (tail[nh * hd + i] != 7.0f) guard = false;
        verdict("16 words past the end stay put", guard, "");
        std::vector<float> got(tail, tail + nh * hd);
        const double rel = rel_l1(want, got);
        std::snprintf(detail, sizeof detail, "rel %.3e", rel);
        verdict("f32 gate vs the double reference", guard && rel <= 1e-6, detail);

        Bytes y16((uint64_t) nh * hd * 2);
        qsa_gate_apply(ba, bq, nh, hd, y16);
        const uint16_t* bits = (const uint16_t*) y16.data();
        int bit_bad = 0;
        double n16 = 0, d16 = 0;
        for (int i = 0; i < nh * hd; ++i) {
            const uint16_t expect = strata::kernels::f16_from_f32(got[(size_t) i]);
            if (bits[i] != expect) ++bit_bad;
            const double dec = strata::kernels::f32_from_f16(bits[i]);
            n16 += std::fabs(dec - want[(size_t) i]);
            d16 += std::fabs(want[(size_t) i]);
        }
        const double rel16 = n16 / (d16 > 1e-30 ? d16 : 1e-30);
        std::snprintf(detail, sizeof detail, "rel %.3e, %d bit mismatches", rel16, bit_bad);
        verdict("fp16 gate vs the double reference", bit_bad == 0 && rel16 <= 1e-3, detail);

        const int nh2 = 4, hd2 = 8;
        std::vector<float> a2((size_t) nh2 * hd2), q2((size_t) nh2 * 2 * hd2);
        for (auto& v : a2) v = u(rng);
        for (auto& v : q2) v = u(rng);
        const std::vector<double> want2 = reference(a2, q2, nh2, hd2, true, true);
        Bytes ba2 = upload(a2.data(), a2.size() * 4);
        Bytes bq2 = upload(q2.data(), q2.size() * 4);
        Bytes y2((uint64_t) nh2 * hd2 * 4);
        qsa_gate_apply_f32(ba2, bq2, nh2, hd2, y2);
        std::vector<float> got2((const float*) y2.data(), (const float*) y2.data() + nh2 * hd2);
        const double rel2 = rel_l1(want2, got2);
        std::snprintf(detail, sizeof detail, "rel %.3e", rel2);
        verdict("4 heads by 8", rel2 <= 1e-6, detail);

        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes a(16), q(16), y(16);
                    qsa_gate_apply_f32(a, q, -1, 8, y);
                }),
                "");
        {
            Bytes a(16), q(16), y(16);
            float* p = (float*) y.data();
            p[0] = 7.0f;
            qsa_gate_apply_f32(a, q, 0, 256, y);
            qsa_gate_apply_f32(a, q, 24, 0, y);
            verdict("zero length leaves the buffer", p[0] == 7.0f, "");
        }

        std::printf("\nqsa_gate: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("qsa_gate_parity OK\n");
    return 0;
}
