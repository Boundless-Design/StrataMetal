// The MoE combine. The routed rows are weighted. The shared output is added plain.
// Two wrong readings are required to sit more than 5% away before the kernel is judged.
// The bound is 1e-6. This GPU has no fp64, so the shader sums a pair of floats.
#include "strata/metal/moe_combine.hpp"
#include "strata/metal/device.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr double kTol = 1e-6;

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
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

std::vector<float> reference(const std::vector<float>& parts, const std::vector<float>& w,
                             const std::vector<float>* shared, int n_embd, int k, bool weight_routed,
                             bool weight_shared) {
    std::vector<float> y((size_t) n_embd, 0.0f);
    for (int j = 0; j < n_embd; ++j) {
        double a = 0;
        for (int e = 0; e < k; ++e) {
            const double ww = weight_routed ? (double) w[(size_t) e] : 1.0;
            a += ww * (double) parts[(size_t) e * n_embd + j];
        }
        if (shared) {
            a += weight_shared ? (double) w[0] * (double) (*shared)[(size_t) j] : (double) (*shared)[(size_t) j];
        }
        y[(size_t) j] = (float) a;
    }
    return y;
}

void fill_case(std::mt19937& rng, int n_embd, int k, std::vector<float>& parts, std::vector<float>& w,
               std::vector<float>& shared) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    parts.resize((size_t) k * n_embd);
    w.resize((size_t) k);
    shared.resize((size_t) n_embd);
    for (auto& v : parts) v = g(rng);
    double wsum = 0;
    for (auto& v : w) {
        v = std::fabs(g(rng)) + 0.05f;
        wsum += v;
    }
    for (auto& v : w) v = (float) (v / wsum);
    for (auto& v : shared) v = g(rng) * 3.0f;
}

double run_kernel(const std::vector<float>& parts, const std::vector<float>& w, const std::vector<float>* shared,
                  int n_embd, int k, std::vector<float>& got) {
    Bytes bp = upload(parts.data(), parts.size() * 4);
    Bytes bw = upload(w.data(), w.size() * 4);
    Bytes by((uint64_t) (n_embd + 16) * 4);
    float* tail = (float*) by.data();
    for (int i = 0; i < n_embd + 16; ++i) tail[i] = 7.0f;
    if (shared) {
        Bytes bs = upload(shared->data(), shared->size() * 4);
        moe_combine(bp, bw, &bs, by, n_embd, k);
    } else {
        moe_combine(bp, bw, nullptr, by, n_embd, k);
    }
    got.assign(tail, tail + n_embd);
    for (int i = 0; i < 16; ++i) {
        if (tail[n_embd + i] != 7.0f) return -1.0;
    }
    return 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-moe-combine [--selftest]\n");
        return 2;
    }
    std::printf("moe_combine: weighted routed sum, shared added plain\n\n");
    try {
        const int n_embd = 256;
        const int k = 10;
        std::mt19937 rng(31337);
        std::vector<float> parts, w, shared;
        fill_case(rng, n_embd, k, parts, w, shared);
        const std::vector<float> want = reference(parts, w, &shared, n_embd, k, true, false);

        for (const auto& trap : {std::make_pair(false, false), std::make_pair(true, true)}) {
            const std::vector<float> wrong = reference(parts, w, &shared, n_embd, k, trap.first, trap.second);
            const double rel = rel_l1(want, wrong);
            const char* what = (!trap.first && !trap.second) ? "routed sum left unweighted"
                                                             : "shared output router-weighted";
            char detail[64];
            std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel * 100.0);
            verdict(what, rel > 0.05, detail);
        }

        std::vector<float> got;
        const double guard = run_kernel(parts, w, &shared, n_embd, k, got);
        verdict("16 words past the end stay put", guard == 0.0, "");
        const double rel_k = guard < 0.0 ? 1.0 : rel_l1(want, got);
        char detail[64];
        std::snprintf(detail, sizeof detail, "rel %.3e", rel_k);
        verdict("moe_combine vs reference", guard == 0.0 && rel_k <= kTol, detail);

        const double guard_null = run_kernel(parts, w, nullptr, n_embd, k, got);
        const double rel_no = guard_null < 0.0 ? 0.0 : rel_l1(want, got);
        std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel_no * 100.0);
        verdict("a null shared is observable", guard_null == 0.0 && rel_no > 0.05, detail);

        std::mt19937 wide(2560);
        std::vector<float> parts2, w2, shared2;
        fill_case(wide, 2560, 10, parts2, w2, shared2);
        const std::vector<float> want2 = reference(parts2, w2, &shared2, 2560, 10, true, false);
        const double guard_w = run_kernel(parts2, w2, &shared2, 2560, 10, got);
        const double rel_w = guard_w < 0.0 ? 1.0 : rel_l1(want2, got);
        std::snprintf(detail, sizeof detail, "rel %.3e", rel_w);
        verdict("model shape, 2560 by 10", guard_w == 0.0 && rel_w <= kTol, detail);

        verdict("k = 65 is refused",
                throws_metal([] {
                    Bytes p(256), w(256), y(256);
                    moe_combine(p, w, nullptr, y, 16, 65);
                }),
                "");
        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes p(16), w(16), y(16);
                    moe_combine(p, w, nullptr, y, -1, 4);
                }),
                "");
        {
            Bytes p(16), w(16), y(16);
            float* out = (float*) y.data();
            out[0] = 7.0f;
            moe_combine(p, w, nullptr, y, 0, 10);
            moe_combine(p, w, nullptr, y, 256, 0);
            verdict("zero length leaves the buffer", out[0] == 7.0f, "");
        }

        std::printf("\nmoe_combine: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("moe_combine_parity OK\n");
    return 0;
}
