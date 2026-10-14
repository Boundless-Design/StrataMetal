// The MoE router. Expert ids are exact. Weights are within 1e-5 of the f64 reference.
// The reference is softmax over every expert, a stable descending sort, then w / max(sum, 2**-14).
// Ties keep the lower index. The clamp cannot fire at 10 experts out of 512.
#include "strata/metal/router.hpp"
#include "strata/metal/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr double kClamp = 6.103515625e-05;
constexpr double kTol = 1e-5;

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

void reference(const float* logits, int n_expert, int k, int* ids, float* w, float* w_raw) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    double sum = 0;
    std::vector<double> p((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = std::exp((double) logits[e] - mx);
        sum += p[(size_t) e];
    }
    for (int e = 0; e < n_expert; ++e) p[(size_t) e] /= sum;

    std::vector<int> idx((size_t) n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t) i];
        w_raw[i] = (float) p[(size_t) idx[(size_t) i]];
        w[i] = w_raw[i];
        s += p[(size_t) idx[(size_t) i]];
    }
    s = std::max(s, kClamp);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / s);
}

int run_case(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k) {
    std::vector<int> h_ids((size_t) n_tokens * k);
    std::vector<float> h_w((size_t) n_tokens * k);
    std::vector<int> r_ids((size_t) n_tokens * k);
    std::vector<float> r_w((size_t) n_tokens * k);
    std::vector<float> r_raw((size_t) n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t) t * n_expert], n_expert, k, &r_ids[(size_t) t * k], &r_w[(size_t) t * k],
                  &r_raw[(size_t) t * k]);
    }

    Bytes d_l = upload(logits.data(), logits.size() * sizeof(float));
    Bytes d_ids((uint64_t) h_ids.size() * sizeof(int));
    Bytes d_w((uint64_t) h_w.size() * sizeof(float));
    router_top10(d_l, n_tokens, n_expert, k, d_ids, d_w);
    std::memcpy(h_ids.data(), d_ids.data(), h_ids.size() * sizeof(int));
    std::memcpy(h_w.data(), d_w.data(), h_w.size() * sizeof(float));

    long long id_bad = 0, w_bad = 0;
    double worst = 0, worst_sum = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
        if (h_ids[i] != r_ids[i]) ++id_bad;
        const double den = std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30;
        const double rel = std::fabs((double) h_w[i] - (double) r_w[i]) / den;
        worst = std::max(worst, rel);
        if (!(rel <= kTol)) ++w_bad;
    }
    for (int t = 0; t < n_tokens; ++t) {
        double s = 0;
        for (int i = 0; i < k; ++i) s += (double) h_w[(size_t) t * k + i];
        worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
    }
    const bool sum_bad = !(worst_sum <= kTol);
    std::printf("  %-26s ids %s (%lld bad)   weights worst rel %.3e (%lld over tol)   |sum-1| %.1e\n", name,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad, worst_sum);
    if (sum_bad) {
        std::printf("  %-26s *** NO ***      |sum-1| %.3e is over %.0e\n", name, worst_sum, kTol);
        ++g_fail;
    }
    g_fail += (int) (id_bad + w_bad);
    return (int) (id_bad + w_bad) + (sum_bad ? 1 : 0);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-router [--selftest]\n");
        return 2;
    }
    std::printf("router: softmax, top-k, lower index on a tie, renorm clamped at 2**-14\n\n");
    try {
        const int NE = 512;
        const int K = 10;
        const int NT = 64;
        std::mt19937 rng(2024);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        int bad = 0;

        std::vector<float> a((size_t) NT * NE);
        for (auto& v : a) v = gauss(rng);
        bad += run_case("random normal", a, NT, NE, K);

        std::vector<float> b((size_t) NT * NE, 0.5f);
        bad += run_case("all equal (ties)", b, NT, NE, K);

        std::vector<float> c((size_t) NT * NE);
        for (int t = 0; t < NT; ++t)
            for (int e = 0; e < NE; ++e) c[(size_t) t * NE + e] = (e < 12) ? 2.0f : gauss(rng);
        bad += run_case("12-way exact tie", c, NT, NE, K);

        std::vector<float> d((size_t) NT * NE, 0.0f);
        for (int t = 0; t < NT; ++t) d[(size_t) t * NE + (t % NE)] = 20.0f;
        bad += run_case("dominant expert", d, NT, NE, K);

        const double min_possible = (double) K / (double) NE;
        std::printf("  %-26s top-%d sum >= k/n = %.5f, clamp is %.3e -> margin %.0fx  %s\n", "clamp reachability",
                    K, min_possible, kClamp, min_possible / kClamp,
                    min_possible > kClamp ? "(CLAMP IS UNREACHABLE, asserted)" : "*** REACHABLE ***");
        if (!(min_possible > kClamp)) {
            std::fprintf(stderr, "the clamp CAN trigger, so it needs a test\n");
            ++bad;
            ++g_fail;
        }

        {
            int ids[10];
            float w[10], raw[10];
            reference(b.data(), NE, K, ids, w, raw);
            int id_diff = 0;
            for (int i = 0; i < K; ++i)
                if (ids[i] != (NE - 1 - i)) ++id_diff;
            const double apart = (double) id_diff / (double) K;
            char detail[64];
            std::snprintf(detail, sizeof detail, "(%.2f%% of ids differ)", apart * 100.0);
            verdict("all equal, highest index on a tie", apart > 0.5 && ids[0] == 0, detail);

            double acc = 0, mass = 0;
            for (int i = 0; i < K; ++i) {
                acc += std::fabs((double) raw[i] - (double) w[i]);
                mass += std::fabs((double) w[i]);
            }
            const double rel = acc / (mass > 1e-30 ? mass : 1e-30);
            std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel * 100.0);
            verdict("all equal, renorm left out", rel > 0.5, detail);
        }

        std::mt19937 extra(7);
        std::vector<float> short_l((size_t) 4 * 7);
        for (auto& v : short_l) v = gauss(extra);
        bad += run_case("7 experts, top-3", short_l, 4, 7, 3);

        std::vector<float> wide((size_t) 4 * 1024);
        for (auto& v : wide) v = gauss(extra);
        bad += run_case("1024 experts, top-10", wide, 4, 1024, 10);

        std::vector<float> fullk((size_t) 2 * 64);
        for (auto& v : fullk) v = gauss(extra);
        bad += run_case("k = 64", fullk, 2, 64, 64);

        verdict("k = 65 is refused",
                throws_metal([] {
                    Bytes logits(256), ids(256), w(256);
                    router_top10(logits, 1, 128, 65, ids, w);
                }),
                "");
        verdict("1025 experts is refused",
                throws_metal([] {
                    Bytes logits(16), ids(16), w(16);
                    router_top10(logits, 1, 1025, 1, ids, w);
                }),
                "");
        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes logits(16), ids(16), w(16);
                    router_top10(logits, -1, 32, 4, ids, w);
                }),
                "");
        verdict("k above n_expert is refused",
                throws_metal([] {
                    Bytes logits(64), ids(64), w(64);
                    router_top10(logits, 1, 4, 8, ids, w);
                }),
                "");
        {
            Bytes logits(64), ids(64), w(64);
            int* p = (int*) ids.data();
            p[0] = 7;
            router_top10(logits, 0, 512, 10, ids, w);
            verdict("zero tokens leaves the id buffer", p[0] == 7, "");
        }

        std::printf("\nrouter_top10: %d failures over 4 distributions x %d tokens x %d experts, top-%d\n", bad, NT,
                    NE, K);
        if (bad || g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("router_top10_parity OK\n");
    return 0;
}
