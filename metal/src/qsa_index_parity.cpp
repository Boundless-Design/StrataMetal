// Indexer cell scores. ReLU is per head, the tail cell gets +1e9, and a dropped bias
// has to sit more than 5% away before the kernel is judged.
#include "strata/metal/qsa_index.hpp"
#include "strata/metal/device.hpp"

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

struct Alt {
    bool relu_per_head = true;
    bool use_relu = true;
    bool use_bias = true;
};

std::vector<double> ref_scores(const std::vector<double>& pooled, int n_bid, const std::vector<double>& q, int n_head,
                               int dim, const std::vector<float>& bias, int n_kv, int r, const Alt& alt) {
    std::vector<double> per((size_t) n_bid + 1, 0.0);
    for (int b = 0; b <= n_bid; ++b) {
        double s = 0.0;
        for (int h = 0; h < n_head; ++h) {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d)
                dot += pooled[(size_t) b * (size_t) dim + (size_t) d] * q[(size_t) h * (size_t) dim + (size_t) d];
            if (!alt.use_relu) s += dot;
            else if (alt.relu_per_head) s += dot > 0.0 ? dot : 0.0;
            else s += dot;
        }
        if (alt.use_relu && !alt.relu_per_head) s = s > 0.0 ? s : 0.0;
        if (alt.use_bias && !bias.empty()) s += (double) bias[(size_t) b];
        per[(size_t) b] = s;
    }
    std::vector<double> cell((size_t) n_kv, 0.0);
    for (int j = 0; j < n_kv; ++j) {
        const int cb = j / r < n_bid ? j / r : n_bid;
        double s = per[(size_t) cb];
        if (cb == n_bid && n_kv % r != 0) s = (double) ((float) s + 1e9f);
        else s = (double) (float) s;
        cell[(size_t) j] = s;
    }
    return cell;
}

double rel_terms(double want, double got, double terms) {
    return std::fabs(want - got) / (terms > 1e-30 ? terms : 1e-30);
}

std::string pct(double rel) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f%% apart", rel * 100.0);
    return b;
}

void check_random(const char* title, int n_head, int dim, int r, int n_kv, uint32_t seed) {
    const int n_bid = n_kv / r;
    std::printf("\n%s  heads %d  dim %d  r %d  n_kv %d\n", title, n_head, dim, r, n_kv);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> pooled((size_t) (n_bid + 1) * (size_t) dim);
    std::vector<float> q((size_t) n_head * (size_t) dim);
    std::vector<float> bias((size_t) n_bid + 1);
    for (float& x : pooled) x = gauss(rng);
    for (float& x : q) x = gauss(rng);
    for (float& x : bias) x = gauss(rng) * 8.0f;
    pooled[0] += 3.0f;

    std::vector<double> p64(pooled.begin(), pooled.end());
    std::vector<double> q64(q.begin(), q.end());
    const std::vector<double> want = ref_scores(p64, n_bid, q64, n_head, dim, bias, n_kv, r, Alt{});

    auto apart = [&](const char* name, Alt alt) {
        const std::vector<double> other = ref_scores(p64, n_bid, q64, n_head, dim, bias, n_kv, r, alt);
        double num = 0, den = 0;
        const int done = n_bid * r;
        for (int j = 0; j < done; ++j) {
            num += std::fabs(other[(size_t) j] - want[(size_t) j]);
            den += std::fabs(want[(size_t) j]);
        }
        const double rel = num / (den > 1e-30 ? den : 1e-30);
        verdict(name, rel > 0.05, pct(rel));
    };
    Alt sum_relu;
    sum_relu.relu_per_head = false;
    apart("relu per head vs relu of the sum is observable", sum_relu);
    Alt no_relu;
    no_relu.use_relu = false;
    apart("dropping the relu is observable", no_relu);
    Alt no_bias;
    no_bias.use_bias = false;
    apart("dropping the per-block bias is observable", no_bias);

    std::vector<float> scores((size_t) n_kv + 16, 7.0f);
    Bytes dp = upload(pooled.data(), pooled.size() * 4u);
    Bytes dq = upload(q.data(), q.size() * 4u);
    Bytes db = upload(bias.data(), bias.size() * 4u);
    Bytes ds = upload(scores.data(), scores.size() * 4u);
    qsa_index(dp, n_bid, dq, &db, n_head, dim, r, n_kv, ds);
    std::vector<float> got((const float*) ds.data(), (const float*) ds.data() + n_kv + 16);

    double worst = 0;
    for (int j = 0; j < n_kv; ++j) {
        const int cb = j / r < n_bid ? j / r : n_bid;
        double terms = std::fabs((double) bias[(size_t) cb]);
        if (cb == n_bid && n_kv % r != 0) terms += 1e9;
        for (int h = 0; h < n_head; ++h) {
            double dot = 0;
            for (int d = 0; d < dim; ++d)
                dot += p64[(size_t) cb * (size_t) dim + (size_t) d] * q64[(size_t) h * (size_t) dim + (size_t) d];
            terms += std::fabs(dot);
        }
        worst = std::max(worst, rel_terms(want[(size_t) j], (double) got[(size_t) j], terms));
    }
    char detail[64];
    std::snprintf(detail, sizeof detail, "rel %.3e", worst);
    verdict("cell scores vs the reference", worst <= 1e-6, detail);

    bool same = true;
    for (int j = 1; j < r && j < n_kv; ++j)
        if (got[(size_t) j] != got[0]) same = false;
    verdict("cells of the first block share one score", same, "");
    if (n_kv > r) {
        bool tail = true;
        const int tail0 = n_bid * r;
        for (int j = tail0; j < n_kv; ++j)
            if (got[(size_t) j] != got[(size_t) tail0]) tail = false;
        const bool differs = n_bid > 0 && got[(size_t) tail0] != got[(size_t) (tail0 - 1)];
        verdict("the incomplete tail takes the spare row", tail && differs, "");
    }
    int guard = 0;
    for (int i = 0; i < 16; ++i)
        if (got[(size_t) n_kv + (size_t) i] != 7.0f) ++guard;
    verdict("16 words past the end stayed put", guard == 0, std::to_string(guard) + " changed");
}

void check_tail_bias(int n_kv) {
    const int r = 4;
    const int n_head = 1;
    const int dim = 4;
    const int n_bid = n_kv / r;
    const int tail = n_kv % r;
    std::vector<float> pooled((size_t) (n_bid + 1) * (size_t) dim, 0.0f);
    std::vector<float> q((size_t) n_head * (size_t) dim, 0.0f);
    std::vector<float> scores((size_t) n_kv, 3.0f);
    Bytes dp = upload(pooled.data(), pooled.size() * 4u);
    Bytes dq = upload(q.data(), q.size() * 4u);
    Bytes ds = upload(scores.data(), scores.size() * 4u);
    qsa_index(dp, n_bid, dq, nullptr, n_head, dim, r, n_kv, ds);
    const float* got = (const float*) ds.data();
    int bad = 0;
    for (int j = 0; j < n_kv; ++j) {
        const float want = j >= n_kv - tail ? 1e9f : 0.0f;
        if (got[j] != want) ++bad;
    }
    verdict(("null bias, n_kv " + std::to_string(n_kv) + ", only the tail gets 1e9").c_str(), bad == 0,
            std::to_string(bad) + " wrong");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-qsa-index [--selftest]\n");
        return 2;
    }
    std::printf("qsa_index: relu per head, then the block bias, then +1e9 on the tail\n");
    try {
        check_random("model shape", 4, 128, 4, 10, 11u);
        check_random("short", 2, 8, 2, 5, 5u);
        std::printf("\n");
        check_tail_bias(10);
        check_tail_bias(2051);

        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes p(64), q(64), s(64);
                    qsa_index(p, 0, q, nullptr, 1, 4, 4, -1, s);
                }),
                "");
        verdict("a block count that is not n_kv / r is refused",
                throws_metal([] {
                    Bytes p(256), q(64), s(64);
                    qsa_index(p, 3, q, nullptr, 1, 4, 4, 10, s);
                }),
                "");
        verdict("more than 32 indexer heads is refused",
                throws_metal([] {
                    Bytes p(64), q(64), s(16);
                    qsa_index(p, 0, q, nullptr, 33, 4, 4, 1, s);
                }),
                "");

        std::printf("\nqsa_index: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("qsa_index_parity OK\n");
    return 0;
}
