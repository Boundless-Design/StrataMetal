// The selection. An independent sort is the reference: score descending, ties by
// ascending index, then the ids sorted ascending. The ids are exact.
#include "strata/metal/topk.hpp"
#include "strata/metal/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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

int width_of(int n_kv, int top_k, int r) {
    const int w = top_k + r - 1;
    return n_kv < w ? n_kv : w;
}

std::vector<int> ref_select(const std::vector<float>& scores, int top_k, int r) {
    const int n = (int) scores.size();
    const int width = width_of(n, top_k, r);
    std::vector<int> idx(n);
    for (int j = 0; j < n; ++j) idx[(size_t) j] = j;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float sa = scores[(size_t) a];
        const float sb = scores[(size_t) b];
        if (sa != sb) return sa > sb;
        return a < b;
    });
    idx.resize((size_t) width);
    std::sort(idx.begin(), idx.end());
    return idx;
}

std::vector<int> run(const std::vector<float>& scores, int top_k, int r, int guard = 16) {
    const int n = (int) scores.size();
    const int width = width_of(n, top_k, r);
    std::vector<int> ids((size_t) width + (size_t) guard, 7);
    Bytes ds = upload(scores.data(), scores.size() * 4u);
    Bytes di = upload(ids.data(), ids.size() * 4u);
    topk_512(ds, n, top_k, r, width, di);
    std::vector<int> got((const int*) di.data(), (const int*) di.data() + width + guard);
    return got;
}

void check_one(const char* name, const std::vector<float>& scores, int top_k, int r) {
    const int width = width_of((int) scores.size(), top_k, r);
    const std::vector<int> want = ref_select(scores, top_k, r);
    const std::vector<int> got = run(scores, top_k, r);
    bool ok = true;
    for (int i = 0; i < width; ++i)
        if (got[(size_t) i] != want[(size_t) i]) ok = false;
    bool asc = width > 0;
    for (int i = 1; i < width; ++i)
        if (got[(size_t) i] <= got[(size_t) i - 1]) asc = false;
    int guard = 0;
    for (int i = 0; i < 16; ++i)
        if (got[(size_t) width + (size_t) i] != 7) ++guard;
    verdict(name, ok && asc && guard == 0,
            std::to_string(width) + " ids" + (ok ? "" : " differ") + (guard ? ", guard" : ""));
    if (!ok) {
        for (int i = 0; i < width; ++i) {
            if (got[(size_t) i] != want[(size_t) i]) {
                std::printf("    first difference at %d: got %d, want %d\n", i, got[(size_t) i], want[(size_t) i]);
                break;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-topk [--selftest]\n");
        return 2;
    }
    std::printf("topk: largest cells, lower index on a tie, ids ascending\n\n");
    try {
        const int top_k = 2048;
        const int r = 4;
        std::mt19937 rng(20260916u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> sc((size_t) kTopkMaxCells);
        for (float& x : sc) x = gauss(rng);

        auto slice = [&](int n) { return std::vector<float>(sc.begin(), sc.begin() + n); };
        check_one("n_kv = 1", slice(1), top_k, r);
        check_one("n_kv = 7", slice(7), top_k, r);
        check_one("n_kv = 2048", slice(2048), top_k, r);
        check_one("n_kv = 2051", slice(2051), top_k, r);
        check_one("n_kv = 2052", slice(2052), top_k, r);
        check_one("n_kv = 5000", slice(5000), top_k, r);
        check_one("n_kv = 32768", sc, top_k, r);

        std::vector<float> eq(2052, 0.25f);
        check_one("2052 equal scores", eq, top_k, r);
        {
            const std::vector<int> got = run(eq, top_k, r);
            bool ident = true;
            for (int j = 0; j < 2051; ++j)
                if (got[(size_t) j] != j) ident = false;
            verdict("equal scores keep 0..width-1", ident, "");
        }

        std::vector<float> cut(5000);
        for (int j = 0; j < 5000; ++j) cut[(size_t) j] = (float) (-(double) j);
        for (int j = 2051 - 50; j < 2051 + 50; ++j) cut[(size_t) j] = -1000.0f;
        check_one("a 100-cell tie straddling the cut", cut, top_k, r);

        for (int n : {2048, 2051}) {
            std::vector<float> idn = slice(n);
            const std::vector<int> got = run(idn, top_k, r);
            bool ident = true;
            for (int j = 0; j < n; ++j)
                if (got[(size_t) j] != j) ident = false;
            verdict(("n_kv = " + std::to_string(n) + " selects every cell").c_str(), ident, "");
        }
        {
            std::vector<float> past = slice(2052);
            const std::vector<int> got = run(past, top_k, r);
            std::vector<char> kept(2052, 0);
            for (int i = 0; i < 2051; ++i) kept[(size_t) got[(size_t) i]] = 1;
            int missing = 0;
            int argmin = 0;
            for (int j = 0; j < 2052; ++j)
                if (past[(size_t) j] < past[(size_t) argmin]) argmin = j;
            for (int j = 0; j < 2052; ++j)
                if (!kept[(size_t) j]) ++missing;
            verdict("n_kv = 2052 drops the smallest score", missing == 1 && !kept[(size_t) argmin],
                    std::to_string(missing) + " dropped");
        }
        {
            std::vector<float> nan8(8, 1.0f);
            nan8[3] = std::numeric_limits<float>::quiet_NaN();
            for (int j = 0; j < 8; ++j) nan8[(size_t) j] = (float) j;
            nan8[3] = std::numeric_limits<float>::quiet_NaN();
            const std::vector<int> got = run(nan8, 3, 2);
            bool has = false;
            for (int i = 0; i < width_of(8, 3, 2); ++i)
                if (got[(size_t) i] == 3) has = true;
            verdict("a NaN is not selected", !has, "");
        }
        {
            std::vector<float> z(4, 0.0f);
            z[1] = -0.0f;
            check_one("-0 matches +0 and the lower index wins", z, 2, 2);
        }

        {
            std::vector<int> ids(4, 7);
            Bytes sc(16), di = upload(ids.data(), ids.size() * 4u);
            topk_512(sc, 0, top_k, r, 4, di);
            verdict("zero length leaves the buffer", ((const int*) di.data())[0] == 7, "");
        }
        verdict("n_kv above 32768 is refused",
                throws_metal([] {
                    Bytes sc(16), ids(16);
                    topk_512(sc, 32769, 2048, 4, 2051, ids);
                }),
                "");
        verdict("a cap below the width is refused",
                throws_metal([] {
                    Bytes sc(64), ids(16);
                    std::vector<float> s(8, 1.0f);
                    std::memcpy(sc.data(), s.data(), s.size() * 4u);
                    topk_512(sc, 8, 2048, 4, 4, ids);
                }),
                "");
        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes sc(16), ids(16);
                    topk_512(sc, -1, 2048, 4, 4, ids);
                }),
                "");

        std::printf("\ntopk: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("topk_parity OK\n");
    return 0;
}
