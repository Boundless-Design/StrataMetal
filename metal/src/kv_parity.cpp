// Paged fp16 K and V. The pool is bit-exact against the shared conversion, the page
// table is reversed, and two rival layouts are counted apart from the declared one.
#include "strata/metal/kv.hpp"
#include "strata/metal/device.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;
using strata::kernels::f16_from_f32;

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

std::string pct(double rel) {
    char b[40];
    std::snprintf(b, sizeof b, "%.2f%% apart", rel * 100.0);
    return b;
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

void fill_u16(Bytes& b, uint16_t v) {
    auto* p = (uint16_t*) b.data();
    const size_t n = (size_t) (b.size() / 2u);
    for (size_t i = 0; i < n; ++i) p[i] = v;
}

void check_page(int page_size, const std::vector<float>& kcur, const std::vector<float>& vcur,
                const std::vector<int>& ids) {
    const int nkv = 2;
    const int hd = 256;
    const int tcells = 1100;
    const int n_pages = (tcells + page_size - 1) / page_size;
    std::vector<int> table((size_t) n_pages);
    for (int i = 0; i < n_pages; ++i) table[(size_t) i] = n_pages - 1 - i;

    const size_t pool_n = (size_t) n_pages * (size_t) nkv * (size_t) page_size * (size_t) hd;
    const int guard = 16;
    Bytes kpool((pool_n + (size_t) guard) * 2u);
    Bytes vpool((pool_n + (size_t) guard) * 2u);
    fill_u16(kpool, 7);
    fill_u16(vpool, 7);
    Bytes dtab = upload(table.data(), table.size() * 4u);

    const int row_n = nkv * hd;
    for (int t = 0; t < tcells; ++t) {
        Bytes kc = upload(kcur.data() + (size_t) t * (size_t) row_n, (size_t) row_n * 4u);
        Bytes vc = upload(vcur.data() + (size_t) t * (size_t) row_n, (size_t) row_n * 4u);
        kv_append(kpool, vpool, dtab, t, kc, vc, nkv, hd, page_size);
    }

    const auto* hk = (const uint16_t*) kpool.data();
    const auto* hv = (const uint16_t*) vpool.data();
    const int64_t rows_avail = (int64_t) (pool_n / (size_t) hd);
    auto elsewhere = [&](int64_t r) { return r < 0 || r >= rows_avail; };

    int mism = 0, mism_linear = 0, mism_head = 0;
    for (int cell = 0; cell < tcells; ++cell) {
        for (int h = 0; h < nkv; ++h) {
            for (int d = 0; d < hd; ++d) {
                const size_t src = ((size_t) cell * (size_t) nkv + (size_t) h) * (size_t) hd + (size_t) d;
                const uint16_t wk = f16_from_f32(kcur[src]);
                const uint16_t wv = f16_from_f32(vcur[src]);
                const int64_t row = ((int64_t) table[(size_t) (cell / page_size)] * nkv + h) * page_size +
                                    (cell % page_size);
                if (hk[(size_t) row * (size_t) hd + (size_t) d] != wk ||
                    hv[(size_t) row * (size_t) hd + (size_t) d] != wv)
                    ++mism;
                const int64_t rl = ((int64_t) cell * nkv + h) * page_size + (cell % page_size);
                if (elsewhere(rl) || hk[(size_t) rl * (size_t) hd + (size_t) d] != wk) ++mism_linear;
                const int64_t rh =
                    ((int64_t) table[(size_t) (cell / page_size)] * page_size + (cell % page_size)) * nkv + h;
                if (elsewhere(rh) || hk[(size_t) rh * (size_t) hd + (size_t) d] != wk) ++mism_head;
            }
        }
    }
    int guard_bad = 0;
    for (int i = 0; i < guard; ++i) {
        if (hk[pool_n + (size_t) i] != 7) ++guard_bad;
        if (hv[pool_n + (size_t) i] != 7) ++guard_bad;
    }
    const double n_tot = (double) tcells * nkv * hd;
    const std::string ps = std::to_string(page_size);
    verdict(("kv_append page_size " + ps + " is bit-exact").c_str(), mism == 0 && guard_bad == 0,
            std::to_string(mism) + " of " + std::to_string((long long) n_tot) + " wrong" +
                (guard_bad ? ", guard" : ""));
    verdict(("page table is observable at page_size " + ps).c_str(), (double) mism_linear / n_tot > 0.5,
            pct((double) mism_linear / n_tot));
    if (page_size >= 4) {
        verdict(("head-major inside the page, page_size " + ps).c_str(), (double) mism_head / n_tot > 0.5,
                pct((double) mism_head / n_tot));
    }

    const size_t scr_n = ids.size() * (size_t) row_n;
    Bytes kscr((scr_n + (size_t) guard) * 2u);
    Bytes vscr((scr_n + (size_t) guard) * 2u);
    fill_u16(kscr, 7);
    fill_u16(vscr, 7);
    Bytes dids = upload(ids.data(), ids.size() * 4u);
    kv_gather(kpool, vpool, dtab, dids, (int) ids.size(), nkv, hd, page_size, kscr, vscr);
    const auto* gk = (const uint16_t*) kscr.data();
    const auto* gv = (const uint16_t*) vscr.data();
    int gm = 0;
    for (size_t j = 0; j < ids.size(); ++j) {
        for (int h = 0; h < nkv; ++h) {
            for (int d = 0; d < hd; ++d) {
                const size_t src = ((size_t) ids[j] * (size_t) nkv + (size_t) h) * (size_t) hd + (size_t) d;
                const size_t dst = (j * (size_t) nkv + (size_t) h) * (size_t) hd + (size_t) d;
                if (gk[dst] != f16_from_f32(kcur[src])) ++gm;
                if (gv[dst] != f16_from_f32(vcur[src])) ++gm;
            }
        }
    }
    int gguard = 0;
    for (int i = 0; i < guard; ++i) {
        if (gk[scr_n + (size_t) i] != 7) ++gguard;
        if (gv[scr_n + (size_t) i] != 7) ++gguard;
    }
    verdict(("kv_gather page_size " + ps + " is bit-exact").c_str(), gm == 0 && gguard == 0,
            std::to_string(gm) + " wrong" + (gguard ? ", guard" : ""));
}

void check_negative_page() {
    const int nkv = 2;
    const int hd = 4;
    const int page_size = 4;
    const int n = nkv * hd;
    std::vector<int> table = {-1, 0};
    std::vector<float> k(n), v(n);
    for (int i = 0; i < n; ++i) {
        k[(size_t) i] = 0.25f * (float) (i + 1);
        v[(size_t) i] = -0.5f * (float) (i + 1);
    }
    const size_t pool_n = (size_t) nkv * (size_t) page_size * (size_t) hd;
    Bytes kpool(pool_n * 2u), vpool(pool_n * 2u);
    fill_u16(kpool, 0xBEEF);
    fill_u16(vpool, 0xBEEF);
    Bytes dtab = upload(table.data(), table.size() * 4u);
    Bytes kc = upload(k.data(), k.size() * 4u);
    Bytes vc = upload(v.data(), v.size() * 4u);
    kv_append(kpool, vpool, dtab, 0, kc, vc, nkv, hd, page_size);
    const auto* hk = (const uint16_t*) kpool.data();
    const auto* hv = (const uint16_t*) vpool.data();
    int touched = 0;
    for (size_t i = 0; i < pool_n; ++i) {
        if (hk[i] != 0xBEEF) ++touched;
        if (hv[i] != 0xBEEF) ++touched;
    }
    kv_append(kpool, vpool, dtab, page_size, kc, vc, nkv, hd, page_size);
    int written = 0, stale = 0;
    for (int h = 0; h < nkv; ++h) {
        for (int d = 0; d < hd; ++d) {
            const int row = (0 * nkv + h) * page_size + 0;
            const size_t at = (size_t) row * (size_t) hd + (size_t) d;
            const size_t src = (size_t) h * (size_t) hd + (size_t) d;
            if (hk[at] == f16_from_f32(k[(size_t) src]) && hv[at] == f16_from_f32(v[(size_t) src])) ++written;
            else ++stale;
        }
    }
    verdict("a negative page writes nothing", touched == 0 && written == n && stale == 0,
            std::to_string(touched) + " touched");
}

void check_refusals() {
    Bytes pool(64), cur(64), table(4), ids(16), scratch(64);
    std::vector<int> one = {0};
    std::memcpy(table.data(), one.data(), 4);
    verdict("a negative position is refused",
            throws_metal([&] { kv_append(pool, pool, table, -1, cur, cur, 1, 4, 4); }), "");
    verdict("head_dim not a multiple of 4 is refused",
            throws_metal([&] { kv_append(pool, pool, table, 0, cur, cur, 1, 6, 4); }), "");
    verdict("a page outside the table is refused",
            throws_metal([&] { kv_append(pool, pool, table, 4, cur, cur, 1, 4, 4); }), "");
    std::vector<int> neg_cell = {-3};
    Bytes bad_ids = upload(neg_cell.data(), 4);
    verdict("a negative cell is refused",
            throws_metal([&] { kv_gather(pool, pool, table, bad_ids, 1, 1, 4, 4, scratch, scratch); }), "");
    std::vector<int> neg_page = {-1};
    Bytes neg_tab = upload(neg_page.data(), 4);
    std::vector<int> cell0 = {0};
    Bytes id0 = upload(cell0.data(), 4);
    verdict("a negative physical page is refused",
            throws_metal([&] { kv_gather(pool, pool, neg_tab, id0, 1, 1, 4, 4, scratch, scratch); }), "");
    fill_u16(scratch, 9);
    Bytes empty(16);
    kv_gather(pool, pool, table, empty, 0, 1, 4, 4, scratch, scratch);
    verdict("a zero-length gather leaves the scratch", ((const uint16_t*) scratch.data())[0] == 9, "");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-kv [--selftest]\n");
        return 2;
    }
    std::printf("kv: paged fp16 append and gather, reversed page table\n\n");
    try {
        const int nkv = 2;
        const int hd = 256;
        const int tcells = 1100;
        std::mt19937 rng(20260916u);
        std::normal_distribution<double> gauss(0.0, 1.0);
        auto rnd = [&](double s) { return (float) (gauss(rng) * s); };
        std::vector<float> kcur((size_t) tcells * nkv * hd), vcur(kcur.size());
        for (float& x : kcur) x = rnd(1.0);
        for (float& x : vcur) x = rnd(1.0);
        const std::vector<int> ids = {0, 1, 2, 3, 5, 8, 13, 21, 34, 39, 1099, 700};

        check_page(1, kcur, vcur, ids);
        check_page(4, kcur, vcur, ids);
        check_page(512, kcur, vcur, ids);
        check_negative_page();
        check_refusals();

        std::printf("\nkv: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("kv_parity OK\n");
    return 0;
}
