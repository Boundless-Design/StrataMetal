// Partial NEOX rotation. The table is the float64 frequencies. The rotation uses that table.
// Pair i with i + n_rot/2. The adjacent pair (2i, 2i+1) is the rival.
#include "strata/metal/rope.hpp"
#include "strata/metal/device.hpp"

#include <cmath>
#include <cstdint>
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

void ref_table(int n_rot, double theta, int max_pos, std::vector<float>& cos_tab, std::vector<float>& sin_tab) {
    const int pairs = n_rot / 2;
    cos_tab.assign((size_t) max_pos * (size_t) pairs, 0.0f);
    sin_tab.assign(cos_tab.size(), 0.0f);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < pairs; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * (size_t) pairs + (size_t) i] = (float) std::cos(ang);
            sin_tab[(size_t) p * (size_t) pairs + (size_t) i] = (float) std::sin(ang);
        }
    }
}

void rotate_neox(const std::vector<float>& x, const std::vector<int>& pos, int head_dim, int n_rot,
                 const std::vector<float>& cos_tab, const std::vector<float>& sin_tab, std::vector<float>& out) {
    const int pairs = n_rot / 2;
    const int rows = (int) pos.size();
    out.resize((size_t) rows * (size_t) head_dim);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + (size_t) r * (size_t) head_dim;
        float* orow = out.data() + (size_t) r * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) orow[d] = xr[d];
        const int p = pos[(size_t) r];
        for (int i = 0; i < pairs; ++i) {
            const float a = xr[i];
            const float b = xr[pairs + i];
            const float c = cos_tab[(size_t) p * (size_t) pairs + (size_t) i];
            const float s = sin_tab[(size_t) p * (size_t) pairs + (size_t) i];
            const float ac = a * c;
            const float bs = b * s;
            const float as_ = a * s;
            const float bc = b * c;
            orow[i] = ac - bs;
            orow[pairs + i] = as_ + bc;
        }
    }
}

void rotate_adjacent(const std::vector<float>& x, const std::vector<int>& pos, int head_dim, int n_rot,
                     const std::vector<float>& cos_tab, const std::vector<float>& sin_tab, std::vector<float>& out) {
    const int pairs = n_rot / 2;
    const int rows = (int) pos.size();
    out.resize((size_t) rows * (size_t) head_dim);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + (size_t) r * (size_t) head_dim;
        float* orow = out.data() + (size_t) r * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) orow[d] = xr[d];
        const int p = pos[(size_t) r];
        for (int i = 0; i < pairs; ++i) {
            const float a = xr[2 * i];
            const float b = xr[2 * i + 1];
            const float c = cos_tab[(size_t) p * (size_t) pairs + (size_t) i];
            const float s = sin_tab[(size_t) p * (size_t) pairs + (size_t) i];
            orow[2 * i] = a * c - b * s;
            orow[2 * i + 1] = a * s + b * c;
        }
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b, int head_dim, int n_rot) {
    double d = 0, m = 0;
    const int rows = (int) (a.size() / (size_t) head_dim);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < n_rot; ++i) {
            const size_t at = (size_t) r * (size_t) head_dim + (size_t) i;
            d += std::fabs((double) a[at] - (double) b[at]);
            m += std::fabs((double) a[at]);
        }
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

void check_shape(const char* title, int head_dim, int n_rot, int rows, int max_pos, uint32_t seed) {
    const double theta = 1.0e7;
    const int pairs = n_rot / 2;
    std::vector<float> cos_tab, sin_tab;
    ref_table(n_rot, theta, max_pos, cos_tab, sin_tab);
    std::vector<float> built_c(cos_tab.size(), 3.0f), built_s(sin_tab.size(), 3.0f);
    build_rope_table(n_rot, theta, max_pos, built_c.data(), built_s.data());
    int table_bad = 0;
    for (size_t i = 0; i < cos_tab.size(); ++i) {
        if (std::memcmp(&built_c[i], &cos_tab[i], 4) != 0) ++table_bad;
        if (std::memcmp(&built_s[i], &sin_tab[i], 4) != 0) ++table_bad;
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * (size_t) head_dim);
    for (float& v : x) v = gauss(rng);
    std::vector<int> pos((size_t) rows);
    for (int r = 0; r < rows; ++r) pos[(size_t) r] = r % max_pos;

    std::vector<float> want;
    rotate_neox(x, pos, head_dim, n_rot, cos_tab, sin_tab, want);
    const int guard = 16;
    std::vector<float> host_out(want.size() + (size_t) guard, 7.0f);
    Bytes dx = upload(x.data(), x.size() * 4u);
    Bytes dout = upload(host_out.data(), host_out.size() * 4u);
    Bytes dc = upload(cos_tab.data(), cos_tab.size() * 4u);
    Bytes ds = upload(sin_tab.data(), sin_tab.size() * 4u);
    Bytes dp = upload(pos.data(), pos.size() * 4u);
    rope_neox_apply(dout, dx, rows, head_dim, n_rot, dc, ds, dp);
    const float* got = (const float*) dout.data();

    int over = 0, tail_bad = 0, bits = 0;
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        double scale = 0;
        for (int d = 0; d < head_dim; ++d)
            scale = std::max(scale, (double) std::fabs(x[(size_t) r * (size_t) head_dim + (size_t) d]));
        if (scale < 1e-30) scale = 1e-30;
        for (int d = 0; d < head_dim; ++d) {
            const size_t at = (size_t) r * (size_t) head_dim + (size_t) d;
            const double rel = std::fabs((double) want[at] - (double) got[at]) / scale;
            if (rel > worst) worst = rel;
            if (rel > 1e-6) ++over;
            if (std::memcmp(&want[at], &got[at], 4) != 0) ++bits;
            if (d >= n_rot && std::memcmp(&x[at], &got[at], 4) != 0) ++tail_bad;
        }
    }
    int guard_bad = 0;
    for (int i = 0; i < guard; ++i)
        if (got[want.size() + (size_t) i] != 7.0f) ++guard_bad;

    std::vector<float> adj;
    rotate_adjacent(x, pos, head_dim, n_rot, cos_tab, sin_tab, adj);
    const double apart = rel_l1(want, adj, head_dim, n_rot);

    char worst_b[32];
    std::snprintf(worst_b, sizeof worst_b, "rel %.3e", worst);
    std::printf("%s  head_dim %d  n_rot %d  rows %d\n", title, head_dim, n_rot, rows);
    verdict("table vs the float64 frequencies", table_bad == 0, std::to_string(table_bad) + " differ");
    verdict("rotation vs the same table", over == 0 && guard_bad == 0,
            std::string(worst_b) + ", " + std::to_string(bits) + " bits differ" + (guard_bad ? ", guard" : ""));
    verdict("the tail past n_rot is copied", tail_bad == 0, std::to_string(tail_bad) + " changed");
    verdict("adjacent pairs are observable", apart > 0.05, pct(apart));

    std::vector<float> inplace = x;
    inplace.resize(x.size() + (size_t) guard, 7.0f);
    Bytes dio = upload(inplace.data(), inplace.size() * 4u);
    rope_neox_apply(dio, dio, rows, head_dim, n_rot, dc, ds, dp);
    const float* ig = (const float*) dio.data();
    int ip_bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (std::memcmp(&ig[i], &got[i], 4) != 0) ++ip_bad;
    verdict("in place matches a separate buffer", ip_bad == 0, std::to_string(ip_bad) + " differ");
}

void check_pairing() {
    const int head_dim = 256;
    const int n_rot = 64;
    const int pairs = n_rot / 2;
    const int max_pos = 32;
    std::vector<float> cos_tab, sin_tab;
    ref_table(n_rot, 1.0e7, max_pos, cos_tab, sin_tab);
    Bytes dc = upload(cos_tab.data(), cos_tab.size() * 4u);
    Bytes ds = upload(sin_tab.data(), sin_tab.size() * 4u);

    std::vector<float> e((size_t) head_dim, 0.0f);
    e[0] = 1.0f;
    int p7 = 7;
    Bytes dx = upload(e.data(), e.size() * 4u);
    Bytes dout((size_t) head_dim * 4u);
    Bytes dp = upload(&p7, 4);
    rope_neox_apply(dout, dx, 1, head_dim, n_rot, dc, ds, dp);
    const float* o = (const float*) dout.data();
    int other = 0;
    for (int d = 0; d < head_dim; ++d) {
        if (d == 0 || d == pairs) continue;
        if (o[d] != 0.0f) ++other;
    }
    const bool neox = o[pairs] != 0.0f;
    const bool adjacent = o[1] != 0.0f;
    char detail[80];
    std::snprintf(detail, sizeof detail, "dim %d moved %.6f, dim 1 is %.6f", pairs, (double) o[pairs], (double) o[1]);
    verdict("a 1 at dim 0 moves dim n_rot/2", neox && !adjacent && other == 0, detail);

    int p0 = 0;
    Bytes dp0 = upload(&p0, 4);
    Bytes d0 = upload(e.data(), e.size() * 4u);
    rope_neox_apply(d0, d0, 1, head_dim, n_rot, dc, ds, dp0);
    const float* z = (const float*) d0.data();
    int id_bad = 0;
    for (int d = 0; d < head_dim; ++d)
        if (z[d] != e[(size_t) d]) ++id_bad;
    verdict("position 0 leaves the row unchanged", id_bad == 0, std::to_string(id_bad) + " changed");
}

void check_refusals() {
    std::vector<float> row(8, 1.0f);
    std::vector<float> tab(8, 1.0f);
    int p = 0;
    Bytes x = upload(row.data(), row.size() * 4u);
    Bytes out = upload(row.data(), row.size() * 4u);
    Bytes c = upload(tab.data(), tab.size() * 4u);
    Bytes s = upload(tab.data(), tab.size() * 4u);
    Bytes pos = upload(&p, 4);
    std::vector<float> sentinel(4, 7.0f);
    Bytes stay = upload(sentinel.data(), sentinel.size() * 4u);
    rope_neox_apply(stay, x, 0, 8, 4, c, s, pos);
    verdict("zero rows leaves the buffer", ((const float*) stay.data())[0] == 7.0f, "");
    verdict("a negative row count is refused",
            throws_metal([&] { rope_neox_apply(out, x, -1, 8, 4, c, s, pos); }), "");
    verdict("an odd rotation width is refused",
            throws_metal([&] { rope_neox_apply(out, x, 1, 8, 3, c, s, pos); }), "");
    verdict("a rotation wider than the row is refused",
            throws_metal([&] { rope_neox_apply(out, x, 1, 8, 16, c, s, pos); }), "");
    int hi = 4;
    Bytes phi = upload(&hi, 4);
    verdict("a position past the table is refused",
            throws_metal([&] { rope_neox_apply(out, x, 1, 8, 4, c, s, phi); }), "");
    int neg = -1;
    Bytes pn = upload(&neg, 4);
    verdict("a negative position is refused",
            throws_metal([&] { rope_neox_apply(out, x, 1, 8, 4, c, s, pn); }), "");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-rope [--selftest]\n");
        return 2;
    }
    std::printf("rope: partial NEOX, host float64 table\n\n");
    try {
        check_shape("model shape", 256, 64, 24 * 8, 32, 11u);
        std::printf("\n");
        check_shape("short", 16, 8, 3, 8, 7u);
        std::printf("\n");
        check_pairing();
        check_refusals();
        std::printf("\nrope: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("rope_parity OK\n");
    return 0;
}
