// The indexer append. The spare key is cell 0, the pool is the mean then the norm, and the
// rotation is at the block's first cell. Each wrong reading has to sit more than 5% away
// before the kernel is judged.
#include "strata/metal/indexer.hpp"
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

double mul_add(double x, double y, double z) {
    const volatile double p = x * y;
    return z + p;
}

void ref_rms(std::vector<double>& x, const std::vector<float>& w, float eps) {
    double ss = 0.0;
    for (double v : x) ss = mul_add(v, v, ss);
    const volatile double q = ss / (double) x.size();
    const volatile double arg = q + (double) eps;
    const volatile double root = std::sqrt(arg);
    const volatile double inv = 1.0 / root;
    for (size_t i = 0; i < x.size(); ++i) {
        const volatile double a = x[i] * inv;
        const volatile double b = a * (w.empty() ? 1.0 : (double) w[i]);
        x[i] = b;
    }
}

void ref_rope(std::vector<double>& x, double pos, int n_rot, double theta, std::vector<double>* terms) {
    const int half = n_rot / 2;
    std::vector<double> a(x.begin(), x.begin() + half);
    std::vector<double> b(x.begin() + half, x.begin() + n_rot);
    if (terms) {
        terms->assign(x.size(), 0.0);
        for (size_t i = (size_t) n_rot; i < x.size(); ++i) (*terms)[i] = std::fabs(x[i]);
    }
    for (int i = 0; i < half; ++i) {
        const double invf = std::pow(theta, -2.0 * (double) i / (double) n_rot);
        const double ang = pos * invf;
        const double c = std::cos(ang);
        const double s = std::sin(ang);
        x[(size_t) i] = a[(size_t) i] * c - b[(size_t) i] * s;
        x[(size_t) half + (size_t) i] = a[(size_t) i] * s + b[(size_t) i] * c;
        if (terms) {
            (*terms)[(size_t) i] = std::fabs(a[(size_t) i] * c) + std::fabs(b[(size_t) i] * s);
            (*terms)[(size_t) half + (size_t) i] = std::fabs(a[(size_t) i] * s) + std::fabs(b[(size_t) i] * c);
        }
    }
}

struct Alt {
    bool rotate_at_start = true;
    bool pool_then_norm = true;
    bool dead_from_first = true;
    bool rope_raw = false;
};

std::vector<double> ref_pooled(const std::vector<std::vector<double>>& raw, const std::vector<double>& pos,
                               const std::vector<float>& w, float eps, int n_rot, double theta, int r,
                               const Alt& alt, std::vector<double>* terms) {
    const int n = (int) raw.size();
    const int dim = (int) raw[0].size();
    const int n_bid = n / r;
    std::vector<std::vector<double>> rows = raw;
    if (alt.rope_raw) {
        for (int j = 0; j < n; ++j) ref_rope(rows[(size_t) j], pos[(size_t) j], n_rot, theta, nullptr);
    }
    std::vector<double> pooled((size_t) (n_bid + 1) * (size_t) dim, 0.0);
    if (terms) terms->assign(pooled.size(), 0.0);
    for (int b = 0; b < n_bid; ++b) {
        std::vector<double> m((size_t) dim, 0.0);
        if (alt.pool_then_norm) {
            for (int d = 0; d < dim; ++d) {
                double acc = 0.0;
                for (int i = 0; i < r; ++i) {
                    const volatile double v = rows[(size_t) (b * r + i)][(size_t) d];
                    acc = acc + v;
                }
                const volatile double div = acc / (double) r;
                m[(size_t) d] = div;
            }
            ref_rms(m, w, eps);
        } else {
            for (int i = 0; i < r; ++i) {
                std::vector<double> one = rows[(size_t) (b * r + i)];
                ref_rms(one, w, eps);
                for (int d = 0; d < dim; ++d) m[(size_t) d] += one[(size_t) d] / (double) r;
            }
        }
        const double rp = alt.rotate_at_start ? pos[(size_t) (b * r)] : pos[(size_t) (b * r + r - 1)];
        std::vector<double> term;
        ref_rope(m, rp, n_rot, theta, terms ? &term : nullptr);
        for (int d = 0; d < dim; ++d) {
            pooled[(size_t) b * (size_t) dim + (size_t) d] = m[(size_t) d];
            if (terms) (*terms)[(size_t) b * (size_t) dim + (size_t) d] = term[(size_t) d];
        }
    }
    std::vector<double> dead = rows[alt.dead_from_first ? 0 : (size_t) n - 1];
    ref_rms(dead, w, eps);
    ref_rope(dead, 0.0, n_rot, theta, nullptr);
    for (int d = 0; d < dim; ++d) pooled[(size_t) n_bid * (size_t) dim + (size_t) d] = dead[(size_t) d];
    if (terms) {
        for (int d = 0; d < dim; ++d)
            (*terms)[(size_t) n_bid * (size_t) dim + (size_t) d] = std::fabs(dead[(size_t) d]);
    }
    return pooled;
}

double sep_of(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(a[i] - b[i]);
        den += std::fabs(a[i]);
    }
    return num / (den > 1e-30 ? den : 1e-30);
}

std::string pct(double rel) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f%% apart", rel * 100.0);
    return b;
}

void build_table(int n_rot, double theta, int rows, std::vector<float>& cos_tab, std::vector<float>& sin_tab) {
    const int half = n_rot / 2;
    cos_tab.assign((size_t) rows * (size_t) half, 0.0f);
    sin_tab.assign((size_t) rows * (size_t) half, 0.0f);
    for (int p = 0; p < rows; ++p) {
        for (int i = 0; i < half; ++i) {
            const double invf = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * invf;
            cos_tab[(size_t) p * (size_t) half + (size_t) i] = (float) std::cos(ang);
            sin_tab[(size_t) p * (size_t) half + (size_t) i] = (float) std::sin(ang);
        }
    }
}

struct Run {
    std::vector<float> pooled;
    std::vector<float> dead;
    std::vector<float> tail;
    int block_pos = 0;
    int block_pos_after_idle = 0;
};

Run append_all(const std::vector<float>& raw, int n_tok, int dim, int r, int n_rot, int pos_base, float eps,
               const std::vector<float>& w, const std::vector<float>& cos_tab, const std::vector<float>& sin_tab,
               int guard) {
    const int n_bid = n_tok / r;
    const size_t pooled_n = (size_t) (n_bid + 1) * (size_t) dim;
    std::vector<float> pooled(pooled_n + (size_t) guard, 7.0f);
    std::vector<float> dead((size_t) dim, 7.0f);
    std::vector<float> tail((size_t) (r - 1) * (size_t) dim, 7.0f);
    int block_pos = 7;
    Bytes draw = upload(raw.data(), (size_t) dim * 4u);
    Bytes dpos = upload(&block_pos, 4);
    Bytes dw = upload(w.data(), w.size() * 4u);
    Bytes dtail = upload(tail.data(), tail.size() * 4u);
    Bytes ddead = upload(dead.data(), dead.size() * 4u);
    Bytes dpooled = upload(pooled.data(), pooled.size() * 4u);
    Bytes dblock = upload(&block_pos, 4);
    Bytes dcos = upload(cos_tab.data(), cos_tab.size() * 4u);
    Bytes dsin = upload(sin_tab.data(), sin_tab.size() * 4u);
    IndexerBuf buf{dtail, ddead, dpooled, dblock};
    int idle = 7;
    for (int t = 0; t < n_tok; ++t) {
        std::memcpy(draw.data(), raw.data() + (size_t) t * (size_t) dim, (size_t) dim * 4u);
        std::memcpy(dpos.data(), &t, 4);
        indexer_key_append(draw, dpos, pos_base, dw, eps, buf, dim, r, n_rot, dcos, dsin);
        if (t == r) std::memcpy(&idle, dblock.data(), 4);
    }
    Run out;
    out.pooled.assign((const float*) dpooled.data(), (const float*) dpooled.data() + pooled.size());
    out.dead.assign((const float*) ddead.data(), (const float*) ddead.data() + dim);
    out.tail.assign((const float*) dtail.data(), (const float*) dtail.data() + tail.size());
    std::memcpy(&out.block_pos, dblock.data(), 4);
    out.block_pos_after_idle = idle;
    return out;
}

double rel_terms(double want, double got, double terms) {
    return std::fabs(want - got) / (terms > 1e-30 ? terms : 1e-30);
}

void check_shape(const char* title, int dim, int r, int n_rot, int n_tok, int pos_base, uint32_t seed, bool traps) {
    std::printf("\n%s  dim %d  r %d  n_rot %d  tokens %d  pos_base %d\n", title, dim, r, n_rot, n_tok, pos_base);
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const float eps = 1e-6f;
    const double theta = 1e7;
    std::vector<std::vector<double>> raw((size_t) n_tok, std::vector<double>((size_t) dim));
    std::vector<float> flat((size_t) n_tok * (size_t) dim);
    std::vector<double> pos((size_t) n_tok);
    for (int t = 0; t < n_tok; ++t) {
        const double scale = std::pow(4.0, (double) (t % r));
        for (int d = 0; d < dim; ++d) {
            const float f = (float) (gauss(rng) * scale);
            flat[(size_t) t * (size_t) dim + (size_t) d] = f;
            raw[(size_t) t][(size_t) d] = (double) f;
        }
        pos[(size_t) t] = (double) (pos_base + t);
    }
    std::vector<float> w((size_t) dim);
    for (float& x : w) x = 1.0f + 0.1f * (float) gauss(rng);
    const int table_rows = pos_base + n_tok + 8;
    std::vector<float> cos_tab, sin_tab;
    build_table(n_rot, theta, table_rows, cos_tab, sin_tab);

    const int n_bid = n_tok / r;
    Alt good;
    std::vector<double> terms;
    const std::vector<double> want = ref_pooled(raw, pos, w, eps, n_rot, theta, r, good, &terms);

    if (traps) {
        auto apart = [&](const char* name, Alt alt, bool index_positions) {
            std::vector<double> p = pos;
            if (index_positions)
                for (int j = 0; j < n_tok; ++j) p[(size_t) j] = (double) j;
            const std::vector<double> altv = ref_pooled(raw, p, w, eps, n_rot, theta, r, alt, nullptr);
            const double rel = sep_of(want, altv);
            verdict(name, rel > 0.05, pct(rel));
        };
        Alt last;
        last.rotate_at_start = false;
        apart("rotating at the block's last cell is observable", last, false);
        Alt norm_first;
        norm_first.pool_then_norm = false;
        apart("norm-then-pool vs pool-then-norm is observable", norm_first, false);
        Alt dead_last;
        dead_last.dead_from_first = false;
        apart("the spare key from the last raw key is observable", dead_last, false);
        Alt rope_raw;
        rope_raw.rope_raw = true;
        apart("rotating the raw indexer keys is observable", rope_raw, false);
        apart("the rotation's position base is observable", good, true);
    }

    const Run got = append_all(flat, n_tok, dim, r, n_rot, pos_base, eps, w, cos_tab, sin_tab, 16);

    int dead_bad = 0;
    for (int d = 0; d < dim; ++d)
        if (got.dead[(size_t) d] != (float) want[(size_t) n_bid * (size_t) dim + (size_t) d]) ++dead_bad;
    verdict("the spare key is bit-exact", dead_bad == 0,
            std::to_string(dead_bad) + " of " + std::to_string(dim) + " wrong");

    double worst = 0;
    for (int b = 0; b < n_bid; ++b) {
        for (int d = 0; d < dim; ++d) {
            const size_t i = (size_t) b * (size_t) dim + (size_t) d;
            worst = std::max(worst, rel_terms(want[i], (double) got.pooled[i], terms[i]));
        }
    }
    char detail[64];
    std::snprintf(detail, sizeof detail, "rel %.3e", worst);
    verdict("pooled block keys vs the reference", worst <= 1e-6, detail);

    int spare_row_bad = 0;
    for (int d = 0; d < dim; ++d)
        if (got.pooled[(size_t) n_bid * (size_t) dim + (size_t) d] != got.dead[(size_t) d]) ++spare_row_bad;
    verdict("the spare key sits at row n_bid", spare_row_bad == 0,
            std::to_string(spare_row_bad) + " mismatched");

    {
        std::vector<double> tm((size_t) dim, 0.0);
        const int tail_n = n_tok - n_bid * r;
        for (int i = n_bid * r; i < n_tok; ++i)
            for (int d = 0; d < dim; ++d) tm[(size_t) d] += raw[(size_t) i][(size_t) d] / (double) tail_n;
        ref_rms(tm, w, eps);
        double num = 0, den = 0;
        for (int d = 0; d < dim; ++d) {
            num += std::fabs(tm[(size_t) d] - (double) got.dead[(size_t) d]);
            den += std::fabs(tm[(size_t) d]);
        }
        const double rel = num / (den > 1e-30 ? den : 1e-30);
        verdict("the incomplete tail is not pooled into the spare key", rel > 0.5, pct(rel));
    }

    int tail_bad = 0;
    const int tail_rows = r - 1;
    for (int s = 0; s < tail_rows; ++s) {
        int src = -1;
        for (int t = n_tok - 1; t >= 0; --t)
            if (t % r == s) {
                src = t;
                break;
            }
        if (src < 0) continue;
        for (int d = 0; d < dim; ++d)
            if (got.tail[(size_t) s * (size_t) dim + (size_t) d] != flat[(size_t) src * (size_t) dim + (size_t) d])
                ++tail_bad;
    }
    verdict("the tail keeps the raw key, and a completing cell is not stored", tail_bad == 0,
            std::to_string(tail_bad) + " wrong");

    const int last_complete = (n_bid - 1) * r + (r - 1);
    const int want_pos = pos_base + (last_complete / r) * r;
    const int idle_pos = pos_base + 0 * r;
    verdict("block_pos is the first cell of the last complete block", got.block_pos == want_pos,
            "got " + std::to_string(got.block_pos) + " want " + std::to_string(want_pos));
    verdict("a cell that does not complete a block leaves block_pos", got.block_pos_after_idle == idle_pos,
            "got " + std::to_string(got.block_pos_after_idle));

    int guard_bad = 0;
    for (int i = 0; i < 16; ++i)
        if (got.pooled[(size_t) (n_bid + 1) * (size_t) dim + (size_t) i] != 7.0f) ++guard_bad;
    verdict("16 words past the end stayed put", guard_bad == 0, std::to_string(guard_bad) + " changed");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-indexer [--selftest]\n");
        return 2;
    }
    std::printf("indexer: raw tail, pool on a block boundary, rotate at the first cell\n");
    try {
        check_shape("model shape", 128, 4, 64, 10, 100, 100u, true);
        check_shape("short block", 16, 2, 8, 5, 7, 7u, false);

        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes raw(64), pos(4), w(64), tail(64), dead(64), pooled(128), block(4), cos(64), sin(64);
                    int cell = 0;
                    std::memcpy(pos.data(), &cell, 4);
                    IndexerBuf buf{tail, dead, pooled, block};
                    indexer_key_append(raw, pos, 0, w, 1e-6f, buf, -1, 4, 4, cos, sin);
                }),
                "");
        verdict("a block shorter than 2 is refused",
                throws_metal([] {
                    Bytes raw(64), pos(4), w(64), tail(64), dead(64), pooled(128), block(4), cos(64), sin(64);
                    int cell = 0;
                    std::memcpy(pos.data(), &cell, 4);
                    IndexerBuf buf{tail, dead, pooled, block};
                    indexer_key_append(raw, pos, 0, w, 1e-6f, buf, 8, 1, 4, cos, sin);
                }),
                "");
        verdict("an odd rotation width is refused",
                throws_metal([] {
                    Bytes raw(64), pos(4), w(64), tail(64), dead(64), pooled(128), block(4), cos(64), sin(64);
                    int cell = 0;
                    std::memcpy(pos.data(), &cell, 4);
                    IndexerBuf buf{tail, dead, pooled, block};
                    indexer_key_append(raw, pos, 0, w, 1e-6f, buf, 8, 4, 3, cos, sin);
                }),
                "");
        verdict("a negative cell index is refused",
                throws_metal([] {
                    Bytes raw(64), pos(4), w(64), tail(96), dead(64), pooled(256), block(4), cos(256), sin(256);
                    int cell = -1;
                    std::memcpy(pos.data(), &cell, 4);
                    IndexerBuf buf{tail, dead, pooled, block};
                    indexer_key_append(raw, pos, 0, w, 1e-6f, buf, 16, 4, 8, cos, sin);
                }),
                "");

        std::printf("\nindexer: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("indexer_parity OK\n");
    return 0;
}
