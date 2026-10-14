// The glue between the GDN projections. Gate bound 1e-6. SiLU bound 1e-7.
// softplus returns x above 20. ssm_a is negative. Scale is an exact multiply.
// The embedding multiply and the add round separately. The expert add is exact.
// The QSA norm divides by the root of the mean of squares, then multiplies by w. Bound 1e-6.
#include "strata/metal/elementwise.hpp"
#include "strata/metal/device.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

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
    Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}
template <class T>
Bytes upload(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
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

double worst_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double m = std::fabs((double) a[i]);
        const double e = std::fabs((double) a[i] - (double) b[i]) / (m > 1e-30 ? m : 1e-30);
        if (e > worst) worst = e;
    }
    return worst;
}

// f64 oracle. use_sum is the DeltaNet reading (no division by the row length, no weight).
std::vector<float> rms_ref(const std::vector<float>& x, const std::vector<float>& w, int64_t rows, int64_t cols,
                           bool use_sum, bool use_w, float eps) {
    std::vector<float> out((size_t) (rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0;
        for (int64_t c = 0; c < cols; ++c) {
            const double v = x[(size_t) (r * cols + c)];
            ss += v * v;
        }
        const double den = std::sqrt((use_sum ? ss : ss / (double) cols) + (double) eps);
        for (int64_t c = 0; c < cols; ++c) {
            const double v = x[(size_t) (r * cols + c)];
            out[(size_t) (r * cols + c)] = (float) ((v / den) * (use_w ? (double) w[(size_t) c] : 1.0));
        }
    }
    return out;
}

std::vector<float> rms_x(int64_t n) {
    std::vector<float> x((size_t) n);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float) std::sin((double) i * 0.017) * 1.7f + 0.4f;
    return x;
}

std::vector<float> rms_w(int64_t cols) {
    std::vector<float> w((size_t) cols);
    for (int64_t c = 0; c < cols; ++c) w[(size_t) c] = 0.5f + 0.002f * (float) (c % 97);
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-elementwise [--selftest]\n");
        return 2;
    }
    std::printf("GDN glue: softplus gate, SiLU, 1/sqrt(S), f16 and bf16\n\n");
    try {
        const int H_V = 48;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> alpha((size_t) H_V), dt((size_t) H_V), a((size_t) H_V), want((size_t) H_V);
        for (int i = 0; i < H_V; ++i) {
            const bool large = (i % 3) == 0;
            alpha[(size_t) i] = large ? (22.0f + 6.0f * (float) (i % 17)) : g(rng) * 3.0f;
            dt[(size_t) i] = g(rng) * 0.5f;
            a[(size_t) i] = -(std::fabs(g(rng)) + 0.1f);
            const double x = (double) alpha[(size_t) i] + (double) dt[(size_t) i];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[(size_t) i] = (float) (sp * (double) a[(size_t) i]);
        }
        int over = 0, past = 0, positive = 0;
        for (int i = 0; i < H_V; ++i) {
            const float x = alpha[(size_t) i] + dt[(size_t) i];
            if (x > 20.0f) ++over;
            if (x > 88.0f) ++past;
            if (a[(size_t) i] >= 0.0f) ++positive;
        }
        char cross[64];
        std::snprintf(cross, sizeof cross, "(%d above 20, %d above 88)", over, past);
        verdict("the fixture crosses the softplus branch", over > 0, cross);
        verdict("the fixture's ssm_a is negative", positive == 0, "");

        Bytes da = upload(alpha), ddt = upload(dt), dsa = upload(a), dg((size_t) H_V * 4);
        gdn_gate(da, ddt, dsa, dg, 1, H_V);
        std::vector<float> got((size_t) H_V);
        std::memcpy(got.data(), dg.data(), got.size() * 4);
        report("gdn_gate vs the double reference", rel_l1(want, got), 1e-6);

        int wrong = 0, differing = 0;
        float first_differ = -1.0f;
        std::vector<float> rival((size_t) H_V);
        for (int i = 0; i < H_V; ++i) {
            const float x = alpha[(size_t) i] + dt[(size_t) i];
            const float sp = std::log1p(std::exp(x));
            rival[(size_t) i] = sp * a[(size_t) i];
            if (!std::isfinite(sp)) ++wrong;
            const float branched = x > 20.0f ? x : std::log1pf(std::exp(x));
            if (sp != branched) {
                ++differing;
                if (first_differ < 0.0f || x < first_differ) first_differ = x;
            }
        }
        const double r = rel_l1(want, rival);
        char branch[96];
        if (std::isfinite(r))
            std::snprintf(branch, sizeof branch, "(%.2f%% apart, %d non-finite, %d heads, first at x = %.1f)", r * 100.0,
                          wrong, differing, (double) first_differ);
        else
            std::snprintf(branch, sizeof branch, "(%d non-finite, %d heads, first at x = %.1f)", wrong, differing,
                          (double) first_differ);
        verdict("the softplus large-x branch is observable", r > 0.05 || wrong > 0, branch);

        int decay = 0;
        for (float v : got)
            if (!(std::exp((double) v) < 1.0)) ++decay;
        verdict("exp(gate) < 1 for every head", decay == 0, "");

        {
            const int n = H_V * 2;
            std::vector<float> a2((size_t) n), w2((size_t) n);
            for (int t = 0; t < 2; ++t) {
                for (int h = 0; h < H_V; ++h) {
                    const int i = t * H_V + h;
                    a2[(size_t) i] = alpha[(size_t) h] + (t ? 1.5f : 0.0f);
                    const double x = (double) a2[(size_t) i] + (double) dt[(size_t) h];
                    const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
                    w2[(size_t) i] = (float) (sp * (double) a[(size_t) h]);
                }
            }
            Bytes da2 = upload(a2), dg2((size_t) n * 4);
            gdn_gate(da2, ddt, dsa, dg2, 2, H_V);
            std::vector<float> g2((size_t) n);
            std::memcpy(g2.data(), dg2.data(), g2.size() * 4);
            report("gdn_gate, two tokens, dt indexed by head", rel_l1(w2, g2), 1e-6);
        }
        {
            Bytes one(16), out(16);
            float* p = (float*) out.data();
            p[0] = 7.f;
            gdn_gate(one, one, one, out, 0, H_V);
            verdict("zero tokens leaves the gate alone", p[0] == 7.f, "");
        }

        std::printf("\n");
        {
            const int n = 4096;
            std::mt19937 rngs(22);
            std::normal_distribution<float> gs(0.0f, 3.0f);
            std::vector<float> x((size_t) n), w((size_t) n);
            for (int i = 0; i < n; ++i) {
                x[(size_t) i] = gs(rngs);
                const double v = (double) x[(size_t) i];
                w[(size_t) i] = (float) (v / (1.0 + std::exp(-v)));
            }
            Bytes dx = upload(x);
            silu_inplace(dx, n);
            std::vector<float> sgot((size_t) n);
            std::memcpy(sgot.data(), dx.data(), sgot.size() * 4);
            report("silu vs the double reference", rel_l1(w, sgot), 1e-7);
        }

        {
            const int n = 1024;
            const int guard = 16;
            std::vector<float> x((size_t) (n + guard)), w((size_t) n);
            for (int i = 0; i < n + guard; ++i) x[(size_t) i] = (float) (i - n / 2) * 0.013f;
            const float s = 1.0f / std::sqrt(128.0f);
            for (int i = 0; i < n; ++i) w[(size_t) i] = x[(size_t) i] * s;
            Bytes dx = upload(x);
            scale_inplace(dx, n, s);
            const float* p = (const float*) dx.data();
            int diff = 0, touched = 0;
            for (int i = 0; i < n; ++i)
                if (p[i] != w[(size_t) i]) ++diff;
            for (int i = n; i < n + guard; ++i)
                if (p[i] != x[(size_t) i]) ++touched;
            char detail[48];
            std::snprintf(detail, sizeof detail, "%d of %d differ, guard %d", diff, n, touched);
            verdict("scale 1/sqrt(128) is exact", diff == 0 && touched == 0, detail);

            std::vector<uint16_t> want16((size_t) n);
            for (int i = 0; i < n; ++i) want16[(size_t) i] = bits::f16_from_f32(p[i]);
            Bytes dy((size_t) n * 2);
            f32_to_f16_bulk(dx, dy, n);
            int hbad = 0;
            const uint16_t* h = (const uint16_t*) dy.data();
            for (int i = 0; i < n; ++i)
                if (h[i] != want16[(size_t) i]) ++hbad;
            verdict("f32_to_f16 matches the shared conversion", hbad == 0, hbad ? std::to_string(hbad) + " differ" : "");
        }

        {
            struct F16Case { float in; uint16_t want; const char* what; };
            const F16Case cases[] = {
                {1e30f, 0x7C00u, "1e30 -> +inf"},
                {-1e30f, 0xFC00u, "-1e30 -> -inf"},
                {65504.0f, 0x7BFFu, "65504 stays finite"},
                {65520.0f, 0x7C00u, "65520 rounds to inf"},
                {5.9604645e-8f, 0x0001u, "2^-24 stays a subnormal"},
            };
            std::vector<float> in;
            for (const F16Case& c : cases) in.push_back(c.in);
            Bytes dx = upload(in), dy(in.size() * 2);
            f32_to_f16_bulk(dx, dy, (int64_t) in.size());
            const uint16_t* h = (const uint16_t*) dy.data();
            int bad = 0;
            for (size_t i = 0; i < in.size(); ++i)
                if (h[i] != cases[i].want) ++bad;
            verdict("f16 overflow saturates and a subnormal stays", bad == 0, "");
        }

        {
            std::vector<uint32_t> bits = {0x7FFFFFFFu, 0xFFFFFFFFu, 0x7F800001u, 0xFF800001u, 0x7FC00000u,
                                          0x7FBFFFFFu, 0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0x00000000u,
                                          0x80000000u, 0x3F808000u, 0x3F818000u, 0x00008000u, 0x7F7F8000u};
            std::mt19937 rb(13);
            while (bits.size() < 4096) bits.push_back((uint32_t) rb());
            Bytes dx((uint64_t) bits.size() * 4), dy((uint64_t) bits.size() * 2);
            std::memcpy(dx.data(), bits.data(), bits.size() * 4);
            f32_to_bf16_bulk(dx, dy, (int64_t) bits.size());
            const uint16_t* h = (const uint16_t*) dy.data();
            int wrong = 0;
            for (size_t i = 0; i < bits.size(); ++i) {
                float f;
                std::memcpy(&f, &bits[i], 4);
                if (h[i] != bits::bf16_from_f32(f)) ++wrong;
            }
            const bool nan_ok = h[0] == 0x7FFFu && h[2] == 0x7FC0u;
            verdict("bf16 matches the shared conversion, NaN included", wrong == 0 && nan_ok,
                    wrong ? std::to_string(wrong) + " differ" : "");
        }

        std::printf("\nembedding, the expert add, and the QSA norm\n");
        {
            std::mt19937 rng(77);
            std::uniform_real_distribution<float> values(-2.0f, 2.0f);
            int mismatches = 0, guards = 0, fma_diff = 0, cases = 0;
            for (const int ebits : {2, 4, 8}) {
                const int bias = ebits == 2 ? -1 : ebits == 4 ? -7 : -16;
                const int per_byte = 8 / ebits;
                const unsigned mask = (1u << ebits) - 1u;
                for (const int group : {16, 32, 64}) {
                    for (const int n : {320, 2560}) {
                        const int row_bytes = n / per_byte;
                        const int row_groups = n / group;
                        std::vector<uint8_t> codes((size_t) 3 * row_bytes, 0);
                        std::vector<float> scales((size_t) 3 * row_groups), offsets(scales.size());
                        std::vector<int> unpacked((size_t) 3 * n);
                        for (size_t i = 0; i < unpacked.size(); ++i) {
                            const unsigned code = (unsigned) (i * 13 + i / (size_t) n * 7) & mask;
                            unpacked[i] = (int) code;
                            codes[i / (size_t) per_byte] |= (uint8_t) (code << ((i % (size_t) per_byte) * (size_t) ebits));
                        }
                        for (size_t i = 0; i < scales.size(); ++i) {
                            scales[i] = values(rng);
                            offsets[i] = values(rng);
                        }
                        scales[0] = -0.0f;
                        offsets[0] = 0.0f;
                        Bytes dc = upload(codes), ds = upload(scales), dof = upload(offsets);
                        for (const bool with_offset : {false, true}) {
                            for (int row = 0; row < 3; ++row) {
                                std::vector<float> want((size_t) n), got((size_t) n + 2, -12345.0f);
                                for (int i = 0; i < n; ++i) {
                                    const size_t gi = (size_t) row * (size_t) row_groups + (size_t) i / (size_t) group;
                                    const float code = (float) (unpacked[(size_t) row * (size_t) n + (size_t) i] + bias);
                                    volatile float product = code * scales[gi];
                                    const float offset = with_offset ? offsets[gi] : 0.0f;
                                    want[(size_t) i] = product + offset;
                                    const float fused = std::fma(code, scales[gi], offset);
                                    if (std::memcmp(&fused, &want[(size_t) i], sizeof(float)) != 0) ++fma_diff;
                                }
                                Bytes dout = upload(got);
                                embedding_gather(dc, ds, with_offset ? &dof : nullptr, n, ebits, bias, group, row, dout,
                                                 1);
                                std::memcpy(got.data(), dout.data(), got.size() * 4);
                                for (int i = 0; i < n; ++i) {
                                    if (std::memcmp(&want[(size_t) i], &got[(size_t) i + 1], sizeof(float)) != 0)
                                        ++mismatches;
                                }
                                if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                                ++cases;
                            }
                        }
                    }
                }
            }
            char detail[128];
            std::snprintf(detail, sizeof detail, "%d row cases, %d bit mismatches, %d guard failures, %d FMA differences",
                          cases, mismatches, guards, fma_diff);
            verdict("embedding gather matches the separate mul and add",
                    mismatches == 0 && guards == 0 && fma_diff != 0, detail);
        }
        {
            Bytes codes(16), scales(16), out(16);
            float* p = (float*) out.data();
            p[0] = -12345.0f;
            embedding_gather(codes, scales, nullptr, 0, 4, -7, 16, 0, out, 0);
            verdict("a zero-length embedding leaves the buffer", p[0] == -12345.0f, "");
        }

        {
            const int n = 1024;
            const int guard = 16;
            std::mt19937 rng(20261014);
            std::uniform_real_distribution<float> u(-8.0f, 8.0f);
            std::vector<float> dst((size_t) (n + guard)), src((size_t) (n + guard)), want((size_t) n);
            for (int i = 0; i < n + guard; ++i) {
                dst[(size_t) i] = u(rng);
                src[(size_t) i] = u(rng);
            }
            for (int i = 0; i < n; ++i) want[(size_t) i] = dst[(size_t) i] + src[(size_t) i];
            Bytes dd = upload(dst), ds = upload(src);
            add_inplace(dd, ds, n);
            const float* p = (const float*) dd.data();
            const float* sp = (const float*) ds.data();
            int diff = 0, touched = 0, src_touched = 0;
            for (int i = 0; i < n; ++i)
                if (std::memcmp(&p[i], &want[(size_t) i], sizeof(float)) != 0) ++diff;
            for (int i = n; i < n + guard; ++i)
                if (p[i] != dst[(size_t) i]) ++touched;
            for (int i = 0; i < n + guard; ++i)
                if (sp[i] != src[(size_t) i]) ++src_touched;
            char detail[80];
            std::snprintf(detail, sizeof detail, "%d of %d differ, guard %d, src %d", diff, n, touched, src_touched);
            verdict("add_inplace is exact", diff == 0 && touched == 0 && src_touched == 0, detail);

            float* z = (float*) dd.data();
            z[0] = 7.0f;
            add_inplace(dd, ds, 0);
            verdict("a zero-length add leaves the buffer", z[0] == 7.0f, "");
        }

        {
            const float eps = 1e-6f;
            const int64_t rows = 24, cols = 256;
            const std::vector<float> x = rms_x(rows * cols);
            const std::vector<float> w = rms_w(cols);
            const std::vector<float> want = rms_ref(x, w, rows, cols, false, true, eps);
            const std::vector<float> rival_sum = rms_ref(x, w, rows, cols, true, false, eps);
            const std::vector<float> rival_now = rms_ref(x, w, rows, cols, false, false, eps);
            const double r_sum = rel_l1(want, rival_sum);
            const double r_now = rel_l1(want, rival_now);
            verdict("sum-of-squares against the mean is observable", r_sum > 0.5, pct(r_sum));
            verdict("leaving the weight out is observable", r_now > 0.5, pct(r_now));

            Bytes dx = upload(x), dw = upload(w);
            rms_norm_weighted(dx, &dw, rows, cols, eps);
            std::vector<float> got((size_t) (rows * cols));
            std::memcpy(got.data(), dx.data(), got.size() * 4);
            const double r_got = rel_l1(want, got);
            const double worst = worst_rel(want, got);
            int nonfinite = 0;
            for (float v : got)
                if (!std::isfinite(v)) ++nonfinite;
            const bool ok = r_got < 1e-6 && nonfinite == 0 && std::isfinite(r_got);
            std::printf("  %-52s rel %.3e  worst %.3e%s\n", "rms_norm_weighted vs the f64 oracle", r_got, worst,
                        ok ? "" : "   *** NO ***");
            if (!ok) ++g_fail;

            // The kernel is in place. The null-weight check has to see the original rows again.
            const std::vector<float> want_null = rival_now;
            dx = upload(x);
            rms_norm_weighted(dx, nullptr, rows, cols, eps);
            std::memcpy(got.data(), dx.data(), got.size() * 4);
            int null_bad = 0;
            for (size_t i = 0; i < got.size(); ++i)
                if (std::fabs((double) got[i] - (double) want_null[i]) > 1e-6) ++null_bad;
            char nd[48];
            std::snprintf(nd, sizeof nd, "%d of %zu differ", null_bad, got.size());
            verdict("a null weight is legal", null_bad == 0, nd);

            auto bounded = [&](const char* name, int64_t nrows, int64_t ncols, int guard) {
                const std::vector<float> xx = rms_x(nrows * ncols);
                const std::vector<float> ww = rms_w(ncols);
                const std::vector<float> yy = rms_ref(xx, ww, nrows, ncols, false, true, eps);
                std::vector<float> buf = xx;
                buf.insert(buf.end(), (size_t) guard, -12345.0f);
                Bytes dxx = upload(buf), dww = upload(ww);
                rms_norm_weighted(dxx, &dww, nrows, ncols, eps);
                std::vector<float> g((size_t) (nrows * ncols));
                std::memcpy(g.data(), dxx.data(), g.size() * 4);
                const float* tail = (const float*) dxx.data() + nrows * ncols;
                int smashed = 0;
                for (int i = 0; i < guard; ++i)
                    if (tail[i] != -12345.0f) ++smashed;
                const double rel = rel_l1(yy, g);
                const bool fine = rel < 1e-6 && smashed == 0 && std::isfinite(rel);
                char detail[80];
                std::snprintf(detail, sizeof detail, "rel %.3e, guard %d", rel, smashed);
                verdict(name, fine, detail);
            };
            bounded("the 2-row norm stays inside its buffer", 2, 256, 16);
            bounded("a row of 100 stays inside its buffer", 1, 100, 32);

            float* idle = (float*) dx.data();
            idle[0] = 7.0f;
            rms_norm_weighted(dx, nullptr, 0, cols, eps);
            verdict("zero rows leave the norm buffer", idle[0] == 7.0f, "");
        }

        std::printf("\n");
        verdict("gdn_gate refuses a negative head count",
                throws_metal([] {
                    Bytes b(16);
                    gdn_gate(b, b, b, b, 1, -1);
                }),
                "");
        verdict("silu refuses a negative length",
                throws_metal([] {
                    Bytes b(16);
                    silu_inplace(b, -1);
                }),
                "");
        verdict("embedding refuses a code width of 3",
                throws_metal([] {
                    Bytes b(64);
                    embedding_gather(b, b, nullptr, 32, 3, 0, 16, 0, b, 0);
                }),
                "");
        verdict("embedding refuses a group that does not divide n",
                throws_metal([] {
                    Bytes b(4096);
                    embedding_gather(b, b, nullptr, 320, 4, -7, 24, 0, b, 0);
                }),
                "");
        verdict("embedding refuses a negative length",
                throws_metal([] {
                    Bytes b(16);
                    embedding_gather(b, b, nullptr, -1, 4, -7, 16, 0, b, 0);
                }),
                "");
        verdict("add_inplace refuses a negative length",
                throws_metal([] {
                    Bytes b(16);
                    add_inplace(b, b, -1);
                }),
                "");
        verdict("rms_norm_weighted refuses a negative row count",
                throws_metal([] {
                    Bytes b(16);
                    rms_norm_weighted(b, nullptr, -1, 256, 1e-6f);
                }),
                "");
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("\nelementwise: %d failures\n", g_fail);
    if (g_fail) return 1;
    std::printf("elementwise_parity OK\n");
    return 0;
}
