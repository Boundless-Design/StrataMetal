// The gated residual. Mixed and inject bound 1e-4. The write bound is 1e-6 on the terms.
// w_norm is stored as (1 + w). /hc is inside the silu. The write's 2*sigmoid centres a zero injection on 1.
// Native MMVF and the fused multi-token read are not in this port.
#include "strata/metal/gr.hpp"
#include "strata/metal/device.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
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

double worst_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double m = std::fabs((double) a[i]);
        const double e = std::fabs((double) a[i] - (double) b[i]) / (m > 1e-30 ? m : 1e-30);
        if (e > worst) worst = e;
    }
    return worst;
}

double rel_terms(double want, double got, double term_a, double term_b) {
    const double den = std::fabs(term_a) + std::fabs(term_b);
    return std::fabs(want - got) / (den > 1e-30 ? den : 1e-30);
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

float to_bf16(float f) { return bits::f32_from_bf16(bits::bf16_from_f32(f)); }

struct Opts {
    bool per_stream_norm = true;
    bool scale_inside_silu = true;
    bool sigmoid_on_gate = true;
    bool mean_over_streams = true;
    bool round_activation = true;
};

float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void reference(const std::vector<float>& R, const std::vector<float>& w_norm, const std::vector<float>& w_down,
               const std::vector<float>& w_up, const std::vector<float>& w_inject, float eps, long long n_embd,
               long long hc, long long hc_lr, const Opts& o, std::vector<float>& mixed, std::vector<float>& inject) {
    const long long hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    if (o.per_stream_norm) {
        for (long long c = 0; c < hc; ++c) {
            double ms = 0;
            for (long long d = 0; d < n_embd; ++d) {
                const double v = R[(size_t) (c * n_embd + d)];
                ms += v * v;
            }
            ms /= (double) n_embd;
            const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                xn[i] = R[i] * rs * w_norm[i];
            }
        }
    } else {
        double ms = 0;
        for (long long i = 0; i < hc_dim; ++i) ms += (double) R[(size_t) i] * (double) R[(size_t) i];
        ms /= (double) hc_dim;
        const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
        for (long long i = 0; i < hc_dim; ++i) xn[(size_t) i] = R[(size_t) i] * rs * w_norm[(size_t) i];
    }

    std::vector<float> act((size_t) hc_dim);
    for (long long i = 0; i < hc_dim; ++i) act[(size_t) i] = o.round_activation ? to_bf16(xn[(size_t) i]) : xn[(size_t) i];

    std::vector<float> lo((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_down[(size_t) (k * hc_dim + i)];
        const float p = (float) a;
        if (o.sigmoid_on_gate)
            lo[(size_t) k] = o.scale_inside_silu ? silu_f(p / (float) hc) : silu_f(p) / (float) hc;
        else
            lo[(size_t) k] = o.scale_inside_silu ? sigmoid_f(p / (float) hc) : sigmoid_f(p) / (float) hc;
    }
    std::vector<float> lq((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k) lq[(size_t) k] = o.round_activation ? to_bf16(lo[(size_t) k]) : lo[(size_t) k];

    mixed.assign((size_t) n_embd, 0.0f);
    for (long long d = 0; d < n_embd; ++d) {
        float m = 0.0f;
        for (long long c = 0; c < hc; ++c) {
            const long long i = c * n_embd + d;
            double a = 0;
            for (long long k = 0; k < hc_lr; ++k) a += (double) lq[(size_t) k] * (double) w_up[(size_t) (i * hc_lr + k)];
            const float g = (float) a;
            const float s = o.sigmoid_on_gate ? sigmoid_f(g) : silu_f(g);
            m += xn[(size_t) i] * s;
        }
        mixed[(size_t) d] = o.mean_over_streams ? m / (float) hc : m;
    }

    inject.assign((size_t) hc, 0.0f);
    for (long long c = 0; c < hc; ++c) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i)
            a += (double) act[(size_t) i] * (double) w_inject[(size_t) (c * hc_dim + i)];
        inject[(size_t) c] = (float) a;
    }
}

std::vector<uint16_t> pack_bf16(const std::vector<float>& w) {
    std::vector<uint16_t> q(w.size());
    for (size_t i = 0; i < w.size(); ++i) q[i] = bits::bf16_from_f32(w[i]);
    return q;
}

struct Fixture {
    long long n_embd, hc, hc_lr;
    float eps;
    std::vector<float> R, w_norm, w_down, w_up, w_inject;
    std::vector<uint16_t> q_down, q_up, q_inject;
};

Fixture make_fixture(long long n_embd, long long hc, long long hc_lr, uint32_t seed) {
    Fixture f;
    f.n_embd = n_embd;
    f.hc = hc;
    f.hc_lr = hc_lr;
    f.eps = 1e-6f;
    const long long hc_dim = hc * n_embd;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    auto bf16_weight = [&](size_t n, float sigma) {
        std::vector<float> v(n);
        for (auto& x : v) x = to_bf16(gauss(rng) * sigma);
        return v;
    };
    f.R.resize((size_t) hc_dim);
    for (auto& x : f.R) x = gauss(rng);
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) f.R[(size_t) (c * n_embd + d)] *= std::pow(4.0f, (float) c);
    f.w_norm.resize((size_t) hc_dim);
    for (auto& x : f.w_norm) x = 1.0f + 0.1f * gauss(rng);
    f.w_down = bf16_weight((size_t) (hc_lr * hc_dim), 0.2f);
    f.w_up = bf16_weight((size_t) (hc_dim * hc_lr), 0.2f);
    f.w_inject = bf16_weight((size_t) (hc * hc_dim), 0.05f);
    f.q_down = pack_bf16(f.w_down);
    f.q_up = pack_bf16(f.w_up);
    f.q_inject = pack_bf16(f.w_inject);
    return f;
}

void run_read(const Fixture& f, bool fp32_act, GrScratch& sc, std::vector<float>& mixed, std::vector<float>& inject) {
    Bytes dR = upload(f.R), dN = upload(f.w_norm), dD = upload(f.q_down), dU = upload(f.q_up), dJ = upload(f.q_inject);
    Bytes dM((uint64_t) f.n_embd * 4), dI((uint64_t) f.hc * 4);
    gr_read(GrShapes{f.n_embd, f.hc, f.hc_lr}, dR, dN, dD, dU, &dJ, f.eps, fp32_act, sc, dM, dI);
    mixed.resize((size_t) f.n_embd);
    inject.resize((size_t) f.hc);
    std::memcpy(mixed.data(), dM.data(), mixed.size() * 4);
    std::memcpy(inject.data(), dI.data(), inject.size() * 4);
}

void check_pair(const char* name, const std::vector<float>& want_m, const std::vector<float>& got_m,
                const std::vector<float>& want_i, const std::vector<float>& got_i) {
    const double rm = rel_l1(want_m, got_m);
    const double ri = rel_l1(want_i, got_i);
    const double wm = worst_rel(want_m, got_m);
    const bool ok = rm <= 1e-4 && ri <= 1e-4 && std::isfinite(rm) && std::isfinite(ri);
    char detail[96];
    std::snprintf(detail, sizeof detail, "mixed rel %.3e worst %.3e, inject rel %.3e", rm, wm, ri);
    verdict(name, ok, detail);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-gr [--selftest]\n");
        return 2;
    }
    std::printf("gated residual: per-stream norm, silu(dot/hc), mean, 2*sigmoid write\n\n");
    try {
        const Fixture f = make_fixture(256, 4, 32, 1234);
        GrScratch sc = make_gr_scratch(GrShapes{f.n_embd, f.hc, f.hc_lr});
        std::vector<float> got_m, got_i;
        run_read(f, false, sc, got_m, got_i);

        Opts right;
        std::vector<float> want_m, want_i;
        reference(f.R, f.w_norm, f.w_down, f.w_up, f.w_inject, f.eps, f.n_embd, f.hc, f.hc_lr, right, want_m, want_i);

        struct Trap {
            const char* name;
            Opts o;
            double floor;
        };
        const Trap traps[] = {
            {"per-stream RMSNorm vs the whole stack", {false, true, true, true, true}, 0.05},
            {"/hc inside the silu vs outside", {true, false, true, true, true}, 0.05},
            {"SiLU on lo and sigmoid on the gate, swapped", {true, true, false, true, true}, 0.05},
            {"mean over streams vs sum", {true, true, true, false, true}, 0.50},
            {"bf16 vs f32 activations", {true, true, true, true, false}, 1e-4},
        };
        for (const Trap& t : traps) {
            std::vector<float> m, inj;
            reference(f.R, f.w_norm, f.w_down, f.w_up, f.w_inject, f.eps, f.n_embd, f.hc, f.hc_lr, t.o, m, inj);
            const double rel = rel_l1(want_m, m);
            char detail[80];
            std::snprintf(detail, sizeof detail, "(%.2f%% apart, floor %.2f%%)", rel * 100.0, t.floor * 100.0);
            verdict(t.name, rel > t.floor, detail);
        }

        check_pair("gr_read vs the f64 reference", want_m, got_m, want_i, got_i);

        {
            Opts fp32;
            fp32.round_activation = false;
            std::vector<float> wm, wi, gm, gi;
            reference(f.R, f.w_norm, f.w_down, f.w_up, f.w_inject, f.eps, f.n_embd, f.hc, f.hc_lr, fp32, wm, wi);
            run_read(f, true, sc, gm, gi);
            check_pair("f32 activations vs the f64 reference", wm, gm, wi, gi);
            std::vector<float> rm, ri;
            run_read(f, false, sc, rm, ri);
            const bool same = rm.size() == got_m.size() && ri.size() == got_i.size() &&
                              std::memcmp(rm.data(), got_m.data(), rm.size() * 4) == 0 &&
                              std::memcmp(ri.data(), got_i.data(), ri.size() * 4) == 0;
            verdict("the bf16 activation contract is unchanged", same, "");
        }

        {
            const long long hc_dim = f.hc * f.n_embd;
            std::vector<uint16_t> tr_down(f.q_down.size()), tr_up(f.q_up.size());
            for (long long k = 0; k < f.hc_lr; ++k)
                for (long long i = 0; i < hc_dim; ++i)
                    tr_down[(size_t) (i * f.hc_lr + k)] = f.q_down[(size_t) (k * hc_dim + i)];
            for (long long i = 0; i < hc_dim; ++i)
                for (long long k = 0; k < f.hc_lr; ++k)
                    tr_up[(size_t) (k * hc_dim + i)] = f.q_up[(size_t) (i * f.hc_lr + k)];
            Bytes dR = upload(f.R), dN = upload(f.w_norm), dD = upload(tr_down), dU = upload(tr_up);
            Bytes dJ = upload(f.q_inject), dM((uint64_t) f.n_embd * 4), dI((uint64_t) f.hc * 4);
            gr_read(GrShapes{f.n_embd, f.hc, f.hc_lr}, dR, dN, dD, dU, &dJ, f.eps, false, sc, dM, dI);
            std::vector<float> bad((size_t) f.n_embd);
            std::memcpy(bad.data(), dM.data(), bad.size() * 4);
            const double rel = rel_l1(want_m, bad);
            verdict("a transposed weight is observable", rel > 0.05, pct(rel));
        }

        {
            Bytes dR = upload(f.R), dN = upload(f.w_norm), dD = upload(f.q_down), dU = upload(f.q_up);
            Bytes dM((uint64_t) f.n_embd * 4), dI(16);
            float* sentinel = (float*) dI.data();
            sentinel[0] = -12345.0f;
            gr_read(GrShapes{f.n_embd, f.hc, f.hc_lr}, dR, dN, dD, dU, nullptr, f.eps, false, sc, dM, dI);
            verdict("a null injection writes nothing", sentinel[0] == -12345.0f, "");
        }

        std::printf("\n");
        {
            std::mt19937 rng(1234);
            std::normal_distribution<float> gauss(0.0f, 1.0f);
            const long long n_embd = f.n_embd, hc = f.hc, hc_dim = hc * n_embd;
            std::vector<float> block((size_t) n_embd);
            for (auto& x : block) x = gauss(rng);
            std::vector<float> zero((size_t) hc, 0.0f);
            Bytes dR = upload(f.R), dB = upload(block), dZ = upload(zero), dO((uint64_t) hc_dim * 4);
            gr_write(GrShapes{n_embd, hc, f.hc_lr}, dR, dB, dZ, dO);
            std::vector<float> got((size_t) hc_dim);
            std::memcpy(got.data(), dO.data(), got.size() * 4);
            double worst_plain = 0;
            for (long long c = 0; c < hc; ++c)
                for (long long d = 0; d < n_embd; ++d) {
                    const size_t i = (size_t) (c * n_embd + d);
                    const double want = (double) f.R[i] + (double) block[(size_t) d];
                    worst_plain = std::max(worst_plain, rel_terms(want, (double) got[i], (double) f.R[i],
                                                                  (double) block[(size_t) d]));
                }
            char plain[48];
            std::snprintf(plain, sizeof plain, "worst rel %.3e", worst_plain);
            verdict("a zero injection is a plain residual add", worst_plain < 1e-6, plain);

            std::vector<float> got_delta((size_t) hc_dim), want_delta((size_t) hc_dim), wrong_delta((size_t) hc_dim);
            for (long long c = 0; c < hc; ++c)
                for (long long d = 0; d < n_embd; ++d) {
                    const size_t i = (size_t) (c * n_embd + d);
                    got_delta[i] = (float) ((double) got[i] - (double) f.R[i]);
                    want_delta[i] = block[(size_t) d];
                    wrong_delta[i] = block[(size_t) d] * 0.5f;
                }
            const double apart = rel_l1(want_delta, wrong_delta);
            const double got_rel = rel_l1(want_delta, got_delta);
            char centre[80];
            std::snprintf(centre, sizeof centre, "(%.2f%% apart, kernel rel %.3e)", apart * 100.0, got_rel);
            verdict("1*sigmoid vs 2*sigmoid is observable", apart > 0.05 && got_rel < 0.05, centre);

            std::vector<float> inj((size_t) hc);
            for (auto& x : inj) x = gauss(rng) * 3.0f;
            Bytes dI = upload(inj);
            gr_write(GrShapes{n_embd, hc, f.hc_lr}, dR, dB, dI, dO);
            std::memcpy(got.data(), dO.data(), got.size() * 4);
            double worst_stream = 0;
            for (long long c = 0; c < hc; ++c) {
                const double w = 2.0 / (1.0 + std::exp(-(double) inj[(size_t) c] / (double) hc));
                for (long long d = 0; d < n_embd; ++d) {
                    const size_t i = (size_t) (c * n_embd + d);
                    const double term = (double) block[(size_t) d] * w;
                    const double want = (double) f.R[i] + term;
                    worst_stream = std::max(worst_stream, rel_terms(want, (double) got[i], (double) f.R[i], term));
                }
            }
            char streams[48];
            std::snprintf(streams, sizeof streams, "worst rel %.3e", worst_stream);
            verdict("every stream adds the same block output", worst_stream < 1e-6, streams);
        }

        std::printf("\n");
        {
            const Fixture real = make_fixture(2560, 4, 320, 99);
            GrScratch big = make_gr_scratch(GrShapes{real.n_embd, real.hc, real.hc_lr});
            for (const bool fp32_act : {false, true}) {
                std::vector<float> gm, gi, wm, wi;
                run_read(real, fp32_act, big, gm, gi);
                Opts o;
                o.round_activation = !fp32_act;
                reference(real.R, real.w_norm, real.w_down, real.w_up, real.w_inject, real.eps, real.n_embd, real.hc,
                          real.hc_lr, o, wm, wi);
                check_pair(fp32_act ? "real 2560/4/320, f32 activations" : "real 2560/4/320, bf16 activations", wm, gm,
                           wi, gi);
            }
        }

        {
            const GrShapes sh{2, 1, 2};
            GrScratch tiny = make_gr_scratch(sh);
            const float ones[] = {1.0f, 1.0f};
            std::vector<uint16_t> down = {bits::bf16_from_f32(0.5f), 0, 0, 0};
            std::vector<uint16_t> up = {bits::bf16_from_f32(1.5f), 0, bits::bf16_from_f32(1.5f), 0};
            std::vector<uint16_t> injw = {bits::bf16_from_f32(2.0f), 0};
            Bytes dR = upload(ones, sizeof ones), dD = upload(down), dU = upload(up), dJ = upload(injw);
            Bytes dM(8), dI(4);
            for (const float gamma : {1.0f, 1.00390625f}) {
                const float gammas[] = {gamma, gamma};
                Bytes dN = upload(gammas, sizeof gammas);
                float mixed_bf16 = 0, mixed_fp32 = 0, inj_bf16 = 0, inj_fp32 = 0;
                for (const bool fp32_act : {false, true}) {
                    gr_read(sh, dR, dN, dD, dU, &dJ, 0.0f, fp32_act, tiny, dM, dI);
                    float mixed[2];
                    std::memcpy(mixed, dM.data(), sizeof mixed);
                    float injection;
                    std::memcpy(&injection, dI.data(), 4);
                    const float activation = fp32_act ? gamma : to_bf16(gamma);
                    const float projection = activation * 0.5f;
                    const float lo = silu_f(projection);
                    const float gate = (fp32_act ? lo : to_bf16(lo)) * 1.5f;
                    const float expected_mixed = gamma * sigmoid_f(gate);
                    const float expected_inject = activation * 2.0f;
                    const bool ok = std::fabs(mixed[0] - expected_mixed) <= 2e-7f && mixed[0] == mixed[1] &&
                                    injection == expected_inject;
                    char detail[96];
                    std::snprintf(detail, sizeof detail, "mixed %.9f, ref %.9f", (double) mixed[0],
                                  (double) expected_mixed);
                    char name[64];
                    std::snprintf(name, sizeof name, "scalar gamma %.8f, %s", (double) gamma,
                                  fp32_act ? "f32" : "bf16");
                    verdict(name, ok, detail);
                    if (!fp32_act) {
                        mixed_bf16 = mixed[0];
                        inj_bf16 = injection;
                    } else {
                        mixed_fp32 = mixed[0];
                        inj_fp32 = injection;
                    }
                    float saved[2];
                    std::memcpy(saved, mixed, sizeof saved);
                    float* slot = (float*) dI.data();
                    slot[0] = -73.25f;
                    gr_read(sh, dR, dN, dD, dU, nullptr, 0.0f, fp32_act, tiny, dM, dI);
                    float again[2];
                    std::memcpy(again, dM.data(), sizeof again);
                    const bool final_ok = std::memcmp(again, saved, sizeof saved) == 0 && slot[0] == -73.25f;
                    std::snprintf(name, sizeof name, "scalar final mixer %.8f, %s", (double) gamma,
                                  fp32_act ? "f32" : "bf16");
                    verdict(name, final_ok, "");
                }
                const bool seen = std::fabs(mixed_fp32 - mixed_bf16) > 1e-5f &&
                                  (gamma == 1.0f || inj_bf16 != inj_fp32);
                char name[64];
                std::snprintf(name, sizeof name, "scalar gamma %.8f, the two contracts differ", (double) gamma);
                verdict(name, seen, "");
            }
        }

        std::printf("\n");
        verdict("gr_read refuses a negative width",
                throws_metal([] {
                    GrShapes z{256, 4, 32};
                    GrScratch sc = make_gr_scratch(z);
                    Bytes b(16);
                    gr_read(GrShapes{-1, 4, 32}, b, b, b, b, nullptr, 1e-6f, false, sc, b, b);
                }),
                "");
        verdict("gr_write refuses a negative width",
                throws_metal([] {
                    Bytes b(16);
                    gr_write(GrShapes{-1, 4, 32}, b, b, b, b);
                }),
                "");
        {
            GrScratch sc = make_gr_scratch(GrShapes{256, 4, 32});
            Bytes b(64);
            float* p = (float*) b.data();
            p[0] = 7.0f;
            gr_read(GrShapes{0, 4, 32}, b, b, b, b, nullptr, 1e-6f, false, sc, b, b);
            verdict("a zero width leaves the read buffer", p[0] == 7.0f, "");
        }
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("\ngr: %d failures\n", g_fail);
    if (g_fail) return 1;
    std::printf("gr_parity OK\n");
    return 0;
}
