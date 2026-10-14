// One shared expert. SiLU is on the gate projection. The other reading is required to sit
// more than 50% away before the kernel is judged. The bound on the structure is 1e-2.
#include "strata/metal/shared_expert.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/sgemv.hpp"

#include "strata/kernels/bf16_bits.hpp"
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

float q8_at(const std::vector<uint8_t>& x, long long i) {
    const uint8_t* blk = x.data() + (size_t) (i / 32) * 34;
    const uint16_t bits = (uint16_t) (blk[0] | (blk[1] << 8));
    return strata::kernels::f32_from_f16(bits) * (float) (int8_t) blk[2 + (i % 32)];
}

float s2_weight(const std::vector<uint8_t>& codes, const std::vector<float>& scales, long long row, long long n_in,
                long long i) {
    const uint8_t byte = codes[(size_t) (row * (n_in / 4) + i / 4)];
    const int code = (byte >> ((i % 4) * 2)) & 3;
    return (float) (code - 1) * scales[(size_t) (row * (n_in / 64) + i / 64)];
}

std::vector<float> project(const std::vector<uint8_t>& act, const std::vector<uint8_t>& codes,
                           const std::vector<float>& scales, long long n_in, long long n_out) {
    std::vector<float> y((size_t) n_out, 0.0f);
    for (long long r = 0; r < n_out; ++r) {
        double a = 0;
        for (long long i = 0; i < n_in; ++i) a += (double) q8_at(act, i) * (double) s2_weight(codes, scales, r, n_in, i);
        y[(size_t) r] = (float) a;
    }
    return y;
}

std::vector<float> quantize_ref(const std::vector<float>& h) {
    std::vector<float> y(h.size(), 0.0f);
    for (size_t b = 0; b < h.size(); b += 32) {
        double amax = 0;
        for (size_t r = b; r < b + 32; ++r) amax = std::fmax(amax, std::fabs((double) h[r]));
        const double dd = amax / 127.0;
        const double id = dd > 0 ? 1.0 / dd : 0.0;
        for (size_t r = b; r < b + 32; ++r) {
            double q = std::round((double) h[r] * id);
            if (q > 127.0) q = 127.0;
            if (q < -127.0) q = -127.0;
            y[r] = (float) (q * dd);
        }
    }
    return y;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-shared-expert [--selftest]\n");
        return 2;
    }
    std::printf("shared_expert: silu on the gate, one scalar, 2-bit Q8_0 projections\n\n");
    try {
        const long long n_embd = 256, n_ff = 64;
        const int tpr = 32;
        std::mt19937 rng(77);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        std::vector<float> fx((size_t) n_embd);
        for (auto& v : fx) v = std::fabs(gauss(rng)) + 0.5f;
        std::vector<uint8_t> gc((size_t) n_ff * n_embd / 4, 0xFF);
        std::vector<uint8_t> uc((size_t) n_ff * n_embd / 4, 0x00);
        std::vector<uint8_t> dc((size_t) n_embd * n_ff / 4);
        for (auto& v : dc) v = (uint8_t) (rng() & 0xFF);
        std::vector<float> gs((size_t) n_ff * (n_embd / 64));
        std::vector<float> us((size_t) n_ff * (n_embd / 64));
        std::vector<float> ds((size_t) n_embd * (n_ff / 64));
        for (auto& s : gs) s = 0.02f + 0.001f * (float) (rng() % 50);
        for (auto& s : us) s = 0.02f + 0.001f * (float) (rng() % 50);
        for (auto& s : ds) s = 0.2f + 0.01f * (float) (rng() % 50);
        std::vector<float> ginp((size_t) n_embd);
        for (auto& v : ginp) v = gauss(rng) * 0.05f;

        Bytes dx = upload(fx.data(), fx.size() * 4);
        Bytes xq(q8_0_bytes(n_embd));
        quantize_q8_0(dx, n_embd, xq);
        std::vector<uint8_t> hx0(xq.size());
        std::memcpy(hx0.data(), xq.data(), hx0.size());

        std::vector<uint16_t> xb((size_t) n_embd), wb((size_t) n_embd);
        for (long long i = 0; i < n_embd; ++i) {
            xb[(size_t) i] = strata::kernels::bf16_from_f32(fx[(size_t) i]);
            wb[(size_t) i] = strata::kernels::bf16_from_f32(ginp[(size_t) i]);
        }

        auto run_ref = [&](bool silu_on_up) {
            const std::vector<float> gp = project(hx0, gc, gs, n_embd, n_ff);
            const std::vector<float> upp = project(hx0, uc, us, n_embd, n_ff);
            std::vector<float> h((size_t) n_ff);
            double hmax = 0;
            for (long long r = 0; r < n_ff; ++r) {
                const double gv = silu_on_up ? (double) upp[(size_t) r] : (double) gp[(size_t) r];
                const double other = silu_on_up ? (double) gp[(size_t) r] : (double) upp[(size_t) r];
                h[(size_t) r] = (float) (gv / (1.0 + std::exp(-gv))) * (float) other;
                hmax = std::fmax(hmax, std::fabs((double) h[(size_t) r]));
            }
            const std::vector<float> h16 = quantize_ref(h);
            std::vector<float> out((size_t) n_embd, 0.0f);
            for (long long o = 0; o < n_embd; ++o) {
                double a = 0;
                for (long long r = 0; r < n_ff; ++r)
                    a += (double) h16[(size_t) r] * (double) s2_weight(dc, ds, o, n_ff, r);
                out[(size_t) o] = (float) a;
            }
            double dot = 0;
            for (long long i = 0; i < n_embd; ++i)
                dot += (double) strata::kernels::f32_from_bf16(xb[(size_t) i]) *
                       (double) strata::kernels::f32_from_bf16(wb[(size_t) i]);
            const double sg = 1.0 / (1.0 + std::exp(-dot));
            for (long long o = 0; o < n_embd; ++o) out[(size_t) o] = (float) ((double) out[(size_t) o] * sg);
            return std::make_pair(out, hmax);
        };

        const auto want_pair = run_ref(false);
        const auto wrong_pair = run_ref(true);
        const std::vector<float>& want = want_pair.first;
        const std::vector<float>& wrong = wrong_pair.first;
        const bool in_range = want_pair.second < 65504.0 && std::isfinite(want_pair.second);
        char detail[96];
        std::snprintf(detail, sizeof detail, "(max |h| %.1f)", want_pair.second);
        verdict("the SwiGLU intermediate stays in fp16 range", in_range, detail);

        double d_wrong = 0, mag = 0;
        for (long long o = 0; o < n_embd; ++o) {
            d_wrong += std::fabs((double) want[(size_t) o] - (double) wrong[(size_t) o]);
            mag += std::fabs((double) want[(size_t) o]);
        }
        const double rel_wrong = d_wrong / (mag > 1e-30 ? mag : 1e-30);
        std::snprintf(detail, sizeof detail, "(%.2f%% apart)", rel_wrong * 100.0);
        verdict("silu on the gate, not on up", rel_wrong > 0.5, detail);

        Bytes xbf = upload(xb.data(), xb.size() * 2);
        Bytes gc_b = upload(gc.data(), gc.size());
        Bytes uc_b = upload(uc.data(), uc.size());
        Bytes dc_b = upload(dc.data(), dc.size());
        Bytes gs_b = upload(gs.data(), gs.size() * 4);
        Bytes us_b = upload(us.data(), us.size() * 4);
        Bytes ds_b = upload(ds.data(), ds.size() * 4);
        Bytes wb_b = upload(wb.data(), wb.size() * 2);
        Bytes gate(n_ff * 4), up(n_ff * 4), hq(q8_0_bytes(n_ff)), g(16);
        Bytes out((n_embd + 16) * 4);
        float* tail = (float*) out.data();
        for (long long i = 0; i < n_embd + 16; ++i) tail[i] = 7.0f;
        shared_expert(xq, xbf, gc_b, gs_b, uc_b, us_b, dc_b, ds_b, wb_b, gate, up, hq, g, out, n_embd, n_ff, tpr);

        bool guard = true;
        for (int i = 0; i < 16; ++i)
            if (tail[n_embd + i] != 7.0f) guard = false;
        verdict("16 words past the end stay put", guard, "");

        double worst = 0, sum = 0;
        long long nonfinite = 0;
        for (long long o = 0; o < n_embd; ++o) {
            const double a = want[(size_t) o], b = tail[o];
            sum += std::fabs(a);
            if (!std::isfinite(a) || !std::isfinite(b)) ++nonfinite;
            const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
            if (!(rel <= worst)) worst = rel;
        }
        std::snprintf(detail, sizeof detail, "worst rel %.3e, %lld non-finite", worst, nonfinite);
        verdict("structure vs reference", guard && nonfinite == 0 && worst <= 1e-2, detail);
        std::printf("  %-52s mean |ref| %.4f\n", "reference magnitude", sum / (double) n_embd);

        verdict("a width that is not a multiple of 64 is refused",
                throws_metal([] {
                    Bytes a(16), b(16), c(16), d(16), e(16), f(16), g0(16), h(16), i(16);
                    Bytes p(16), q(16), r(16), s(16), o(16);
                    shared_expert(a, b, c, d, e, f, g0, h, i, p, q, r, s, o, 32, 64, 32);
                }),
                "");
        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes a(16), b(16), c(16), d(16), e(16), f(16), g0(16), h(16), i(16);
                    Bytes p(16), q(16), r(16), s(16), o(16);
                    shared_expert(a, b, c, d, e, f, g0, h, i, p, q, r, s, o, -64, 64, 32);
                }),
                "");
        {
            Bytes a(16), b(16), c(16), d(16), e(16), f(16), g0(16), h(16), i(16);
            Bytes p(16), q(16), r(16), s(16), o(16);
            float* y = (float*) o.data();
            y[0] = 7.0f;
            shared_expert(a, b, c, d, e, f, g0, h, i, p, q, r, s, o, 0, 64, 32);
            verdict("zero length leaves the buffer", y[0] == 7.0f, "");
        }

        std::printf("\nshared_expert: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("shared_expert_parity OK\n");
    return 0;
}
