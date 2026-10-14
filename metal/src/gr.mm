// metal/src/gr.mm - one token of the gated residual.
#include "strata/metal/gr.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct GrNormP {
    int n_embd;
    int hc;
    float eps;
};
struct GrGemvP {
    int n_in;
    int n_out;
};
struct GrSiluP {
    int n;
    int hc;
};
struct GrNP {
    int n;
};
struct GrMeanP {
    int n_embd;
    int hc;
};
struct GrWriteP {
    int n_embd;
    int hc;
};
static_assert(sizeof(GrNormP) == 12, "GrNorm must match the shader");
static_assert(sizeof(GrGemvP) == 8, "GrGemv must match the shader");
static_assert(sizeof(GrSiluP) == 8, "GrSilu must match the shader");
static_assert(sizeof(GrNP) == 4, "GrN must match the shader");
static_assert(sizeof(GrMeanP) == 8, "GrMean must match the shader");
static_assert(sizeof(GrWriteP) == 8, "GrWrite must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

int fit(const char* what, int64_t v) {
    if (v < 0 || v > 0x7fffffff) throw MetalError(std::string(what) + " does not fit in int");
    return (int) v;
}

uint32_t groups_for(int n) { return (uint32_t) ((n + 255) / 256); }

void require_positive_read(const GrShapes& s) {
    if (s.n_embd < 0 || s.hc < 0 || s.hc_lr < 0) throw MetalError("gr_read: negative size");
}

void f32_bf16_gemv(const Bytes& x, const Bytes& w, Bytes& y, int n_in, int n_out) {
    const GrGemvP dim{n_in, n_out};
    const Bytes* bufs[] = {&x, &w, &y};
    if (n_out >= 64) {
        const uint32_t groups = (uint32_t) ((n_out + 7) / 8);
        launch("gr_f32_gemv_warp", bufs, 3, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
    } else {
        launch("gr_f32_gemv_naive", bufs, 3, &dim, sizeof dim, Grid{groups_for(n_out), 1, 1, 256, 1, 1});
    }
}

}  // namespace

GrScratch make_gr_scratch(const GrShapes& s) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) throw MetalError("gr scratch: non-positive geometry");
    if (s.hc > 0x7fffffff / s.n_embd) throw MetalError("gr scratch: hc * n_embd overflows");
    const uint64_t hc_dim = (uint64_t) s.hc * (uint64_t) s.n_embd;
    return GrScratch{
        Bytes(hc_dim * 4), Bytes(hc_dim * 2), Bytes((uint64_t) s.hc_lr * 4), Bytes((uint64_t) s.hc_lr * 2),
        Bytes(hc_dim * 4),
    };
}

void gr_read(const GrShapes& s, const Bytes& R, const Bytes& w_norm, const Bytes& w_down, const Bytes& w_up,
             const Bytes* w_inject, float eps, bool fp32_act, GrScratch& sc, Bytes& mixed, Bytes& inject) {
    require_positive_read(s);
    if (s.n_embd == 0 || s.hc == 0 || s.hc_lr == 0) return;
    if (s.hc > 0x7fffffff / s.n_embd) throw MetalError("gr_read: hc * n_embd overflows");
    const int n_embd = fit("n_embd", s.n_embd);
    const int hc = fit("hc", s.hc);
    const int hc_lr = fit("hc_lr", s.hc_lr);
    const int hc_dim = n_embd * hc;
    const uint64_t dim = (uint64_t) hc_dim;
    need("gr R", R, dim * 4);
    need("gr w_norm", w_norm, dim * 4);
    need("gr w_down", w_down, (uint64_t) hc_lr * dim * 2);
    need("gr w_up", w_up, dim * (uint64_t) hc_lr * 2);
    need("gr mixed", mixed, (uint64_t) n_embd * 4);
    need("gr xn", sc.xn, dim * 4);
    need("gr proj", sc.proj, (uint64_t) hc_lr * 4);
    need("gr gate", sc.gate, dim * 4);
    if (!fp32_act) {
        need("gr xq", sc.xq, dim * 2);
        need("gr lq", sc.lq, (uint64_t) hc_lr * 2);
    }
    if (w_inject) {
        need("gr w_inject", *w_inject, (uint64_t) hc * dim * 2);
        need("gr inject", inject, (uint64_t) hc * 4);
    }

    const GrNormP nd{n_embd, hc, eps};
    const Bytes* nb[] = {&R, &w_norm, &sc.xn};
    launch("gr_norm", nb, 3, &nd, sizeof nd, Grid{(uint32_t) hc, 1, 1, 32, 1, 1});
    if (!fp32_act) f32_to_bf16_bulk(sc.xn, sc.xq, hc_dim);

    if (fp32_act) f32_bf16_gemv(sc.xn, w_down, sc.proj, hc_dim, hc_lr);
    else bf16_gemv(sc.xq, w_down, sc.proj, hc_dim, hc_lr);
    const GrSiluP sd{hc_lr, hc};
    const Bytes* sb[] = {&sc.proj};
    launch("gr_silu_hc", sb, 1, &sd, sizeof sd, Grid{groups_for(hc_lr), 1, 1, 256, 1, 1});
    if (!fp32_act) f32_to_bf16_bulk(sc.proj, sc.lq, hc_lr);

    if (fp32_act) f32_bf16_gemv(sc.proj, w_up, sc.gate, hc_lr, hc_dim);
    else bf16_gemv(sc.lq, w_up, sc.gate, hc_lr, hc_dim);
    const GrNP ad{hc_dim};
    const Bytes* ab[] = {&sc.xn, &sc.gate};
    launch("gr_sigmoid_mul", ab, 2, &ad, sizeof ad, Grid{groups_for(hc_dim), 1, 1, 256, 1, 1});

    const GrMeanP md{n_embd, hc};
    const Bytes* mb[] = {&sc.gate, &mixed};
    launch("gr_mean", mb, 2, &md, sizeof md, Grid{groups_for(n_embd), 1, 1, 256, 1, 1});

    if (w_inject) {
        if (fp32_act) f32_bf16_gemv(sc.xn, *w_inject, inject, hc_dim, hc);
        else bf16_gemv(sc.xq, *w_inject, inject, hc_dim, hc);
    }
}

void gr_write(const GrShapes& s, const Bytes& R, const Bytes& block_out, const Bytes& inject, Bytes& R_out) {
    if (s.n_embd < 0 || s.hc < 0) throw MetalError("gr_write: negative size");
    if (s.n_embd == 0 || s.hc == 0) return;
    if (s.hc > 0x7fffffff / s.n_embd) throw MetalError("gr_write: hc * n_embd overflows");
    const int n_embd = fit("n_embd", s.n_embd);
    const int hc = fit("hc", s.hc);
    const int n = n_embd * hc;
    need("gr write R", R, (uint64_t) n * 4);
    need("gr write block", block_out, (uint64_t) n_embd * 4);
    need("gr write inject", inject, (uint64_t) hc * 4);
    need("gr write out", R_out, (uint64_t) n * 4);
    const GrWriteP dim{n_embd, hc};
    const Bytes* bufs[] = {&R, &block_out, &inject, &R_out};
    launch("gr_write", bufs, 4, &dim, sizeof dim, Grid{groups_for(n), 1, 1, 256, 1, 1});
}

}  // namespace strata::metal
