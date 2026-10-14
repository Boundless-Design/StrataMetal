// metal/src/gdn.mm - launch the delta-net kernels. S is staged in 128 floats, as the CUDA kernel is.
#include "strata/metal/gdn.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " buffer is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void gdn_step(Bytes& state, const Bytes& q, const Bytes& k, const Bytes& v, const Bytes& gate, const Bytes& beta,
              Bytes& o, const GdnShapes& s) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) throw MetalError("gdn_step: S exceeds the staged 128");
    if (s.h_v > 0x7fffffff || s.S > 0x7fffffff) throw MetalError("gdn_step: dimension does not fit");
    const uint64_t S = (uint64_t) s.S, hk = (uint64_t) s.h_k, hv = (uint64_t) s.h_v;
    need("gdn state", state, S * hv * S * 4);
    need("gdn q", q, hk * S * 4);
    need("gdn k", k, hk * S * 4);
    need("gdn v", v, hv * S * 4);
    need("gdn gate", gate, hv * 4);
    need("gdn beta", beta, hv * 4);
    need("gdn o", o, hv * S * 4);
    const GdnDim dim{(int32_t) s.S, (int32_t) s.h_k, (int32_t) s.h_v};
    const Bytes* bufs[] = {&state, &q, &k, &v, &gate, &beta, &o};
    const uint32_t gx = (uint32_t) ((s.S + 32 - 1) / 32);
    launch("gdn_step", bufs, 7, &dim, sizeof(dim), Grid{gx, (uint32_t) s.h_v, 1, 32, 1, 1});
}

void gdn_conv_step(Bytes& conv_state, const Bytes& x, const Bytes& kW, Bytes& out, int64_t channels, int64_t d_conv) {
    if (channels <= 0) return;
    if (d_conv < 2) throw MetalError("gdn_conv_step: d_conv must be at least 2");
    need("conv state", conv_state, (uint64_t) channels * (uint64_t) (d_conv - 1) * 4);
    need("conv x", x, (uint64_t) channels * 4);
    need("conv weight", kW, (uint64_t) channels * (uint64_t) d_conv * 4);
    need("conv out", out, (uint64_t) channels * 4);
    const ConvDim dim{(int32_t) channels, (int32_t) d_conv};
    const Bytes* bufs[] = {&conv_state, &x, &kW, &out};
    const uint32_t groups = (uint32_t) ((channels + 255) / 256);
    launch("gdn_conv", bufs, 4, &dim, sizeof(dim), Grid{groups, 1, 1, 256, 1, 1});
}

void gdn_l2_norm(Bytes& x, int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) throw MetalError("gdn_l2_norm: cols exceed the 1024-wide reduction");
    need("l2", x, (uint64_t) rows * (uint64_t) cols * 4);
    const L2Dim dim{(int32_t) cols, eps};
    const Bytes* bufs[] = {&x};
    launch("gdn_l2", bufs, 1, &dim, sizeof(dim), Grid{(uint32_t) rows, 1, 1, 32, 1, 1});
}

void gdn_beta_gate(Bytes& beta, int64_t h_v) {
    if (h_v <= 0) return;
    need("beta", beta, (uint64_t) h_v * 4);
    const BetaDim dim{(int32_t) h_v};
    const Bytes* bufs[] = {&beta};
    const uint32_t groups = (uint32_t) ((h_v + 127) / 128);
    launch("gdn_beta", bufs, 1, &dim, sizeof(dim), Grid{groups, 1, 1, 128, 1, 1});
}

void gdn_out_norm(const Bytes& o, const Bytes& z, const Bytes& ssm_norm, Bytes& y, int64_t h_v, int64_t S, float eps) {
    if (h_v <= 0 || S <= 0) return;
    if (S > 1024) throw MetalError("gdn_out_norm: S exceeds the 1024-wide reduction");
    need("out o", o, (uint64_t) h_v * (uint64_t) S * 4);
    need("out z", z, (uint64_t) h_v * (uint64_t) S * 4);
    need("out norm", ssm_norm, (uint64_t) S * 4);
    need("out y", y, (uint64_t) h_v * (uint64_t) S * 4);
    const OutDim dim{(int32_t) S, eps};
    const Bytes* bufs[] = {&o, &z, &ssm_norm, &y};
    launch("gdn_out_norm", bufs, 4, &dim, sizeof(dim), Grid{(uint32_t) h_v, 1, 1, 32, 1, 1});
}

}  // namespace strata::metal
