// metal/include/strata/metal/gdn.hpp - gated delta-net, the part between the projections.
//
// The state layout and the contracts are include/strata/kernels/gdn.hpp. This is that, on a shared buffer.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

struct GdnShapes {
    int64_t S = 0;
    int64_t h_k = 0;
    int64_t h_v = 0;
};

// Parameter blocks the shaders read. Layout is the Metal struct, asserted in the .mm.
struct GdnDim { int32_t S, h_k, h_v; };
struct ConvDim { int32_t C, dc; };
struct L2Dim { int32_t cols; float eps; };
struct OutDim { int32_t S; float eps; };
struct BetaDim { int32_t n; };

void gdn_step(Bytes& state, const Bytes& q, const Bytes& k, const Bytes& v, const Bytes& gate, const Bytes& beta,
              Bytes& o, const GdnShapes& s);
void gdn_conv_step(Bytes& conv_state, const Bytes& x, const Bytes& kW, Bytes& out, int64_t channels, int64_t d_conv);
void gdn_l2_norm(Bytes& x, int64_t rows, int64_t cols, float eps);
void gdn_beta_gate(Bytes& beta, int64_t h_v);
void gdn_out_norm(const Bytes& o, const Bytes& z, const Bytes& ssm_norm, Bytes& y, int64_t h_v, int64_t S, float eps);

}  // namespace strata::metal
