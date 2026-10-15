// metal/include/strata/metal/gdn_layer.hpp - one GDN layer, the non-fused order.
//
// quantize the activation, attn_qkv, conv, SiLU on the whole conv output, L2 on q and k,
// 1/sqrt(S) on q only, ssm_alpha and ssm_beta, the beta sigmoid, the softplus gate, the step,
// attn_gate, the output norm, then ssm_out. The three dense projections use a Q8_K activation.
// A 2-bit row has no Q8_K contract. gdn_layer_iq is that same order on ggml blocks: the
// activation is q8_1, and attn_qkv, attn_gate, and ssm_out are the bytes iq_mmvq reads.
// Fused kernels, the router, and the residual are not here.
//
// Conv weights are already channel-major: tap i of channel c is kW[c * d_conv + i], which is
// the pack's ne = [d_conv, channels]. ssm_norm is S floats, the same vector on every head.
// q and k are their own buffers: the L2 kernel reads from the start of the buffer it is given.
#pragma once

#include "strata/metal/bytes.hpp"
#include "strata/metal/sgemv.hpp"

#include <cstdint>

namespace strata::metal {

struct GdnLayerGeom {
    int64_t n_embd = 0;
    int64_t S = 0;
    int64_t h_k = 0;
    int64_t h_v = 0;
    int64_t d_conv = 0;
};

// One S-form plane. `offset` is read only when `form.has_offset` is set.
struct GdnPlane {
    SForm form;
    const Bytes* codes = nullptr;
    const Bytes* scales = nullptr;
    const Bytes* offset = nullptr;
};

struct GdnLayerWeights {
    GdnPlane qkv;        // n_embd -> C, C = S * (2 * h_k + h_v)
    GdnPlane attn_gate;  // n_embd -> V, V = S * h_v
    GdnPlane ssm_out;    // V -> n_embd
    const Bytes* alpha = nullptr;     // bf16, h_v rows of n_embd
    const Bytes* beta = nullptr;      // bf16, h_v rows of n_embd
    const Bytes* conv = nullptr;      // f32, C * d_conv, channel-major
    const Bytes* ssm_norm = nullptr;  // f32, S
    const Bytes* dt = nullptr;        // f32, h_v
    const Bytes* ssm_a = nullptr;     // f32, h_v, negative
};

// One ggml block plane. `type` is the id iq_mmvq dispatches. The bytes are that block
// layout, not an S-form row. Q4_K and Q5_K stay out of the expert wrapper.
struct GdnIqPlane {
    int type = 0;
    const Bytes* weights = nullptr;
};

struct GdnIqWeights {
    GdnIqPlane qkv;        // n_embd -> C, C = S * (2 * h_k + h_v)
    GdnIqPlane attn_gate;  // n_embd -> V, V = S * h_v
    GdnIqPlane ssm_out;    // V -> n_embd
    const Bytes* alpha = nullptr;     // bf16, h_v rows of n_embd
    const Bytes* beta = nullptr;      // bf16, h_v rows of n_embd
    const Bytes* conv = nullptr;      // f32, C * d_conv, channel-major
    const Bytes* ssm_norm = nullptr;  // f32, S
    const Bytes* dt = nullptr;        // f32, h_v
    const Bytes* ssm_a = nullptr;     // f32, h_v, negative
};

struct GdnLayerState {
    Bytes* state = nullptr;       // f32, (S, h_v, S), j fastest
    Bytes* conv_state = nullptr;  // f32, C * (d_conv - 1), channel-major, oldest tap first
};

// Scratch the layer reuses. `make_gdn_layer_scratch` sizes it from the geometry.
struct GdnLayerScratch {
    Bytes x_q8k;
    Bytes x_bf16;
    Bytes qkv;
    Bytes conv_out;
    Bytes q;
    Bytes k;
    Bytes v;
    Bytes alpha;
    Bytes beta;
    Bytes gate;
    Bytes o;
    Bytes z;
    Bytes y;
    Bytes y_q8k;
};

GdnLayerScratch make_gdn_layer_scratch(const GdnLayerGeom& g);

// One token. `x` is n_embd floats. `out` is n_embd floats. State is updated in place.
void gdn_layer(const GdnLayerGeom& g, const GdnLayerWeights& w, GdnLayerState& st, GdnLayerScratch& sc,
               const Bytes& x, Bytes& out);

// The same token on ggml blocks. The q8_1 rows are local. `sc` is still make_gdn_layer_scratch,
// and its Q8_K activation slots stay unused. This does not bind an S-form plane.
void gdn_layer_iq(const GdnLayerGeom& g, const GdnIqWeights& w, GdnLayerState& st, GdnLayerScratch& sc,
                  const Bytes& x, Bytes& out);

}  // namespace strata::metal
