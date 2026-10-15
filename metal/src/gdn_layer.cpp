// One GDN layer on the kernels that already pass. The order is the non-fused path in
// src/core/layer.cpp: project, conv, SiLU, L2, scale q, alpha and beta, gate, step, z, norm, out.
#include "strata/metal/gdn_layer.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/gdn.hpp"
#include "strata/metal/iq.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

constexpr float kRmsEps = 1e-6f;
constexpr int kThreadsPerRow = 32;

struct Dims {
    int64_t qk;
    int64_t V;
    int64_t C;
};

void check_geom(const GdnLayerGeom& g) {
    if (g.n_embd <= 0 || g.S <= 0 || g.h_k <= 0 || g.h_v <= 0 || g.d_conv < 2)
        throw MetalError("gdn_layer: geometry must be positive, and d_conv at least 2");
    if (g.S > 128) throw MetalError("gdn_layer: S exceeds the staged 128");
    if (g.h_v % g.h_k != 0) throw MetalError("gdn_layer: h_v is not a multiple of h_k");
    if (g.n_embd % 256 != 0) throw MetalError("gdn_layer: n_embd is not a multiple of 256");
    if (g.h_v > 0x7fffffff / g.S) throw MetalError("gdn_layer: value dim overflows");
    const int64_t V = g.S * g.h_v;
    if (V % 256 != 0) throw MetalError("gdn_layer: value dim is not a multiple of 256");
    if (g.h_k > 0x7fffffff / g.S) throw MetalError("gdn_layer: qk overflows");
    const int64_t qk = g.S * g.h_k;
    if (V > 0x7fffffff || qk > (0x7fffffff - V) / 2) throw MetalError("gdn_layer: channels overflow");
}

Dims dims_of(const GdnLayerGeom& g) {
    Dims d;
    d.qk = g.S * g.h_k;
    d.V = g.S * g.h_v;
    d.C = d.qk + d.qk + d.V;
    return d;
}

void need(const char* what, const void* p) {
    if (p == nullptr) throw MetalError(std::string("gdn_layer: ") + what + " is missing");
}

const Bytes* plane_offset(const char* name, const GdnPlane& p) {
    if (p.codes == nullptr) throw MetalError(std::string("gdn_layer: ") + name + " codes are missing");
    if (p.scales == nullptr) throw MetalError(std::string("gdn_layer: ") + name + " scales are missing");
    if (p.form.has_offset && p.offset == nullptr)
        throw MetalError(std::string("gdn_layer: ") + name + " offset is missing");
    return p.form.has_offset ? p.offset : nullptr;
}

}  // namespace

GdnLayerScratch make_gdn_layer_scratch(const GdnLayerGeom& g) {
    check_geom(g);
    const Dims d = dims_of(g);
    return GdnLayerScratch{
        Bytes(q8k_bytes(g.n_embd)), Bytes((uint64_t) g.n_embd * 2), Bytes((uint64_t) d.C * 4),
        Bytes((uint64_t) d.C * 4),  Bytes((uint64_t) d.qk * 4),     Bytes((uint64_t) d.qk * 4),
        Bytes((uint64_t) d.V * 4),  Bytes((uint64_t) g.h_v * 4),    Bytes((uint64_t) g.h_v * 4),
        Bytes((uint64_t) g.h_v * 4), Bytes((uint64_t) d.V * 4),     Bytes((uint64_t) d.V * 4),
        Bytes((uint64_t) d.V * 4),  Bytes(q8k_bytes(d.V)),
    };
}

void gdn_layer(const GdnLayerGeom& g, const GdnLayerWeights& w, GdnLayerState& st, GdnLayerScratch& sc,
               const Bytes& x, Bytes& out) {
    check_geom(g);
    const Dims d = dims_of(g);
    need("state", st.state);
    need("conv state", st.conv_state);
    need("conv weight", w.conv);
    need("ssm_norm", w.ssm_norm);
    need("dt", w.dt);
    need("ssm_a", w.ssm_a);
    need("alpha weight", w.alpha);
    need("beta weight", w.beta);
    const Bytes* qkv_off = plane_offset("attn_qkv", w.qkv);
    const Bytes* gate_off = plane_offset("attn_gate", w.attn_gate);
    const Bytes* out_off = plane_offset("ssm_out", w.ssm_out);

    quantize_q8_k(x, g.n_embd, sc.x_q8k);
    s_gemv_q8k_split(sc.x_q8k, *w.qkv.codes, *w.qkv.scales, qkv_off, sc.qkv, g.n_embd, d.C, w.qkv.form);

    // The kernel reads kW[c * d_conv + i]. The caller stored the weight that way.
    gdn_conv_step(*st.conv_state, sc.qkv, *w.conv, sc.conv_out, d.C, g.d_conv);
    silu_inplace(sc.conv_out, d.C);

    // L2 and the step start at buffer offset 0, so q and k cannot be views into conv_out.
    const float* conv = static_cast<const float*>(sc.conv_out.data());
    std::memcpy(sc.q.data(), conv, (size_t) d.qk * sizeof(float));
    std::memcpy(sc.k.data(), conv + d.qk, (size_t) d.qk * sizeof(float));
    std::memcpy(sc.v.data(), conv + 2 * d.qk, (size_t) d.V * sizeof(float));

    gdn_l2_norm(sc.q, g.h_k, g.S, kRmsEps);
    gdn_l2_norm(sc.k, g.h_k, g.S, kRmsEps);
    scale_inplace(sc.q, d.qk, 1.0f / std::sqrt((float) g.S));

    f32_to_bf16_bulk(x, sc.x_bf16, g.n_embd);
    bf16_gemv_split(sc.x_bf16, *w.alpha, sc.alpha, g.n_embd, g.h_v, kThreadsPerRow);
    bf16_gemv_split(sc.x_bf16, *w.beta, sc.beta, g.n_embd, g.h_v, kThreadsPerRow);
    gdn_beta_gate(sc.beta, g.h_v);
    gdn_gate(sc.alpha, *w.dt, *w.ssm_a, sc.gate, 1, g.h_v);

    const GdnShapes sh{g.S, g.h_k, g.h_v};
    gdn_step(*st.state, sc.q, sc.k, sc.v, sc.gate, sc.beta, sc.o, sh);

    s_gemv_q8k_split(sc.x_q8k, *w.attn_gate.codes, *w.attn_gate.scales, gate_off, sc.z, g.n_embd, d.V,
                     w.attn_gate.form);
    gdn_out_norm(sc.o, sc.z, *w.ssm_norm, sc.y, g.h_v, g.S, kRmsEps);

    quantize_q8_k(sc.y, d.V, sc.y_q8k);
    s_gemv_q8k_split(sc.y_q8k, *w.ssm_out.codes, *w.ssm_out.scales, out_off, out, d.V, g.n_embd, w.ssm_out.form);
}

void gdn_layer_iq(const GdnLayerGeom& g, const GdnIqWeights& w, GdnLayerState& st, GdnLayerScratch& sc,
                  const Bytes& x, Bytes& out) {
    check_geom(g);
    const Dims d = dims_of(g);
    need("state", st.state);
    need("conv state", st.conv_state);
    need("conv weight", w.conv);
    need("ssm_norm", w.ssm_norm);
    need("dt", w.dt);
    need("ssm_a", w.ssm_a);
    need("alpha weight", w.alpha);
    need("beta weight", w.beta);
    need("attn_qkv weights", w.qkv.weights);
    need("attn_gate weights", w.attn_gate.weights);
    need("ssm_out weights", w.ssm_out.weights);

    // q8_1 is 36 bytes a block. The scratch slots are Q8_K and are the wrong width.
    Bytes xq(iq_q8_1_bytes(g.n_embd));
    iq_quantize_q8_1(x, g.n_embd, xq);
    iq_mmvq(w.qkv.type, *w.qkv.weights, 0, xq, g.n_embd, d.C, 1, sc.qkv);

    gdn_conv_step(*st.conv_state, sc.qkv, *w.conv, sc.conv_out, d.C, g.d_conv);
    silu_inplace(sc.conv_out, d.C);

    const float* conv = static_cast<const float*>(sc.conv_out.data());
    std::memcpy(sc.q.data(), conv, (size_t) d.qk * sizeof(float));
    std::memcpy(sc.k.data(), conv + d.qk, (size_t) d.qk * sizeof(float));
    std::memcpy(sc.v.data(), conv + 2 * d.qk, (size_t) d.V * sizeof(float));

    gdn_l2_norm(sc.q, g.h_k, g.S, kRmsEps);
    gdn_l2_norm(sc.k, g.h_k, g.S, kRmsEps);
    scale_inplace(sc.q, d.qk, 1.0f / std::sqrt((float) g.S));

    f32_to_bf16_bulk(x, sc.x_bf16, g.n_embd);
    bf16_gemv_split(sc.x_bf16, *w.alpha, sc.alpha, g.n_embd, g.h_v, kThreadsPerRow);
    bf16_gemv_split(sc.x_bf16, *w.beta, sc.beta, g.n_embd, g.h_v, kThreadsPerRow);
    gdn_beta_gate(sc.beta, g.h_v);
    gdn_gate(sc.alpha, *w.dt, *w.ssm_a, sc.gate, 1, g.h_v);

    const GdnShapes sh{g.S, g.h_k, g.h_v};
    gdn_step(*st.state, sc.q, sc.k, sc.v, sc.gate, sc.beta, sc.o, sh);

    iq_mmvq(w.attn_gate.type, *w.attn_gate.weights, 0, xq, g.n_embd, d.V, 1, sc.z);
    gdn_out_norm(sc.o, sc.z, *w.ssm_norm, sc.y, g.h_v, g.S, kRmsEps);

    Bytes yq(iq_q8_1_bytes(d.V));
    iq_quantize_q8_1(sc.y, d.V, yq);
    iq_mmvq(w.ssm_out.type, *w.ssm_out.weights, 0, yq, d.V, g.n_embd, 1, out);
}

}  // namespace strata::metal
