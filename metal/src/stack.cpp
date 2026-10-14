// Several layers, one token. The feed-forward half matches token.cpp. The mixer follows stack_is_qsa.
#include "strata/metal/stack.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/moe_combine.hpp"
#include "strata/metal/router.hpp"
#include "strata/metal/sgemv.hpp"
#include "strata/metal/shared_expert.hpp"

#include <cstring>
#include <string>

namespace strata::metal {
namespace {

constexpr float kEps = 1e-6f;

void need(const char* what, const void* p) {
    if (p == nullptr) throw MetalError(std::string("stack: ") + what + " is missing");
}

void check_geom(const StackGeom& g) {
    if (g.n_layers < 1) throw MetalError("stack: n_layers must be positive");
    if (g.qsa_interval < 1) throw MetalError("stack: qsa_interval must be positive");
    if (g.gdn.n_embd != g.token.n_embd) throw MetalError("stack: the GDN width does not match");
}

void check_layer(const StackLayerWeights& w) {
    need("attention norm", w.attn.norm);
    need("attention down", w.attn.down);
    need("attention up", w.attn.up);
    need("attention inject", w.attn.inject);
    need("ffn norm", w.ffn.norm);
    need("ffn down", w.ffn.down);
    need("ffn up", w.ffn.up);
    need("ffn inject", w.ffn.inject);
    need("router", w.router);
    need("experts", w.experts);
    need("shared gate codes", w.sh_gate_codes);
    need("shared gate scales", w.sh_gate_scales);
    need("shared up codes", w.sh_up_codes);
    need("shared up scales", w.sh_up_scales);
    need("shared down codes", w.sh_down_codes);
    need("shared down scales", w.sh_down_scales);
    need("shared gate input", w.sh_gate_inp);
}

void one_layer(const StackGeom& g, const StackLayerWeights& lw, StackLayerState& ls, StackState& st,
               StackScratch& sc, int pos, int pos_base, bool qsa) {
    check_layer(lw);
    if (qsa) {
        if (lw.qsa.k.codes == nullptr) throw MetalError("stack: a full-attention layer is missing its weights");
    } else if (lw.gdn.qkv.codes == nullptr) {
        throw MetalError("stack: a GDN layer is missing its weights");
    }

    const TokenGeom& tg = g.token;
    const GrShapes gs{tg.n_embd, tg.hc, tg.hc_lr};
    gr_read(gs, *st.R, *lw.attn.norm, *lw.attn.down, *lw.attn.up, lw.attn.inject, kEps, false, sc.tok.gr,
            sc.tok.mixed, sc.tok.inject);
    if (qsa)
        qsa_layer(tg.qsa, lw.qsa, ls.qsa, sc.tok.qsa, pos, pos_base, sc.tok.mixed, sc.tok.block_out);
    else
        gdn_layer(g.gdn, lw.gdn, ls.gdn, sc.gdn, sc.tok.mixed, sc.tok.block_out);
    gr_write(gs, *st.R, sc.tok.block_out, sc.tok.inject, *st.R);

    gr_read(gs, *st.R, *lw.ffn.norm, *lw.ffn.down, *lw.ffn.up, lw.ffn.inject, kEps, false, sc.tok.gr,
            sc.tok.mixed, sc.tok.inject_ffn);
    f32_to_bf16_bulk(sc.tok.mixed, sc.tok.x_bf16, tg.n_embd);
    bf16_gemv(sc.tok.x_bf16, *lw.router, sc.tok.router_logits, tg.n_embd, tg.n_expert);
    router_top10(sc.tok.router_logits, 1, tg.n_expert, tg.k, sc.tok.route_ids, sc.tok.route_w);

    iq_quantize_q8_1(sc.tok.mixed, tg.n_embd, sc.tok.x_q8_1);
    const int* ids = static_cast<const int*>(sc.tok.route_ids.data());
    auto* parts = static_cast<float*>(sc.tok.parts.data());
    for (int e = 0; e < tg.k; ++e) {
        const int id = ids[e];
        if (id < 0 || id >= tg.n_expert) throw MetalError("stack: a routed expert id is outside the pool");
        iq_expert(lw.experts[id], sc.tok.x_q8_1, tg.expert, 1, sc.tok.one_expert);
        std::memcpy(parts + (size_t) e * (size_t) tg.n_embd, sc.tok.one_expert.data(),
                    (size_t) tg.n_embd * sizeof(float));
    }

    quantize_q8_0(sc.tok.mixed, tg.n_embd, sc.tok.x_q8_0);
    shared_expert(sc.tok.x_q8_0, sc.tok.x_bf16, *lw.sh_gate_codes, *lw.sh_gate_scales, *lw.sh_up_codes,
                  *lw.sh_up_scales, *lw.sh_down_codes, *lw.sh_down_scales, *lw.sh_gate_inp, sc.tok.sh_gate,
                  sc.tok.sh_up, sc.tok.sh_hq, sc.tok.sh_g, sc.tok.shared, tg.n_embd, tg.n_ff_shared, tg.shared_tpr);
    moe_combine(sc.tok.parts, sc.tok.route_w, &sc.tok.shared, sc.tok.block_out, (int) tg.n_embd, tg.k);
    gr_write(gs, *st.R, sc.tok.block_out, sc.tok.inject_ffn, *st.R);
}

}  // namespace

StackScratch make_stack_scratch(const StackGeom& g) {
    check_geom(g);
    return StackScratch{make_token_scratch(g.token), make_gdn_layer_scratch(g.gdn)};
}

int generate_stack_token(const StackGeom& g, const StackWeights& w, StackState& st, StackScratch& sc, int token,
                         int pos, int pos_base, const SamplerParams& sp, const Bytes* history, int history_len) {
    check_geom(g);
    need("layers", w.layers);
    need("residual", st.R);
    need("layer state", st.layers);
    if (pos < 0) throw MetalError("stack: the cell index is negative");
    TokenWeights shell;
    shell.emb_codes = w.emb_codes;
    shell.emb_scales = w.emb_scales;
    shell.emb_offsets = w.emb_offsets;
    TokenState ts;
    ts.R = st.R;
    embed_into_residual(g.token, shell, ts, sc.tok, token);
    for (int64_t layer = 0; layer < g.n_layers; ++layer)
        one_layer(g, w.layers[layer], st.layers[layer], st, sc, pos, pos_base, stack_is_qsa(g, layer));
    shell.head = w.head;
    shell.out = w.out;
    lm_head_logits(g.token, shell, ts, sc.tok);
    sample_tokens(sc.tok.logits, 1, g.token.n_vocab, history, history_len, sp, sc.tok.sampled);
    return static_cast<const int*>(sc.tok.sampled.data())[0];
}

}  // namespace strata::metal
