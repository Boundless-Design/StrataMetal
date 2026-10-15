// One token: embed, one block, the output mix, one sample.
// One draft step: project the next token and the caller's residual, then the same block.
#include "strata/metal/token.hpp"

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
    if (p == nullptr) throw MetalError(std::string("token: ") + what + " is missing");
}

void check_geom(const TokenGeom& g) {
    if (g.n_embd <= 0 || g.n_embd % 256 != 0) throw MetalError("token: n_embd must be a positive multiple of 256");
    if (g.hc <= 0 || g.hc_lr <= 0) throw MetalError("token: the gated residual is not set up");
    if (g.k < 1 || g.k > 64) throw MetalError("token: k must be 1..64");
    if (g.n_expert < g.k || g.n_expert > 1024) throw MetalError("token: n_expert must be from k through 1024");
    if (g.n_vocab <= 0) throw MetalError("token: n_vocab must be positive");
    if (g.n_ff_shared <= 0 || g.shared_tpr <= 0) throw MetalError("token: the shared expert is not set up");
    if (g.qsa.n_embd != g.n_embd) throw MetalError("token: the attention block width does not match");
    if (g.expert.n_embd != g.n_embd || g.expert.n_ff <= 0) throw MetalError("token: the routed expert does not match");
    if (!iq_type_supported(g.expert.gu_type) || !iq_type_supported(g.expert.d_type))
        throw MetalError("token: the routed expert type is not a Coder expert");
    if (iq_block_values(g.expert.gu_type) != 256) throw MetalError("token: the gate quant is a down type");
}

void check_weights(const TokenWeights& w) {
    need("embedding codes", w.emb_codes);
    need("embedding scales", w.emb_scales);
    need("attention norm", w.attn.norm);
    need("attention down", w.attn.down);
    need("attention up", w.attn.up);
    need("attention inject", w.attn.inject);
    need("ffn norm", w.ffn.norm);
    need("ffn down", w.ffn.down);
    need("ffn up", w.ffn.up);
    need("ffn inject", w.ffn.inject);
    need("output norm", w.head.norm);
    need("output down", w.head.down);
    need("output up", w.head.up);
    need("router", w.router);
    need("experts", w.experts);
    need("shared gate codes", w.sh_gate_codes);
    need("shared gate scales", w.sh_gate_scales);
    need("shared up codes", w.sh_up_codes);
    need("shared up scales", w.sh_up_scales);
    need("shared down codes", w.sh_down_codes);
    need("shared down scales", w.sh_down_scales);
    need("shared gate input", w.sh_gate_inp);
    need("output codes", w.out.codes);
    need("output scales", w.out.scales);
}

void gr_half(const TokenGeom& g, const TokenGr& half, const Bytes* inject, TokenState& st, TokenScratch& sc,
             Bytes& inj) {
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    gr_read(gs, *st.R, *half.norm, *half.down, *half.up, inject, kEps, false, sc.gr, sc.mixed, inj);
}

void need_fc(const char* name, const DraftFc& fc) {
    if (fc.codes == nullptr) throw MetalError(std::string("draft: ") + name + " codes are missing");
    if (fc.scales == nullptr) throw MetalError(std::string("draft: ") + name + " scales are missing");
    if (fc.norm == nullptr) throw MetalError(std::string("draft: ") + name + " norm is missing");
    if (fc.form.has_offset && fc.offset == nullptr)
        throw MetalError(std::string("draft: ") + name + " offset is missing");
    if (fc.form.code_bits != 4 && fc.form.code_bits != 8)
        throw MetalError(std::string("draft: ") + name + " code_bits has no Q8_0 contract");
}

void project_q8_0(const Bytes& x, const DraftFc& fc, int64_t n, Bytes& q8, Bytes& y) {
    quantize_q8_0(x, n, q8);
    const Bytes* off = fc.form.has_offset ? fc.offset : nullptr;
    s_gemv_q8_0_split(q8, *fc.codes, *fc.scales, off, y, n, n, fc.form);
}

// The hidden norm is one row of hc * n_embd. Each stream of that row is then its own projection.
void mix_draft(const TokenGeom& g, const TokenWeights& w, const DraftWeights& d, const Bytes& residual,
               TokenState& st, TokenScratch& sc, int token) {
    const uint64_t nbytes = (uint64_t) g.hc * (uint64_t) g.n_embd * 4u;
    if (residual.size() < nbytes) throw MetalError("draft: the residual is short");
    if (st.R == nullptr) throw MetalError("draft: the layer residual is missing");
    if (st.R->size() < nbytes) throw MetalError("draft: the layer residual is short");
    if (token < 0 || token >= g.n_vocab) throw MetalError("draft: the token id is outside the vocabulary");
    need_fc("fc_embedding", d.embedding);
    need_fc("fc_hidden", d.hidden);
    if (d.embedding.norm->size() < (uint64_t) g.n_embd * 4u) throw MetalError("draft: fc_embedding norm is short");
    if (d.hidden.norm->size() < nbytes) throw MetalError("draft: fc_hidden norm is short");

    embedding_gather(*w.emb_codes, *w.emb_scales, w.emb_offsets, g.n_embd, g.emb_code_bits, g.emb_code_bias,
                     g.emb_group, token, sc.emb, 0);
    rms_norm_weighted(sc.emb, d.embedding.norm, 1, g.n_embd, kEps);
    project_q8_0(sc.emb, d.embedding, g.n_embd, sc.draft_q8, sc.draft_e);

    std::memcpy(sc.draft_hidden.data(), residual.data(), (size_t) nbytes);
    rms_norm_weighted(sc.draft_hidden, d.hidden.norm, 1, g.hc * g.n_embd, kEps);

    auto* R = static_cast<float*>(st.R->data());
    const auto* hidden = static_cast<const float*>(sc.draft_hidden.data());
    const auto* e = static_cast<const float*>(sc.draft_e.data());
    const int64_t n = g.n_embd;
    for (int64_t c = 0; c < g.hc; ++c) {
        std::memcpy(sc.draft_row.data(), hidden + c * n, (size_t) n * sizeof(float));
        project_q8_0(sc.draft_row, d.hidden, n, sc.draft_q8, sc.draft_h);
        const auto* h = static_cast<const float*>(sc.draft_h.data());
        for (int64_t i = 0; i < n; ++i) R[c * n + i] = h[i] + e[i];
    }
}

}  // namespace

TokenScratch make_token_scratch(const TokenGeom& g) {
    check_geom(g);
    const int64_t n = g.n_embd;
    const int64_t hc = g.hc;
    const int64_t ff = g.n_ff_shared;
    const int guard = 16;
    return TokenScratch{
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) hc * 4u),
        Bytes((uint64_t) hc * 4u),
        make_gr_scratch(GrShapes{n, hc, g.hc_lr}),
        make_qsa_layer_scratch(g.qsa),
        Bytes((uint64_t) g.n_expert * 4u),
        Bytes((uint64_t) g.k * 4u),
        Bytes((uint64_t) g.k * 4u),
        Bytes(iq_q8_1_bytes(n)),
        Bytes(q8_0_bytes(n)),
        Bytes((uint64_t) n * 2u),
        Bytes((uint64_t) g.k * (uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) ff * 4u),
        Bytes((uint64_t) ff * 4u),
        Bytes(q8_0_bytes(ff)),
        Bytes(16),
        Bytes(q8k_bytes(n)),
        Bytes((uint64_t) (g.n_vocab + guard) * 4u),
        Bytes(4),
        Bytes((uint64_t) hc * (uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes((uint64_t) n * 4u),
        Bytes(q8_0_bytes(n)),
    };
}

void embed_into_residual(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int token) {
    check_geom(g);
    need("residual", st.R);
    need("embedding codes", w.emb_codes);
    need("embedding scales", w.emb_scales);
    if (token < 0 || token >= g.n_vocab) throw MetalError("token: the token id is outside the vocabulary");
    embedding_gather(*w.emb_codes, *w.emb_scales, w.emb_offsets, g.n_embd, g.emb_code_bits, g.emb_code_bias,
                     g.emb_group, token, sc.emb, 0);
    auto* R = static_cast<float*>(st.R->data());
    const auto* emb = static_cast<const float*>(sc.emb.data());
    for (int64_t c = 0; c < g.hc; ++c)
        std::memcpy(R + c * g.n_embd, emb, (size_t) g.n_embd * sizeof(float));
}

void block_forward(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int pos,
                   int pos_base) {
    check_geom(g);
    check_weights(w);
    need("residual", st.R);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};

    gr_half(g, w.attn, w.attn.inject, st, sc, sc.inject);
    qsa_layer(g.qsa, w.qsa, st.qsa, sc.qsa, pos, pos_base, sc.mixed, sc.block_out);
    gr_write(gs, *st.R, sc.block_out, sc.inject, *st.R);

    gr_half(g, w.ffn, w.ffn.inject, st, sc, sc.inject_ffn);
    f32_to_bf16_bulk(sc.mixed, sc.x_bf16, g.n_embd);
    bf16_gemv(sc.x_bf16, *w.router, sc.router_logits, g.n_embd, g.n_expert);
    router_top10(sc.router_logits, 1, g.n_expert, g.k, sc.route_ids, sc.route_w);

    iq_quantize_q8_1(sc.mixed, g.n_embd, sc.x_q8_1);
    const int* ids = static_cast<const int*>(sc.route_ids.data());
    auto* parts = static_cast<float*>(sc.parts.data());
    for (int e = 0; e < g.k; ++e) {
        const int id = ids[e];
        if (id < 0 || id >= g.n_expert) throw MetalError("token: a routed expert id is outside the pool");
        iq_expert(w.experts[id], sc.x_q8_1, g.expert, 1, sc.one_expert);
        std::memcpy(parts + (size_t) e * (size_t) g.n_embd, sc.one_expert.data(), (size_t) g.n_embd * sizeof(float));
    }

    quantize_q8_0(sc.mixed, g.n_embd, sc.x_q8_0);
    shared_expert(sc.x_q8_0, sc.x_bf16, *w.sh_gate_codes, *w.sh_gate_scales, *w.sh_up_codes, *w.sh_up_scales,
                  *w.sh_down_codes, *w.sh_down_scales, *w.sh_gate_inp, sc.sh_gate, sc.sh_up, sc.sh_hq, sc.sh_g,
                  sc.shared, g.n_embd, g.n_ff_shared, g.shared_tpr);
    moe_combine(sc.parts, sc.route_w, &sc.shared, sc.block_out, (int) g.n_embd, g.k);
    gr_write(gs, *st.R, sc.block_out, sc.inject_ffn, *st.R);
}

void lm_head_logits(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc) {
    check_geom(g);
    need("residual", st.R);
    need("output norm", w.head.norm);
    need("output down", w.head.down);
    need("output up", w.head.up);
    need("output codes", w.out.codes);
    need("output scales", w.out.scales);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    gr_read(gs, *st.R, *w.head.norm, *w.head.down, *w.head.up, nullptr, kEps, false, sc.gr, sc.mixed, sc.inject);
    quantize_q8_k(sc.mixed, g.n_embd, sc.head_q8k);
    const Bytes* off = w.out.form.has_offset ? w.out.offset : nullptr;
    s_gemv_q8k_split(sc.head_q8k, *w.out.codes, *w.out.scales, off, sc.logits, g.n_embd, g.n_vocab, w.out.form);
}

int generate_token(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int token, int pos,
                   int pos_base, const SamplerParams& sp, const Bytes* history, int history_len) {
    embed_into_residual(g, w, st, sc, token);
    block_forward(g, w, st, sc, pos, pos_base);
    lm_head_logits(g, w, st, sc);
    sample_tokens(sc.logits, 1, g.n_vocab, history, history_len, sp, sc.sampled);
    return static_cast<const int*>(sc.sampled.data())[0];
}

int draft_token(const TokenGeom& g, const TokenWeights& w, const DraftWeights& d, const Bytes& residual,
                TokenState& st, TokenScratch& sc, int token, int pos, int pos_base, const SamplerParams& sp,
                const Bytes* history, int history_len) {
    check_geom(g);
    check_weights(w);
    mix_draft(g, w, d, residual, st, sc, token);
    block_forward(g, w, st, sc, pos, pos_base);
    lm_head_logits(g, w, st, sc);
    sample_tokens(sc.logits, 1, g.n_vocab, history, history_len, sp, sc.sampled);
    return static_cast<const int*>(sc.sampled.data())[0];
}

}  // namespace strata::metal
