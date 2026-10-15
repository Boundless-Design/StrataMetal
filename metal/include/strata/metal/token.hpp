// metal/include/strata/metal/token.hpp - one token on the kernels that already pass.
//
// The embedding replaces every residual stream. One block is the attention gated residual,
// one QSA attention block, the write, the feed-forward gated residual, the router, the
// routed experts, the 2-bit shared expert, the combine, and the write. The output mix is a
// gated residual with no injection, then a Q8_K projection. The sampler draws the id.
//
// draft_token is a different call. It pairs one caller residual with the next token, builds
// the layer residual, and then runs that same block. It is one cell. It is not a loaded
// MTP runtime.
//
// This is one block. It is not the 48-layer model, and it does not start the Python server.
#pragma once

#include "strata/metal/gr.hpp"
#include "strata/metal/iq.hpp"
#include "strata/metal/qsa_layer.hpp"
#include "strata/metal/sampler.hpp"

#include <cstdint>

namespace strata::metal {

struct TokenGeom {
    int64_t n_embd = 0;
    int64_t hc = 0;
    int64_t hc_lr = 0;
    int n_expert = 0;
    int k = 0;
    int64_t n_ff_shared = 0;
    int shared_tpr = 32;
    int n_vocab = 0;
    int emb_code_bits = 4;
    int emb_code_bias = 0;
    int emb_group = 32;
    QsaLayerGeom qsa;
    IqExpert expert;
};

// One gated-residual half. inject is null on the output mix.
struct TokenGr {
    const Bytes* norm = nullptr;
    const Bytes* down = nullptr;
    const Bytes* up = nullptr;
    const Bytes* inject = nullptr;
};

struct TokenWeights {
    const Bytes* emb_codes = nullptr;
    const Bytes* emb_scales = nullptr;
    const Bytes* emb_offsets = nullptr;
    TokenGr attn;
    TokenGr ffn;
    TokenGr head;
    QsaLayerWeights qsa;
    const Bytes* router = nullptr;  // bf16, n_expert rows of n_embd
    const Bytes* experts = nullptr;  // n_expert blobs, each iq_expert_bytes
    const Bytes* sh_gate_codes = nullptr;
    const Bytes* sh_gate_scales = nullptr;
    const Bytes* sh_up_codes = nullptr;
    const Bytes* sh_up_scales = nullptr;
    const Bytes* sh_down_codes = nullptr;
    const Bytes* sh_down_scales = nullptr;
    const Bytes* sh_gate_inp = nullptr;  // bf16, n_embd
    QsaPlane out;
};

// One square projection. The activation is Q8_0. The norm is stored already as 1 + w.
// embedding.norm has n_embd values. hidden.norm has hc * n_embd values.
struct DraftFc {
    SForm form;
    const Bytes* codes = nullptr;
    const Bytes* scales = nullptr;
    const Bytes* offset = nullptr;
    const Bytes* norm = nullptr;
};

struct DraftWeights {
    DraftFc embedding;
    DraftFc hidden;
};

struct TokenState {
    Bytes* R = nullptr;  // hc * n_embd floats
    QsaLayerState qsa;
};

struct TokenScratch {
    Bytes emb;
    Bytes mixed;
    Bytes block_out;
    Bytes inject;
    Bytes inject_ffn;
    GrScratch gr;
    QsaLayerScratch qsa;
    Bytes router_logits;
    Bytes route_ids;
    Bytes route_w;
    Bytes x_q8_1;
    Bytes x_q8_0;
    Bytes x_bf16;
    Bytes parts;
    Bytes one_expert;
    Bytes shared;
    Bytes sh_gate;
    Bytes sh_up;
    Bytes sh_hq;
    Bytes sh_g;
    Bytes head_q8k;
    Bytes logits;
    Bytes sampled;
    Bytes draft_hidden;
    Bytes draft_row;
    Bytes draft_e;
    Bytes draft_h;
    Bytes draft_q8;
};

TokenScratch make_token_scratch(const TokenGeom& g);

// Copies the embedding row into every residual stream. The previous residual is replaced.
void embed_into_residual(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int token);

// One block at cell `pos`. Rope position is pos_base + pos. The indexer cell is `pos`.
void block_forward(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int pos,
                   int pos_base);

// Gated residual with a null injection, then the output projection. Logits stay in sc.logits.
void lm_head_logits(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc);

// Embed, one block, the output projection, one sample. Returns the id.
int generate_token(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int token, int pos,
                   int pos_base, const SamplerParams& sp, const Bytes* history, int history_len);

// One draft step. `residual` is the caller's final multi-stream residual at this cell and is
// not written. `token` is the token at the next position. Rope position is pos_base + pos.
// The layer's own K/V is st.qsa. The sampled id is the draft for the position after that.
//
// The embedding row is normalized and projected. The whole residual is normalized, then each
// stream is projected by the same hidden weight. The layer residual is that projection plus
// the embedding projection on every stream. One block and the output projection follow.
// The two projections are 4-bit or 8-bit S-form with a Q8_0 activation. The shared expert
// is the 2-bit function. This is one cell, not a loaded MTP runtime.
int draft_token(const TokenGeom& g, const TokenWeights& w, const DraftWeights& d, const Bytes& residual,
                TokenState& st, TokenScratch& sc, int token, int pos, int pos_base, const SamplerParams& sp,
                const Bytes* history, int history_len);

}  // namespace strata::metal
