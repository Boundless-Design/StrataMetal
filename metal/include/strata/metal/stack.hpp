// metal/include/strata/metal/stack.hpp - several layers, one token, on the kernels that already pass.
//
// The pack rule is layer % qsa_interval == qsa_interval - 1. Those layers are the QSA attention
// block. The others are one GDN layer. Every layer then does the same feed-forward half as the
// one-block token: gated residual, router, routed experts, the 2-bit shared expert, combine, write.
// The embedding replaces every residual stream at the start of the token. The output mix is the
// token's gated residual with no injection, then the Q8_K projection, then the sampler.
//
// Recurrence across tokens is the GDN state and the QSA pools. The residual is replaced from the
// embedding on every token. This is the schedule, at whatever layer count the caller passes. It is
// not the 48-layer model, and it does not load a GGUF or start the draft model or the Python server.
#pragma once

#include "strata/metal/gdn_layer.hpp"
#include "strata/metal/token.hpp"

#include <cstdint>

namespace strata::metal {

struct StackGeom {
    int64_t n_layers = 0;
    int64_t qsa_interval = 4;
    TokenGeom token;
    GdnLayerGeom gdn;
};

// The pack rule. A non-positive interval is not full attention.
inline bool stack_is_qsa(const StackGeom& g, int64_t layer) {
    return g.qsa_interval > 0 && layer >= 0 && layer % g.qsa_interval == g.qsa_interval - 1;
}

// One layer. The GDN weights are read when the layer is not full attention. The QSA weights are
// read when it is. The other set may be null.
struct StackLayerWeights {
    TokenGr attn;
    TokenGr ffn;
    GdnLayerWeights gdn;
    QsaLayerWeights qsa;
    const Bytes* router = nullptr;
    const Bytes* experts = nullptr;
    const Bytes* sh_gate_codes = nullptr;
    const Bytes* sh_gate_scales = nullptr;
    const Bytes* sh_up_codes = nullptr;
    const Bytes* sh_up_scales = nullptr;
    const Bytes* sh_down_codes = nullptr;
    const Bytes* sh_down_scales = nullptr;
    const Bytes* sh_gate_inp = nullptr;
};

struct StackLayerState {
    GdnLayerState gdn;
    QsaLayerState qsa;
};

struct StackWeights {
    const Bytes* emb_codes = nullptr;
    const Bytes* emb_scales = nullptr;
    const Bytes* emb_offsets = nullptr;
    TokenGr head;
    QsaPlane out;
    const StackLayerWeights* layers = nullptr;  // n_layers
};

struct StackState {
    Bytes* R = nullptr;
    StackLayerState* layers = nullptr;  // n_layers
};

struct StackScratch {
    TokenScratch tok;
    GdnLayerScratch gdn;
};

StackScratch make_stack_scratch(const StackGeom& g);

// Embed, every layer, the output projection, one sample. Returns the id.
// pos is the cell index. Rope position is pos_base + pos. The same cell is used on every layer.
int generate_stack_token(const StackGeom& g, const StackWeights& w, StackState& st, StackScratch& sc, int token,
                         int pos, int pos_base, const SamplerParams& sp, const Bytes* history, int history_len);

}  // namespace strata::metal
