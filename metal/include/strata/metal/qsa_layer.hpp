// metal/include/strata/metal/qsa_layer.hpp - one QSA attention block, the direct calls.
//
// The order is the fp16 path: indexer key, K and V, norm and rotate K, append K and V,
// the query, the first half normed and rotated, the indexer query, scores, top-k, paged
// attention, the gate, then the output projection. Q4, Q8, and hybrid KV are not this
// function. The residual, the router, and the experts are not this function.
#pragma once

#include "strata/metal/bytes.hpp"
#include "strata/metal/sgemv.hpp"

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::metal {

struct QsaLayerGeom {
    int64_t n_embd = 0;
    int64_t max_cells = 0;
    strata::kernels::QsaShapes qsa;
};

struct QsaPlane {
    SForm form;
    const Bytes* codes = nullptr;
    const Bytes* scales = nullptr;
    const Bytes* offset = nullptr;
};

struct QsaLayerWeights {
    QsaPlane k;
    QsaPlane v;
    QsaPlane q;
    QsaPlane out;
    const Bytes* idx_k = nullptr;       // bf16, idx_dim rows of n_embd
    const Bytes* idx_q = nullptr;       // bf16, idx_n_head * idx_dim rows of n_embd
    const Bytes* k_norm = nullptr;      // f32, head_dim
    const Bytes* q_norm = nullptr;      // f32, head_dim
    const Bytes* idx_k_norm = nullptr;  // f32, idx_dim
    const Bytes* idx_q_norm = nullptr;  // f32, idx_dim
    const Bytes* cos_tab = nullptr;
    const Bytes* sin_tab = nullptr;
};

// The pools and the indexer rows live for the sequence. The page table is not the identity.
struct QsaLayerState {
    Bytes* k_pool = nullptr;
    Bytes* v_pool = nullptr;
    const Bytes* page_table = nullptr;
    Bytes* idx_tail = nullptr;
    Bytes* idx_dead = nullptr;
    Bytes* idx_pooled = nullptr;
    Bytes* idx_block_pos = nullptr;
};

struct QsaLayerScratch {
    Bytes x_q8k;
    Bytes x_bf16;
    Bytes idx_raw;
    Bytes kcur;
    Bytes vcur;
    Bytes q_full;
    Bytes qcur;
    Bytes q_idx;
    Bytes cell_scores;
    Bytes ids;
    Bytes step;
    Bytes pos;
    Bytes attn_scratch;
    Bytes attn;
    Bytes gated;
    Bytes gated_q8k;
};

QsaLayerScratch make_qsa_layer_scratch(const QsaLayerGeom& g);

// One token at cell `pos`. Rope reads position pos_base + pos. The indexer cell index is `pos`.
void qsa_layer(const QsaLayerGeom& g, const QsaLayerWeights& w, QsaLayerState& st, QsaLayerScratch& sc, int pos,
               int pos_base, const Bytes& x, Bytes& out);

// One ggml block plane. `type` is the id iq_mmvq dispatches. The bytes are that block layout,
// not an S-form row. Q2_0, Q3_K, Q4_K, and Q5_K stay out of the expert wrapper.
struct QsaIqPlane {
    int type = 0;
    const Bytes* weights = nullptr;
};

// The same block on ggml rows. The activation is q8_1. The indexer projections stay bf16.
// Scratch is still make_qsa_layer_scratch, and its Q8_K slots stay unused.
struct QsaIqWeights {
    QsaIqPlane k;
    QsaIqPlane v;
    QsaIqPlane q;
    QsaIqPlane out;
    const Bytes* idx_k = nullptr;
    const Bytes* idx_q = nullptr;
    const Bytes* k_norm = nullptr;
    const Bytes* q_norm = nullptr;
    const Bytes* idx_k_norm = nullptr;
    const Bytes* idx_q_norm = nullptr;
    const Bytes* cos_tab = nullptr;
    const Bytes* sin_tab = nullptr;
};

void qsa_layer_iq(const QsaLayerGeom& g, const QsaIqWeights& w, QsaLayerState& st, QsaLayerScratch& sc, int pos,
                  int pos_base, const Bytes& x, Bytes& out);

}  // namespace strata::metal
