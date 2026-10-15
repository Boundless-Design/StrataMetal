// metal/include/strata/metal/layer_exec.hpp - one layer step, shared by the walk and the forward.
//
// The walk's checks and the engine's forward pass call these. A GDN step carries the conv and the
// recurrent state. A QSA step carries the fp16 KV pages and the indexer rows. The feed-forward step
// has no sequence state: each of the ten selected experts is read from the mapping for that token.
//
// `reset` true is the walk: position 0, empty cache, state and pages zeroed before the kernels.
// `reset` false leaves that state where the previous position put it.
// Weights are found by name. The quant type is the one the file stored on the tensor.
#pragma once

#include "strata/metal/gdn_layer.hpp"
#include "strata/metal/gr.hpp"
#include "strata/metal/layer_tensors.hpp"
#include "strata/metal/qsa_layer.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {
struct ModelGeometry;
}

namespace strata::metal {

// One expert's rows inside the three pools, or all-zero counts when the planes are absent or the
// byte count is not n_expert of those rows. A type with no Metal product yields a zero count.
struct ExpertSpans {
    const LayerTensor* gate = nullptr;
    const LayerTensor* up = nullptr;
    const LayerTensor* down = nullptr;
    uint64_t gate_n = 0, up_n = 0, down_n = 0;
};

ExpertSpans expert_spans(const ModelTensors& model, int layer, int64_t n_embd, int64_t n_ff, int64_t n_expert);

// The attention half of a GDN layer. `run` reads `residual`, writes `y`, and updates `state` and `conv`.
struct GdnExec {
    int qkv_ty = -1, gate_ty = -1, ssm_ty = -1;
    int n_embd = 0;
    int hc = 0;
    const Bytes* qkv_override = nullptr;
    Bytes qkv, gate, ssm;
    Bytes conv_w, alpha_w, beta_w, ssm_a, dt_w, ssm_norm;
    Bytes hc_down, hc_up, hc_inj, hc_norm;
    Bytes state, conv, mixed, inject, block;
    GrShapes shapes{};
    GdnLayerGeom geom{};
    GrScratch gr;
    GdnLayerScratch scratch;

    static GdnExec load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& geom);
    void set_qkv(const Bytes* override_or_null);
    // `y` is filled with 7 first, then the residual write. A guard float past hc * n_embd stays 7.
    void run(const Bytes& residual, Bytes& y, bool reset_state);
};

// The attention half of a QSA layer. `max_cells` is how far `block_at` may be called. `rope_rows` is
// the rope table, one row per absolute position the table must hold.
struct QsaExec {
    int q_ty = -1, k_ty = -1, v_ty = -1, o_ty = -1;
    int n_embd = 0;
    int hc = 0;
    int hc_lr = 0;
    int max_cells = 0;
    const Bytes* v_override = nullptr;
    Bytes q, k, v, o;
    Bytes q_norm, k_norm, idx_q_norm, idx_k_norm, idx_q, idx_k;
    Bytes hc_down, hc_up, hc_inj, hc_norm;
    Bytes cos_tab, sin_tab;
    Bytes page_table, k_pool, v_pool;
    Bytes idx_tail, idx_dead, idx_pooled, idx_bpos;
    Bytes mixed, inject, block;
    GrShapes shapes{};
    QsaLayerGeom geom{};
    GrScratch gr;
    QsaLayerScratch scratch;

    static QsaExec load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& geom, int max_cells,
                        int rope_rows);
    void set_v(const Bytes* override_or_null);
    // The hyper-connection read of this residual. The cache is zeroed only when `reset_cache` is set.
    // A later `block_at` reuses this mixed vector and does not read the residual again.
    void begin(const Bytes& residual, bool reset_cache);
    // One cell. `out` is filled with 7 first. Rope reads `pos_base + pos`. The indexer cell is `pos`.
    void block_at(int pos, int pos_base, Bytes& out);
    // `begin`, position 0 into `block`, then the residual write into `y`. The walk's attention write.
    void attend_write(const Bytes& residual, Bytes& y, bool reset_cache);
};

// What one feed-forward pass produced, for the host check. The kernels do not read it back.
struct FfnTrace {
    std::vector<float> mixed, inject, residual, shared, combined, layer;
    std::vector<float> sgate, sup, sdown, route;
    std::vector<uint8_t> exq, sh_xq, sh_hq;
    float scale = 0;
    std::vector<int> ids;
    std::vector<float> route_w;
    std::vector<std::vector<float>> egate, eup, edown;
    std::vector<std::vector<uint8_t>> ehq, blob;
};

// The feed-forward half. `with_shared` false does not load the shared planes. `run` still evaluates
// the shared expert when the planes were loaded, and adds it only when `add_shared` is set.
struct FfnExec {
    const ModelTensors* model = nullptr;
    int n_embd = 0;
    int n_ff = 0;
    int n_expert = 0;
    int hc = 0;
    int hc_lr = 0;
    int k = 10;
    bool with_shared = false;
    int eg_ty = -1, eu_ty = -1, ed_ty = -1;
    int sg_ty = -1, su_ty = -1, sd_ty = -1;
    uint64_t gate_n = 0, up_n = 0, down_n = 0;
    LayerTensor gate, up, down;
    Bytes hc_down, hc_up, hc_inj, hc_norm;
    Bytes router, ginp, sgate, sup, sdown;
    GrShapes shapes{};
    GrScratch gr;
    Bytes expert_y, shared_y, combined, mixed, inject;
    Bytes x_bf16, route, exq, egate_y, eup_y, ehq;
    Bytes sh_xq, sh_gate_y, sh_up_y, sh_hq, sh_down_y, sh_scalar;
    Bytes parts, ids, route_w;

    static FfnExec load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& geom, bool with_shared);
    // `out` is the residual stream, filled with 7 first. `trace` may be null.
    void run(const Bytes& residual, Bytes& out, bool add_shared, FfnTrace* trace);
};

// The output mixer: hyper-connection read with no inject weight. Both buffers are filled with 7 first,
// and a null inject weight leaves `inject` at 7.
void output_mix(const GrShapes& shapes, GrScratch& scratch, const Bytes& residual, const Bytes& norm,
                const Bytes& down, const Bytes& up, Bytes& mixed, Bytes& inject);

// The output rows. Quantizes the first `n_embd` floats of `mixed`, fills `logits` with 7, then the
// matrix-vector product of `ggml_type`. Returns the q8_1 activation.
Bytes output_project(int ggml_type, const Bytes& weight, const Bytes& mixed, int n_embd, int n_vocab, Bytes& logits);

}  // namespace strata::metal
