// metal/include/strata/metal/forward.hpp - one token, all the way to the logit row.
//
// The same layer steps the walk checks. A position replaces the residual with the token embedding on
// every hyper-connection stream, runs each layer in order, and carries the GDN recurrence, the QSA
// cache, and the PLE convolution history. The routed experts and the n-gram table stay on disk: one
// token reads its ten experts and its sixteen table rows through the mapping.
#pragma once

#include "strata/metal/layer_exec.hpp"

#include "strata/core/layout.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::metal {

class Session;

// The GGUF, both shards, and the weights that are not sequence state. The experts and the n-gram
// table are not copied in. A Session keeps pointers into this object, so it has to stay where it was
// constructed.
class Model {
public:
    explicit Model(const std::string& any_shard);

    const ModelTensors& tensors() const { return tensors_; }
    const strata::core::ModelGeometry& geom() const { return geom_; }
    int n_vocab() const { return n_vocab_; }
    int n_layers() const { return (int) geom_.n_layers; }
    int ple_layer() const { return ple_layer_; }

private:
    friend class Session;

    struct Weights {
        Bytes key, value, out_w, out_norm, out_down, out_up;
    };

    ModelTensors tensors_;
    strata::core::ModelGeometry geom_{};
    int n_vocab_ = 0;
    int ple_layer_ = -1;
    int head_dim_ = 0;
    int n_heads_ = 0;
    int ngram_ = 0;
    int conv_k_ = 0;
    int hist_ = 0;
    int32_t eos_ = 0;
    int emb_type_ = -1;
    int key_type_ = -1;
    int value_type_ = -1;
    int out_type_ = -1;
    int table_type_ = -1;
    uint64_t emb_row_bytes_ = 0;
    uint64_t table_rows_ = 0;
    uint64_t table_row_bytes_ = 0;
    std::vector<uint64_t> mult_, vocab_hash_, offset_;
    LayerTensor emb_{};
    LayerTensor table_{};
    std::vector<float> norm_key_, norm_query_, norm_conv_, conv_;
    std::optional<Weights> weights_;
};

// Per-layer recurrence for `max_positions` cells, which is at least 512: p5's prompt plus 32 new
// tokens is 339 positions, and the rope table has one row per absolute position. `forward` writes
// n_vocab floats. Position 0 clears the recurrence, the cache, and the PLE history. Any other
// position has to be the one after the position this session just ran.
class Session {
public:
    explicit Session(Model& model, int max_positions = 512);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void forward(int32_t token, int pos, float* logits);
    int n_vocab() const { return model_->n_vocab(); }
    int max_positions() const { return max_pos_; }

private:
    struct Slot {
        bool qsa = false;
        std::optional<GdnExec> gdn;
        std::optional<QsaExec> qsa_exec;
        std::optional<FfnExec> ffn;
    };
    struct Scratch {
        Bytes cur, nxt;
        Bytes emb_f32, emb_bf16, emb_q8;
        Bytes key_y, value_y;
        Bytes mixed, inject, logits;
        GrScratch head;
    };

    void project(const float* x, int type, const Bytes& weight, int n_out, Bytes& y, float* host);
    void apply_ple(Bytes& residual, int32_t token);

    Model* model_ = nullptr;
    int max_pos_ = 0;
    int next_pos_ = 0;
    // Oldest first. Length is ngram_size - 1. -1 is a missing predecessor, which cuts the n-gram.
    std::vector<int32_t> ple_prev_;
    std::vector<float> history_;
    std::vector<Slot> layers_;
    std::optional<Scratch> scratch_;
};

}  // namespace strata::metal
