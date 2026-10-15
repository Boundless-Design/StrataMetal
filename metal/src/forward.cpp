// One token's forward pass. The layer bodies are GdnExec, QsaExec, and FfnExec. This file is the wiring
// those steps do not own: the embedding broadcast, the layer-1 PLE, the residual handed from one layer
// to the next, and the output mixer plus the head.
#include "strata/metal/forward.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/iq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::metal {
namespace {

constexpr int kF32 = 0;
constexpr int kF16 = 1;
constexpr int kBf16 = 30;
constexpr float kRmsEps = 1e-6f;

std::string shape_text(const LayerTensor& t) {
    std::string s = t.name + " " + t.type_name() + " [";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(t.shape[i]);
    }
    return s + "]";
}

void require_shape(const LayerTensor& t, std::initializer_list<uint64_t> want) {
    if (t.shape.size() != want.size() || !std::equal(want.begin(), want.end(), t.shape.begin())) {
        std::string expected = "[";
        bool first = true;
        for (uint64_t d : want) {
            if (!first) expected += ',';
            first = false;
            expected += std::to_string(d);
        }
        throw MetalError(shape_text(t) + " does not match " + expected + "]");
    }
}

uint64_t row_bytes_of(int type, int64_t n, const char* what) {
    int elems = 0, bytes = 0;
    if (!strata::block_geometry((uint32_t) type, elems, bytes) || elems <= 0 || n <= 0 || n % elems)
        throw MetalError(std::string(what) + " has no row layout for " + strata::ggml_type_name((uint32_t) type));
    return (uint64_t) (n / elems) * (uint64_t) bytes;
}

void dequant_row(int type, const uint8_t* src, int n, float* dst) {
    int elems = 0, bytes = 0;
    if (!strata::block_geometry((uint32_t) type, elems, bytes) || elems <= 0 || n <= 0 || n % elems)
        throw MetalError(std::string("cannot dequant ") + strata::ggml_type_name((uint32_t) type));
    const int blocks = n / elems;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = src + (size_t) b * (size_t) bytes;
        float* o = dst + (size_t) b * (size_t) elems;
        switch (type) {
        case kF32: strata::dequantize_f32(p, o, elems); break;
        case kF16: strata::dequantize_f16(p, o, elems); break;
        case kBf16: strata::dequantize_bf16(p, o, elems); break;
        case kQ4_0: strata::dequantize_q4_0(p, o); break;
        case kQ5_0: strata::dequantize_q5_0(p, o); break;
        case kQ8_0: strata::dequantize_q8_0(p, o); break;
        case kQ3_K: strata::dequantize_q3_K(p, o); break;
        case kQ4_K: strata::dequantize_q4_K(p, o); break;
        case kQ5_K: strata::dequantize_q5_K(p, o); break;
        case kQ6_K: strata::dequantize_q6_K(p, o); break;
        case kIq4Nl: strata::dequantize_iq4_nl(p, o); break;
        case kIq4Xs: strata::dequantize_iq4_xs(p, o); break;
        case kQ2_0: strata::dequantize_q2_0(p, o); break;
        default: throw MetalError(std::string("no dequantizer for ") + strata::ggml_type_name((uint32_t) type));
        }
    }
}

std::vector<float> load_float_plane(const ModelTensors& model, const LayerTensor& t, int n) {
    const auto raw = model.read(t);
    std::vector<float> out((size_t) n);
    if (t.type == kF32) {
        if (raw.size() != (size_t) n * 4u) throw MetalError(t.name + " is not " + std::to_string(n) + " f32 values");
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }
    if ((int) raw.size() != (int) row_bytes_of(t.type, n, t.name.c_str()))
        throw MetalError(shape_text(t) + " does not cover " + std::to_string(n) + " values");
    dequant_row(t.type, raw.data(), n, out.data());
    return out;
}

int64_t meta_int(const ModelTensors& model, const char* key) {
    const strata::MetaValue* v = model.meta(key);
    if (!v) throw MetalError(std::string(key) + " is missing");
    switch (v->type) {
    case strata::MetaType::U8:
    case strata::MetaType::I8:
    case strata::MetaType::U16:
    case strata::MetaType::I16:
    case strata::MetaType::U32:
    case strata::MetaType::I32:
    case strata::MetaType::U64:
    case strata::MetaType::I64: return (int64_t) v->u;
    default: throw MetalError(std::string(key) + " is not an integer");
    }
}

std::vector<uint64_t> meta_u64s(const ModelTensors& model, const char* key, size_t n) {
    const strata::MetaValue* v = model.meta(key);
    if (!v || v->type != strata::MetaType::ARRAY || v->count != n || v->items.size() != n)
        throw MetalError(std::string(key) + " is not " + std::to_string(n) + " integers");
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        const strata::MetaValue& e = v->items[i];
        if (!e.is_num() || e.type == strata::MetaType::F32 || e.type == strata::MetaType::F64)
            throw MetalError(std::string(key) + " entry is not an integer");
        out[i] = e.u;
    }
    return out;
}

Bytes load_weight(const ModelTensors& model, const LayerTensor& t) {
    const auto raw = model.read(t);
    Bytes b((uint64_t) raw.size());
    if (!raw.empty()) std::memcpy(b.data(), raw.data(), raw.size());
    return b;
}

// Grouped RMS. One scale per stream of n_embd, then the per-element gamma. Sum of squares is a multiply
// followed by an add: a contracted FMA is a different rounding than the CUDA postops.
#pragma clang fp contract(off)
void grouped_rms(const float* x, const float* gamma, float* y, int n_embd, int hc) {
    for (int c = 0; c < hc; ++c) {
        const float* row = x + (size_t) c * (size_t) n_embd;
        const float* g = gamma + (size_t) c * (size_t) n_embd;
        float* o = y + (size_t) c * (size_t) n_embd;
        float sumsq = 0.f;
        for (int d = 0; d < n_embd; ++d) {
            const float v = row[d];
            const float sq = v * v;
            sumsq = sumsq + sq;
        }
        const float scale = 1.f / std::sqrt(sumsq / (float) n_embd + kRmsEps);
        for (int d = 0; d < n_embd; ++d) {
            const float scaled = scale * row[d];
            o[d] = scaled * g[d];
        }
    }
}

// The convolution reads the normalized row, not the gated row. History is channel-major and row-fastest:
// index c * hist + row, oldest at row 0. Taps sit `dilation` rows apart, and the current row is the last tap.
#pragma clang fp contract(off)
void ple_postops(float* hidden, const float* key_proj, const float* value, const float* history, const float* norm_key,
                 const float* norm_query, const float* norm_conv, const float* conv_w, float* normalized, int n_embd,
                 int hc, int conv_k, int hist, int dilation) {
    const int hd = hc * n_embd;
    std::vector<float> key((size_t) hd), query((size_t) hd), gated((size_t) hd);
    std::vector<float> gate((size_t) hc);
    grouped_rms(key_proj, norm_key, key.data(), n_embd, hc);
    grouped_rms(hidden, norm_query, query.data(), n_embd, hc);
    const float dot_scale = 1.f / std::sqrt((float) n_embd);
    for (int c = 0; c < hc; ++c) {
        float sum = 0.f;
        const float* ks = key.data() + (size_t) c * (size_t) n_embd;
        const float* qs = query.data() + (size_t) c * (size_t) n_embd;
        for (int d = 0; d < n_embd; ++d) {
            const float p = ks[d] * qs[d];
            sum = sum + p;
        }
        const float s = dot_scale * sum;
        const float mag = std::sqrt(std::max(std::fabs(s), kRmsEps));
        const float sign = (float) ((s > 0.f) - (s < 0.f));
        const float x = sign * mag;
        gate[(size_t) c] = 1.f / (1.f + std::exp(-x));
    }
    for (int c = 0; c < hc; ++c) {
        const float g = gate[(size_t) c];
        float* row = gated.data() + (size_t) c * (size_t) n_embd;
        for (int d = 0; d < n_embd; ++d) {
            const float p = value[d] * g;
            row[d] = p;
        }
    }
    grouped_rms(gated.data(), norm_conv, normalized, n_embd, hc);
    for (int c = 0; c < hd; ++c) {
        float sum = 0.f;
        for (int k = 0; k < conv_k; ++k) {
            const float tap = (k == conv_k - 1) ? normalized[c] : history[(size_t) c * (size_t) hist + (size_t) dilation * (size_t) k];
            const float w = conv_w[(size_t) c * (size_t) conv_k + (size_t) k];
            const float term = tap * w;
            sum = (k == 0) ? term : sum + term;
        }
        const float activation = sum / (1.f + std::exp(-sum));
        const float with_gate = gated[(size_t) c] + activation;
        hidden[c] = hidden[c] + with_gate;
    }
}
#pragma clang fp contract(on)

void history_advance(float* history, const float* normalized, int hist, int channels) {
    for (int c = 0; c < channels; ++c) {
        float* row = history + (size_t) c * (size_t) hist;
        for (int r = 0; r + 1 < hist; ++r) row[r] = row[r + 1];
        row[hist - 1] = normalized[c];
    }
}

void hash_rows(int ngram, int n_heads, int32_t eos, const uint64_t* mult, const uint64_t* vocab, const uint64_t* offset,
               uint64_t table_rows, int32_t token, const int32_t* prev, uint32_t* out) {
    const int n_prev = ngram - 1;
    const int heads_per = n_heads / n_prev;
    std::vector<int64_t> ctx((size_t) ngram);
    ctx[0] = token;
    bool cut = false;
    for (int s = 1; s < ngram; ++s) {
        const int32_t t = cut ? (int32_t) -1 : prev[n_prev - s];
        cut = cut || t < 0 || t == eos;
        ctx[(size_t) s] = cut ? (int64_t) eos : (int64_t) t;
    }
    for (int n = 2; n <= ngram; ++n) {
        uint64_t mixed = (uint64_t) ctx[0] * mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[(size_t) j] * mult[j];
        const int base = (n - 2) * heads_per;
        for (int g = 0; g < heads_per; ++g) {
            const int h = base + g;
            const uint64_t row = mixed % vocab[h] + offset[h];
            if (row >= table_rows) throw MetalError("a PLE row is outside the table");
            out[h] = (uint32_t) row;
        }
    }
}

}  // namespace

Model::Model(const std::string& any_shard) : tensors_(any_shard) {
    if (meta_int(tensors_, "qwen4exp.embedding_length") != geom_.n_embd)
        throw MetalError("qwen4exp.embedding_length does not match the geometry");
    if (tensors_.n_layers() != (int) geom_.n_layers)
        throw MetalError("qwen4exp.block_count does not match the geometry");
    if (!tensors_.shard_present(1)) throw MetalError("shard 2 is not on disk; the PLE table lives there");

    emb_ = tensors_.global("token_embd.weight");
    const LayerTensor& out = tensors_.global("output.weight");
    if (emb_.shape.size() != 2 || emb_.shape[0] != (uint64_t) geom_.n_embd)
        throw MetalError(shape_text(emb_) + " is not [n_embd, n_vocab]");
    if (out.shape.size() != 2 || out.shape[0] != (uint64_t) geom_.n_embd || out.shape[1] != emb_.shape[1])
        throw MetalError(shape_text(out) + " does not match the embedding vocab");
    n_vocab_ = (int) emb_.shape[1];
    if (n_vocab_ <= 0 || (uint64_t) n_vocab_ != emb_.shape[1]) throw MetalError("n_vocab does not fit");
    emb_type_ = emb_.type;
    out_type_ = out.type;
    emb_row_bytes_ = row_bytes_of(emb_type_, geom_.n_embd, "token_embd.weight");
    if (emb_.bytes != emb_row_bytes_ * (uint64_t) n_vocab_) throw MetalError("token_embd.weight has a short payload");
    const uint64_t out_row = row_bytes_of(out_type_, geom_.n_embd, "output.weight");
    if (out.bytes != out_row * (uint64_t) n_vocab_) throw MetalError("output.weight has a short payload");

    const uint64_t hc_dim = (uint64_t) geom_.hc_dim();
    const LayerTensor& on = tensors_.global("output_hc_norm.weight");
    const LayerTensor& od = tensors_.global("output_hc_down.weight");
    const LayerTensor& ou = tensors_.global("output_hc_up.weight");
    require_shape(on, {hc_dim});
    require_shape(od, {hc_dim, (uint64_t) geom_.hc_lr});
    require_shape(ou, {(uint64_t) geom_.hc_lr, hc_dim});

    const auto layers = meta_u64s(tensors_, "qwen4exp.ple.layers", 1);
    ple_layer_ = (int) layers[0];
    if (ple_layer_ < 0 || ple_layer_ >= (int) geom_.n_layers) throw MetalError("qwen4exp.ple.layers is outside the model");
    ngram_ = (int) meta_int(tensors_, "qwen4exp.ple.ngram_size");
    const int heads_per = (int) meta_int(tensors_, "qwen4exp.ple.heads_per_ngram");
    conv_k_ = (int) meta_int(tensors_, "qwen4exp.ple.conv_kernel");
    eos_ = (int32_t) meta_int(tensors_, "qwen4exp.ple.eos_token_id");
    head_dim_ = (int) meta_int(tensors_, "qwen4exp.embedding_length_per_layer_input");
    if (ngram_ < 2 || heads_per < 1 || conv_k_ < 1 || head_dim_ < 1 || geom_.n_embd % head_dim_ != 0)
        throw MetalError("the PLE metadata does not describe a table that covers n_embd");
    n_heads_ = (int) (geom_.n_embd / head_dim_);
    if (n_heads_ != heads_per * (ngram_ - 1)) throw MetalError("PLE head count does not cover n_embd");
    hist_ = (conv_k_ - 1) * ngram_;
    if (hist_ < conv_k_) throw MetalError("the PLE history is shorter than the convolution");

    mult_ = meta_u64s(tensors_, "qwen4exp.ple.layer_multipliers", (size_t) ngram_);
    vocab_hash_ = meta_u64s(tensors_, "qwen4exp.ple.head_vocab_sizes", (size_t) n_heads_);
    offset_ = meta_u64s(tensors_, "qwen4exp.ple.head_offsets", (size_t) n_heads_);
    for (int h = 0; h < n_heads_; ++h)
        if (vocab_hash_[(size_t) h] == 0) throw MetalError("a PLE head vocab size is 0");

    table_ = tensors_.global("per_layer_token_embd.weight");
    if (table_.shape.size() != 2 || table_.shape[0] != (uint64_t) head_dim_)
        throw MetalError(shape_text(table_) + " is not [head_dim, rows]");
    table_rows_ = table_.shape[1];
    table_type_ = table_.type;
    table_row_bytes_ = row_bytes_of(table_type_, head_dim_, "per_layer_token_embd.weight");
    if (table_.bytes != table_row_bytes_ * table_rows_) throw MetalError("the PLE table payload does not match its shape");

    const LayerTensor& key = tensors_.at(ple_layer_, "ple_key.weight");
    const LayerTensor& value = tensors_.at(ple_layer_, "ple_value.weight");
    const LayerTensor& nk = tensors_.at(ple_layer_, "ple_norm_key.weight");
    const LayerTensor& nq = tensors_.at(ple_layer_, "ple_norm_query.weight");
    const LayerTensor& nc = tensors_.at(ple_layer_, "ple_norm_conv.weight");
    const LayerTensor& cw = tensors_.at(ple_layer_, "ple_conv1d.weight");
    require_shape(key, {(uint64_t) geom_.n_embd, hc_dim});
    require_shape(value, {(uint64_t) geom_.n_embd, (uint64_t) geom_.n_embd});
    require_shape(nk, {hc_dim});
    require_shape(nq, {hc_dim});
    require_shape(nc, {hc_dim});
    require_shape(cw, {(uint64_t) conv_k_, hc_dim});
    key_type_ = key.type;
    value_type_ = value.type;
    if (key_type_ != kBf16) (void) row_bytes_of(key_type_, geom_.n_embd, "ple_key.weight");
    if (value_type_ != kBf16) (void) row_bytes_of(value_type_, geom_.n_embd, "ple_value.weight");

    norm_key_ = load_float_plane(tensors_, nk, (int) hc_dim);
    norm_query_ = load_float_plane(tensors_, nq, (int) hc_dim);
    norm_conv_ = load_float_plane(tensors_, nc, (int) hc_dim);
    conv_ = load_float_plane(tensors_, cw, conv_k_ * (int) hc_dim);

    std::fprintf(stderr, "%s\n", shape_text(emb_).c_str());
    std::fprintf(stderr, "%s\n", shape_text(out).c_str());
    std::fprintf(stderr, "%s\n", shape_text(table_).c_str());
    std::fprintf(stderr, "%s\n", shape_text(key).c_str());
    std::fprintf(stderr, "%s\n", shape_text(value).c_str());
    std::fprintf(stderr, "PLE layer %d  heads %d x %d  history %d  eos %d  vocab %d\n", ple_layer_, n_heads_, head_dim_,
                 hist_, eos_, n_vocab_);

    weights_ = Weights{.key = load_weight(tensors_, key),
                       .value = load_weight(tensors_, value),
                       .out_w = load_weight(tensors_, out),
                       .out_norm = load_weight(tensors_, on),
                       .out_down = load_weight(tensors_, od),
                       .out_up = load_weight(tensors_, ou)};
}

Session::Session(Model& model, int max_positions) : model_(&model), max_pos_(max_positions) {
    // 512 covers the longest fixture: 307 prompt tokens plus 32 generated positions is 339.
    if (max_positions < 512) throw MetalError("the session needs at least 512 positions");
    const strata::core::ModelGeometry& g = model.geom();
    const int n_layers = (int) g.n_layers;
    layers_.resize((size_t) n_layers);
    for (int L = 0; L < n_layers; ++L) {
        const bool qsa = model.tensors().kind(L) == LayerKind::Qsa;
        if (qsa != strata::core::is_qsa_layer(g, L))
            throw MetalError("layer " + std::to_string(L) + " does not match the QSA schedule");
        std::fprintf(stderr, "layer %d/%d %s\n", L, n_layers, qsa ? "QSA" : "GDN");
        std::fflush(stderr);
        layers_[(size_t) L].qsa = qsa;
        if (qsa)
            layers_[(size_t) L].qsa_exec.emplace(QsaExec::load(model.tensors(), L, g, max_positions, max_positions));
        else
            layers_[(size_t) L].gdn.emplace(GdnExec::load(model.tensors(), L, g));
        layers_[(size_t) L].ffn.emplace(FfnExec::load(model.tensors(), L, g, true));
    }
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int64_t hc_dim = g.hc_dim();
    const GrShapes shapes{g.n_embd, g.hc, g.hc_lr};
    scratch_ = Scratch{.cur = Bytes((uint64_t) hc_dim * 4u),
                       .nxt = Bytes((uint64_t) hc_dim * 4u),
                       .emb_f32 = Bytes((uint64_t) n_embd * 4u),
                       .emb_bf16 = Bytes((uint64_t) n_embd * 2u),
                       .emb_q8 = Bytes(iq_q8_1_bytes(n_embd)),
                       .key_y = Bytes((uint64_t) hc_dim * 4u),
                       .value_y = Bytes((uint64_t) n_embd * 4u),
                       .mixed = Bytes((uint64_t) n_embd * 4u),
                       .inject = Bytes((uint64_t) hc * 4u),
                       .logits = Bytes((uint64_t) model.n_vocab() * 4u),
                       .head = make_gr_scratch(shapes)};
    ple_prev_.assign((size_t) model.ngram_ - 1, -1);
    history_.assign((size_t) model.hist_ * (size_t) hc_dim, 0.f);

    uint64_t layer_bytes = 0;
    for (const Slot& layer : layers_) {
        if (layer.qsa) layer_bytes += layer.qsa_exec->gpu_bytes();
        else layer_bytes += layer.gdn->gpu_bytes();
        layer_bytes += layer.ffn->gpu_bytes();
    }
    const Model::Weights& w = *model.weights_;
    const uint64_t head_bytes = w.out_w.size() + w.out_norm.size() + w.out_down.size() + w.out_up.size();
    const uint64_t ple_bytes = w.key.size() + w.value.size();
    const Scratch& sc = *scratch_;
    const uint64_t scratch_bytes = sc.cur.size() + sc.nxt.size() + sc.emb_f32.size() + sc.emb_bf16.size() +
                                   sc.emb_q8.size() + sc.key_y.size() + sc.value_y.size() + sc.mixed.size() +
                                   sc.inject.size() + sc.logits.size() + sc.head.xn.size() + sc.head.xq.size() +
                                   sc.head.proj.size() + sc.head.lq.size() + sc.head.gate.size();
    const uint64_t resident = layer_bytes + head_bytes;
    const uint64_t working_set = 28ull << 30;
    auto gib = [](uint64_t n) { return (double) n / (1024.0 * 1024.0 * 1024.0); };
    std::fprintf(stderr, "resident GPU %.3f GiB of 28 GiB (48 layer execs %.3f GiB, head %.3f GiB)\n", gib(resident),
                 gib(layer_bytes), gib(head_bytes));
    std::fprintf(stderr, "also resident: PLE projections %.3f GiB, session scratch %.3f GiB; experts and the PLE table stay on disk\n",
                 gib(ple_bytes), gib(scratch_bytes));
    if (resident + ple_bytes + scratch_bytes >= working_set)
        throw MetalError("resident GPU buffers reach the 28 GiB working set");
}

void Session::project(const float* x, int type, const Bytes& weight, int n_out, Bytes& y, float* host) {
    const int n_embd = (int) model_->geom_.n_embd;
    std::memcpy(scratch_->emb_f32.data(), x, (size_t) n_embd * 4u);
    if (type == kBf16) {
        f32_to_bf16_bulk(scratch_->emb_f32, scratch_->emb_bf16, n_embd);
        bf16_gemv(scratch_->emb_bf16, weight, y, n_embd, n_out);
    } else {
        iq_quantize_q8_1(scratch_->emb_f32, n_embd, scratch_->emb_q8);
        iq_mmvq(type, weight, 0, scratch_->emb_q8, n_embd, n_out, 1, y);
    }
    std::memcpy(host, y.data(), (size_t) n_out * 4u);
}

void Session::apply_ple(Bytes& residual, int32_t token) {
    const int n_embd = (int) model_->geom_.n_embd;
    const int hc = (int) model_->geom_.hc;
    const int hd = hc * n_embd;
    std::vector<float> emb((size_t) n_embd);
    std::vector<uint32_t> rows((size_t) model_->n_heads_);
    hash_rows(model_->ngram_, model_->n_heads_, model_->eos_, model_->mult_.data(), model_->vocab_hash_.data(),
              model_->offset_.data(), model_->table_rows_, token, ple_prev_.data(), rows.data());
    for (int h = 0; h < model_->n_heads_; ++h) {
        const uint64_t off = (uint64_t) rows[(size_t) h] * model_->table_row_bytes_;
        const auto raw = model_->tensors_.read(model_->table_, off, model_->table_row_bytes_);
        dequant_row(model_->table_type_, raw.data(), model_->head_dim_, emb.data() + (size_t) h * (size_t) model_->head_dim_);
    }
    std::vector<float> key((size_t) hd), value((size_t) n_embd), hidden((size_t) hd), normalized((size_t) hd);
    project(emb.data(), model_->key_type_, model_->weights_->key, hd, scratch_->key_y, key.data());
    project(emb.data(), model_->value_type_, model_->weights_->value, n_embd, scratch_->value_y, value.data());
    std::memcpy(hidden.data(), residual.data(), (size_t) hd * 4u);
    ple_postops(hidden.data(), key.data(), value.data(), history_.data(), model_->norm_key_.data(),
                model_->norm_query_.data(), model_->norm_conv_.data(), model_->conv_.data(), normalized.data(), n_embd, hc,
                model_->conv_k_, model_->hist_, model_->ngram_);
    std::memcpy(residual.data(), hidden.data(), (size_t) hd * 4u);
    history_advance(history_.data(), normalized.data(), model_->hist_, hd);
}

namespace {

// Diagnostic only. Off unless STRATA_LOG_MARGINS names a file. The ranking matches the router:
// softmax probability, strict greater-than, so a tie keeps the lower expert index. The margin is
// the 10th probability minus the 11th, before those ten weights are divided by their sum.
struct RtDS {
    float hi;
    float lo;
};

RtDS rt_add(RtDS a, float b) {
    const float s = a.hi + b;
    const float z = s - a.hi;
    const float e = (a.hi - (s - z)) + (b - z) + a.lo;
    const float z2 = s + e;
    return RtDS{z2, e - (z2 - s)};
}

void log_router_margin(int pos, int layer, const float* logits, int n_expert, int k) {
    static FILE* out = nullptr;
    static bool ready = false;
    if (!ready) {
        ready = true;
        const char* path = std::getenv("STRATA_LOG_MARGINS");
        if (path && path[0]) {
            out = std::fopen(path, "w");
            if (!out)
                std::fprintf(stderr, "router margins: cannot write %s\n", path);
            else
                std::fprintf(out, "pos\tlayer\tmargin\tlogit_gap\tid10\tid11\tp10\tp11\n");
        }
    }
    if (!out || k < 1 || n_expert < k + 1) return;
    float mx = -INFINITY;
    for (int e = 0; e < n_expert; ++e) mx = std::max(mx, logits[e]);
    std::vector<float> prob((size_t) n_expert);
    {
        std::vector<float> ex((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) ex[(size_t) e] = expf(logits[e] - mx);
        RtDS sum{0.f, 0.f};
        for (int e = 0; e < n_expert; ++e) sum = rt_add(sum, ex[(size_t) e]);
        float inv = 1.f / sum.hi;
        const float corr = sum.lo * inv;
        inv = inv - corr * inv;
        for (int e = 0; e < n_expert; ++e) prob[(size_t) e] = ex[(size_t) e] * inv;
    }
    std::vector<char> taken((size_t) n_expert, 0);
    int id10 = -1, id11 = -1;
    float p10 = 0.f, p11 = 0.f;
    for (int i = 0; i < k + 1; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
        for (int e = 0; e < n_expert; ++e) {
            if (taken[(size_t) e]) continue;
            if (prob[(size_t) e] > bv) {
                bv = prob[(size_t) e];
                bi = e;
            }
        }
        taken[(size_t) bi] = 1;
        if (i == k - 1) {
            id10 = bi;
            p10 = bv;
        }
        if (i == k) {
            id11 = bi;
            p11 = bv;
        }
    }
    const float margin = p10 - p11;
    const float logit_gap = logits[id10] - logits[id11];
    std::fprintf(out, "%d\t%d\t%.9g\t%.9g\t%d\t%d\t%.9g\t%.9g\n", pos, layer, margin, logit_gap, id10, id11, p10,
                 p11);
    std::fflush(out);
}

bool perturb_ulp() {
    static const bool on = [] {
        const char* p = std::getenv("STRATA_PERTURB_ULP");
        return p && p[0] == '1' && p[1] == '\0';
    }();
    return on;
}

}  // namespace

void Session::forward(int32_t token, int pos, float* logits) {
    if (!logits) throw MetalError("forward needs a logit row");
    const int n_vocab = model_->n_vocab_;
    const int n_embd = (int) model_->geom_.n_embd;
    const int hc = (int) model_->geom_.hc;
    if (token < 0 || token >= n_vocab) throw MetalError("token " + std::to_string(token) + " is outside the vocab");
    if (pos < 0 || pos >= max_pos_) throw MetalError("position " + std::to_string(pos) + " is outside the session");
    if (pos == 0) {
        std::fill(ple_prev_.begin(), ple_prev_.end(), -1);
        std::fill(history_.begin(), history_.end(), 0.f);
        next_pos_ = 0;
    }
    if (pos != next_pos_) throw MetalError("position " + std::to_string(pos) + " is not the next position");

    const uint64_t row_off = (uint64_t) token * model_->emb_row_bytes_;
    const auto row = model_->tensors_.read(model_->emb_, row_off, model_->emb_row_bytes_);
    std::vector<float> emb((size_t) n_embd);
    dequant_row(model_->emb_type_, row.data(), n_embd, emb.data());
    auto* cur = static_cast<float*>(scratch_->cur.data());
    for (int c = 0; c < hc; ++c) std::memcpy(cur + (size_t) c * (size_t) n_embd, emb.data(), (size_t) n_embd * 4u);
    if (perturb_ulp()) {
        const int n = hc * n_embd;
        auto* bits = reinterpret_cast<uint32_t*>(cur);
        for (int i = 0; i < n; ++i) bits[i] ^= 1u;
        if (pos == 0)
            std::fprintf(stderr, "perturb: flipped the low mantissa bit of %d layer-0 residual floats\n", n);
    }

    const bool reset = pos == 0;
    for (int L = 0; L < (int) layers_.size(); ++L) {
        Slot& layer = layers_[(size_t) L];
        if (L == model_->ple_layer_) apply_ple(scratch_->cur, token);
        if (layer.qsa) {
            layer.qsa_exec->attend_write(scratch_->cur, scratch_->nxt, pos, reset);
        } else {
            layer.gdn->run(scratch_->cur, scratch_->nxt, reset);
        }
        layer.ffn->run(scratch_->nxt, scratch_->cur, true, nullptr);
        log_router_margin(pos, L, static_cast<const float*>(layer.ffn->route.data()), layer.ffn->n_expert,
                          layer.ffn->k);
    }

    const GrShapes shapes{model_->geom_.n_embd, model_->geom_.hc, model_->geom_.hc_lr};
    output_mix(shapes, scratch_->head, scratch_->cur, model_->weights_->out_norm, model_->weights_->out_down,
               model_->weights_->out_up, scratch_->mixed, scratch_->inject);
    output_project(model_->out_type_, model_->weights_->out_w, scratch_->mixed, n_embd, n_vocab, scratch_->logits);
    std::memcpy(logits, scratch_->logits.data(), (size_t) n_vocab * 4u);

    if (ple_prev_.size() >= 1) {
        for (size_t i = 0; i + 1 < ple_prev_.size(); ++i) ple_prev_[i] = ple_prev_[i + 1];
        ple_prev_.back() = token;
    }
    next_pos_ = pos + 1;
}

}  // namespace strata::metal
