// The layer step the walk checks and the forward pass runs. See layer_exec.hpp.
#include "strata/metal/layer_exec.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/iq.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/moe_combine.hpp"
#include "strata/metal/rope.hpp"
#include "strata/metal/router.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace strata::metal {
namespace {

Bytes copy_bytes(const std::vector<uint8_t>& raw) {
    Bytes b((uint64_t) raw.size());
    if (!raw.empty()) std::memcpy(b.data(), raw.data(), raw.size());
    return b;
}

Bytes load_tensor(const ModelTensors& model, const LayerTensor& tensor) { return copy_bytes(model.read(tensor)); }

void zero_bytes(Bytes& b) {
    if (b.size() != 0) std::memset(b.data(), 0, (size_t) b.size());
}

void fill_f32(Bytes& b, float v) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4u), v); }

size_t row_or_0(int type, int64_t n_in) {
    try {
        return iq_row_bytes(type, n_in);
    } catch (const MetalError&) {
        return 0;
    }
}

// indexer_key_append writes one pooled row at cell 0, and rows [b, b + 1] when a block completes.
// max_cells 1 and 2 never complete a block of 4, so this is one row, the size the walk allocates.
int pooled_rows(int max_cells, int idx_block) {
    if (max_cells < 1 || idx_block < 2) throw MetalError("qsa cache: indexer size is not set up");
    const int last = max_cells - 1;
    int last_complete = (last / idx_block) * idx_block + (idx_block - 1);
    if (last_complete > last) last_complete -= idx_block;
    if (last_complete < 0) return 1;
    return last_complete / idx_block + 2;
}

GdnIqWeights gdn_weights(const GdnExec& e) {
    GdnIqWeights w;
    w.qkv = GdnIqPlane{e.qkv_ty, e.qkv_override ? e.qkv_override : &e.qkv};
    w.attn_gate = GdnIqPlane{e.gate_ty, &e.gate};
    w.ssm_out = GdnIqPlane{e.ssm_ty, &e.ssm};
    w.alpha = &e.alpha_w;
    w.beta = &e.beta_w;
    w.conv = &e.conv_w;
    w.ssm_norm = &e.ssm_norm;
    w.dt = &e.dt_w;
    w.ssm_a = &e.ssm_a;
    return w;
}

QsaIqWeights qsa_weights(const QsaExec& e) {
    QsaIqWeights w;
    w.k = QsaIqPlane{e.k_ty, &e.k};
    w.v = QsaIqPlane{e.v_ty, e.v_override ? e.v_override : &e.v};
    w.q = QsaIqPlane{e.q_ty, &e.q};
    w.out = QsaIqPlane{e.o_ty, &e.o};
    w.idx_k = &e.idx_k;
    w.idx_q = &e.idx_q;
    w.k_norm = &e.k_norm;
    w.q_norm = &e.q_norm;
    w.idx_k_norm = &e.idx_k_norm;
    w.idx_q_norm = &e.idx_q_norm;
    w.cos_tab = &e.cos_tab;
    w.sin_tab = &e.sin_tab;
    return w;
}

QsaLayerState qsa_state(QsaExec& e) {
    QsaLayerState st;
    st.k_pool = &e.k_pool;
    st.v_pool = &e.v_pool;
    st.page_table = &e.page_table;
    st.idx_tail = &e.idx_tail;
    st.idx_dead = &e.idx_dead;
    st.idx_pooled = &e.idx_pooled;
    st.idx_block_pos = &e.idx_bpos;
    return st;
}

struct SeN {
    int n;
    int pad0;
    int pad1;
    int pad2;
};
static_assert(sizeof(SeN) == 16, "SeN must match the shader");

}  // namespace

ExpertSpans expert_spans(const ModelTensors& model, int layer, int64_t n_embd, int64_t n_ff, int64_t n_expert) {
    ExpertSpans e;
    e.gate = model.find(layer, "ffn_gate_exps.weight");
    e.up = model.find(layer, "ffn_up_exps.weight");
    e.down = model.find(layer, "ffn_down_exps.weight");
    if (e.gate && e.up && e.down && n_expert > 0) {
        e.gate_n = row_or_0(e.gate->type, n_embd) * (uint64_t) n_ff;
        e.up_n = row_or_0(e.up->type, n_embd) * (uint64_t) n_ff;
        e.down_n = row_or_0(e.down->type, n_ff) * (uint64_t) n_embd;
        if (e.gate->bytes != e.gate_n * (uint64_t) n_expert || e.up->bytes != e.up_n * (uint64_t) n_expert ||
            e.down->bytes != e.down_n * (uint64_t) n_expert || e.gate_n == 0 || e.up_n == 0 || e.down_n == 0)
            e.gate_n = e.up_n = e.down_n = 0;
    }
    return e;
}

GdnExec GdnExec::load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& g) {
    const LayerTensor& qkv_t = model.at(layer, "attn_qkv.weight");
    const LayerTensor& gate_t = model.at(layer, "attn_gate.weight");
    const LayerTensor& ssm_t = model.at(layer, "ssm_out.weight");
    Bytes qkv = load_tensor(model, qkv_t);
    Bytes gate = load_tensor(model, gate_t);
    Bytes ssm = load_tensor(model, ssm_t);
    Bytes conv_w = load_tensor(model, model.at(layer, "ssm_conv1d.weight"));
    Bytes alpha_w = load_tensor(model, model.at(layer, "ssm_alpha.weight"));
    Bytes beta_w = load_tensor(model, model.at(layer, "ssm_beta.weight"));
    Bytes ssm_a = load_tensor(model, model.at(layer, "ssm_a"));
    Bytes dt_w = load_tensor(model, model.at(layer, "ssm_dt.bias"));
    Bytes ssm_norm = load_tensor(model, model.at(layer, "ssm_norm.weight"));
    Bytes hc_down = load_tensor(model, model.at(layer, "hc_attn_down.weight"));
    Bytes hc_up = load_tensor(model, model.at(layer, "hc_attn_up.weight"));
    Bytes hc_inj = load_tensor(model, model.at(layer, "hc_attn_inject.weight"));
    Bytes hc_norm = load_tensor(model, model.at(layer, "hc_attn_norm.weight"));

    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    GdnLayerGeom geom{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
    GrShapes shapes{g.n_embd, g.hc, g.hc_lr};
    const int state_n = (int) geom.S * (int) geom.h_v * (int) geom.S;
    const int conv_n = (int) g.ssm_conv_channels * (int) (geom.d_conv - 1);
    Bytes state((uint64_t) state_n * 4u);
    Bytes conv((uint64_t) conv_n * 4u);
    zero_bytes(state);
    zero_bytes(conv);
    Bytes mixed((uint64_t) n_embd * 4u), inject((uint64_t) hc * 4u), block((uint64_t) n_embd * 4u);
    return GdnExec{.qkv_ty = qkv_t.type,
                   .gate_ty = gate_t.type,
                   .ssm_ty = ssm_t.type,
                   .n_embd = n_embd,
                   .hc = hc,
                   .qkv = std::move(qkv),
                   .gate = std::move(gate),
                   .ssm = std::move(ssm),
                   .conv_w = std::move(conv_w),
                   .alpha_w = std::move(alpha_w),
                   .beta_w = std::move(beta_w),
                   .ssm_a = std::move(ssm_a),
                   .dt_w = std::move(dt_w),
                   .ssm_norm = std::move(ssm_norm),
                   .hc_down = std::move(hc_down),
                   .hc_up = std::move(hc_up),
                   .hc_inj = std::move(hc_inj),
                   .hc_norm = std::move(hc_norm),
                   .state = std::move(state),
                   .conv = std::move(conv),
                   .mixed = std::move(mixed),
                   .inject = std::move(inject),
                   .block = std::move(block),
                   .shapes = shapes,
                   .geom = geom,
                   .gr = make_gr_scratch(shapes),
                   .scratch = make_gdn_layer_scratch(geom)};
}

void GdnExec::set_qkv(const Bytes* override_or_null) { qkv_override = override_or_null; }

void GdnExec::run(const Bytes& residual, Bytes& y, bool reset_state) {
    if (reset_state) {
        zero_bytes(state);
        zero_bytes(conv);
    }
    fill_f32(block, 7.f);
    fill_f32(y, 7.f);
    const GdnIqWeights w = gdn_weights(*this);
    GdnLayerState st{&state, &conv};
    gr_read(shapes, residual, hc_norm, hc_down, hc_up, &hc_inj, 1e-6f, false, gr, mixed, inject);
    gdn_layer_iq(geom, w, st, scratch, mixed, block);
    gr_write(shapes, residual, block, inject, y);
}

QsaExec QsaExec::load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& g, int max_cells,
                      int rope_rows) {
    if (max_cells < 1 || rope_rows < 1) throw MetalError("qsa cache: max_cells and rope rows must be positive");
    const LayerTensor& q_t = model.at(layer, "attn_q.weight");
    const LayerTensor& k_t = model.at(layer, "attn_k.weight");
    const LayerTensor& v_t = model.at(layer, "attn_v.weight");
    const LayerTensor& o_t = model.at(layer, "attn_output.weight");
    Bytes q = load_tensor(model, q_t);
    Bytes k = load_tensor(model, k_t);
    Bytes v = load_tensor(model, v_t);
    Bytes o = load_tensor(model, o_t);
    Bytes q_norm = load_tensor(model, model.at(layer, "attn_q_norm.weight"));
    Bytes k_norm = load_tensor(model, model.at(layer, "attn_k_norm.weight"));
    Bytes idx_q_norm = load_tensor(model, model.at(layer, "indexer.q_norm.weight"));
    Bytes idx_k_norm = load_tensor(model, model.at(layer, "indexer.k_norm.weight"));
    Bytes idx_q = load_tensor(model, model.at(layer, "indexer.q_proj.weight"));
    Bytes idx_k = load_tensor(model, model.at(layer, "indexer.k_proj.weight"));
    Bytes hc_down = load_tensor(model, model.at(layer, "hc_attn_down.weight"));
    Bytes hc_up = load_tensor(model, model.at(layer, "hc_attn_up.weight"));
    Bytes hc_inj = load_tensor(model, model.at(layer, "hc_attn_inject.weight"));
    Bytes hc_norm = load_tensor(model, model.at(layer, "hc_attn_norm.weight"));

    const auto sh = strata::kernels::qsa_real_shapes();
    const int pairs = (int) sh.n_rot / 2;
    std::vector<float> cos_h((size_t) rope_rows * (size_t) pairs), sin_h(cos_h.size());
    build_rope_table((int) sh.n_rot, strata::kernels::qsa_freq_base(), rope_rows, cos_h.data(), sin_h.data());
    Bytes cos_tab((uint64_t) cos_h.size() * 4u), sin_tab((uint64_t) sin_h.size() * 4u);
    std::memcpy(cos_tab.data(), cos_h.data(), (size_t) cos_tab.size());
    std::memcpy(sin_tab.data(), sin_h.data(), (size_t) sin_tab.size());

    const int page_size = (int) sh.page_size;
    const int n_pages = (max_cells + page_size - 1) / page_size;
    Bytes page_table((uint64_t) n_pages * 4u);
    int* pages = static_cast<int*>(page_table.data());
    for (int i = 0; i < n_pages; ++i) pages[i] = i;
    const size_t pool_n =
        (size_t) sh.n_head_kv * (size_t) n_pages * (size_t) page_size * (size_t) sh.head_dim;
    Bytes k_pool(pool_n * 2u), v_pool(pool_n * 2u);
    const int idx_dim = (int) sh.idx_dim;
    const int idx_block = (int) sh.idx_block;
    Bytes idx_tail((uint64_t) (idx_block - 1) * (uint64_t) idx_dim * 4u);
    Bytes idx_dead((uint64_t) idx_dim * 4u);
    Bytes idx_pooled((uint64_t) pooled_rows(max_cells, idx_block) * (uint64_t) idx_dim * 4u);
    Bytes idx_bpos(4);
    zero_bytes(k_pool);
    zero_bytes(v_pool);
    zero_bytes(idx_tail);
    zero_bytes(idx_dead);
    zero_bytes(idx_pooled);
    zero_bytes(idx_bpos);

    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    GrShapes shapes{g.n_embd, g.hc, g.hc_lr};
    QsaLayerGeom geom;
    geom.n_embd = n_embd;
    geom.max_cells = max_cells;
    geom.qsa = sh;
    Bytes mixed((uint64_t) n_embd * 4u), inject((uint64_t) hc * 4u), block((uint64_t) n_embd * 4u);
    return QsaExec{.q_ty = q_t.type,
                   .k_ty = k_t.type,
                   .v_ty = v_t.type,
                   .o_ty = o_t.type,
                   .n_embd = n_embd,
                   .hc = hc,
                   .hc_lr = (int) g.hc_lr,
                   .max_cells = max_cells,
                   .q = std::move(q),
                   .k = std::move(k),
                   .v = std::move(v),
                   .o = std::move(o),
                   .q_norm = std::move(q_norm),
                   .k_norm = std::move(k_norm),
                   .idx_q_norm = std::move(idx_q_norm),
                   .idx_k_norm = std::move(idx_k_norm),
                   .idx_q = std::move(idx_q),
                   .idx_k = std::move(idx_k),
                   .hc_down = std::move(hc_down),
                   .hc_up = std::move(hc_up),
                   .hc_inj = std::move(hc_inj),
                   .hc_norm = std::move(hc_norm),
                   .cos_tab = std::move(cos_tab),
                   .sin_tab = std::move(sin_tab),
                   .page_table = std::move(page_table),
                   .k_pool = std::move(k_pool),
                   .v_pool = std::move(v_pool),
                   .idx_tail = std::move(idx_tail),
                   .idx_dead = std::move(idx_dead),
                   .idx_pooled = std::move(idx_pooled),
                   .idx_bpos = std::move(idx_bpos),
                   .mixed = std::move(mixed),
                   .inject = std::move(inject),
                   .block = std::move(block),
                   .shapes = shapes,
                   .geom = geom,
                   .gr = make_gr_scratch(shapes),
                   .scratch = make_qsa_layer_scratch(geom)};
}

void QsaExec::set_v(const Bytes* override_or_null) { v_override = override_or_null; }

void QsaExec::begin(const Bytes& residual, bool reset_cache) {
    if (reset_cache) {
        zero_bytes(k_pool);
        zero_bytes(v_pool);
        zero_bytes(idx_tail);
        zero_bytes(idx_dead);
        zero_bytes(idx_pooled);
        zero_bytes(idx_bpos);
    }
    fill_f32(mixed, 7.f);
    gr_read(shapes, residual, hc_norm, hc_down, hc_up, &hc_inj, 1e-6f, false, gr, mixed, inject);
}

void QsaExec::block_at(int pos, int pos_base, Bytes& out) {
    fill_f32(out, 7.f);
    const QsaIqWeights w = qsa_weights(*this);
    QsaLayerState st = qsa_state(*this);
    qsa_layer_iq(geom, w, st, scratch, pos, pos_base, mixed, out);
}

void QsaExec::attend_write(const Bytes& residual, Bytes& y, bool reset_cache) {
    begin(residual, reset_cache);
    fill_f32(block, 7.f);
    fill_f32(y, 7.f);
    block_at(0, 0, block);
    gr_write(shapes, residual, block, inject, y);
}

FfnExec FfnExec::load(const ModelTensors& model, int layer, const strata::core::ModelGeometry& g, bool with_shared) {
    const ExpertSpans spans = expert_spans(model, layer, g.n_embd, g.n_ff, g.n_expert);
    if (!spans.gate || !spans.up || !spans.down || spans.gate_n == 0)
        throw MetalError("routed expert planes are missing");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int k = 10;
    const int guard = 1;
    GrShapes shapes{g.n_embd, g.hc, g.hc_lr};
    Bytes hc_down = load_tensor(model, model.at(layer, "hc_ffn_down.weight"));
    Bytes hc_up = load_tensor(model, model.at(layer, "hc_ffn_up.weight"));
    Bytes hc_inj = load_tensor(model, model.at(layer, "hc_ffn_inject.weight"));
    Bytes hc_norm = load_tensor(model, model.at(layer, "hc_ffn_norm.weight"));
    Bytes router = load_tensor(model, model.at(layer, "ffn_gate_inp.weight"));
    // The routed-only pass does not load the shared planes. A 4-byte stand-in keeps the buffers valid.
    int sg_ty = -1, su_ty = -1, sd_ty = -1;
    Bytes ginp(4), sgate(4), sup(4), sdown(4);
    if (with_shared) {
        const LayerTensor& gin = model.at(layer, "ffn_gate_inp_shexp.weight");
        const LayerTensor& sg = model.at(layer, "ffn_gate_shexp.weight");
        const LayerTensor& su = model.at(layer, "ffn_up_shexp.weight");
        const LayerTensor& sd = model.at(layer, "ffn_down_shexp.weight");
        sg_ty = sg.type;
        su_ty = su.type;
        sd_ty = sd.type;
        ginp = load_tensor(model, gin);
        sgate = load_tensor(model, sg);
        sup = load_tensor(model, su);
        sdown = load_tensor(model, sd);
    }
    return FfnExec{.model = &model,
                   .n_embd = n_embd,
                   .n_ff = n_ff,
                   .n_expert = n_expert,
                   .hc = hc,
                   .hc_lr = (int) g.hc_lr,
                   .k = k,
                   .with_shared = with_shared,
                   .eg_ty = spans.gate->type,
                   .eu_ty = spans.up->type,
                   .ed_ty = spans.down->type,
                   .sg_ty = sg_ty,
                   .su_ty = su_ty,
                   .sd_ty = sd_ty,
                   .gate_n = spans.gate_n,
                   .up_n = spans.up_n,
                   .down_n = spans.down_n,
                   .gate = *spans.gate,
                   .up = *spans.up,
                   .down = *spans.down,
                   .hc_down = std::move(hc_down),
                   .hc_up = std::move(hc_up),
                   .hc_inj = std::move(hc_inj),
                   .hc_norm = std::move(hc_norm),
                   .router = std::move(router),
                   .ginp = std::move(ginp),
                   .sgate = std::move(sgate),
                   .sup = std::move(sup),
                   .sdown = std::move(sdown),
                   .shapes = shapes,
                   .gr = make_gr_scratch(shapes),
                   .expert_y = Bytes((uint64_t) (n_embd + guard) * 4u),
                   .shared_y = Bytes((uint64_t) (n_embd + guard) * 4u),
                   .combined = Bytes((uint64_t) (n_embd + guard) * 4u),
                   .mixed = Bytes((uint64_t) n_embd * 4u),
                   .inject = Bytes((uint64_t) hc * 4u),
                   .x_bf16 = Bytes((uint64_t) n_embd * 2u),
                   .route = Bytes((uint64_t) n_expert * 4u),
                   .exq = Bytes(iq_q8_1_bytes(n_embd)),
                   .egate_y = Bytes((uint64_t) n_ff * 4u),
                   .eup_y = Bytes((uint64_t) n_ff * 4u),
                   .ehq = Bytes(iq_q8_1_bytes(n_ff)),
                   .sh_xq = Bytes(iq_q8_1_bytes(n_embd)),
                   .sh_gate_y = Bytes((uint64_t) n_ff * 4u),
                   .sh_up_y = Bytes((uint64_t) n_ff * 4u),
                   .sh_hq = Bytes(iq_q8_1_bytes(n_ff)),
                   .sh_down_y = Bytes((uint64_t) (n_embd + guard) * 4u),
                   .sh_scalar = Bytes(4),
                   .parts = Bytes((uint64_t) k * (uint64_t) n_embd * 4u),
                   .ids = Bytes((uint64_t) (k + guard) * 4u),
                   .route_w = Bytes((uint64_t) (k + guard) * 4u)};
}

void FfnExec::run(const Bytes& residual, Bytes& out, bool add_shared, FfnTrace* trace) {
    const int guard = 1;
    std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
    std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
    fill_f32(out, 7.f);
    std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
    std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
    gr_read(shapes, residual, hc_norm, hc_down, hc_up, &hc_inj, 1e-6f, false, gr, mixed, inject);
    f32_to_bf16_bulk(mixed, x_bf16, n_embd);
    bf16_gemv(x_bf16, router, route, n_embd, n_expert);
    router_top10(route, 1, n_expert, k, ids, route_w);
    iq_quantize_q8_1(mixed, n_embd, exq);
    const int* idp = static_cast<const int*>(ids.data());
    if (trace) {
        trace->egate.assign((size_t) k, {});
        trace->eup.assign((size_t) k, {});
        trace->edown.assign((size_t) k, {});
        trace->ehq.assign((size_t) k, {});
        trace->blob.assign((size_t) k, {});
        trace->exq.assign((size_t) exq.size(), 0);
        std::memcpy(trace->exq.data(), exq.data(), trace->exq.size());
    }
    const uint64_t blob_n = gate_n + up_n + down_n;
    for (int e = 0; e < k; ++e) {
        const int id = idp[e];
        if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
        std::vector<uint8_t> raw((size_t) blob_n);
        const auto gate_b = model->read(gate, (uint64_t) id * gate_n, gate_n);
        const auto up_b = model->read(up, (uint64_t) id * up_n, up_n);
        const auto down_b = model->read(down, (uint64_t) id * down_n, down_n);
        std::memcpy(raw.data(), gate_b.data(), (size_t) gate_n);
        std::memcpy(raw.data() + gate_n, up_b.data(), (size_t) up_n);
        std::memcpy(raw.data() + gate_n + up_n, down_b.data(), (size_t) down_n);
        Bytes expert_w((uint64_t) raw.size());
        std::memcpy(expert_w.data(), raw.data(), raw.size());
        iq_mmvq(eg_ty, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
        iq_mmvq(eu_ty, expert_w, (int64_t) gate_n, exq, n_embd, n_ff, 1, eup_y);
        iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
        iq_mmvq(ed_ty, expert_w, (int64_t) (gate_n + up_n), ehq, n_ff, n_embd, 1, expert_y);
        std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(), (size_t) n_embd * 4u);
        if (trace) {
            trace->egate[(size_t) e].resize((size_t) n_ff);
            trace->eup[(size_t) e].resize((size_t) n_ff);
            trace->edown[(size_t) e].resize((size_t) n_embd);
            trace->ehq[(size_t) e].resize(ehq.size());
            trace->blob[(size_t) e] = raw;
            std::memcpy(trace->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4u);
            std::memcpy(trace->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4u);
            std::memcpy(trace->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4u);
            std::memcpy(trace->ehq[(size_t) e].data(), ehq.data(), ehq.size());
        }
    }
    if (with_shared) {
        iq_quantize_q8_1(mixed, n_embd, sh_xq);
        iq_mmvq(sg_ty, sgate, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
        iq_mmvq(su_ty, sup, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
        iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
        iq_mmvq(sd_ty, sdown, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
        std::fill_n(static_cast<float*>(shared_y.data()), (size_t) (n_embd + guard), 7.f);
        std::memcpy(shared_y.data(), sh_down_y.data(), (size_t) n_embd * 4u);
        Bytes x_bf16_s((uint64_t) n_embd * 2u);
        f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
        bf16_gemv(x_bf16_s, ginp, sh_scalar, n_embd, 1);
        const SeN dim{n_embd, 0, 0, 0};
        const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
        const Bytes* scb[] = {&shared_y, &sh_scalar};
        launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
    }
    moe_combine(parts, route_w, with_shared && add_shared ? &shared_y : nullptr, combined, n_embd, k);
    gr_write(shapes, residual, combined, inject, out);
    if (!trace) return;
    trace->mixed.resize((size_t) n_embd);
    trace->inject.resize((size_t) hc);
    trace->residual.resize((size_t) hc * n_embd);
    trace->combined.resize((size_t) n_embd);
    trace->layer.resize((size_t) hc * n_embd);
    trace->route.resize((size_t) n_expert);
    trace->ids.assign(idp, idp + k);
    trace->route_w.resize((size_t) k);
    std::memcpy(trace->mixed.data(), mixed.data(), (size_t) n_embd * 4u);
    std::memcpy(trace->inject.data(), inject.data(), (size_t) hc * 4u);
    std::memcpy(trace->residual.data(), residual.data(), (size_t) hc * n_embd * 4u);
    std::memcpy(trace->combined.data(), combined.data(), (size_t) n_embd * 4u);
    std::memcpy(trace->layer.data(), out.data(), (size_t) hc * n_embd * 4u);
    std::memcpy(trace->route.data(), route.data(), (size_t) n_expert * 4u);
    std::memcpy(trace->route_w.data(), route_w.data(), (size_t) k * 4u);
    if (with_shared) {
        trace->shared.resize((size_t) n_embd);
        trace->sgate.resize((size_t) n_ff);
        trace->sup.resize((size_t) n_ff);
        trace->sdown.resize((size_t) n_embd);
        trace->sh_xq.resize(sh_xq.size());
        trace->sh_hq.resize(sh_hq.size());
        std::memcpy(trace->shared.data(), shared_y.data(), (size_t) n_embd * 4u);
        std::memcpy(trace->sgate.data(), sh_gate_y.data(), (size_t) n_ff * 4u);
        std::memcpy(trace->sup.data(), sh_up_y.data(), (size_t) n_ff * 4u);
        std::memcpy(trace->sdown.data(), sh_down_y.data(), (size_t) n_embd * 4u);
        std::memcpy(trace->sh_xq.data(), sh_xq.data(), sh_xq.size());
        std::memcpy(trace->sh_hq.data(), sh_hq.data(), sh_hq.size());
        trace->scale = static_cast<const float*>(sh_scalar.data())[0];
    }
}

void output_mix(const GrShapes& shapes, GrScratch& scratch, const Bytes& residual, const Bytes& norm, const Bytes& down,
                const Bytes& up, Bytes& mixed, Bytes& inject) {
    fill_f32(inject, 7.f);
    fill_f32(mixed, 7.f);
    gr_read(shapes, residual, norm, down, up, nullptr, 1e-6f, false, scratch, mixed, inject);
}

Bytes output_project(int ggml_type, const Bytes& weight, const Bytes& mixed, int n_embd, int n_vocab, Bytes& logits) {
    Bytes act((uint64_t) n_embd * 4u);
    std::memcpy(act.data(), mixed.data(), (size_t) act.size());
    Bytes q(iq_q8_1_bytes(n_embd));
    iq_quantize_q8_1(act, n_embd, q);
    fill_f32(logits, 7.f);
    iq_mmvq(ggml_type, weight, 0, q, n_embd, n_vocab, 1, logits);
    return q;
}

}  // namespace strata::metal
