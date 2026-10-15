// One QSA attention block on the kernels that already pass. fp16 KV only.
#include "strata/metal/qsa_layer.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/iq.hpp"
#include "strata/metal/indexer.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/qsa.hpp"
#include "strata/metal/qsa_gate.hpp"
#include "strata/metal/qsa_index.hpp"
#include "strata/metal/rope.hpp"
#include "strata/metal/topk.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

constexpr float kEps = 1e-6f;

int fit(const char* what, int64_t n) {
    if (n < 0 || n > 0x7fffffff) throw MetalError(std::string("qsa_layer: ") + what + " does not fit");
    return (int) n;
}

void need(const char* what, const void* p) {
    if (p == nullptr) throw MetalError(std::string("qsa_layer: ") + what + " is missing");
}

const Bytes* plane_offset(const char* name, const QsaPlane& p) {
    if (p.codes == nullptr) throw MetalError(std::string("qsa_layer: ") + name + " codes are missing");
    if (p.scales == nullptr) throw MetalError(std::string("qsa_layer: ") + name + " scales are missing");
    if (p.form.has_offset && p.offset == nullptr)
        throw MetalError(std::string("qsa_layer: ") + name + " offset is missing");
    return p.form.has_offset ? p.offset : nullptr;
}

void check_geom(const QsaLayerGeom& g) {
    const auto& s = g.qsa;
    if (g.n_embd <= 0 || g.n_embd % 256 != 0) throw MetalError("qsa_layer: n_embd must be a positive multiple of 256");
    if (g.max_cells <= 0) throw MetalError("qsa_layer: max_cells must be positive");
    if (s.head_dim != 256 || s.n_head_kv <= 0 || s.n_head != 12 * s.n_head_kv)
        throw MetalError("qsa_layer: attention geometry is not the decode kernel's");
    if (s.n_rot < 2 || (s.n_rot % 2) != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim)
        throw MetalError("qsa_layer: n_rot must be even and no wider than the rows it rotates");
    if (s.idx_n_head < 1 || s.idx_n_head > 32 || s.idx_dim < 2 || s.idx_block < 2 || s.idx_top_k < 1 || s.page_size < 1)
        throw MetalError("qsa_layer: indexer geometry is not set up");
    if (s.head_dim % 4 != 0) throw MetalError("qsa_layer: head_dim must be a multiple of 4");
}

void fill_pos(Bytes& pos, int n, int value) {
    int* p = (int*) pos.data();
    for (int i = 0; i < n; ++i) p[i] = value;
}

void split_q(const Bytes& q_full, Bytes& qcur, int n_head, int head_dim) {
    const float* src = (const float*) q_full.data();
    float* dst = (float*) qcur.data();
    for (int h = 0; h < n_head; ++h)
        std::memcpy(dst + (size_t) h * (size_t) head_dim,
                    src + (size_t) h * (size_t) head_dim * 2u, (size_t) head_dim * sizeof(float));
}

}  // namespace

QsaLayerScratch make_qsa_layer_scratch(const QsaLayerGeom& g) {
    check_geom(g);
    const auto& s = g.qsa;
    const int64_t kv = s.n_head_kv * s.head_dim;
    const int64_t q_full = s.n_head * s.head_dim * 2;
    const int64_t q = s.n_head * s.head_dim;
    const int64_t iq = s.idx_n_head * s.idx_dim;
    const int cap = (int) strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int heads = (int) std::max(s.n_head, std::max(s.n_head_kv, s.idx_n_head));
    return QsaLayerScratch{
        Bytes(q8k_bytes(g.n_embd)),
        Bytes((uint64_t) g.n_embd * 2u),
        Bytes((uint64_t) s.idx_dim * 4u),
        Bytes((uint64_t) kv * 4u),
        Bytes((uint64_t) kv * 4u),
        Bytes((uint64_t) q_full * 4u),
        Bytes((uint64_t) q * 4u),
        Bytes((uint64_t) iq * 4u),
        Bytes((uint64_t) g.max_cells * 4u),
        Bytes((uint64_t) cap * 4u),
        Bytes((uint64_t) strata::kernels::kStepCount * 4u),
        Bytes((uint64_t) heads * 4u),
        Bytes(qsa_decode_attn_scratch_floats(cap, s) * 4u),
        Bytes((uint64_t) q * 4u),
        Bytes((uint64_t) q * 4u),
        Bytes(q8k_bytes(q)),
    };
}

void qsa_layer(const QsaLayerGeom& g, const QsaLayerWeights& w, QsaLayerState& st, QsaLayerScratch& sc, int pos,
               int pos_base, const Bytes& x, Bytes& out) {
    check_geom(g);
    if (pos < 0 || pos >= g.max_cells) throw MetalError("qsa_layer: pos is outside the sequence");
    if (pos_base < 0) throw MetalError("qsa_layer: pos_base < 0");
    const int rotary = pos_base + pos;
    if (rotary < pos) throw MetalError("qsa_layer: the rope position overflows");
    need("k pool", st.k_pool);
    need("v pool", st.v_pool);
    need("page table", st.page_table);
    need("indexer tail", st.idx_tail);
    need("indexer dead", st.idx_dead);
    need("indexer pooled", st.idx_pooled);
    need("indexer block pos", st.idx_block_pos);
    need("indexer k weight", w.idx_k);
    need("indexer q weight", w.idx_q);
    need("k norm", w.k_norm);
    need("q norm", w.q_norm);
    need("indexer k norm", w.idx_k_norm);
    need("indexer q norm", w.idx_q_norm);
    need("cos", w.cos_tab);
    need("sin", w.sin_tab);
    const Bytes* k_off = plane_offset("attn_k", w.k);
    const Bytes* v_off = plane_offset("attn_v", w.v);
    const Bytes* q_off = plane_offset("attn_q", w.q);
    const Bytes* o_off = plane_offset("attn_output", w.out);

    const auto& s = g.qsa;
    const int n_embd = fit("n_embd", g.n_embd);
    const int n_head = fit("n_head", s.n_head);
    const int n_head_kv = fit("n_head_kv", s.n_head_kv);
    const int head_dim = fit("head_dim", s.head_dim);
    const int n_rot = fit("n_rot", s.n_rot);
    const int idx_n_head = fit("idx_n_head", s.idx_n_head);
    const int idx_dim = fit("idx_dim", s.idx_dim);
    const int idx_block = fit("idx_block", s.idx_block);
    const int idx_top_k = fit("idx_top_k", s.idx_top_k);
    const int page_size = fit("page_size", s.page_size);
    const int kv = n_head_kv * head_dim;
    const int q_full_n = n_head * head_dim * 2;
    const int q_n = n_head * head_dim;
    const int iq = idx_n_head * idx_dim;
    const int n_kv = pos + 1;
    const int n_bid = n_kv / idx_block;
    const int width = (int) strata::kernels::qsa_selection_width(n_kv, s);
    const int cap = (int) strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);

    quantize_q8_k(x, n_embd, sc.x_q8k);
    f32_to_bf16_bulk(x, sc.x_bf16, n_embd);
    bf16_gemv(sc.x_bf16, *w.idx_k, sc.idx_raw, n_embd, idx_dim);
    s_gemv_q8k_split(sc.x_q8k, *w.k.codes, *w.k.scales, k_off, sc.kcur, n_embd, kv, w.k.form);
    s_gemv_q8k_split(sc.x_q8k, *w.v.codes, *w.v.scales, v_off, sc.vcur, n_embd, kv, w.v.form);

    rms_norm_weighted(sc.kcur, w.k_norm, n_head_kv, head_dim, kEps);
    fill_pos(sc.pos, n_head_kv, rotary);
    rope_neox_apply(sc.kcur, sc.kcur, n_head_kv, head_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);
    kv_append(*st.k_pool, *st.v_pool, *st.page_table, pos, sc.kcur, sc.vcur, n_head_kv, head_dim, page_size);

    fill_pos(sc.pos, 1, pos);
    IndexerBuf ib{*st.idx_tail, *st.idx_dead, *st.idx_pooled, *st.idx_block_pos};
    indexer_key_append(sc.idx_raw, sc.pos, pos_base, *w.idx_k_norm, kEps, ib, idx_dim, idx_block, n_rot, *w.cos_tab,
                       *w.sin_tab);

    s_gemv_q8k_split(sc.x_q8k, *w.q.codes, *w.q.scales, q_off, sc.q_full, n_embd, q_full_n, w.q.form);
    split_q(sc.q_full, sc.qcur, n_head, head_dim);
    rms_norm_weighted(sc.qcur, w.q_norm, n_head, head_dim, kEps);
    fill_pos(sc.pos, n_head, rotary);
    rope_neox_apply(sc.qcur, sc.qcur, n_head, head_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);

    bf16_gemv(sc.x_bf16, *w.idx_q, sc.q_idx, n_embd, iq);
    rms_norm_weighted(sc.q_idx, w.idx_q_norm, idx_n_head, idx_dim, kEps);
    fill_pos(sc.pos, idx_n_head, rotary);
    rope_neox_apply(sc.q_idx, sc.q_idx, idx_n_head, idx_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);

    qsa_index(*st.idx_pooled, n_bid, sc.q_idx, nullptr, idx_n_head, idx_dim, idx_block, n_kv, sc.cell_scores);
    topk_512(sc.cell_scores, n_kv, idx_top_k, idx_block, cap, sc.ids);

    int step[strata::kernels::kStepCount];
    step[strata::kernels::kStepPos] = pos;
    step[strata::kernels::kStepNKv] = n_kv;
    step[strata::kernels::kStepNBid] = n_bid;
    step[strata::kernels::kStepWidth] = width;
    std::memcpy(sc.step.data(), step, sizeof step);

    QsaPools pools;
    pools.k_f16 = st.k_pool;
    pools.v_f16 = st.v_pool;
    pools.page_table = st.page_table;
    qsa_decode_attn(sc.qcur, pools, sc.ids, sc.step, cap, s, sc.attn_scratch, sc.attn, 1);
    qsa_gate_apply_f32(sc.attn, sc.q_full, n_head, head_dim, sc.gated);
    quantize_q8_k(sc.gated, q_n, sc.gated_q8k);
    s_gemv_q8k_split(sc.gated_q8k, *w.out.codes, *w.out.scales, o_off, out, q_n, n_embd, w.out.form);
}

void qsa_layer_iq(const QsaLayerGeom& g, const QsaIqWeights& w, QsaLayerState& st, QsaLayerScratch& sc, int pos,
                  int pos_base, const Bytes& x, Bytes& out) {
    check_geom(g);
    if (pos < 0 || pos >= g.max_cells) throw MetalError("qsa_layer: pos is outside the sequence");
    if (pos_base < 0) throw MetalError("qsa_layer: pos_base < 0");
    const int rotary = pos_base + pos;
    if (rotary < pos) throw MetalError("qsa_layer: the rope position overflows");
    need("k pool", st.k_pool);
    need("v pool", st.v_pool);
    need("page table", st.page_table);
    need("indexer tail", st.idx_tail);
    need("indexer dead", st.idx_dead);
    need("indexer pooled", st.idx_pooled);
    need("indexer block pos", st.idx_block_pos);
    need("indexer k weight", w.idx_k);
    need("indexer q weight", w.idx_q);
    need("k norm", w.k_norm);
    need("q norm", w.q_norm);
    need("indexer k norm", w.idx_k_norm);
    need("indexer q norm", w.idx_q_norm);
    need("cos", w.cos_tab);
    need("sin", w.sin_tab);
    need("attn_k", w.k.weights);
    need("attn_v", w.v.weights);
    need("attn_q", w.q.weights);
    need("attn_output", w.out.weights);

    const auto& s = g.qsa;
    const int n_embd = fit("n_embd", g.n_embd);
    const int n_head = fit("n_head", s.n_head);
    const int n_head_kv = fit("n_head_kv", s.n_head_kv);
    const int head_dim = fit("head_dim", s.head_dim);
    const int n_rot = fit("n_rot", s.n_rot);
    const int idx_n_head = fit("idx_n_head", s.idx_n_head);
    const int idx_dim = fit("idx_dim", s.idx_dim);
    const int idx_block = fit("idx_block", s.idx_block);
    const int idx_top_k = fit("idx_top_k", s.idx_top_k);
    const int page_size = fit("page_size", s.page_size);
    const int kv = n_head_kv * head_dim;
    const int q_full_n = n_head * head_dim * 2;
    const int q_n = n_head * head_dim;
    const int iq = idx_n_head * idx_dim;
    const int n_kv = pos + 1;
    const int n_bid = n_kv / idx_block;
    const int width = (int) strata::kernels::qsa_selection_width(n_kv, s);
    const int cap = (int) strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);

    // q8_1 is 36 bytes a block. The scratch slots are Q8_K and are the wrong width.
    Bytes xq(iq_q8_1_bytes(g.n_embd));
    iq_quantize_q8_1(x, g.n_embd, xq);
    f32_to_bf16_bulk(x, sc.x_bf16, n_embd);
    bf16_gemv(sc.x_bf16, *w.idx_k, sc.idx_raw, n_embd, idx_dim);
    iq_mmvq(w.k.type, *w.k.weights, 0, xq, g.n_embd, kv, 1, sc.kcur);
    iq_mmvq(w.v.type, *w.v.weights, 0, xq, g.n_embd, kv, 1, sc.vcur);

    rms_norm_weighted(sc.kcur, w.k_norm, n_head_kv, head_dim, kEps);
    fill_pos(sc.pos, n_head_kv, rotary);
    rope_neox_apply(sc.kcur, sc.kcur, n_head_kv, head_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);
    kv_append(*st.k_pool, *st.v_pool, *st.page_table, pos, sc.kcur, sc.vcur, n_head_kv, head_dim, page_size);

    fill_pos(sc.pos, 1, pos);
    IndexerBuf ib{*st.idx_tail, *st.idx_dead, *st.idx_pooled, *st.idx_block_pos};
    indexer_key_append(sc.idx_raw, sc.pos, pos_base, *w.idx_k_norm, kEps, ib, idx_dim, idx_block, n_rot, *w.cos_tab,
                       *w.sin_tab);

    iq_mmvq(w.q.type, *w.q.weights, 0, xq, g.n_embd, q_full_n, 1, sc.q_full);
    split_q(sc.q_full, sc.qcur, n_head, head_dim);
    rms_norm_weighted(sc.qcur, w.q_norm, n_head, head_dim, kEps);
    fill_pos(sc.pos, n_head, rotary);
    rope_neox_apply(sc.qcur, sc.qcur, n_head, head_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);

    bf16_gemv(sc.x_bf16, *w.idx_q, sc.q_idx, n_embd, iq);
    rms_norm_weighted(sc.q_idx, w.idx_q_norm, idx_n_head, idx_dim, kEps);
    fill_pos(sc.pos, idx_n_head, rotary);
    rope_neox_apply(sc.q_idx, sc.q_idx, idx_n_head, idx_dim, n_rot, *w.cos_tab, *w.sin_tab, sc.pos);

    qsa_index(*st.idx_pooled, n_bid, sc.q_idx, nullptr, idx_n_head, idx_dim, idx_block, n_kv, sc.cell_scores);
    topk_512(sc.cell_scores, n_kv, idx_top_k, idx_block, cap, sc.ids);

    int step[strata::kernels::kStepCount];
    step[strata::kernels::kStepPos] = pos;
    step[strata::kernels::kStepNKv] = n_kv;
    step[strata::kernels::kStepNBid] = n_bid;
    step[strata::kernels::kStepWidth] = width;
    std::memcpy(sc.step.data(), step, sizeof step);

    QsaPools pools;
    pools.k_f16 = st.k_pool;
    pools.v_f16 = st.v_pool;
    pools.page_table = st.page_table;
    qsa_decode_attn(sc.qcur, pools, sc.ids, sc.step, cap, s, sc.attn_scratch, sc.attn, 1);
    qsa_gate_apply_f32(sc.attn, sc.q_full, n_head, head_dim, sc.gated);
    Bytes gq(iq_q8_1_bytes(q_n));
    iq_quantize_q8_1(sc.gated, q_n, gq);
    iq_mmvq(w.out.type, *w.out.weights, 0, gq, q_n, n_embd, 1, out);
}

}  // namespace strata::metal
