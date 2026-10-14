// Four synthetic layers on the pack schedule. The spec calls each kernel itself.
#include "strata/metal/stack.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/iq.hpp"
#include "strata/metal/moe_combine.hpp"
#include "strata/metal/rope.hpp"
#include "strata/metal/router.hpp"
#include "strata/metal/sgemv.hpp"
#include "strata/metal/shared_expert.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/qsa.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

std::string pct(double rel) {
    char b[40];
    std::snprintf(b, sizeof b, "%.2f%% apart", rel * 100.0);
    return b;
}

strata::metal::Bytes upload(const void* p, size_t n) {
    strata::metal::Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}

template <class F>
bool throws_metal(F&& fn) {
    try {
        fn();
        return false;
    } catch (const strata::metal::MetalError&) {
        return true;
    }
}

double rel_l1(const float* a, const float* b, int n) {
    double d = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

void copy_bytes(strata::metal::Bytes& dst, const strata::metal::Bytes& src) {
    if (dst.size() != src.size()) throw strata::metal::MetalError("stack parity: state size");
    if (dst.size()) std::memcpy(dst.data(), src.data(), (size_t) dst.size());
}

using namespace strata::metal;

SForm form_of(int bits, int bias, bool offset) {
    SForm f;
    f.code_bits = bits;
    f.code_bias = bias;
    f.group_elems = 32;
    f.codebook = Codebook::Affine;
    f.has_offset = offset;
    return f;
}

struct PlaneBuf {
    SForm form;
    Bytes codes;
    Bytes scales;
    Bytes offs;
};

PlaneBuf make_plane(SForm form, int n_in, int n_out, std::mt19937& rng, float off_scale) {
    const int per_byte = 8 / form.code_bits;
    std::vector<uint8_t> codes((size_t) n_out * (size_t) (n_in / per_byte));
    for (uint8_t& v : codes) v = (uint8_t) (rng() & 0xFF);
    const int n_groups = n_in / form.group_elems;
    std::vector<float> scales((size_t) n_out * (size_t) n_groups);
    std::vector<float> offs(scales.size());
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& v : scales) v = 0.01f + 0.001f * (float) (rng() % 20);
    for (float& v : offs) v = off_scale * g(rng);
    return PlaneBuf{form, upload(codes.data(), codes.size()), upload(scales.data(), scales.size() * 4u),
                    upload(offs.data(), offs.size() * 4u)};
}

QsaPlane qsa_view(const PlaneBuf& p) { return QsaPlane{p.form, &p.codes, &p.scales, p.form.has_offset ? &p.offs : nullptr}; }

GdnPlane gdn_view(const PlaneBuf& p) { return GdnPlane{p.form, &p.codes, &p.scales, p.form.has_offset ? &p.offs : nullptr}; }

void put_f16(uint8_t* p, uint16_t h) {
    p[0] = (uint8_t) h;
    p[1] = (uint8_t) (h >> 8);
}

std::vector<uint8_t> random_rows(int t, int rows, int n_in, std::mt19937& rng) {
    const size_t rb = iq_row_bytes(t, n_in);
    const size_t bs = iq_block_bytes(t);
    std::vector<uint8_t> w((size_t) rows * rb);
    for (uint8_t& b : w) b = (uint8_t) (rng() & 0xFF);
    std::uniform_int_distribution<int> ex(7, 11);
    std::uniform_int_distribution<int> man(0, 1023);
    std::uniform_int_distribution<int> sgn(0, 3);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        put_f16(w.data() + o, h);
    }
    return w;
}

Bytes bf16_matrix(int rows, int cols, float scale, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<uint16_t> w((size_t) rows * (size_t) cols);
    for (uint16_t& v : w) v = strata::kernels::bf16_from_f32(scale * g(rng));
    return upload(w.data(), w.size() * 2u);
}

struct GrBuf {
    Bytes norm;
    Bytes down;
    Bytes up;
    Bytes inject;
};

GrBuf make_gr(int n_embd, int hc, int hc_lr, std::mt19937& rng) {
    const int dim = n_embd * hc;
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> norm((size_t) dim);
    for (float& v : norm) v = 1.0f + 0.05f * g(rng);
    return GrBuf{upload(norm.data(), norm.size() * 4u), bf16_matrix(hc_lr, dim, 0.01f, rng),
                 bf16_matrix(dim, hc_lr, 0.01f, rng), bf16_matrix(hc, dim, 0.01f, rng)};
}

TokenGr gr_view(const GrBuf& b) { return TokenGr{&b.norm, &b.down, &b.up, &b.inject}; }

struct LayerOwned {
    GrBuf attn;
    GrBuf ffn;
    PlaneBuf qkv;
    PlaneBuf gate;
    PlaneBuf outp;
    Bytes alpha;
    Bytes beta;
    Bytes conv;
    Bytes ssm_norm;
    Bytes dt;
    Bytes ssm_a;
    bool has_qsa;
    PlaneBuf pk;
    PlaneBuf pv;
    PlaneBuf pq;
    PlaneBuf pout;
    Bytes idx_k;
    Bytes idx_q;
    Bytes k_norm;
    Bytes q_norm;
    Bytes idx_k_norm;
    Bytes idx_q_norm;
    Bytes router;
    std::vector<Bytes> experts;
    Bytes sh_gc;
    Bytes sh_gs;
    Bytes sh_uc;
    Bytes sh_us;
    Bytes sh_dc;
    Bytes sh_ds;
    Bytes sh_ginp;
};

struct LayerStateBytes {
    Bytes gdn_state;
    Bytes gdn_conv;
    Bytes k_pool;
    Bytes v_pool;
    Bytes idx_tail;
    Bytes idx_dead;
    Bytes idx_pooled;
    Bytes idx_block_pos;
};

struct Owned {
    Bytes R;
    std::vector<LayerStateBytes> layers;
};

enum class Trap { None, SkipGdn, QsaAsGdn, ZeroGdn, ExpertZero, ForgetKv };

const int kNEmbd = 256;
const int kHc = 4;
const int kHcLr = 32;
const int kNExpert = 8;
const int kTop = 2;
const int kFf = 64;
const int kFfShared = 64;
const int kVocab = 32;
const int kHead = 24;
const int kHeadKv = 2;
const int kHeadDim = 256;
const int kRot = 64;
const int kIdxHeads = 4;
const int kIdxDim = 64;
const int kBlock = 4;
const int kIdxTop = 2;
const int kPage = 2;
const int kMaxCells = 4;
const int kPages = (kMaxCells + kPage - 1) / kPage;
const int kS = 128;
const int kHk = 2;
const int kHv = 4;
const int kDConv = 4;
const int kQk = kS * kHk;
const int kV = kS * kHv;
const int kC = kQk + kQk + kV;
const int kLayers = 4;
const int kInterval = 4;
const int kGuard = 16;
const int kSkipLayer = 1;

StackGeom geom_of() {
    StackGeom g;
    g.n_layers = kLayers;
    g.qsa_interval = kInterval;
    g.token.n_embd = kNEmbd;
    g.token.hc = kHc;
    g.token.hc_lr = kHcLr;
    g.token.n_expert = kNExpert;
    g.token.k = kTop;
    g.token.n_ff_shared = kFfShared;
    g.token.shared_tpr = 32;
    g.token.n_vocab = kVocab;
    g.token.qsa.n_embd = kNEmbd;
    g.token.qsa.max_cells = kMaxCells;
    g.token.qsa.qsa.n_head = kHead;
    g.token.qsa.qsa.n_head_kv = kHeadKv;
    g.token.qsa.qsa.head_dim = kHeadDim;
    g.token.qsa.qsa.n_rot = kRot;
    g.token.qsa.qsa.idx_n_head = kIdxHeads;
    g.token.qsa.qsa.idx_dim = kIdxDim;
    g.token.qsa.qsa.idx_block = kBlock;
    g.token.qsa.qsa.idx_top_k = kIdxTop;
    g.token.qsa.qsa.page_size = kPage;
    g.token.expert = IqExpert{kIq4Xs, kQ2_0, kNEmbd, kFf};
    g.gdn = GdnLayerGeom{kNEmbd, kS, kHk, kHv, kDConv};
    return g;
}

LayerOwned make_layer(bool qsa, std::mt19937& rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    LayerOwned L{make_gr(kNEmbd, kHc, kHcLr, rng),
                 make_gr(kNEmbd, kHc, kHcLr, rng),
                 make_plane(form_of(4, -8, false), kNEmbd, kC, rng, 0.05f),
                 make_plane(form_of(4, 0, true), kNEmbd, kV, rng, 0.05f),
                 make_plane(form_of(8, 0, true), kV, kNEmbd, rng, 0.05f),
                 bf16_matrix(kHv, kNEmbd, 0.05f, rng),
                 bf16_matrix(kHv, kNEmbd, 0.05f, rng),
                 Bytes((uint64_t) kC * (uint64_t) kDConv * 4u),
                 Bytes((uint64_t) kS * 4u),
                 Bytes((uint64_t) kHv * 4u),
                 Bytes((uint64_t) kHv * 4u),
                 qsa,
                 qsa ? make_plane(form_of(4, 0, true), kNEmbd, kHeadKv * kHeadDim, rng, 0.02f) : PlaneBuf{SForm{}, Bytes(16), Bytes(16), Bytes(16)},
                 qsa ? make_plane(form_of(4, 0, true), kNEmbd, kHeadKv * kHeadDim, rng, 0.02f) : PlaneBuf{SForm{}, Bytes(16), Bytes(16), Bytes(16)},
                 qsa ? make_plane(form_of(4, 0, true), kNEmbd, kHead * kHeadDim * 2, rng, 0.02f) : PlaneBuf{SForm{}, Bytes(16), Bytes(16), Bytes(16)},
                 qsa ? make_plane(form_of(4, 0, true), kHead * kHeadDim, kNEmbd, rng, 0.02f) : PlaneBuf{SForm{}, Bytes(16), Bytes(16), Bytes(16)},
                 qsa ? bf16_matrix(kIdxDim, kNEmbd, 0.05f, rng) : Bytes(16),
                 qsa ? bf16_matrix(kIdxHeads * kIdxDim, kNEmbd, 0.05f, rng) : Bytes(16),
                 Bytes((uint64_t) kHeadDim * 4u),
                 Bytes((uint64_t) kHeadDim * 4u),
                 Bytes((uint64_t) kIdxDim * 4u),
                 Bytes((uint64_t) kIdxDim * 4u),
                 bf16_matrix(kNExpert, kNEmbd, 0.02f, rng),
                 {},
                 Bytes(16),
                 Bytes(16),
                 Bytes(16),
                 Bytes(16),
                 Bytes(16),
                 Bytes(16),
                 Bytes(16)};

    auto* conv = static_cast<float*>(L.conv.data());
    auto* norm = static_cast<float*>(L.ssm_norm.data());
    auto* dt = static_cast<float*>(L.dt.data());
    auto* a = static_cast<float*>(L.ssm_a.data());
    for (int i = 0; i < kC * kDConv; ++i) conv[i] = 0.25f * gauss(rng);
    for (int i = 0; i < kS; ++i) norm[i] = 1.0f + 0.1f * gauss(rng);
    dt[0] = 22.0f;
    dt[1] = 120.0f;
    for (int h = 2; h < kHv; ++h) dt[h] = 0.2f * gauss(rng);
    for (int h = 0; h < kHv; ++h) a[h] = -(std::fabs(gauss(rng)) + 0.2f);

    if (qsa) {
        auto* kn = static_cast<float*>(L.k_norm.data());
        auto* qn = static_cast<float*>(L.q_norm.data());
        auto* ik = static_cast<float*>(L.idx_k_norm.data());
        auto* iq = static_cast<float*>(L.idx_q_norm.data());
        for (int d = 0; d < kHeadDim; ++d) {
            kn[d] = 1.0f + 0.1f * gauss(rng);
            qn[d] = 1.0f + 0.1f * gauss(rng);
        }
        for (int d = kRot; d < kHeadDim; ++d) {
            kn[d] = 0;
            qn[d] = 0;
        }
        for (int d = 0; d < kIdxDim; ++d) {
            ik[d] = 1.0f + 0.1f * gauss(rng);
            iq[d] = 1.0f + 0.1f * gauss(rng);
        }
    }

    const size_t blob_n = iq_expert_bytes(IqExpert{kIq4Xs, kQ2_0, kNEmbd, kFf});
    L.experts.reserve((size_t) kNExpert);
    for (int e = 0; e < kNExpert; ++e) {
        auto gate = random_rows(kIq4Xs, 2 * kFf, kNEmbd, rng);
        auto down = random_rows(kQ2_0, kNEmbd, kFf, rng);
        gate.insert(gate.end(), down.begin(), down.end());
        if (gate.size() != blob_n) throw MetalError("stack parity: expert blob size");
        L.experts.push_back(upload(gate.data(), gate.size()));
    }

    const size_t sh_gu = (size_t) kFfShared * (size_t) kNEmbd / 4u;
    const size_t sh_dn = (size_t) kNEmbd * (size_t) kFfShared / 4u;
    std::vector<uint8_t> gc(sh_gu, 0xFF), uc(sh_gu, 0x00), dc(sh_dn);
    for (uint8_t& v : dc) v = (uint8_t) (rng() & 0xFF);
    std::vector<float> gs((size_t) kFfShared * (size_t) (kNEmbd / 64));
    std::vector<float> us(gs.size()), ds((size_t) kNEmbd * (size_t) (kFfShared / 64));
    for (float& s : gs) s = 0.02f + 0.001f * (float) (rng() % 50);
    for (float& s : us) s = 0.02f + 0.001f * (float) (rng() % 50);
    // The routed experts dominate a 0.02 scale. This scale keeps a routed-slot change visible.
    for (float& s : ds) s = 8.0f + 0.1f * (float) (rng() % 50);
    std::vector<uint16_t> ginp((size_t) kNEmbd);
    for (uint16_t& v : ginp) v = strata::kernels::bf16_from_f32(0.05f * gauss(rng));
    L.sh_gc = upload(gc.data(), gc.size());
    L.sh_gs = upload(gs.data(), gs.size() * 4u);
    L.sh_uc = upload(uc.data(), uc.size());
    L.sh_us = upload(us.data(), us.size() * 4u);
    L.sh_dc = upload(dc.data(), dc.size());
    L.sh_ds = upload(ds.data(), ds.size() * 4u);
    L.sh_ginp = upload(ginp.data(), ginp.size() * 2u);
    return L;
}

StackLayerWeights view_layer(LayerOwned& L, const Bytes& cos, const Bytes& sin) {
    StackLayerWeights w;
    w.attn = gr_view(L.attn);
    w.ffn = gr_view(L.ffn);
    w.gdn.qkv = gdn_view(L.qkv);
    w.gdn.attn_gate = gdn_view(L.gate);
    w.gdn.ssm_out = gdn_view(L.outp);
    w.gdn.alpha = &L.alpha;
    w.gdn.beta = &L.beta;
    w.gdn.conv = &L.conv;
    w.gdn.ssm_norm = &L.ssm_norm;
    w.gdn.dt = &L.dt;
    w.gdn.ssm_a = &L.ssm_a;
    if (L.has_qsa) {
        w.qsa.k = qsa_view(L.pk);
        w.qsa.v = qsa_view(L.pv);
        w.qsa.q = qsa_view(L.pq);
        w.qsa.out = qsa_view(L.pout);
        w.qsa.idx_k = &L.idx_k;
        w.qsa.idx_q = &L.idx_q;
        w.qsa.k_norm = &L.k_norm;
        w.qsa.q_norm = &L.q_norm;
        w.qsa.idx_k_norm = &L.idx_k_norm;
        w.qsa.idx_q_norm = &L.idx_q_norm;
        w.qsa.cos_tab = &cos;
        w.qsa.sin_tab = &sin;
    }
    w.router = &L.router;
    w.experts = L.experts.data();
    w.sh_gate_codes = &L.sh_gc;
    w.sh_gate_scales = &L.sh_gs;
    w.sh_up_codes = &L.sh_uc;
    w.sh_up_scales = &L.sh_us;
    w.sh_down_codes = &L.sh_dc;
    w.sh_down_scales = &L.sh_ds;
    w.sh_gate_inp = &L.sh_ginp;
    return w;
}

LayerStateBytes make_layer_state() {
    const size_t pool_n = (size_t) kPages * (size_t) kHeadKv * (size_t) kPage * (size_t) kHeadDim;
    return LayerStateBytes{Bytes((uint64_t) kS * (uint64_t) kHv * (uint64_t) kS * 4u),
                           Bytes((uint64_t) kC * (uint64_t) (kDConv - 1) * 4u),
                           Bytes(pool_n * 2u),
                           Bytes(pool_n * 2u),
                           Bytes((size_t) (kBlock - 1) * (size_t) kIdxDim * 4u),
                           Bytes((size_t) kIdxDim * 4u),
                           Bytes((size_t) (kMaxCells / kBlock + 1) * (size_t) kIdxDim * 4u),
                           Bytes(4)};
}

void fill_gdn(LayerStateBytes& b, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    auto* s = static_cast<float*>(b.gdn_state.data());
    auto* c = static_cast<float*>(b.gdn_conv.data());
    const int ns = (int) (b.gdn_state.size() / 4u);
    const int nc = (int) (b.gdn_conv.size() / 4u);
    for (int i = 0; i < ns; ++i) s[i] = 0.02f * g(rng);
    for (int i = 0; i < nc; ++i) c[i] = 0.1f * g(rng);
    std::memset(b.k_pool.data(), 0, (size_t) b.k_pool.size());
    std::memset(b.v_pool.data(), 0, (size_t) b.v_pool.size());
    std::memset(b.idx_tail.data(), 0, (size_t) b.idx_tail.size());
    std::memset(b.idx_dead.data(), 0, (size_t) b.idx_dead.size());
    std::memset(b.idx_pooled.data(), 0, (size_t) b.idx_pooled.size());
    std::memset(b.idx_block_pos.data(), 0, (size_t) b.idx_block_pos.size());
}

void copy_layer_state(LayerStateBytes& dst, const LayerStateBytes& src) {
    copy_bytes(dst.gdn_state, src.gdn_state);
    copy_bytes(dst.gdn_conv, src.gdn_conv);
    copy_bytes(dst.k_pool, src.k_pool);
    copy_bytes(dst.v_pool, src.v_pool);
    copy_bytes(dst.idx_tail, src.idx_tail);
    copy_bytes(dst.idx_dead, src.idx_dead);
    copy_bytes(dst.idx_pooled, src.idx_pooled);
    copy_bytes(dst.idx_block_pos, src.idx_block_pos);
}

void copy_owned(Owned& dst, const Owned& src) {
    copy_bytes(dst.R, src.R);
    for (size_t i = 0; i < dst.layers.size(); ++i) copy_layer_state(dst.layers[i], src.layers[i]);
}

Owned make_owned() {
    Owned o{Bytes((uint64_t) kHc * (uint64_t) kNEmbd * 4u), {}};
    o.layers.reserve((size_t) kLayers);
    for (int i = 0; i < kLayers; ++i) o.layers.push_back(make_layer_state());
    std::memset(o.R.data(), 0, (size_t) o.R.size());
    return o;
}

StackLayerState bind_layer(LayerStateBytes& b, const Bytes& table, bool qsa) {
    StackLayerState s;
    s.gdn.state = &b.gdn_state;
    s.gdn.conv_state = &b.gdn_conv;
    if (qsa) {
        s.qsa.k_pool = &b.k_pool;
        s.qsa.v_pool = &b.v_pool;
        s.qsa.page_table = &table;
        s.qsa.idx_tail = &b.idx_tail;
        s.qsa.idx_dead = &b.idx_dead;
        s.qsa.idx_pooled = &b.idx_pooled;
        s.qsa.idx_block_pos = &b.idx_block_pos;
    }
    return s;
}

struct MixerCap {
    std::vector<float> mixer;
    std::vector<float> attn;
};

void run_spec(const StackGeom& g, const StackWeights& w, StackState& st, StackScratch& sc, int token, int pos,
              int pos_base, Trap trap, MixerCap* cap = nullptr) {
    if (trap == Trap::ZeroGdn) {
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            Bytes* s = st.layers[layer].gdn.state;
            Bytes* c = st.layers[layer].gdn.conv_state;
            if (s) std::memset(s->data(), 0, (size_t) s->size());
            if (c) std::memset(c->data(), 0, (size_t) c->size());
        }
    }
    if (trap == Trap::ForgetKv) {
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            if (!stack_is_qsa(g, layer)) continue;
            Bytes* k = st.layers[layer].qsa.k_pool;
            Bytes* v = st.layers[layer].qsa.v_pool;
            if (k) std::memset(k->data(), 0, (size_t) k->size());
            if (v) std::memset(v->data(), 0, (size_t) v->size());
        }
    }

    embedding_gather(*w.emb_codes, *w.emb_scales, w.emb_offsets, g.token.n_embd, g.token.emb_code_bits,
                     g.token.emb_code_bias, g.token.emb_group, token, sc.tok.emb, 0);
    auto* R = static_cast<float*>(st.R->data());
    const auto* emb = static_cast<const float*>(sc.tok.emb.data());
    for (int c = 0; c < (int) g.token.hc; ++c)
        std::memcpy(R + (size_t) c * (size_t) g.token.n_embd, emb, (size_t) g.token.n_embd * 4u);

    const GrShapes gs{g.token.n_embd, g.token.hc, g.token.hc_lr};
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        const bool qsa = stack_is_qsa(g, layer);
        if (trap == Trap::SkipGdn && !qsa && layer == kSkipLayer) continue;
        const StackLayerWeights& lw = w.layers[layer];
        StackLayerState& ls = st.layers[layer];
        gr_read(gs, *st.R, *lw.attn.norm, *lw.attn.down, *lw.attn.up, lw.attn.inject, 1e-6f, false, sc.tok.gr,
                sc.tok.mixed, sc.tok.inject);
        if (qsa && trap != Trap::QsaAsGdn)
            qsa_layer(g.token.qsa, lw.qsa, ls.qsa, sc.tok.qsa, pos, pos_base, sc.tok.mixed, sc.tok.block_out);
        else
            gdn_layer(g.gdn, lw.gdn, ls.gdn, sc.gdn, sc.tok.mixed, sc.tok.block_out);
        if (cap && qsa && trap != Trap::QsaAsGdn) {
            const auto* mix = static_cast<const float*>(sc.tok.block_out.data());
            cap->mixer.assign(mix, mix + g.token.n_embd);
            const auto* attn = static_cast<const float*>(sc.tok.qsa.attn.data());
            cap->attn.assign(attn, attn + (size_t) kHead * (size_t) kHeadDim);
        }
        gr_write(gs, *st.R, sc.tok.block_out, sc.tok.inject, *st.R);

        gr_read(gs, *st.R, *lw.ffn.norm, *lw.ffn.down, *lw.ffn.up, lw.ffn.inject, 1e-6f, false, sc.tok.gr,
                sc.tok.mixed, sc.tok.inject_ffn);
        f32_to_bf16_bulk(sc.tok.mixed, sc.tok.x_bf16, g.token.n_embd);
        bf16_gemv(sc.tok.x_bf16, *lw.router, sc.tok.router_logits, g.token.n_embd, g.token.n_expert);
        router_top10(sc.tok.router_logits, 1, g.token.n_expert, g.token.k, sc.tok.route_ids, sc.tok.route_w);
        if (trap == Trap::ExpertZero && layer == g.n_layers - 1) {
            int* ids = static_cast<int*>(sc.tok.route_ids.data());
            for (int e = 0; e < g.token.k; ++e) ids[e] = 0;
        }
        iq_quantize_q8_1(sc.tok.mixed, g.token.n_embd, sc.tok.x_q8_1);
        const int* ids = static_cast<const int*>(sc.tok.route_ids.data());
        auto* parts = static_cast<float*>(sc.tok.parts.data());
        for (int e = 0; e < g.token.k; ++e) {
            iq_expert(lw.experts[ids[e]], sc.tok.x_q8_1, g.token.expert, 1, sc.tok.one_expert);
            std::memcpy(parts + (size_t) e * (size_t) g.token.n_embd, sc.tok.one_expert.data(),
                        (size_t) g.token.n_embd * 4u);
        }
        quantize_q8_0(sc.tok.mixed, g.token.n_embd, sc.tok.x_q8_0);
        shared_expert(sc.tok.x_q8_0, sc.tok.x_bf16, *lw.sh_gate_codes, *lw.sh_gate_scales, *lw.sh_up_codes,
                      *lw.sh_up_scales, *lw.sh_down_codes, *lw.sh_down_scales, *lw.sh_gate_inp, sc.tok.sh_gate,
                      sc.tok.sh_up, sc.tok.sh_hq, sc.tok.sh_g, sc.tok.shared, g.token.n_embd, g.token.n_ff_shared,
                      g.token.shared_tpr);
        moe_combine(sc.tok.parts, sc.tok.route_w, &sc.tok.shared, sc.tok.block_out, (int) g.token.n_embd, g.token.k);
        gr_write(gs, *st.R, sc.tok.block_out, sc.tok.inject_ffn, *st.R);
    }
    gr_read(gs, *st.R, *w.head.norm, *w.head.down, *w.head.up, nullptr, 1e-6f, false, sc.tok.gr, sc.tok.mixed,
            sc.tok.inject);
    quantize_q8_k(sc.tok.mixed, g.token.n_embd, sc.tok.head_q8k);
    const Bytes* off = w.out.form.has_offset ? w.out.offset : nullptr;
    s_gemv_q8k_split(sc.tok.head_q8k, *w.out.codes, *w.out.scales, off, sc.tok.logits, g.token.n_embd,
                     g.token.n_vocab, w.out.form);
}

int count_diff_f(const Bytes& a, const Bytes& b) {
    const int n = (int) (a.size() / 4u);
    const auto* x = static_cast<const float*>(a.data());
    const auto* y = static_cast<const float*>(b.data());
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (x[i] != y[i]) ++bad;
    return bad;
}

}  // namespace

using namespace strata::metal;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-stack [--selftest]\n");
        return 2;
    }
    std::printf("stack: four layers, pack schedule, one greedy sample, synthetic weights\n\n");
    try {
        StackGeom sched;
        sched.n_layers = 8;
        sched.qsa_interval = 4;
        bool sched_ok = true;
        for (int64_t layer = 0; layer < 8; ++layer) {
            const bool want = layer == 3 || layer == 7;
            if (stack_is_qsa(sched, layer) != want) sched_ok = false;
        }
        sched.qsa_interval = 1;
        if (!stack_is_qsa(sched, 0) || !stack_is_qsa(sched, 5)) sched_ok = false;
        verdict("full attention is every interval's last layer", sched_ok, "");

        verdict("no layers is refused",
                throws_metal([] {
                    StackGeom g = geom_of();
                    g.n_layers = 0;
                    make_stack_scratch(g);
                }),
                "");
        verdict("a zero interval is refused",
                throws_metal([] {
                    StackGeom g = geom_of();
                    g.qsa_interval = 0;
                    make_stack_scratch(g);
                }),
                "");
        verdict("a GDN width that does not match is refused",
                throws_metal([] {
                    StackGeom g = geom_of();
                    g.gdn.n_embd = 512;
                    make_stack_scratch(g);
                }),
                "");

        const StackGeom g = geom_of();
        std::mt19937 rng(11u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        const int per_byte = 2;
        std::vector<uint8_t> emb_codes((size_t) kVocab * (size_t) (kNEmbd / per_byte));
        for (uint8_t& v : emb_codes) v = (uint8_t) (rng() & 0xFF);
        std::vector<float> emb_scales((size_t) kVocab * (size_t) (kNEmbd / 32));
        std::vector<float> emb_offs(emb_scales.size());
        for (float& v : emb_scales) v = 0.05f + 0.001f * (float) (rng() % 20);
        for (float& v : emb_offs) v = 0.02f * gauss(rng);
        Bytes b_emb_c = upload(emb_codes.data(), emb_codes.size());
        Bytes b_emb_s = upload(emb_scales.data(), emb_scales.size() * 4u);
        Bytes b_emb_o = upload(emb_offs.data(), emb_offs.size() * 4u);
        GrBuf head = make_gr(kNEmbd, kHc, kHcLr, rng);
        // The output mix has no injection. The buffer is still allocated so the struct stays uniform.
        PlaneBuf phead = make_plane(form_of(4, 0, true), kNEmbd, kVocab, rng, 0.02f);

        const int pairs = kRot / 2;
        std::vector<float> cos_tab((size_t) kMaxCells * (size_t) pairs), sin_tab(cos_tab.size());
        build_rope_table(kRot, 100.0, kMaxCells, cos_tab.data(), sin_tab.data());
        std::vector<int> table((size_t) kPages);
        for (int i = 0; i < kPages; ++i) table[(size_t) i] = kPages - 1 - i;
        Bytes b_cos = upload(cos_tab.data(), cos_tab.size() * 4u);
        Bytes b_sin = upload(sin_tab.data(), sin_tab.size() * 4u);
        Bytes b_table = upload(table.data(), table.size() * 4u);

        std::vector<LayerOwned> owned;
        owned.reserve((size_t) kLayers);
        for (int layer = 0; layer < kLayers; ++layer) owned.push_back(make_layer(stack_is_qsa(g, layer), rng));
        std::vector<StackLayerWeights> layer_w;
        layer_w.reserve((size_t) kLayers);
        for (LayerOwned& L : owned) layer_w.push_back(view_layer(L, b_cos, b_sin));

        StackWeights w;
        w.emb_codes = &b_emb_c;
        w.emb_scales = &b_emb_s;
        w.emb_offsets = &b_emb_o;
        w.head = TokenGr{&head.norm, &head.down, &head.up, nullptr};
        w.out = qsa_view(phead);
        w.layers = layer_w.data();

        Owned initial = make_owned();
        for (LayerStateBytes& b : initial.layers) fill_gdn(b, rng);
        Owned prod = make_owned();
        Owned spec = make_owned();
        Owned trap = make_owned();
        copy_owned(prod, initial);
        copy_owned(spec, initial);
        copy_owned(trap, initial);

        auto bind_all = [&](Owned& o) {
            std::vector<StackLayerState> layers;
            layers.reserve((size_t) kLayers);
            for (int i = 0; i < kLayers; ++i) layers.push_back(bind_layer(o.layers[(size_t) i], b_table, stack_is_qsa(g, i)));
            return layers;
        };
        // bind_all's vector must stay put. Named locals, not a temporary.
        std::vector<StackLayerState> prod_layers = bind_all(prod);
        std::vector<StackLayerState> spec_layers = bind_all(spec);
        std::vector<StackLayerState> trap_layers = bind_all(trap);
        StackState st_prod{&prod.R, prod_layers.data()};
        StackState st_spec{&spec.R, spec_layers.data()};
        StackState st_trap{&trap.R, trap_layers.data()};

        StackScratch sc_prod = make_stack_scratch(g);
        StackScratch sc_spec = make_stack_scratch(g);
        StackScratch sc_trap = make_stack_scratch(g);
        auto fill_guard = [&](Bytes& logits) {
            auto* p = static_cast<float*>(logits.data());
            for (int i = 0; i < kVocab + kGuard; ++i) p[i] = 7.0f;
        };
        fill_guard(sc_prod.tok.logits);
        fill_guard(sc_spec.tok.logits);

        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0;
        const int token0 = 3;
        const int id = generate_stack_token(g, w, st_prod, sc_prod, token0, 0, 0, sp, nullptr, 0);
        run_spec(g, w, st_spec, sc_spec, token0, 0, 0, Trap::None);
        const float* got = static_cast<const float*>(sc_prod.tok.logits.data());
        const float* want = static_cast<const float*>(sc_spec.tok.logits.data());
        int logit_bad = 0;
        double worst = 0;
        double scale = 0;
        for (int i = 0; i < kVocab; ++i) scale = std::max(scale, std::fabs((double) want[i]));
        if (scale < 1e-30) scale = 1e-30;
        for (int i = 0; i < kVocab; ++i) {
            if (got[i] != want[i]) ++logit_bad;
            const double rel = std::fabs((double) want[i] - (double) got[i]) / scale;
            if (rel > worst) worst = rel;
        }
        int guard_bad = 0;
        for (int i = 0; i < kGuard; ++i)
            if (got[kVocab + i] != 7.0f) ++guard_bad;
        const int want_id = argmax(want, kVocab);
        int id_bad = 0;
        const int* a = static_cast<const int*>(sc_prod.tok.route_ids.data());
        const int* b = static_cast<const int*>(sc_spec.tok.route_ids.data());
        for (int e = 0; e < kTop; ++e)
            if (a[e] != b[e]) ++id_bad;
        const int width = (int) strata::kernels::qsa_selection_width(1, g.token.qsa.qsa);
        int sel_bad = 0;
        const int* sa = static_cast<const int*>(sc_prod.tok.qsa.ids.data());
        const int* sb = static_cast<const int*>(sc_spec.tok.qsa.ids.data());
        for (int i = 0; i < width; ++i)
            if (sa[i] != sb[i]) ++sel_bad;
        const int state_bad = count_diff_f(prod.layers[0].gdn_state, spec.layers[0].gdn_state);
        const int conv_bad = count_diff_f(prod.layers[0].gdn_conv, spec.layers[0].gdn_conv);
        const int state_n = (int) (prod.layers[0].gdn_state.size() / 4u);
        const int moved = count_diff_f(prod.layers[0].gdn_state, initial.layers[0].gdn_state);
        const int qsa_gdn_moved = count_diff_f(prod.layers[3].gdn_state, initial.layers[3].gdn_state);

        char wb[64];
        std::snprintf(wb, sizeof wb, "rel %.3e, %d differ", worst, logit_bad);
        verdict("the logits match the kernel sequence", logit_bad == 0, wb);
        verdict("the last layer's routed ids match", id_bad == 0, std::to_string(id_bad) + " differ");
        verdict("the full-attention ids match", sel_bad == 0, std::to_string(sel_bad) + " differ");
        verdict("the greedy id is the argmax", id == want_id, "id " + std::to_string(id));
        verdict("16 words past the logits stayed put", guard_bad == 0, std::to_string(guard_bad) + " changed");
        verdict("the GDN state matches the kernel sequence", state_bad == 0 && conv_bad == 0,
                std::to_string(state_bad) + " state, " + std::to_string(conv_bad) + " conv");
        verdict("a GDN layer wrote its state", moved * 2 > state_n, std::to_string(moved) + " of " + std::to_string(state_n));
        verdict("the full-attention layer left the GDN state", qsa_gdn_moved == 0,
                std::to_string(qsa_gdn_moved) + " changed");

        std::vector<float> good(want, want + kVocab);
        auto trap_of = [&](Trap t, const char* name) {
            copy_owned(trap, initial);
            fill_guard(sc_trap.tok.logits);
            run_spec(g, w, st_trap, sc_trap, token0, 0, 0, t);
            const float* wrong = static_cast<const float*>(sc_trap.tok.logits.data());
            const double apart = rel_l1(good.data(), wrong, kVocab);
            verdict(name, apart > 0.05, pct(apart));
        };
        trap_of(Trap::SkipGdn, "skipping a GDN layer is observable");
        trap_of(Trap::QsaAsGdn, "running full attention as GDN is observable");
        trap_of(Trap::ZeroGdn, "the GDN state is observable");
        trap_of(Trap::ExpertZero, "routing the last layer to expert 0 is observable");

        const int id2 = generate_stack_token(g, w, st_prod, sc_prod, id, 1, 0, sp, nullptr, 0);
        run_spec(g, w, st_spec, sc_spec, id, 1, 0, Trap::None);
        const float* got2 = static_cast<const float*>(sc_prod.tok.logits.data());
        const float* want2 = static_cast<const float*>(sc_spec.tok.logits.data());
        int bad2 = 0;
        for (int i = 0; i < kVocab; ++i)
            if (got2[i] != want2[i]) ++bad2;
        const int want2_id = argmax(want2, kVocab);
        verdict("the second token matches the kernel sequence", bad2 == 0 && id2 == want2_id,
                "id " + std::to_string(id2) + ", " + std::to_string(bad2) + " logits differ");

        // One earlier cell is a small share of the softmax. Three earlier cells are the history the
        // pool is for. The output projection shrinks the gap, so this is the attention tensor.
        Owned carry = make_owned();
        Owned wiped = make_owned();
        copy_owned(carry, initial);
        std::vector<StackLayerState> carry_layers = bind_all(carry);
        std::vector<StackLayerState> wiped_layers = bind_all(wiped);
        StackState st_carry{&carry.R, carry_layers.data()};
        StackState st_wiped{&wiped.R, wiped_layers.data()};
        StackScratch sc_carry = make_stack_scratch(g);
        StackScratch sc_wiped = make_stack_scratch(g);
        const int hist[3] = {3, 5, 7};
        for (int p = 0; p < 3; ++p) run_spec(g, w, st_carry, sc_carry, hist[p], p, 0, Trap::None);
        copy_owned(wiped, carry);
        MixerCap kept;
        MixerCap forgotten;
        run_spec(g, w, st_carry, sc_carry, 9, 3, 0, Trap::None, &kept);
        run_spec(g, w, st_wiped, sc_wiped, 9, 3, 0, Trap::ForgetKv, &forgotten);
        const double pool_attn = rel_l1(kept.attn.data(), forgotten.attn.data(), kHead * kHeadDim);
        verdict("forgetting the KV pool is observable", pool_attn > 0.05, pct(pool_attn));

        verdict("a negative token is refused",
                throws_metal([&] { generate_stack_token(g, w, st_prod, sc_prod, -1, 0, 0, sp, nullptr, 0); }), "");
        verdict("a negative cell is refused",
                throws_metal([&] { generate_stack_token(g, w, st_prod, sc_prod, token0, -1, 0, sp, nullptr, 0); }), "");

        std::printf("\nstack: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("stack_parity OK\n");
    return 0;
}
