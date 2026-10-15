// One synthetic token: the spec calls each kernel itself, and generate_token has to land on it.
// --serve answers the engine lines on that same fixture. It does not load a model.
#include "strata/metal/token.hpp"

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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cerrno>
#include <csignal>
#include <cstdlib>

#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

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

using namespace strata::metal;

SForm affine4() {
    SForm f;
    f.code_bits = 4;
    f.code_bias = 0;
    f.group_elems = 32;
    f.codebook = Codebook::Affine;
    f.has_offset = true;
    return f;
}

struct PlaneBuf {
    SForm form;
    Bytes codes;
    Bytes scales;
    Bytes offs;
};

PlaneBuf make_plane(int n_in, int n_out, std::mt19937& rng, float scale = 0.01f) {
    const int per_byte = 2;
    std::vector<uint8_t> codes((size_t) n_out * (size_t) (n_in / per_byte));
    for (uint8_t& v : codes) v = (uint8_t) (rng() & 0xFF);
    const int n_groups = n_in / 32;
    std::vector<float> scales((size_t) n_out * (size_t) n_groups);
    std::vector<float> offs(scales.size());
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& v : scales) v = scale + 0.001f * (float) (rng() % 20);
    for (float& v : offs) v = 0.02f * g(rng);
    return PlaneBuf{affine4(), upload(codes.data(), codes.size()), upload(scales.data(), scales.size() * 4u),
                    upload(offs.data(), offs.size() * 4u)};
}

QsaPlane view(const PlaneBuf& p) { return QsaPlane{p.form, &p.codes, &p.scales, &p.offs}; }

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

GrBuf make_gr(int n_embd, int hc, int hc_lr, bool with_inject, std::mt19937& rng) {
    const int dim = n_embd * hc;
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> norm((size_t) dim);
    for (float& v : norm) v = 1.0f + 0.05f * g(rng);
    GrBuf b{upload(norm.data(), norm.size() * 4u), bf16_matrix(hc_lr, dim, 0.01f, rng),
            bf16_matrix(dim, hc_lr, 0.01f, rng), with_inject ? bf16_matrix(hc, dim, 0.01f, rng) : Bytes(16)};
    return b;
}

TokenGr gr_view(const GrBuf& b, bool with_inject) {
    return TokenGr{&b.norm, &b.down, &b.up, with_inject ? &b.inject : nullptr};
}

enum class Trap { None, OneStream, SkipAttnWrite, ExpertZero, ScaleShared };

struct Owned {
    Bytes R;
    Bytes k_pool;
    Bytes v_pool;
    Bytes idx_tail;
    Bytes idx_dead;
    Bytes idx_pooled;
    Bytes idx_block_pos;
};

void zero_owned(Owned& o) {
    std::memset(o.R.data(), 0, (size_t) o.R.size());
    std::memset(o.k_pool.data(), 0, (size_t) o.k_pool.size());
    std::memset(o.v_pool.data(), 0, (size_t) o.v_pool.size());
    std::memset(o.idx_tail.data(), 0, (size_t) o.idx_tail.size());
    std::memset(o.idx_dead.data(), 0, (size_t) o.idx_dead.size());
    std::memset(o.idx_pooled.data(), 0, (size_t) o.idx_pooled.size());
    std::memset(o.idx_block_pos.data(), 0, (size_t) o.idx_block_pos.size());
}

TokenState bind_state(Owned& o, const Bytes& table) {
    TokenState st;
    st.R = &o.R;
    st.qsa.k_pool = &o.k_pool;
    st.qsa.v_pool = &o.v_pool;
    st.qsa.page_table = &table;
    st.qsa.idx_tail = &o.idx_tail;
    st.qsa.idx_dead = &o.idx_dead;
    st.qsa.idx_pooled = &o.idx_pooled;
    st.qsa.idx_block_pos = &o.idx_block_pos;
    return st;
}

void run_block_spec(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int pos,
                    int pos_base, Trap trap) {
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    gr_read(gs, *st.R, *w.attn.norm, *w.attn.down, *w.attn.up, w.attn.inject, 1e-6f, false, sc.gr, sc.mixed,
            sc.inject);
    qsa_layer(g.qsa, w.qsa, st.qsa, sc.qsa, pos, pos_base, sc.mixed, sc.block_out);
    if (trap != Trap::SkipAttnWrite) gr_write(gs, *st.R, sc.block_out, sc.inject, *st.R);
    gr_read(gs, *st.R, *w.ffn.norm, *w.ffn.down, *w.ffn.up, w.ffn.inject, 1e-6f, false, sc.gr, sc.mixed,
            sc.inject_ffn);
    f32_to_bf16_bulk(sc.mixed, sc.x_bf16, g.n_embd);
    bf16_gemv(sc.x_bf16, *w.router, sc.router_logits, g.n_embd, g.n_expert);
    router_top10(sc.router_logits, 1, g.n_expert, g.k, sc.route_ids, sc.route_w);
    if (trap == Trap::ExpertZero) {
        int* ids = static_cast<int*>(sc.route_ids.data());
        for (int e = 0; e < g.k; ++e) ids[e] = 0;
    }
    iq_quantize_q8_1(sc.mixed, g.n_embd, sc.x_q8_1);
    const int* ids = static_cast<const int*>(sc.route_ids.data());
    auto* parts = static_cast<float*>(sc.parts.data());
    for (int e = 0; e < g.k; ++e) {
        iq_expert(w.experts[ids[e]], sc.x_q8_1, g.expert, 1, sc.one_expert);
        std::memcpy(parts + (size_t) e * (size_t) g.n_embd, sc.one_expert.data(), (size_t) g.n_embd * 4u);
    }
    quantize_q8_0(sc.mixed, g.n_embd, sc.x_q8_0);
    shared_expert(sc.x_q8_0, sc.x_bf16, *w.sh_gate_codes, *w.sh_gate_scales, *w.sh_up_codes, *w.sh_up_scales,
                  *w.sh_down_codes, *w.sh_down_scales, *w.sh_gate_inp, sc.sh_gate, sc.sh_up, sc.sh_hq, sc.sh_g,
                  sc.shared, g.n_embd, g.n_ff_shared, g.shared_tpr);
    if (trap == Trap::ScaleShared) {
        const float s = static_cast<const float*>(sc.route_w.data())[0];
        auto* sh = static_cast<float*>(sc.shared.data());
        for (int i = 0; i < (int) g.n_embd; ++i) sh[i] *= s;
    }
    moe_combine(sc.parts, sc.route_w, &sc.shared, sc.block_out, (int) g.n_embd, g.k);
    gr_write(gs, *st.R, sc.block_out, sc.inject_ffn, *st.R);
    gr_read(gs, *st.R, *w.head.norm, *w.head.down, *w.head.up, nullptr, 1e-6f, false, sc.gr, sc.mixed, sc.inject);
    quantize_q8_k(sc.mixed, g.n_embd, sc.head_q8k);
    const Bytes* off = w.out.form.has_offset ? w.out.offset : nullptr;
    s_gemv_q8k_split(sc.head_q8k, *w.out.codes, *w.out.scales, off, sc.logits, g.n_embd, g.n_vocab, w.out.form);
}

void run_spec(const TokenGeom& g, const TokenWeights& w, TokenState& st, TokenScratch& sc, int token, int pos,
              int pos_base, Trap trap) {
    embedding_gather(*w.emb_codes, *w.emb_scales, w.emb_offsets, g.n_embd, g.emb_code_bits, g.emb_code_bias,
                     g.emb_group, token, sc.emb, 0);
    auto* R = static_cast<float*>(st.R->data());
    const auto* emb = static_cast<const float*>(sc.emb.data());
    const int streams = trap == Trap::OneStream ? 1 : (int) g.hc;
    for (int c = 0; c < (int) g.hc; ++c) {
        if (c < streams) std::memcpy(R + (size_t) c * (size_t) g.n_embd, emb, (size_t) g.n_embd * 4u);
        else std::memset(R + (size_t) c * (size_t) g.n_embd, 0, (size_t) g.n_embd * 4u);
    }
    run_block_spec(g, w, st, sc, pos, pos_base, trap);
}

enum class DraftTrap { None, OneStream, SkipHidden };

void run_draft_spec(const TokenGeom& g, const TokenWeights& w, const DraftWeights& d, const Bytes& residual,
                    TokenState& st, TokenScratch& sc, int token, int pos, int pos_base, DraftTrap trap) {
    const int64_t n = g.n_embd;
    const uint64_t nbytes = (uint64_t) g.hc * (uint64_t) n * 4u;
    embedding_gather(*w.emb_codes, *w.emb_scales, w.emb_offsets, g.n_embd, g.emb_code_bits, g.emb_code_bias,
                     g.emb_group, token, sc.emb, 0);
    rms_norm_weighted(sc.emb, d.embedding.norm, 1, n, 1e-6f);
    Bytes q8(q8_0_bytes(n));
    Bytes ebuf((uint64_t) n * 4u);
    quantize_q8_0(sc.emb, n, q8);
    const Bytes* eoff = d.embedding.form.has_offset ? d.embedding.offset : nullptr;
    s_gemv_q8_0_split(q8, *d.embedding.codes, *d.embedding.scales, eoff, ebuf, n, n, d.embedding.form);

    Bytes hidden(nbytes);
    std::memcpy(hidden.data(), residual.data(), (size_t) nbytes);
    rms_norm_weighted(hidden, d.hidden.norm, 1, g.hc * n, 1e-6f);

    Bytes row((uint64_t) n * 4u);
    Bytes hbuf((uint64_t) n * 4u);
    auto* R = static_cast<float*>(st.R->data());
    const auto* src = static_cast<const float*>(hidden.data());
    const auto* e = static_cast<const float*>(ebuf.data());
    for (int64_t c = 0; c < g.hc; ++c) {
        const float* h = e;
        if (trap != DraftTrap::SkipHidden) {
            std::memcpy(row.data(), src + c * n, (size_t) n * 4u);
            quantize_q8_0(row, n, q8);
            const Bytes* hoff = d.hidden.form.has_offset ? d.hidden.offset : nullptr;
            s_gemv_q8_0_split(q8, *d.hidden.codes, *d.hidden.scales, hoff, hbuf, n, n, d.hidden.form);
            h = static_cast<const float*>(hbuf.data());
        }
        for (int64_t i = 0; i < n; ++i) {
            const float add = (trap == DraftTrap::OneStream && c > 0) ? 0.0f : e[i];
            R[c * n + i] = (trap == DraftTrap::SkipHidden ? 0.0f : h[i]) + add;
        }
    }
    run_block_spec(g, w, st, sc, pos, pos_base, Trap::None);
}

void fill_residual(Bytes& b, int kind, int64_t hc, int64_t n) {
    auto* r = static_cast<float*>(b.data());
    std::memset(r, 0, (size_t) hc * (size_t) n * sizeof(float));
    if (kind == 0) {
        for (int64_t c = 0; c < hc; ++c) {
            float* s = r + c * n;
            const float sign = (c % 2 == 0) ? 1.0f : -1.0f;
            for (int i = 0; i < 32; ++i) s[(int) c * 8 + i] = sign;
        }
    } else if (kind == 1) {
        float* s = r + n;
        for (int i = 0; i < 16; ++i) s[i] = 1.0f;
        for (int i = 64; i < 96; ++i) s[i] = -0.5f;
    }
}

int n_differ(const float* a, const float* b, int n) {
    int d = 0;
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i]) ++d;
    return d;
}

template <class F>
bool throws_with(F&& fn, const char* needle) {
    try {
        fn();
        return false;
    } catch (const strata::metal::MetalError& e) {
        return std::strstr(e.what(), needle) != nullptr;
    }
}

// The weights the token check draws, kept alive for a --serve process. Pointers in `w` alias these
// buffers, so the vectors do not grow after build_live returns.
struct Live {
    int n_embd = 256;
    int hc = 4;
    int hc_lr = 32;
    int n_expert = 8;
    int k = 2;
    int n_ff = 64;
    int n_ff_shared = 64;
    // Each shared down scale is base + 0.1 * (0..49). The small fixture uses 8 so that term shows
    // up in the sum. A wider shared expert keeps the same draw and a smaller base.
    float shared_down_base = 8.0f;
    // Added span on the shared down scales: base + span * (0..49). 0.1 is the original step.
    float shared_down_span = 0.1f;
    int n_vocab = 32;
    int n_head = 24;
    int n_head_kv = 2;
    int head_dim = 256;
    int n_rot = 64;
    int idx_n_head = 4;
    int idx_dim = 64;
    int block = 4;
    int top_k = 2;
    int page_size = 2;
    int max_cells = 4;
    int n_pages = 2;
    int guard = 16;
    TokenGeom g;
    std::mt19937 rng{11u};
    std::normal_distribution<float> gauss{0.0f, 1.0f};
    std::unique_ptr<Bytes> emb_c, emb_s, emb_o;
    std::unique_ptr<GrBuf> attn, ffn, head_gr;
    std::unique_ptr<PlaneBuf> pk, pv, pq, po, phead;
    std::unique_ptr<Bytes> idx_k, idx_q, kn, qn, ik, iq, cosb, sinb, table, router;
    std::unique_ptr<Bytes> gc, gs, uc, us, dc, ds, ginp;
    std::vector<Bytes> experts;
    TokenWeights w;
    SamplerParams sp;
};

Owned make_live_owned(const Live& L) {
    const size_t pool_n = (size_t) L.n_pages * (size_t) L.n_head_kv * (size_t) L.page_size * (size_t) L.head_dim;
    return Owned{Bytes((uint64_t) L.hc * (uint64_t) L.n_embd * 4u), Bytes(pool_n * 2u), Bytes(pool_n * 2u),
                 Bytes((size_t) (L.block - 1) * (size_t) L.idx_dim * 4u), Bytes((size_t) L.idx_dim * 4u),
                 Bytes((size_t) (L.max_cells / L.block + 1) * (size_t) L.idx_dim * 4u), Bytes(4)};
}

void build_live(Live& L) {
    L.g.n_embd = L.n_embd;
    L.g.hc = L.hc;
    L.g.hc_lr = L.hc_lr;
    L.g.n_expert = L.n_expert;
    L.g.k = L.k;
    L.g.n_ff_shared = L.n_ff_shared;
    L.g.shared_tpr = 32;
    L.g.n_vocab = L.n_vocab;
    L.g.qsa.n_embd = L.n_embd;
    L.g.qsa.max_cells = L.max_cells;
    L.g.qsa.qsa.n_head = L.n_head;
    L.g.qsa.qsa.n_head_kv = L.n_head_kv;
    L.g.qsa.qsa.head_dim = L.head_dim;
    L.g.qsa.qsa.n_rot = L.n_rot;
    L.g.qsa.qsa.idx_n_head = L.idx_n_head;
    L.g.qsa.qsa.idx_dim = L.idx_dim;
    L.g.qsa.qsa.idx_block = L.block;
    L.g.qsa.qsa.idx_top_k = L.top_k;
    L.g.qsa.qsa.page_size = L.page_size;
    L.g.expert = IqExpert{kIq4Xs, kQ2_0, L.n_embd, L.n_ff};
    L.n_pages = (L.max_cells + L.page_size - 1) / L.page_size;

    const int per_byte = 2;
    std::vector<uint8_t> emb_codes((size_t) L.n_vocab * (size_t) (L.n_embd / per_byte));
    for (uint8_t& v : emb_codes) v = (uint8_t) (L.rng() & 0xFF);
    std::vector<float> emb_scales((size_t) L.n_vocab * (size_t) (L.n_embd / 32));
    std::vector<float> emb_offs(emb_scales.size());
    for (float& v : emb_scales) v = 0.05f + 0.001f * (float) (L.rng() % 20);
    for (float& v : emb_offs) v = 0.02f * L.gauss(L.rng);
    L.emb_c.reset(new Bytes(upload(emb_codes.data(), emb_codes.size())));
    L.emb_s.reset(new Bytes(upload(emb_scales.data(), emb_scales.size() * 4u)));
    L.emb_o.reset(new Bytes(upload(emb_offs.data(), emb_offs.size() * 4u)));

    L.attn.reset(new GrBuf(make_gr(L.n_embd, L.hc, L.hc_lr, true, L.rng)));
    L.ffn.reset(new GrBuf(make_gr(L.n_embd, L.hc, L.hc_lr, true, L.rng)));
    L.head_gr.reset(new GrBuf(make_gr(L.n_embd, L.hc, L.hc_lr, false, L.rng)));
    L.pk.reset(new PlaneBuf(make_plane(L.n_embd, L.n_head_kv * L.head_dim, L.rng)));
    L.pv.reset(new PlaneBuf(make_plane(L.n_embd, L.n_head_kv * L.head_dim, L.rng)));
    L.pq.reset(new PlaneBuf(make_plane(L.n_embd, L.n_head * L.head_dim * 2, L.rng)));
    L.po.reset(new PlaneBuf(make_plane(L.n_head * L.head_dim, L.n_embd, L.rng)));
    L.phead.reset(new PlaneBuf(make_plane(L.n_embd, L.n_vocab, L.rng)));

    std::vector<uint16_t> w_idx_k((size_t) L.idx_dim * (size_t) L.n_embd);
    std::vector<uint16_t> w_idx_q((size_t) L.idx_n_head * (size_t) L.idx_dim * (size_t) L.n_embd);
    for (uint16_t& v : w_idx_k) v = strata::kernels::bf16_from_f32(0.05f * L.gauss(L.rng));
    for (uint16_t& v : w_idx_q) v = strata::kernels::bf16_from_f32(0.05f * L.gauss(L.rng));
    std::vector<float> w_kn((size_t) L.head_dim), w_qn((size_t) L.head_dim), w_ik((size_t) L.idx_dim),
        w_iq((size_t) L.idx_dim);
    for (float& v : w_kn) v = 1.0f + 0.1f * L.gauss(L.rng);
    for (float& v : w_qn) v = 1.0f + 0.1f * L.gauss(L.rng);
    for (float& v : w_ik) v = 1.0f + 0.1f * L.gauss(L.rng);
    for (float& v : w_iq) v = 1.0f + 0.1f * L.gauss(L.rng);
    for (int d = L.n_rot; d < L.head_dim; ++d) {
        w_kn[(size_t) d] = 0;
        w_qn[(size_t) d] = 0;
    }
    // The model indexer row is 128 wide and this block rotates 64. Zero the tail so the
    // dot sits in the rotated pairs. The 64-wide fixture has no tail, and this loop draws nothing.
    for (int d = L.n_rot; d < L.idx_dim; ++d) {
        w_ik[(size_t) d] = 0;
        w_iq[(size_t) d] = 0;
    }
    const int pairs = L.n_rot / 2;
    std::vector<float> cos_tab((size_t) L.max_cells * (size_t) pairs), sin_tab(cos_tab.size());
    build_rope_table(L.n_rot, 100.0, L.max_cells, cos_tab.data(), sin_tab.data());
    std::vector<int> page((size_t) L.n_pages);
    for (int i = 0; i < L.n_pages; ++i) page[(size_t) i] = L.n_pages - 1 - i;

    L.idx_k.reset(new Bytes(upload(w_idx_k.data(), w_idx_k.size() * 2u)));
    L.idx_q.reset(new Bytes(upload(w_idx_q.data(), w_idx_q.size() * 2u)));
    L.kn.reset(new Bytes(upload(w_kn.data(), w_kn.size() * 4u)));
    L.qn.reset(new Bytes(upload(w_qn.data(), w_qn.size() * 4u)));
    L.ik.reset(new Bytes(upload(w_ik.data(), w_ik.size() * 4u)));
    L.iq.reset(new Bytes(upload(w_iq.data(), w_iq.size() * 4u)));
    L.cosb.reset(new Bytes(upload(cos_tab.data(), cos_tab.size() * 4u)));
    L.sinb.reset(new Bytes(upload(sin_tab.data(), sin_tab.size() * 4u)));
    L.table.reset(new Bytes(upload(page.data(), page.size() * 4u)));
    L.router.reset(new Bytes(bf16_matrix(L.n_expert, L.n_embd, 0.02f, L.rng)));

    const size_t blob_n = iq_expert_bytes(L.g.expert);
    L.experts.reserve((size_t) L.n_expert);
    for (int e = 0; e < L.n_expert; ++e) {
        auto gate = random_rows(kIq4Xs, 2 * L.n_ff, L.n_embd, L.rng);
        auto down = random_rows(kQ2_0, L.n_embd, L.n_ff, L.rng);
        gate.insert(gate.end(), down.begin(), down.end());
        if (gate.size() != blob_n) throw MetalError("token parity: expert blob size");
        L.experts.push_back(upload(gate.data(), gate.size()));
    }

    const size_t sh_gu = (size_t) L.n_ff_shared * (size_t) L.n_embd / 4u;
    const size_t sh_dn = (size_t) L.n_embd * (size_t) L.n_ff_shared / 4u;
    std::vector<uint8_t> gc(sh_gu, 0xFF), uc(sh_gu, 0x00), dc(sh_dn);
    for (uint8_t& v : dc) v = (uint8_t) (L.rng() & 0xFF);
    std::vector<float> gsc((size_t) L.n_ff_shared * (size_t) (L.n_embd / 64));
    std::vector<float> usc(gsc.size()), dsc((size_t) L.n_embd * (size_t) (L.n_ff_shared / 64));
    for (float& s : gsc) s = 0.02f + 0.001f * (float) (L.rng() % 50);
    for (float& s : usc) s = 0.02f + 0.001f * (float) (L.rng() % 50);
    for (float& s : dsc) s = L.shared_down_base + L.shared_down_span * (float) (L.rng() % 50);
    std::vector<uint16_t> ginv((size_t) L.n_embd);
    for (uint16_t& v : ginv) v = strata::kernels::bf16_from_f32(0.05f * L.gauss(L.rng));
    L.gc.reset(new Bytes(upload(gc.data(), gc.size())));
    L.gs.reset(new Bytes(upload(gsc.data(), gsc.size() * 4u)));
    L.uc.reset(new Bytes(upload(uc.data(), uc.size())));
    L.us.reset(new Bytes(upload(usc.data(), usc.size() * 4u)));
    L.dc.reset(new Bytes(upload(dc.data(), dc.size())));
    L.ds.reset(new Bytes(upload(dsc.data(), dsc.size() * 4u)));
    L.ginp.reset(new Bytes(upload(ginv.data(), ginv.size() * 2u)));

    L.w.emb_codes = L.emb_c.get();
    L.w.emb_scales = L.emb_s.get();
    L.w.emb_offsets = L.emb_o.get();
    L.w.attn = gr_view(*L.attn, true);
    L.w.ffn = gr_view(*L.ffn, true);
    L.w.head = gr_view(*L.head_gr, false);
    L.w.qsa.k = view(*L.pk);
    L.w.qsa.v = view(*L.pv);
    L.w.qsa.q = view(*L.pq);
    L.w.qsa.out = view(*L.po);
    L.w.qsa.idx_k = L.idx_k.get();
    L.w.qsa.idx_q = L.idx_q.get();
    L.w.qsa.k_norm = L.kn.get();
    L.w.qsa.q_norm = L.qn.get();
    L.w.qsa.idx_k_norm = L.ik.get();
    L.w.qsa.idx_q_norm = L.iq.get();
    L.w.qsa.cos_tab = L.cosb.get();
    L.w.qsa.sin_tab = L.sinb.get();
    L.w.router = L.router.get();
    L.w.experts = L.experts.data();
    L.w.sh_gate_codes = L.gc.get();
    L.w.sh_gate_scales = L.gs.get();
    L.w.sh_up_codes = L.uc.get();
    L.w.sh_up_scales = L.us.get();
    L.w.sh_down_codes = L.dc.get();
    L.w.sh_down_scales = L.ds.get();
    L.w.sh_gate_inp = L.ginp.get();
    L.w.out = view(*L.phead);
    L.sp.greedy = true;
    L.sp.temperature = 0;
}

std::vector<int> greedy_ids(const Live& L, const std::vector<int>& prompt, int max_new) {
    Owned o = make_live_owned(L);
    zero_owned(o);
    TokenState st = bind_state(o, *L.table);
    TokenScratch sc = make_token_scratch(L.g);
    for (int i = 0; i + 1 < (int) prompt.size(); ++i)
        generate_token(L.g, L.w, st, sc, prompt[(size_t) i], i, 0, L.sp, nullptr, 0);
    std::vector<int> out;
    int tok = prompt.back();
    int pos = (int) prompt.size() - 1;
    for (int n = 0; n < max_new; ++n) {
        tok = generate_token(L.g, L.w, st, sc, tok, pos, 0, L.sp, nullptr, 0);
        out.push_back(tok);
        ++pos;
    }
    return out;
}

void say(const char* line) {
    std::fputs(line, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// One request. The prompt tokens fill the layer's own K and V. The sample at the last prompt
// position is the first new id. Each request starts from an empty sequence.
bool serve_request(const Live& L, int max_new, const std::vector<int>& ids, std::string& err) {
    if (max_new < 1) {
        err = "max_new";
        return false;
    }
    if (ids.empty()) {
        err = "the prompt is empty";
        return false;
    }
    for (int id : ids) {
        if (id < 0 || id >= L.n_vocab) {
            err = "a token id is outside the vocabulary";
            return false;
        }
    }
    const int last = (int) ids.size() + max_new - 2;
    if (last >= L.max_cells) {
        err = "the request does not fit in the context";
        return false;
    }
    Owned o = make_live_owned(L);
    zero_owned(o);
    TokenState st = bind_state(o, *L.table);
    TokenScratch sc = make_token_scratch(L.g);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i + 1 < (int) ids.size(); ++i)
        generate_token(L.g, L.w, st, sc, ids[(size_t) i], i, 0, L.sp, nullptr, 0);
    const auto t1 = std::chrono::steady_clock::now();
    int tok = ids.back();
    int pos = (int) ids.size() - 1;
    for (int n = 0; n < max_new; ++n) {
        tok = generate_token(L.g, L.w, st, sc, tok, pos, 0, L.sp, nullptr, 0);
        std::printf("T %d\n", tok);
        std::fflush(stdout);
        ++pos;
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("DONE %d %d %.3f %.3f length\n", max_new, (int) ids.size(), prompt_ms, decode_ms);
    std::fflush(stdout);
    return true;
}

bool parse_request(const std::string& line, int& max_new, std::vector<int>& ids, std::string& err) {
    if (line.rfind("GENI ", 0) == 0) {
        err = "this engine has no vision";
        return false;
    }
    if (line.rfind("GEN ", 0) != 0) {
        err = "expected: GEN <max_new> <id,id,...>";
        return false;
    }
    const char* p = line.c_str() + 4;
    char* end = nullptr;
    const long long mn = std::strtoll(p, &end, 10);
    if (end == p || mn > 0x7fffffff) {
        err = "max_new";
        return false;
    }
    max_new = (int) mn;
    p = end;
    for (;;) {
        while (*p == ' ') ++p;
        if (*p == '\0') break;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, p);
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = start;
            break;
        }
        const std::string key = tok.substr(0, eq);
        if (key == "temperature") {
            char* tend = nullptr;
            const float t = std::strtof(tok.c_str() + eq + 1, &tend);
            if (tend == tok.c_str() + eq + 1 || *tend != '\0' || t > 0.0f) {
                err = "this engine is greedy";
                return false;
            }
        } else {
            err = "this engine is greedy";
            return false;
        }
    }
    while (*p == ' ') ++p;
    ids.clear();
    if (*p == '\0') {
        err = "the prompt is empty";
        return false;
    }
    while (*p) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) {
            err = "bad id";
            return false;
        }
        ids.push_back((int) v);
        p = e;
        if (*p == ',') ++p;
        else if (*p != '\0') {
            err = "bad id";
            return false;
        }
    }
    return true;
}

int serve_loop(const Live& L) {
    std::printf("INFO vocab=%d n_embd=%d\n", L.n_vocab, L.n_embd);
    std::printf("READY %d\n", L.max_cells);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") return 0;
        if (line.rfind("VRAM", 0) == 0) {
            say("ERR this engine has no elastic cache");
            continue;
        }
        if (line == "STOP" || line.rfind("BGEN", 0) == 0 || line.rfind("BSTOP", 0) == 0) {
            say("ERR this engine runs one request at a time");
            continue;
        }
        int max_new = 0;
        std::vector<int> ids;
        std::string err;
        if (!parse_request(line, max_new, ids, err) || !serve_request(L, max_new, ids, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}

struct PipeChild {
    pid_t pid = -1;
    int in_fd = -1;
    int out_fd = -1;
    ~PipeChild() {
        if (in_fd >= 0) close(in_fd);
        if (out_fd >= 0) close(out_fd);
        if (pid > 0) {
            int status = 0;
            if (waitpid(pid, &status, WNOHANG) == 0) {
                kill(pid, SIGTERM);
                waitpid(pid, &status, 0);
            }
        }
    }
};

bool read_line(int fd, std::string& line, int timeout_ms) {
    line.clear();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        struct pollfd p;
        p.fd = fd;
        p.events = POLLIN;
        p.revents = 0;
        const int left = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                              deadline - std::chrono::steady_clock::now())
                              .count();
        const int pr = poll(&p, 1, left > 0 ? left : 0);
        if (pr < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (pr == 0) return false;
        char c = 0;
        const ssize_t n = read(fd, &c, 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        if (c == '\n') return true;
        if (c != '\r') line.push_back(c);
    }
    return false;
}

void write_line(int fd, const std::string& line) {
    const std::string s = line + "\n";
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = write(fd, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw MetalError("the engine pipe closed");
        }
        if (n == 0) throw MetalError("the engine pipe closed");
        off += (size_t) n;
    }
}

struct IgnorePipe {
    void (*prev)(int);
    IgnorePipe() : prev(std::signal(SIGPIPE, SIG_IGN)) {}
    ~IgnorePipe() { std::signal(SIGPIPE, prev); }
};

bool spawn_engine(const char* exe, PipeChild& child, std::string& err) {
    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0) {
        err = "pipe";
        return false;
    }
    if (pipe(out_pipe) != 0) {
        err = "pipe";
        close(in_pipe[0]);
        close(in_pipe[1]);
        return false;
    }
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        err = "file actions";
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        return false;
    }
    const int dup_in = posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
    const int dup_out = posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, in_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, in_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
    std::string exe_s(exe ? exe : "");
    std::vector<char> a0(exe_s.begin(), exe_s.end());
    a0.push_back('\0');
    char serve[] = "--serve";
    char ctx[] = "--max-context";
    char num[] = "8192";
    char* args[] = {a0.data(), serve, ctx, num, nullptr};
    pid_t pid = -1;
    const int rc = (dup_in == 0 && dup_out == 0) ? posix_spawnp(&pid, exe, &actions, nullptr, args, environ) : EINVAL;
    posix_spawn_file_actions_destroy(&actions);
    close(in_pipe[0]);
    close(out_pipe[1]);
    if (rc != 0) {
        close(in_pipe[1]);
        close(out_pipe[0]);
        err = "posix_spawnp failed";
        return false;
    }
    child.pid = pid;
    child.in_fd = in_pipe[1];
    child.out_fd = out_pipe[0];
    return true;
}

bool reap_child(PipeChild& child, int timeout_ms, int& status) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const pid_t r = waitpid(child.pid, &status, WNOHANG);
        if (r == child.pid) return true;
        if (r < 0) return false;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        poll(nullptr, 0, 20);
    }
}

bool parse_tid(const std::string& line, int& id) {
    if (line.rfind("T ", 0) != 0) return false;
    char* end = nullptr;
    const long v = std::strtol(line.c_str() + 2, &end, 10);
    if (end == line.c_str() + 2 || *end != '\0') return false;
    id = (int) v;
    return true;
}

// DONE carries generated, prompt length, both times, and the finish word. Nothing after that.
bool parse_done_line(const std::string& line, int expect_gen, int expect_prompt, std::string& detail) {
    int gen = 0, prompt = 0;
    double pms = 0, dms = 0;
    char finish[32] = {};
    if (std::sscanf(line.c_str(), "DONE %d %d %lf %lf %31s", &gen, &prompt, &pms, &dms, finish) != 5) {
        detail = line;
        return false;
    }
    const char* p = line.c_str();
    for (int i = 0; i < 6; ++i) {
        while (*p == ' ') ++p;
        if (*p == '\0') {
            detail = line;
            return false;
        }
        while (*p && *p != ' ') ++p;
    }
    while (*p == ' ') ++p;
    if (*p != '\0' || gen != expect_gen || prompt != expect_prompt || std::strcmp(finish, "length") != 0 || pms < 0.0 ||
        dms < 0.0) {
        detail = line;
        return false;
    }
    detail = "generated " + std::to_string(gen) + ", prompt " + std::to_string(prompt) + ", " + finish;
    return true;
}

bool run_gen(PipeChild& child, const std::string& req, const std::vector<int>& ids, int prompt_n, int first_ms,
             std::string& detail) {
    write_line(child.in_fd, req);
    std::string line;
    std::string shown;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!read_line(child.out_fd, line, i == 0 ? first_ms : 60000)) {
            detail = shown.empty() ? "no token line" : shown;
            return false;
        }
        int got = -1;
        if (!parse_tid(line, got) || got != ids[i]) {
            detail = line;
            return false;
        }
        if (!shown.empty()) shown += ",";
        shown += std::to_string(got);
    }
    if (!read_line(child.out_fd, line, 60000)) {
        detail = shown + " then no DONE";
        return false;
    }
    std::string done;
    if (!parse_done_line(line, (int) ids.size(), prompt_n, done)) {
        detail = shown + " " + line;
        return false;
    }
    detail = "id " + shown + ", " + done;
    return true;
}

bool run_err(PipeChild& child, const std::string& req, const char* needle, std::string& detail) {
    write_line(child.in_fd, req);
    std::string line;
    if (!read_line(child.out_fd, line, 60000)) {
        detail = "no reply";
        return false;
    }
    detail = line;
    return line.rfind("ERR ", 0) == 0 && std::strstr(line.c_str(), needle) != nullptr;
}

// The parent has already used Metal, so the engine is a new process. Its ids have to match a fresh
// draw of the same fixture, and a refused line has to leave that process up.
void check_serve(const char* exe) {
    IgnorePipe ignore_pipe;
    PipeChild child;
    try {
        Live live;
        build_live(live);
        const std::vector<int> one = greedy_ids(live, {3}, 1);
        const std::vector<int> two = greedy_ids(live, {3}, 2);
        const std::vector<int> pref = greedy_ids(live, {3, 5}, 1);
        std::string err;
        if (!spawn_engine(exe, child, err)) {
            verdict("the engine binary starts", false, err);
            return;
        }
        std::string line;
        const bool info = read_line(child.out_fd, line, 60000) && line == "INFO vocab=32 n_embd=256";
        std::string ready;
        const bool ready_ok = info && read_line(child.out_fd, ready, 60000) && ready == "READY 4";
        verdict("the engine prints INFO then READY 4", info && ready_ok, info ? ready : line);
        if (!ready_ok) return;

        std::string detail;
        const bool g1 = run_gen(child, "GEN 1 3", one, 1, 180000, detail);
        verdict("GEN 1 3 matches the in-process sample", g1, detail);
        if (!g1) return;
        const bool again = run_gen(child, "GEN 1 3", one, 1, 60000, detail);
        verdict("a second GEN 1 3 matches", again, detail);
        if (!again) return;
        const bool both = run_gen(child, "GEN 2 3", two, 1, 60000, detail);
        verdict("GEN 2 3 matches both samples", both, detail);
        if (!both) return;
        const bool prefix = run_gen(child, "GEN 1 3,5", pref, 2, 60000, detail);
        verdict("a prefix token matches the in-process sample", prefix, detail);
        if (!prefix) return;
        const bool greedy = run_gen(child, "GEN 1 temperature=0 3", one, 1, 60000, detail);
        verdict("temperature=0 stays greedy", greedy, detail);
        if (!greedy) return;

        const bool outside = run_err(child, "GEN 1 99", "outside the vocabulary", detail);
        verdict("an id outside the vocabulary is refused", outside, detail);
        if (!outside) return;
        const bool still = run_gen(child, "GEN 1 3", one, 1, 60000, detail);
        verdict("the engine still answers after a refusal", still, detail);
        if (!still) return;
        const bool fit = run_err(child, "GEN 5 3", "does not fit", detail);
        verdict("a request past the context is refused", fit, detail);
        if (!fit) return;
        const bool bad = run_err(child, "HELLO", "expected: GEN", detail);
        verdict("a line that is not GEN is refused", bad, detail);
        if (!bad) return;
        const bool warm = run_err(child, "GEN 1 temperature=0.7 3", "this engine is greedy", detail);
        verdict("temperature=0.7 is refused", warm, detail);
        if (!warm) return;
        const bool key = run_err(child, "GEN 1 top_p=0.9 3", "this engine is greedy", detail);
        verdict("another sampling key is refused", key, detail);
        if (!key) return;
        const bool vram = run_err(child, "VRAM 64", "no elastic cache", detail);
        verdict("VRAM is refused", vram, detail);
        if (!vram) return;
        const bool stop = run_err(child, "STOP", "one request at a time", detail);
        verdict("STOP is refused", stop, detail);
        if (!stop) return;
        const bool bgen = run_err(child, "BGEN 0 1 3", "one request at a time", detail);
        verdict("BGEN is refused", bgen, detail);
        if (!bgen) return;
        const bool bstop = run_err(child, "BSTOP 0", "one request at a time", detail);
        verdict("BSTOP is refused", bstop, detail);
        if (!bstop) return;
        const bool vision = run_err(child, "GENI 1 embed.bin", "no vision", detail);
        verdict("GENI is refused", vision, detail);
        if (!vision) return;

        write_line(child.in_fd, "QUIT");
        close(child.in_fd);
        child.in_fd = -1;
        int status = 0;
        const bool gone = reap_child(child, 10000, status);
        const bool clean = gone && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (gone) child.pid = -1;
        const int code = gone && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        verdict("QUIT ends the engine", clean, gone ? "exit " + std::to_string(code) : "still running");
    } catch (const MetalError& e) {
        verdict("the engine pipe stays open", false, e.what());
    }
}

// `tag` is the verdict prefix. The draft and the serve loop do not run here.
void check_live_token(Live& L, const char* tag) {
    Owned prod = make_live_owned(L);
    Owned spec = make_live_owned(L);
    Owned trap = make_live_owned(L);
    zero_owned(prod);
    zero_owned(spec);
    zero_owned(trap);
    TokenState st_prod = bind_state(prod, *L.table);
    TokenState st_spec = bind_state(spec, *L.table);
    TokenState st_trap = bind_state(trap, *L.table);
    TokenScratch sc_prod = make_token_scratch(L.g);
    TokenScratch sc_spec = make_token_scratch(L.g);
    TokenScratch sc_trap = make_token_scratch(L.g);

    const int n_vocab = L.n_vocab;
    const int guard = L.guard;
    auto fill_guard = [&](Bytes& logits) {
        auto* p = static_cast<float*>(logits.data());
        for (int i = 0; i < n_vocab + guard; ++i) p[i] = 7.0f;
    };
    fill_guard(sc_prod.logits);
    fill_guard(sc_spec.logits);

    const int token0 = 3;
    const int id = generate_token(L.g, L.w, st_prod, sc_prod, token0, 0, 0, L.sp, nullptr, 0);
    run_spec(L.g, L.w, st_spec, sc_spec, token0, 0, 0, Trap::None);
    const float* got = static_cast<const float*>(sc_prod.logits.data());
    const float* want = static_cast<const float*>(sc_spec.logits.data());
    int logit_bad = 0;
    double worst = 0;
    double scale = 0;
    for (int i = 0; i < n_vocab; ++i) scale = std::max(scale, std::fabs((double) want[i]));
    if (scale < 1e-30) scale = 1e-30;
    for (int i = 0; i < n_vocab; ++i) {
        if (got[i] != want[i]) ++logit_bad;
        const double rel = std::fabs((double) want[i] - (double) got[i]) / scale;
        if (rel > worst) worst = rel;
    }
    int guard_bad = 0;
    for (int i = 0; i < guard; ++i)
        if (got[n_vocab + i] != 7.0f) ++guard_bad;
    const int want_id = argmax(want, n_vocab);
    int id_bad = 0;
    const int* a = static_cast<const int*>(sc_prod.route_ids.data());
    const int* b = static_cast<const int*>(sc_spec.route_ids.data());
    for (int e = 0; e < L.k; ++e)
        if (a[e] != b[e]) ++id_bad;
    char wb[64];
    std::snprintf(wb, sizeof wb, "rel %.3e, %d differ", worst, logit_bad);
    const std::string prefix = std::string("the ") + tag + " ";
    verdict((prefix + "logits match the kernel sequence").c_str(), logit_bad == 0, wb);
    verdict((prefix + "routed ids match the kernel sequence").c_str(), id_bad == 0,
            std::to_string(id_bad) + " differ");
    verdict((prefix + "greedy id is the argmax").c_str(), id == want_id, "id " + std::to_string(id));
    verdict((std::string("16 words past the ") + tag + " logits stayed put").c_str(), guard_bad == 0,
            std::to_string(guard_bad) + " changed");

    std::vector<float> good(want, want + n_vocab);
    zero_owned(trap);
    fill_guard(sc_trap.logits);
    run_spec(L.g, L.w, st_trap, sc_trap, token0, 0, 0, Trap::SkipAttnWrite);
    const double apart = rel_l1(good.data(), static_cast<const float*>(sc_trap.logits.data()), n_vocab);
    verdict((std::string("skipping the ") + tag + " attention write is observable").c_str(), apart > 0.05, pct(apart));
}

// One greedy token at the model embedding. Its own seed. Rank, expert count, hidden size, and
// vocabulary stay on the small fixture.
void check_wide_token() {
    std::printf("\nmodel width  n_embd 2560  idx_dim 128\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    build_live(L);
    check_live_token(L, "wide");
}

// The same token at the model residual rank. Expert count, hidden size, and vocabulary stay put.
void check_rank_token() {
    std::printf("\nmodel rank  n_embd 2560  hc_lr 320\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    build_live(L);
    check_live_token(L, "rank");
}

// The same token with the model routed-expert hidden size. The shared expert stays at 64.
void check_hidden_token() {
    std::printf("\nmodel hidden  n_embd 2560  hc_lr 320  n_ff 640\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    L.n_ff = 640;
    build_live(L);
    check_live_token(L, "hidden");
}

// The same token at the model expert count. Shared hidden, vocabulary, and top-k stay put.
void check_expert_token() {
    std::printf("\nmodel experts  n_embd 2560  hc_lr 320  n_ff 640  n_expert 512\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    L.n_ff = 640;
    L.n_expert = 512;
    build_live(L);
    check_live_token(L, "expert");
}

// The same token at the model shared-expert hidden size. Vocabulary and top-k stay put.
void check_shared_token() {
    std::printf("\nmodel shared  n_embd 2560  n_ff_shared 640  n_expert 512\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    L.n_ff = 640;
    L.n_expert = 512;
    L.n_ff_shared = 640;
    // Hidden 640 with down scales from 8 left the attention write 4.07% apart. A base of 1
    // keeps that term in the sum without covering the write.
    L.shared_down_base = 1.0f;
    build_live(L);
    check_live_token(L, "shared");
}

// The same token at the model vocabulary. Top-k stays 2.
void check_vocab_token() {
    std::printf("\nmodel vocab  n_embd 2560  n_vocab 248320\n");
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    L.n_ff = 640;
    L.n_expert = 512;
    L.n_ff_shared = 640;
    L.shared_down_base = 1.0f;
    L.n_vocab = 248320;
    build_live(L);
    check_live_token(L, "vocab");
}

// Sixteen cells at that same shape. Top-k stays 2, so the last cell keeps 5 of 16.
// A fresh cache at top-k 2048 keeps all 16. The filled cache is not replayed: a second
// pass would append every cell again.
void check_long_token() {
    constexpr int kCells = 16;
    Live L;
    L.n_embd = 2560;
    L.idx_dim = 128;
    L.hc_lr = 320;
    L.n_ff = 640;
    L.n_expert = 512;
    L.n_ff_shared = 640;
    // Shared down scales 0.05 to 0.54, so the expert sum does not cover the value mix.
    L.shared_down_base = 0.05f;
    L.shared_down_span = 0.01f;
    L.n_vocab = 248320;
    L.max_cells = kCells;
    strata::kernels::QsaShapes shapes;
    shapes.idx_block = L.block;
    shapes.idx_top_k = L.top_k;
    const int w2 = (int) strata::kernels::qsa_selection_width(kCells, shapes);
    shapes.idx_top_k = 2048;
    const int w2048 = (int) strata::kernels::qsa_selection_width(kCells, shapes);
    std::printf("\nmodel sequence  cells %d  n_kv %d  top-k 2 width %d  top-k 2048 width %d\n", kCells, kCells, w2,
                w2048);
    build_live(L);
    // A zero key plane flattens the softmax. Random 4-bit value rows share a positive mean,
    // so every token lands on that same vector: raw attention was 0.22% apart and the logits
    // 0.01%. Bias -8 and paired nibbles 7 and 9 center each value row.
    std::memset(L.pk->codes.data(), 0, (size_t) L.pk->codes.size());
    std::memset(L.pk->scales.data(), 0, (size_t) L.pk->scales.size());
    std::memset(L.pk->offs.data(), 0, (size_t) L.pk->offs.size());
    L.w.qsa.v.form.code_bias = -8;
    {
        auto* codes = static_cast<uint8_t*>(L.pv->codes.data());
        const int row_bytes = L.n_embd / 2;
        const int n_out = L.n_head_kv * L.head_dim;
        for (int row = 0; row < n_out; ++row) {
            uint8_t* r = codes + (size_t) row * (size_t) row_bytes;
            for (int j = 0; j < row_bytes; ++j) {
                const uint32_t h = (uint32_t) row * 1315423911u ^ (uint32_t) j * 2654435761u;
                const uint8_t lo = (h & 1u) ? 7 : 9;
                const uint8_t hi = (h & 2u) ? 7 : 9;
                r[j] = (uint8_t) (lo | (hi << 4));
            }
        }
        auto* scales = static_cast<float*>(L.pv->scales.data());
        for (size_t i = 0; i < (size_t) L.pv->scales.size() / sizeof(float); ++i) scales[i] = 400.0f;
        auto* offs = static_cast<float*>(L.pv->offs.data());
        for (size_t i = 0; i < (size_t) L.pv->offs.size() / sizeof(float); ++i) offs[i] = 0.0f;
    }
    L.w.qsa.out.form.code_bias = -8;
    L.w.out.form.code_bias = -8;

    Owned prod = make_live_owned(L);
    Owned spec = make_live_owned(L);
    zero_owned(prod);
    zero_owned(spec);
    TokenState st_prod = bind_state(prod, *L.table);
    TokenState st_spec = bind_state(spec, *L.table);
    TokenScratch sc_prod = make_token_scratch(L.g);
    TokenScratch sc_spec = make_token_scratch(L.g);

    const int n_vocab = L.n_vocab;
    const int guard = L.guard;
    auto fill_guard = [&](Bytes& logits) {
        auto* p = static_cast<float*>(logits.data());
        for (int i = 0; i < n_vocab + guard; ++i) p[i] = 7.0f;
    };

    int logit_bad = 0;
    double worst = 0;
    int route_bad = 0;
    int argmax_bad = 0;
    int guard_bad = 0;
    std::vector<int> inputs((size_t) kCells);
    std::string in_list;
    for (int pos = 0; pos < kCells; ++pos) {
        inputs[(size_t) pos] = 3 + pos * 15000;
        if (!in_list.empty()) in_list.push_back(' ');
        in_list += std::to_string(inputs[(size_t) pos]);
    }
    std::printf("  inputs %s\n", in_list.c_str());
    std::string id_list;
    std::vector<float> last((size_t) n_vocab);
    for (int pos = 0; pos < kCells; ++pos) {
        fill_guard(sc_prod.logits);
        fill_guard(sc_spec.logits);
        const int id = generate_token(L.g, L.w, st_prod, sc_prod, inputs[(size_t) pos], pos, 0, L.sp, nullptr, 0);
        run_spec(L.g, L.w, st_spec, sc_spec, inputs[(size_t) pos], pos, 0, Trap::None);
        const float* got = static_cast<const float*>(sc_prod.logits.data());
        const float* want = static_cast<const float*>(sc_spec.logits.data());
        double scale = 0;
        for (int i = 0; i < n_vocab; ++i) scale = std::max(scale, std::fabs((double) want[i]));
        if (scale < 1e-30) scale = 1e-30;
        for (int i = 0; i < n_vocab; ++i) {
            if (got[i] != want[i]) ++logit_bad;
            const double rel = std::fabs((double) want[i] - (double) got[i]) / scale;
            if (rel > worst) worst = rel;
        }
        for (int i = 0; i < guard; ++i)
            if (got[n_vocab + i] != 7.0f) ++guard_bad;
        if (id != argmax(want, n_vocab)) ++argmax_bad;
        const int* a = static_cast<const int*>(sc_prod.route_ids.data());
        const int* b = static_cast<const int*>(sc_spec.route_ids.data());
        for (int e = 0; e < L.k; ++e)
            if (a[e] != b[e]) ++route_bad;
        if (!id_list.empty()) id_list.push_back(' ');
        id_list += std::to_string(id);
        if (pos + 1 == kCells) std::memcpy(last.data(), want, (size_t) n_vocab * 4u);
    }

    char wb[64];
    std::snprintf(wb, sizeof wb, "rel %.3e, %d differ", worst, logit_bad);
    verdict("the sequence logits match the kernel sequence", logit_bad == 0, wb);
    verdict("the sequence routed ids match the kernel sequence", route_bad == 0, std::to_string(route_bad) + " differ");
    verdict("the sequence greedy ids are the argmax", argmax_bad == 0, id_list);
    verdict("16 words past the sequence logits stayed put", guard_bad == 0, std::to_string(guard_bad) + " changed");

    TokenGeom wide = L.g;
    wide.qsa.qsa.idx_top_k = 2048;
    Owned kept = make_live_owned(L);
    zero_owned(kept);
    TokenState st_wide = bind_state(kept, *L.table);
    TokenScratch sc_wide = make_token_scratch(wide);
    for (int pos = 0; pos < kCells; ++pos)
        run_spec(wide, L.w, st_wide, sc_wide, inputs[(size_t) pos], pos, 0, Trap::None);
    const double apart = rel_l1(last.data(), static_cast<const float*>(sc_wide.logits.data()), n_vocab);
    std::string kept_ids;
    const int* narrow_ids = static_cast<const int*>(sc_spec.qsa.ids.data());
    for (int i = 0; i < w2; ++i) {
        if (i) kept_ids.push_back(' ');
        kept_ids += std::to_string(narrow_ids[i]);
    }
    verdict("top-k 2048 is observable", apart > 0.05, pct(apart) + ", top-k 2 kept " + kept_ids);
}

}  // namespace

using namespace strata::metal;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 2 && std::strcmp(argv[1], "--serve") == 0) {
        try {
            Live live;
            build_live(live);
            return serve_loop(live);
        } catch (const MetalError& e) {
            std::fprintf(stderr, "Metal error: %s\n", e.what());
            return 2;
        }
    }
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-token [--selftest|--serve]\n");
        return 2;
    }
    std::printf(
        "token: one block, one greedy sample, one draft step, one model-width token, one model-rank token, "
        "one model-hidden token, one model-expert token, one model-shared token, one model-vocab token, "
        "one model sequence, synthetic weights\n\n");
    try {
        const int n_embd = 256;
        const int hc = 4;
        const int hc_lr = 32;
        const int n_expert = 8;
        const int k = 2;
        const int n_ff = 64;
        const int n_ff_shared = 64;
        const int n_vocab = 32;
        const int n_head = 24;
        const int n_head_kv = 2;
        const int head_dim = 256;
        const int n_rot = 64;
        const int idx_n_head = 4;
        const int idx_dim = 64;
        const int r = 4;
        const int top_k = 2;
        const int page_size = 2;
        const int max_cells = 4;
        const int n_pages = (max_cells + page_size - 1) / page_size;
        const int guard = 16;

        TokenGeom g;
        g.n_embd = n_embd;
        g.hc = hc;
        g.hc_lr = hc_lr;
        g.n_expert = n_expert;
        g.k = k;
        g.n_ff_shared = n_ff_shared;
        g.shared_tpr = 32;
        g.n_vocab = n_vocab;
        g.qsa.n_embd = n_embd;
        g.qsa.max_cells = max_cells;
        g.qsa.qsa.n_head = n_head;
        g.qsa.qsa.n_head_kv = n_head_kv;
        g.qsa.qsa.head_dim = head_dim;
        g.qsa.qsa.n_rot = n_rot;
        g.qsa.qsa.idx_n_head = idx_n_head;
        g.qsa.qsa.idx_dim = idx_dim;
        g.qsa.qsa.idx_block = r;
        g.qsa.qsa.idx_top_k = top_k;
        g.qsa.qsa.page_size = page_size;
        g.expert = IqExpert{kIq4Xs, kQ2_0, n_embd, n_ff};

        std::mt19937 rng(11u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        const int per_byte = 2;
        std::vector<uint8_t> emb_codes((size_t) n_vocab * (size_t) (n_embd / per_byte));
        for (uint8_t& v : emb_codes) v = (uint8_t) (rng() & 0xFF);
        std::vector<float> emb_scales((size_t) n_vocab * (size_t) (n_embd / 32));
        std::vector<float> emb_offs(emb_scales.size());
        for (float& v : emb_scales) v = 0.05f + 0.001f * (float) (rng() % 20);
        for (float& v : emb_offs) v = 0.02f * gauss(rng);
        Bytes b_emb_c = upload(emb_codes.data(), emb_codes.size());
        Bytes b_emb_s = upload(emb_scales.data(), emb_scales.size() * 4u);
        Bytes b_emb_o = upload(emb_offs.data(), emb_offs.size() * 4u);

        GrBuf attn = make_gr(n_embd, hc, hc_lr, true, rng);
        GrBuf ffn = make_gr(n_embd, hc, hc_lr, true, rng);
        GrBuf head = make_gr(n_embd, hc, hc_lr, false, rng);

        PlaneBuf pk = make_plane(n_embd, n_head_kv * head_dim, rng);
        PlaneBuf pv = make_plane(n_embd, n_head_kv * head_dim, rng);
        PlaneBuf pq = make_plane(n_embd, n_head * head_dim * 2, rng);
        PlaneBuf po = make_plane(n_head * head_dim, n_embd, rng);
        PlaneBuf phead = make_plane(n_embd, n_vocab, rng);

        std::vector<uint16_t> w_idx_k((size_t) idx_dim * (size_t) n_embd);
        std::vector<uint16_t> w_idx_q((size_t) idx_n_head * (size_t) idx_dim * (size_t) n_embd);
        for (uint16_t& v : w_idx_k) v = strata::kernels::bf16_from_f32(0.05f * gauss(rng));
        for (uint16_t& v : w_idx_q) v = strata::kernels::bf16_from_f32(0.05f * gauss(rng));
        std::vector<float> w_kn((size_t) head_dim), w_qn((size_t) head_dim), w_ik((size_t) idx_dim),
            w_iq((size_t) idx_dim);
        for (float& v : w_kn) v = 1.0f + 0.1f * gauss(rng);
        for (float& v : w_qn) v = 1.0f + 0.1f * gauss(rng);
        for (float& v : w_ik) v = 1.0f + 0.1f * gauss(rng);
        for (float& v : w_iq) v = 1.0f + 0.1f * gauss(rng);
        for (int d = n_rot; d < head_dim; ++d) {
            w_kn[(size_t) d] = 0;
            w_qn[(size_t) d] = 0;
        }
        const int pairs = n_rot / 2;
        std::vector<float> cos_tab((size_t) max_cells * (size_t) pairs), sin_tab(cos_tab.size());
        build_rope_table(n_rot, 100.0, max_cells, cos_tab.data(), sin_tab.data());
        std::vector<int> table((size_t) n_pages);
        for (int i = 0; i < n_pages; ++i) table[(size_t) i] = n_pages - 1 - i;

        Bytes b_idx_k = upload(w_idx_k.data(), w_idx_k.size() * 2u);
        Bytes b_idx_q = upload(w_idx_q.data(), w_idx_q.size() * 2u);
        Bytes b_kn = upload(w_kn.data(), w_kn.size() * 4u);
        Bytes b_qn = upload(w_qn.data(), w_qn.size() * 4u);
        Bytes b_ik = upload(w_ik.data(), w_ik.size() * 4u);
        Bytes b_iq = upload(w_iq.data(), w_iq.size() * 4u);
        Bytes b_cos = upload(cos_tab.data(), cos_tab.size() * 4u);
        Bytes b_sin = upload(sin_tab.data(), sin_tab.size() * 4u);
        Bytes b_table = upload(table.data(), table.size() * 4u);
        Bytes b_router = bf16_matrix(n_expert, n_embd, 0.02f, rng);

        const size_t blob_n = iq_expert_bytes(g.expert);
        std::vector<Bytes> experts;
        experts.reserve((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) {
            auto gate = random_rows(kIq4Xs, 2 * n_ff, n_embd, rng);
            auto down = random_rows(kQ2_0, n_embd, n_ff, rng);
            gate.insert(gate.end(), down.begin(), down.end());
            if (gate.size() != blob_n) throw MetalError("token parity: expert blob size");
            experts.push_back(upload(gate.data(), gate.size()));
        }

        const size_t sh_gu = (size_t) n_ff_shared * (size_t) n_embd / 4u;
        const size_t sh_dn = (size_t) n_embd * (size_t) n_ff_shared / 4u;
        std::vector<uint8_t> gc(sh_gu, 0xFF), uc(sh_gu, 0x00), dc(sh_dn);
        for (uint8_t& v : dc) v = (uint8_t) (rng() & 0xFF);
        std::vector<float> gs((size_t) n_ff_shared * (size_t) (n_embd / 64));
        std::vector<float> us(gs.size()), ds((size_t) n_embd * (size_t) (n_ff_shared / 64));
        for (float& s : gs) s = 0.02f + 0.001f * (float) (rng() % 50);
        for (float& s : us) s = 0.02f + 0.001f * (float) (rng() % 50);
        // The routed experts dominate a 0.02 scale. This scale keeps the shared term visible in the sum.
        for (float& s : ds) s = 8.0f + 0.1f * (float) (rng() % 50);
        std::vector<uint16_t> ginp((size_t) n_embd);
        for (uint16_t& v : ginp) v = strata::kernels::bf16_from_f32(0.05f * gauss(rng));
        Bytes b_gc = upload(gc.data(), gc.size());
        Bytes b_gs = upload(gs.data(), gs.size() * 4u);
        Bytes b_uc = upload(uc.data(), uc.size());
        Bytes b_us = upload(us.data(), us.size() * 4u);
        Bytes b_dc = upload(dc.data(), dc.size());
        Bytes b_ds = upload(ds.data(), ds.size() * 4u);
        Bytes b_ginp = upload(ginp.data(), ginp.size() * 2u);

        TokenWeights w;
        w.emb_codes = &b_emb_c;
        w.emb_scales = &b_emb_s;
        w.emb_offsets = &b_emb_o;
        w.attn = gr_view(attn, true);
        w.ffn = gr_view(ffn, true);
        w.head = gr_view(head, false);
        w.qsa.k = view(pk);
        w.qsa.v = view(pv);
        w.qsa.q = view(pq);
        w.qsa.out = view(po);
        w.qsa.idx_k = &b_idx_k;
        w.qsa.idx_q = &b_idx_q;
        w.qsa.k_norm = &b_kn;
        w.qsa.q_norm = &b_qn;
        w.qsa.idx_k_norm = &b_ik;
        w.qsa.idx_q_norm = &b_iq;
        w.qsa.cos_tab = &b_cos;
        w.qsa.sin_tab = &b_sin;
        w.router = &b_router;
        w.experts = experts.data();
        w.sh_gate_codes = &b_gc;
        w.sh_gate_scales = &b_gs;
        w.sh_up_codes = &b_uc;
        w.sh_up_scales = &b_us;
        w.sh_down_codes = &b_dc;
        w.sh_down_scales = &b_ds;
        w.sh_gate_inp = &b_ginp;
        w.out = view(phead);

        const size_t pool_n = (size_t) n_pages * (size_t) n_head_kv * (size_t) page_size * (size_t) head_dim;
        auto make_owned = [&]() {
            return Owned{Bytes((uint64_t) hc * (uint64_t) n_embd * 4u), Bytes(pool_n * 2u), Bytes(pool_n * 2u),
                         Bytes((size_t) (r - 1) * (size_t) idx_dim * 4u), Bytes((size_t) idx_dim * 4u),
                         Bytes((size_t) (max_cells / r + 1) * (size_t) idx_dim * 4u), Bytes(4)};
        };
        Owned prod = make_owned();
        Owned spec = make_owned();
        Owned trap = make_owned();
        zero_owned(prod);
        zero_owned(spec);
        zero_owned(trap);
        TokenState st_prod = bind_state(prod, b_table);
        TokenState st_spec = bind_state(spec, b_table);
        TokenState st_trap = bind_state(trap, b_table);
        TokenScratch sc_prod = make_token_scratch(g);
        TokenScratch sc_spec = make_token_scratch(g);
        TokenScratch sc_trap = make_token_scratch(g);
        auto fill_guard = [&](Bytes& logits) {
            auto* p = static_cast<float*>(logits.data());
            for (int i = 0; i < n_vocab + guard; ++i) p[i] = 7.0f;
        };
        fill_guard(sc_prod.logits);
        fill_guard(sc_spec.logits);

        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0;
        const int token0 = 3;
        const int id = generate_token(g, w, st_prod, sc_prod, token0, 0, 0, sp, nullptr, 0);
        run_spec(g, w, st_spec, sc_spec, token0, 0, 0, Trap::None);
        const float* got = static_cast<const float*>(sc_prod.logits.data());
        const float* want = static_cast<const float*>(sc_spec.logits.data());
        int logit_bad = 0;
        double worst = 0;
        double scale = 0;
        for (int i = 0; i < n_vocab; ++i) scale = std::max(scale, std::fabs((double) want[i]));
        if (scale < 1e-30) scale = 1e-30;
        for (int i = 0; i < n_vocab; ++i) {
            if (got[i] != want[i]) ++logit_bad;
            const double rel = std::fabs((double) want[i] - (double) got[i]) / scale;
            if (rel > worst) worst = rel;
        }
        int guard_bad = 0;
        for (int i = 0; i < guard; ++i)
            if (got[n_vocab + i] != 7.0f) ++guard_bad;
        const int want_id = argmax(want, n_vocab);
        int id_bad = 0;
        const int* a = static_cast<const int*>(sc_prod.route_ids.data());
        const int* b = static_cast<const int*>(sc_spec.route_ids.data());
        for (int e = 0; e < k; ++e)
            if (a[e] != b[e]) ++id_bad;
        char wb[64];
        std::snprintf(wb, sizeof wb, "rel %.3e, %d differ", worst, logit_bad);
        verdict("the logits match the kernel sequence", logit_bad == 0, wb);
        verdict("the routed ids match the kernel sequence", id_bad == 0, std::to_string(id_bad) + " differ");
        verdict("the greedy id is the argmax", id == want_id, "id " + std::to_string(id));
        verdict("16 words past the logits stayed put", guard_bad == 0, std::to_string(guard_bad) + " changed");

        std::vector<float> good(want, want + n_vocab);
        auto trap_of = [&](Trap t, const char* name) {
            zero_owned(trap);
            fill_guard(sc_trap.logits);
            run_spec(g, w, st_trap, sc_trap, token0, 0, 0, t);
            const float* wrong = static_cast<const float*>(sc_trap.logits.data());
            const double apart = rel_l1(good.data(), wrong, n_vocab);
            verdict(name, apart > 0.05, pct(apart));
        };
        trap_of(Trap::OneStream, "one residual stream is observable");
        trap_of(Trap::SkipAttnWrite, "skipping the attention write is observable");
        trap_of(Trap::ExpertZero, "routing every slot to expert 0 is observable");
        {
            zero_owned(trap);
            run_spec(g, w, st_trap, sc_trap, token0, 0, 0, Trap::ScaleShared);
            const double apart = rel_l1(static_cast<const float*>(sc_spec.block_out.data()),
                                        static_cast<const float*>(sc_trap.block_out.data()), n_embd);
            verdict("weighting the shared expert is observable", apart > 0.05, pct(apart));
        }

        const int id2 = generate_token(g, w, st_prod, sc_prod, id, 1, 0, sp, nullptr, 0);
        run_spec(g, w, st_spec, sc_spec, id, 1, 0, Trap::None);
        const float* got2 = static_cast<const float*>(sc_prod.logits.data());
        const float* want2 = static_cast<const float*>(sc_spec.logits.data());
        int bad2 = 0;
        for (int i = 0; i < n_vocab; ++i)
            if (got2[i] != want2[i]) ++bad2;
        const int want2_id = argmax(want2, n_vocab);
        verdict("the second token matches the kernel sequence", bad2 == 0 && id2 == want2_id,
                "id " + std::to_string(id2) + ", " + std::to_string(bad2) + " logits differ");

        verdict("a negative token is refused",
                throws_metal([&] { generate_token(g, w, st_prod, sc_prod, -1, 0, 0, sp, nullptr, 0); }), "");
        TokenGeom bad = g;
        bad.n_embd = 200;
        bad.qsa.n_embd = 200;
        verdict("an embedding that is not a multiple of 256 is refused",
                throws_metal([&] { make_token_scratch(bad); }), "");

        PlaneBuf fc_e = make_plane(n_embd, n_embd, rng);
        PlaneBuf fc_h = make_plane(n_embd, n_embd, rng, 0.4f);
        std::vector<float> nemb((size_t) n_embd), nhid((size_t) hc * (size_t) n_embd);
        for (float& v : nemb) v = 1.0f + 0.05f * gauss(rng);
        for (float& v : nhid) v = 1.0f + 0.05f * gauss(rng);
        Bytes b_nemb = upload(nemb.data(), nemb.size() * 4u);
        Bytes b_nhid = upload(nhid.data(), nhid.size() * 4u);
        DraftWeights dw;
        dw.embedding = DraftFc{fc_e.form, &fc_e.codes, &fc_e.scales, &fc_e.offs, &b_nemb};
        dw.hidden = DraftFc{fc_h.form, &fc_h.codes, &fc_h.scales, &fc_h.offs, &b_nhid};

        Bytes res_a((uint64_t) hc * (uint64_t) n_embd * 4u);
        Bytes res_b((uint64_t) hc * (uint64_t) n_embd * 4u);
        fill_residual(res_a, 0, hc, n_embd);
        fill_residual(res_b, 1, hc, n_embd);
        std::vector<float> res_saved((size_t) hc * (size_t) n_embd);
        std::memcpy(res_saved.data(), res_a.data(), res_saved.size() * 4u);

        Owned d_prod = make_owned();
        Owned d_spec = make_owned();
        Owned d_trap = make_owned();
        Owned d_fresh = make_owned();
        zero_owned(d_prod);
        zero_owned(d_spec);
        zero_owned(d_trap);
        zero_owned(d_fresh);
        TokenState dst_prod = bind_state(d_prod, b_table);
        TokenState dst_spec = bind_state(d_spec, b_table);
        TokenState dst_trap = bind_state(d_trap, b_table);
        TokenState dst_fresh = bind_state(d_fresh, b_table);
        fill_guard(sc_prod.logits);
        fill_guard(sc_spec.logits);
        fill_guard(sc_trap.logits);

        const int draft_tok = 3;
        const int draft_id = draft_token(g, w, dw, res_a, dst_prod, sc_prod, draft_tok, 0, 0, sp, nullptr, 0);
        run_draft_spec(g, w, dw, res_a, dst_spec, sc_spec, draft_tok, 0, 0, DraftTrap::None);
        const float* dgot = static_cast<const float*>(sc_prod.logits.data());
        const float* dwant = static_cast<const float*>(sc_spec.logits.data());
        const int draft_bad = n_differ(dgot, dwant, n_vocab);
        int draft_guard = 0;
        for (int i = 0; i < guard; ++i)
            if (dgot[n_vocab + i] != 7.0f) ++draft_guard;
        const int draft_argmax = argmax(dwant, n_vocab);
        std::vector<float> draft_good(dwant, dwant + n_vocab);
        verdict("one draft matches the kernel sequence", draft_bad == 0 && draft_id == draft_argmax,
                "id " + std::to_string(draft_id) + ", " + std::to_string(draft_bad) + " of " +
                    std::to_string(n_vocab));
        verdict("16 words past the draft logits stayed put", draft_guard == 0,
                std::to_string(draft_guard) + " changed");
        const bool residual_kept = std::memcmp(res_a.data(), res_saved.data(), res_saved.size() * 4u) == 0;
        verdict("the caller's residual stayed put", residual_kept, "");

        const int cell_id = draft_token(g, w, dw, res_b, dst_prod, sc_prod, 5, 1, 0, sp, nullptr, 0);
        run_draft_spec(g, w, dw, res_b, dst_spec, sc_spec, 5, 1, 0, DraftTrap::None);
        const int cell_bad = n_differ(static_cast<const float*>(sc_prod.logits.data()),
                                      static_cast<const float*>(sc_spec.logits.data()), n_vocab);
        const int cell_argmax = argmax(static_cast<const float*>(sc_spec.logits.data()), n_vocab);
        verdict("a second draft cell matches the kernel sequence", cell_bad == 0 && cell_id == cell_argmax,
                "id " + std::to_string(cell_id) + ", " + std::to_string(cell_bad) + " of " +
                    std::to_string(n_vocab));

        zero_owned(d_fresh);
        fill_guard(sc_prod.logits);
        const int again_id = draft_token(g, w, dw, res_a, dst_fresh, sc_prod, draft_tok, 0, 0, sp, nullptr, 0);
        const int again_bad = n_differ(static_cast<const float*>(sc_prod.logits.data()), draft_good.data(), n_vocab);
        verdict("a second draw from a fresh state matches", again_bad == 0 && again_id == draft_id,
                std::to_string(again_bad) + " of " + std::to_string(n_vocab));

        zero_owned(d_fresh);
        fill_guard(sc_prod.logits);
        draft_token(g, w, dw, res_a, dst_fresh, sc_prod, 7, 0, 0, sp, nullptr, 0);
        const int tok_bad = n_differ(static_cast<const float*>(sc_prod.logits.data()), draft_good.data(), n_vocab);
        verdict("a different token moves the draft", tok_bad > 0,
                "differed on " + std::to_string(tok_bad) + " of " + std::to_string(n_vocab));

        zero_owned(d_fresh);
        fill_guard(sc_prod.logits);
        draft_token(g, w, dw, res_b, dst_fresh, sc_prod, draft_tok, 0, 0, sp, nullptr, 0);
        const int res_bad = n_differ(static_cast<const float*>(sc_prod.logits.data()), draft_good.data(), n_vocab);
        verdict("a different residual moves the draft", res_bad > 0,
                "differed on " + std::to_string(res_bad) + " of " + std::to_string(n_vocab));

        auto draft_trap = [&](DraftTrap t, const char* name) {
            zero_owned(d_trap);
            fill_guard(sc_trap.logits);
            run_draft_spec(g, w, dw, res_a, dst_trap, sc_trap, draft_tok, 0, 0, t);
            const double apart = rel_l1(draft_good.data(), static_cast<const float*>(sc_trap.logits.data()), n_vocab);
            verdict(name, apart > 0.05, pct(apart));
        };
        draft_trap(DraftTrap::OneStream, "adding the embedding to one stream is observable");
        draft_trap(DraftTrap::SkipHidden, "skipping the hidden projection is observable");

        verdict("a draft token outside the vocabulary is refused",
                throws_with([&] { draft_token(g, w, dw, res_a, dst_fresh, sc_prod, -1, 0, 0, sp, nullptr, 0); },
                            "outside the vocabulary") &&
                    throws_with([&] { draft_token(g, w, dw, res_a, dst_fresh, sc_prod, n_vocab, 0, 0, sp, nullptr, 0); },
                                "outside the vocabulary"),
                "");
        Bytes short_res(16);
        verdict("a short residual is refused",
                throws_with([&] { draft_token(g, w, dw, short_res, dst_fresh, sc_prod, draft_tok, 0, 0, sp, nullptr, 0); },
                            "residual is short"),
                "");
        DraftWeights missing = dw;
        missing.embedding.norm = nullptr;
        verdict("a missing embedding norm is refused",
                throws_with(
                    [&] { draft_token(g, w, missing, res_a, dst_fresh, sc_prod, draft_tok, 0, 0, sp, nullptr, 0); },
                    "fc_embedding norm is missing"),
                "");
        DraftWeights bits2 = dw;
        bits2.hidden.form.code_bits = 2;
        bits2.hidden.form.group_elems = 64;
        bits2.hidden.form.has_offset = false;
        verdict("a 2-bit hidden projection is refused",
                throws_with([&] { draft_token(g, w, bits2, res_a, dst_fresh, sc_prod, draft_tok, 0, 0, sp, nullptr, 0); },
                            "no Q8_0 contract"),
                "");

        check_wide_token();
        check_rank_token();
        check_hidden_token();
        check_expert_token();
        check_shared_token();
        check_vocab_token();
        check_long_token();

        check_serve(argv[0]);

        std::printf("\ntoken: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("token_parity OK\n");
    return 0;
}
