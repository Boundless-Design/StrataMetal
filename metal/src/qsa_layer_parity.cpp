// One QSA attention block. The projections, the rotation, the page table, the selection,
// and the gate each have to show up in the result. This is not a generate loop.
#include "strata/metal/qsa_layer.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/rope.hpp"
#include "strata/metal/sgemv.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;
using strata::kernels::bf16_from_f32;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

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

Bytes upload(const void* p, size_t n) {
    Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}

template <class F>
bool throws_metal(F&& fn) {
    try {
        fn();
        return false;
    } catch (const MetalError&) {
        return true;
    }
}

struct Plane {
    SForm form;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<float> offs;
};

SForm affine4() {
    SForm f;
    f.code_bits = 4;
    f.code_bias = 0;
    f.group_elems = 32;
    f.codebook = Codebook::Affine;
    f.has_offset = true;
    return f;
}

Plane make_plane(int n_in, int n_out, std::mt19937& rng) {
    Plane p;
    p.form = affine4();
    const int per_byte = 2;
    p.codes.resize((size_t) n_out * (size_t) (n_in / per_byte));
    for (uint8_t& v : p.codes) v = (uint8_t) (rng() & 0xFF);
    const int n_groups = n_in / 32;
    p.scales.resize((size_t) n_out * (size_t) n_groups);
    p.offs.resize(p.scales.size());
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& v : p.scales) v = 0.01f + 0.001f * (float) (rng() % 20);
    for (float& v : p.offs) v = 0.02f * g(rng);
    return p;
}

double weight_at(const Plane& p, int n_in, int row, int i) {
    const int per_byte = 2;
    const int codes_per_row = n_in / per_byte;
    const uint8_t* c = p.codes.data() + (size_t) row * (size_t) codes_per_row;
    const int code = (c[i / per_byte] >> ((i % per_byte) * 4)) & 15;
    const int g = i / 32;
    const int n_groups = n_in / 32;
    return (double) code * (double) p.scales[(size_t) row * (size_t) n_groups + (size_t) g] +
           (double) p.offs[(size_t) row * (size_t) n_groups + (size_t) g];
}

double q8k_at(const std::vector<uint8_t>& x, int i) {
    const uint8_t* blk = x.data() + (size_t) (i / 256) * 292u;
    float d;
    std::memcpy(&d, blk, 4);
    const int8_t q = (int8_t) blk[4 + (i % 256)];
    return (double) d * (double) q;
}

std::vector<float> gemv_ref(const Plane& p, const std::vector<uint8_t>& act, int n_in, int n_out) {
    std::vector<float> y((size_t) n_out);
    for (int row = 0; row < n_out; ++row) {
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += weight_at(p, n_in, row, i) * q8k_at(act, i);
        y[(size_t) row] = (float) acc;
    }
    return y;
}

std::vector<uint8_t> quantize(const std::vector<float>& x) {
    Bytes in = upload(x.data(), x.size() * 4u);
    Bytes blocks(q8k_bytes((int64_t) x.size()));
    quantize_q8_k(in, (int64_t) x.size(), blocks);
    return std::vector<uint8_t>((const uint8_t*) blocks.data(), (const uint8_t*) blocks.data() + blocks.size());
}

std::vector<float> bf16_ref(const std::vector<float>& x, const std::vector<uint16_t>& w, int n_in, int n_out) {
    std::vector<float> y((size_t) n_out);
    for (int row = 0; row < n_out; ++row) {
        double acc = 0;
        const uint16_t* wr = w.data() + (size_t) row * (size_t) n_in;
        for (int i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(bf16_from_f32(x[(size_t) i])) * (double) f32_from_bf16(wr[i]);
        y[(size_t) row] = (float) acc;
    }
    return y;
}

void rms_rows(std::vector<float>& x, const std::vector<float>& w, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float part[32];
        for (int i = 0; i < 32; ++i) part[i] = 0;
        float* row = x.data() + (size_t) r * (size_t) cols;
        for (int c = 0; c < cols; ++c) part[c & 31] += row[c] * row[c];
        for (int off = 16; off > 0; off >>= 1) {
            float next[32];
            for (int i = 0; i < 32; ++i) {
                const float other = (i + off < 32) ? part[i + off] : 0.0f;
                next[i] = part[i] + other;
            }
            for (int i = 0; i < 32; ++i) part[i] = next[i];
        }
        const float inv = 1.0f / std::sqrt(part[0] / (float) cols + 1e-6f);
        for (int c = 0; c < cols; ++c) row[c] = row[c] * w[(size_t) c] * inv;
    }
}

void rope_rows(std::vector<float>& x, int rows, int head_dim, int n_rot, int pos, const std::vector<float>& cos_tab,
               const std::vector<float>& sin_tab) {
    const int pairs = n_rot / 2;
    for (int r = 0; r < rows; ++r) {
        float* row = x.data() + (size_t) r * (size_t) head_dim;
        for (int i = 0; i < pairs; ++i) {
            const float a = row[i];
            const float b = row[pairs + i];
            const float c = cos_tab[(size_t) pos * (size_t) pairs + (size_t) i];
            const float s = sin_tab[(size_t) pos * (size_t) pairs + (size_t) i];
            const float ac = a * c;
            const float bs = b * s;
            const float as_ = a * s;
            const float bc = b * c;
            row[i] = ac - bs;
            row[pairs + i] = as_ + bc;
        }
    }
}

struct HostIdx {
    int dim = 0;
    int r = 0;
    int n_rot = 0;
    std::vector<float> tail;
    std::vector<float> dead;
    std::vector<float> pooled;
    std::vector<float> w;
    std::vector<float> cos_tab;
    std::vector<float> sin_tab;
};

void idx_spare(HostIdx& ix, const std::vector<float>& raw) {
    const int n = ix.dim;
    double ss = 0;
    for (int i = 0; i < n; ++i) {
        const double v = (double) raw[(size_t) i];
        const volatile double p = v * v;
        ss = ss + p;
    }
    const volatile double q = ss / (double) n;
    const volatile double arg = q + 1e-6;
    const volatile double root = std::sqrt(arg);
    const volatile double inv = 1.0 / root;
    const int pairs = ix.n_rot / 2;
    ix.dead.assign((size_t) n, 0);
    if (ix.pooled.size() < (size_t) n) ix.pooled.resize((size_t) n);
    for (int d = 0; d < n; ++d) {
        const volatile double a = (double) raw[(size_t) d] * inv;
        const volatile double b = a * (double) ix.w[(size_t) d];
        float y = (float) b;
        if (d < ix.n_rot) y *= ix.cos_tab[(size_t) (d % pairs)];
        ix.dead[(size_t) d] = y;
        ix.pooled[(size_t) d] = y;
    }
}

void idx_block(HostIdx& ix, const std::vector<float>& raw, int b, int pos_base) {
    const int n = ix.dim;
    const int r = ix.r;
    std::vector<float> row((size_t) n);
    for (int d = 0; d < n; ++d) {
        double m = 0;
        for (int j = 0; j < r - 1; ++j) {
            const volatile double v = (double) ix.tail[(size_t) j * (size_t) n + (size_t) d];
            m = m + v;
        }
        const volatile double rawd = (double) raw[(size_t) d];
        m = m + rawd;
        const volatile double div = m / (double) r;
        row[(size_t) d] = (float) div;
    }
    double ss = 0;
    for (int i = 0; i < n; ++i) {
        const double v = (double) row[(size_t) i];
        const volatile double p = v * v;
        ss = ss + p;
    }
    const volatile double q = ss / (double) n;
    const volatile double arg = q + 1e-6;
    const volatile double root = std::sqrt(arg);
    const volatile double inv = 1.0 / root;
    for (int d = 0; d < n; ++d) {
        const volatile double a = (double) row[(size_t) d] * inv;
        const volatile double bw = a * (double) ix.w[(size_t) d];
        row[(size_t) d] = (float) bw;
    }
    const int rotary = pos_base + b * r;
    const int pairs = ix.n_rot / 2;
    for (int i = 0; i < pairs; ++i) {
        const float a = row[(size_t) i];
        const float bb = row[(size_t) pairs + (size_t) i];
        const float c = ix.cos_tab[(size_t) rotary * (size_t) pairs + (size_t) i];
        const float s = ix.sin_tab[(size_t) rotary * (size_t) pairs + (size_t) i];
        const float ac = a * c;
        const float bs = bb * s;
        const float as_ = a * s;
        const float bc = bb * c;
        row[(size_t) i] = ac - bs;
        row[(size_t) pairs + (size_t) i] = as_ + bc;
    }
    const size_t need = (size_t) (b + 2) * (size_t) n;
    if (ix.pooled.size() < need) ix.pooled.resize(need);
    std::memcpy(ix.pooled.data() + (size_t) b * (size_t) n, row.data(), (size_t) n * 4u);
    std::memcpy(ix.pooled.data() + (size_t) (b + 1) * (size_t) n, ix.dead.data(), (size_t) n * 4u);
}

void idx_append(HostIdx& ix, const std::vector<float>& raw, int cell, int pos_base) {
    const int slot = cell % ix.r;
    if (slot < ix.r - 1)
        std::memcpy(ix.tail.data() + (size_t) slot * (size_t) ix.dim, raw.data(), (size_t) ix.dim * 4u);
    if (cell == 0) idx_spare(ix, raw);
    if (slot == ix.r - 1) idx_block(ix, raw, cell / ix.r, pos_base);
}

std::vector<float> index_scores(const HostIdx& ix, const std::vector<float>& q_idx, int n_head, int n_kv) {
    const int n_bid = n_kv / ix.r;
    std::vector<float> scores((size_t) n_kv);
    for (int j = 0; j < n_kv; ++j) {
        const int cb = std::min(j / ix.r, n_bid);
        double sum = 0;
        for (int h = 0; h < n_head; ++h) {
            double dot = 0;
            const float* row = ix.pooled.data() + (size_t) cb * (size_t) ix.dim;
            const float* q = q_idx.data() + (size_t) h * (size_t) ix.dim;
            for (int d = 0; d < ix.dim; ++d) dot += (double) row[d] * (double) q[d];
            if (dot > 0) sum += dot;
        }
        float sc = (float) sum;
        if (j >= n_bid * ix.r) sc += 1e9f;
        scores[(size_t) j] = sc;
    }
    return scores;
}

std::vector<int> select_ids(const std::vector<float>& scores, int top_k, int r) {
    const int n = (int) scores.size();
    const int width = std::min(n, top_k + r - 1);
    std::vector<int> idx(n);
    for (int j = 0; j < n; ++j) idx[(size_t) j] = j;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float sa = scores[(size_t) a];
        const float sb = scores[(size_t) b];
        if (sa != sb) return sa > sb;
        return a < b;
    });
    idx.resize((size_t) width);
    std::sort(idx.begin(), idx.end());
    return idx;
}

std::vector<float> attend(const std::vector<float>& q, const std::vector<uint16_t>& k, const std::vector<uint16_t>& v,
                          const std::vector<int>& ids, int n_head, int n_head_kv, int head_dim, bool divide) {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    const int group = n_head / n_head_kv;
    std::vector<float> out((size_t) n_head * (size_t) head_dim, 0);
    for (int h = 0; h < n_head; ++h) {
        const int kv = divide ? h / group : h % n_head_kv;
        std::vector<double> sc(ids.size());
        for (size_t j = 0; j < ids.size(); ++j) {
            double dot = 0;
            for (int d = 0; d < head_dim; ++d) {
                const size_t at = ((size_t) ids[j] * (size_t) n_head_kv + (size_t) kv) * (size_t) head_dim + (size_t) d;
                dot += (double) f32_from_f16(k[at]) * (double) q[(size_t) h * (size_t) head_dim + (size_t) d];
            }
            sc[j] = (double) scale * dot;
        }
        const double mx = *std::max_element(sc.begin(), sc.end());
        double sum = 0;
        for (double& s : sc) {
            s = std::exp(s - mx);
            sum += s;
        }
        for (double& s : sc) s /= sum;
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0;
            for (size_t j = 0; j < ids.size(); ++j) {
                const size_t at = ((size_t) ids[j] * (size_t) n_head_kv + (size_t) kv) * (size_t) head_dim + (size_t) d;
                acc += sc[j] * (double) f32_from_f16(v[at]);
            }
            out[(size_t) h * (size_t) head_dim + (size_t) d] = (float) acc;
        }
    }
    return out;
}

std::vector<float> gate_of(const std::vector<float>& attn, const std::vector<float>& q_full, int n_head, int head_dim,
                           bool second) {
    std::vector<float> out((size_t) n_head * (size_t) head_dim);
    const int half = second ? head_dim : 0;
    for (int h = 0; h < n_head; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            const float g = q_full[(size_t) h * (size_t) head_dim * 2u + (size_t) half + (size_t) d];
            const float s = 1.0f / (1.0f + std::exp(-g));
            out[(size_t) h * (size_t) head_dim + (size_t) d] = attn[(size_t) h * (size_t) head_dim + (size_t) d] * s;
        }
    }
    return out;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

double worst_rel(const std::vector<float>& want, const float* got) {
    double worst = 0;
    double scale = 0;
    for (float v : want) scale = std::max(scale, std::fabs((double) v));
    if (scale < 1e-30) scale = 1e-30;
    for (size_t i = 0; i < want.size(); ++i) {
        const double rel = std::fabs((double) want[i] - (double) got[i]) / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

// The first call is the block already measured. The second is the model embedding and the
// model indexer width. Top-k stays 2: eight cells do not exercise the artifact's 2048.
void check_block(const char* title, int n_embd, int idx_dim, bool refusals) {
    std::printf("%s  n_embd %d  idx_dim %d\n", title, n_embd, idx_dim);
    QsaLayerGeom g;
    g.n_embd = n_embd;
    g.max_cells = 8;
    g.qsa.n_head = 24;
    g.qsa.n_head_kv = 2;
    g.qsa.head_dim = 256;
    g.qsa.n_rot = 64;
    g.qsa.idx_n_head = 4;
    g.qsa.idx_dim = idx_dim;
    g.qsa.idx_block = 4;
    g.qsa.idx_top_k = 2;
    g.qsa.page_size = 4;
    const int n_head = 24;
    const int n_head_kv = 2;
    const int head_dim = 256;
    const int n_rot = 64;
    const int idx_n_head = 4;
    const int r = 4;
        const int top_k = 2;
        const int page_size = 4;
        const int tokens = 8;
        const int pos_base = 5;
        const int n_pages = (tokens + page_size - 1) / page_size;
        const int pairs = n_rot / 2;
        const int table_rows = pos_base + tokens;

        std::mt19937 rng(11u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        Plane pk = make_plane(n_embd, n_head_kv * head_dim, rng);
        Plane pv = make_plane(n_embd, n_head_kv * head_dim, rng);
        Plane pq = make_plane(n_embd, n_head * head_dim * 2, rng);
        Plane po = make_plane(n_head * head_dim, n_embd, rng);

        std::vector<uint16_t> w_idx_k((size_t) idx_dim * (size_t) n_embd);
        std::vector<uint16_t> w_idx_q((size_t) idx_n_head * (size_t) idx_dim * (size_t) n_embd);
        for (uint16_t& v : w_idx_k) v = bf16_from_f32(0.05f * gauss(rng));
        for (uint16_t& v : w_idx_q) v = bf16_from_f32(0.05f * gauss(rng));
        std::vector<float> w_kn((size_t) head_dim), w_qn((size_t) head_dim), w_ik((size_t) idx_dim), w_iq((size_t) idx_dim);
        for (float& v : w_kn) v = 1.0f + 0.1f * gauss(rng);
        for (float& v : w_qn) v = 1.0f + 0.1f * gauss(rng);
        // The decode head is 256 wide and this fixture rotates 64. Zero the tail weights so the
        // dot lives in the rotated pairs; otherwise a skipped rotation does not move attention.
        for (int d = n_rot; d < head_dim; ++d) {
            w_kn[(size_t) d] = 0;
            w_qn[(size_t) d] = 0;
        }
        for (float& v : w_ik) v = 1.0f + 0.1f * gauss(rng);
        for (float& v : w_iq) v = 1.0f + 0.1f * gauss(rng);
        // The model indexer row is 128 wide and this block rotates 64. Zero the tail so the
        // dot sits in the rotated pairs. The 64-wide fixture has no tail.
        for (int d = n_rot; d < idx_dim; ++d) {
            w_ik[(size_t) d] = 0;
            w_iq[(size_t) d] = 0;
        }

        std::vector<float> cos_tab((size_t) table_rows * (size_t) pairs), sin_tab(cos_tab.size());
        // 1e7 barely turns across eight consecutive positions, so a skipped rotation stays under the
        // bar. This table uses 100 so neighboring cells land on different angles.
        build_rope_table(n_rot, 100.0, table_rows, cos_tab.data(), sin_tab.data());

        std::vector<int> table((size_t) n_pages);
        for (int i = 0; i < n_pages; ++i) table[(size_t) i] = n_pages - 1 - i;
        const size_t pool_n = (size_t) n_pages * (size_t) n_head_kv * (size_t) page_size * (size_t) head_dim;

        Bytes codes_k = upload(pk.codes.data(), pk.codes.size());
        Bytes scales_k = upload(pk.scales.data(), pk.scales.size() * 4u);
        Bytes offs_k = upload(pk.offs.data(), pk.offs.size() * 4u);
        Bytes codes_v = upload(pv.codes.data(), pv.codes.size());
        Bytes scales_v = upload(pv.scales.data(), pv.scales.size() * 4u);
        Bytes offs_v = upload(pv.offs.data(), pv.offs.size() * 4u);
        Bytes codes_q = upload(pq.codes.data(), pq.codes.size());
        Bytes scales_q = upload(pq.scales.data(), pq.scales.size() * 4u);
        Bytes offs_q = upload(pq.offs.data(), pq.offs.size() * 4u);
        Bytes codes_o = upload(po.codes.data(), po.codes.size());
        Bytes scales_o = upload(po.scales.data(), po.scales.size() * 4u);
        Bytes offs_o = upload(po.offs.data(), po.offs.size() * 4u);
        Bytes b_idx_k = upload(w_idx_k.data(), w_idx_k.size() * 2u);
        Bytes b_idx_q = upload(w_idx_q.data(), w_idx_q.size() * 2u);
        Bytes b_kn = upload(w_kn.data(), w_kn.size() * 4u);
        Bytes b_qn = upload(w_qn.data(), w_qn.size() * 4u);
        Bytes b_ik = upload(w_ik.data(), w_ik.size() * 4u);
        Bytes b_iq = upload(w_iq.data(), w_iq.size() * 4u);
        Bytes b_cos = upload(cos_tab.data(), cos_tab.size() * 4u);
        Bytes b_sin = upload(sin_tab.data(), sin_tab.size() * 4u);
        Bytes b_table = upload(table.data(), table.size() * 4u);
        Bytes k_pool(pool_n * 2u), v_pool(pool_n * 2u);
        std::memset(k_pool.data(), 0, (size_t) k_pool.size());
        std::memset(v_pool.data(), 0, (size_t) v_pool.size());
        Bytes idx_tail((size_t) (r - 1) * (size_t) idx_dim * 4u);
        Bytes idx_dead((size_t) idx_dim * 4u);
        Bytes idx_pooled((size_t) (tokens / r + 1) * (size_t) idx_dim * 4u);
        Bytes idx_block_pos(4);
        std::memset(idx_tail.data(), 0, (size_t) idx_tail.size());
        std::memset(idx_dead.data(), 0, (size_t) idx_dead.size());
        std::memset(idx_pooled.data(), 0, (size_t) idx_pooled.size());

        QsaLayerWeights w;
        w.k = QsaPlane{pk.form, &codes_k, &scales_k, &offs_k};
        w.v = QsaPlane{pv.form, &codes_v, &scales_v, &offs_v};
        w.q = QsaPlane{pq.form, &codes_q, &scales_q, &offs_q};
        w.out = QsaPlane{po.form, &codes_o, &scales_o, &offs_o};
        w.idx_k = &b_idx_k;
        w.idx_q = &b_idx_q;
        w.k_norm = &b_kn;
        w.q_norm = &b_qn;
        w.idx_k_norm = &b_ik;
        w.idx_q_norm = &b_iq;
        w.cos_tab = &b_cos;
        w.sin_tab = &b_sin;
        QsaLayerState st;
        st.k_pool = &k_pool;
        st.v_pool = &v_pool;
        st.page_table = &b_table;
        st.idx_tail = &idx_tail;
        st.idx_dead = &idx_dead;
        st.idx_pooled = &idx_pooled;
        st.idx_block_pos = &idx_block_pos;
        QsaLayerScratch sc = make_qsa_layer_scratch(g);

        HostIdx ix;
        ix.dim = idx_dim;
        ix.r = r;
        ix.n_rot = n_rot;
        ix.tail.assign((size_t) (r - 1) * (size_t) idx_dim, 0);
        ix.w = w_ik;
        ix.cos_tab = cos_tab;
        ix.sin_tab = sin_tab;
        std::vector<uint16_t> href_k((size_t) tokens * (size_t) n_head_kv * (size_t) head_dim);
        std::vector<uint16_t> href_k_norot(href_k.size());
        std::vector<uint16_t> href_v(href_k.size());

        double worst = 0;
        int id_bad = 0;
        std::vector<float> last_attn, last_norot, last_first, last_mod, last_gated;
        for (int t = 0; t < tokens; ++t) {
            std::vector<float> x((size_t) n_embd);
            for (float& v : x) v = 0.25f * gauss(rng);
            const std::vector<uint8_t> act = quantize(x);
            std::vector<float> k = gemv_ref(pk, act, n_embd, n_head_kv * head_dim);
            std::vector<float> v = gemv_ref(pv, act, n_embd, n_head_kv * head_dim);
            std::vector<float> q_full = gemv_ref(pq, act, n_embd, n_head * head_dim * 2);
            std::vector<float> raw = bf16_ref(x, w_idx_k, n_embd, idx_dim);
            rms_rows(k, w_kn, n_head_kv, head_dim);
            std::vector<float> k_norot = k;
            rope_rows(k, n_head_kv, head_dim, n_rot, pos_base + t, cos_tab, sin_tab);
            for (int h = 0; h < n_head_kv; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t src = (size_t) h * (size_t) head_dim + (size_t) d;
                    const size_t cell = ((size_t) t * (size_t) n_head_kv + (size_t) h) * (size_t) head_dim + (size_t) d;
                    href_k[cell] = f16_from_f32(k[src]);
                    href_k_norot[cell] = f16_from_f32(k_norot[(size_t) h * (size_t) head_dim + (size_t) d]);
                    href_v[cell] = f16_from_f32(v[src]);
                }
            }
            idx_append(ix, raw, t, pos_base);
            std::vector<float> q((size_t) n_head * (size_t) head_dim);
            for (int h = 0; h < n_head; ++h)
                std::memcpy(q.data() + (size_t) h * (size_t) head_dim,
                            q_full.data() + (size_t) h * (size_t) head_dim * 2u, (size_t) head_dim * 4u);
            rms_rows(q, w_qn, n_head, head_dim);
            rope_rows(q, n_head, head_dim, n_rot, pos_base + t, cos_tab, sin_tab);
            std::vector<float> q_idx = bf16_ref(x, w_idx_q, n_embd, idx_n_head * idx_dim);
            rms_rows(q_idx, w_iq, idx_n_head, idx_dim);
            rope_rows(q_idx, idx_n_head, idx_dim, n_rot, pos_base + t, cos_tab, sin_tab);
            const std::vector<float> scores = index_scores(ix, q_idx, idx_n_head, t + 1);
            const std::vector<int> ids = select_ids(scores, top_k, r);

            const int guard = 16;
            std::vector<float> host_out((size_t) n_embd + (size_t) guard, 7.0f);
            Bytes bx = upload(x.data(), x.size() * 4u);
            Bytes bout = upload(host_out.data(), host_out.size() * 4u);
            qsa_layer(g, w, st, sc, t, pos_base, bx, bout);
            const std::vector<float> attn = attend(q, href_k, href_v, ids, n_head, n_head_kv, head_dim, true);
            const std::vector<float> gated = gate_of(attn, q_full, n_head, head_dim, true);
            const std::vector<float> y = gemv_ref(po, quantize(gated), n_head * head_dim, n_embd);
            const double rel = worst_rel(y, (const float*) bout.data());
            if (rel > worst) worst = rel;
            const int* got_ids = (const int*) sc.ids.data();
            for (size_t i = 0; i < ids.size(); ++i)
                if (got_ids[i] != ids[(size_t) i]) ++id_bad;
            if (t == tokens - 1) {
                last_attn = attn;
                last_gated = gated;
                last_norot = attend(q, href_k_norot, href_v, ids, n_head, n_head_kv, head_dim, true);
                last_first = gate_of(attn, q_full, n_head, head_dim, false);
                last_mod = attend(q, href_k, href_v, ids, n_head, n_head_kv, head_dim, false);
                int gbad = 0;
                const float* gp = (const float*) bout.data();
                for (int i = 0; i < guard; ++i)
                    if (gp[n_embd + i] != 7.0f) ++gbad;
                if (gbad) ++g_fail;
                verdict("16 words past the end stayed put", gbad == 0, std::to_string(gbad) + " changed");
            }
        }
        const uint16_t* hk = (const uint16_t*) k_pool.data();
        const uint16_t* hv = (const uint16_t*) v_pool.data();
        const int64_t rows_avail = (int64_t) (pool_n / (size_t) head_dim);
        int pool_bad = 0, pool_k_bad = 0, pool_v_bad = 0, pool_linear = 0;
        const int n_pool = tokens * n_head_kv * head_dim;
        for (int cell = 0; cell < tokens; ++cell) {
            for (int h = 0; h < n_head_kv; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t src =
                        ((size_t) cell * (size_t) n_head_kv + (size_t) h) * (size_t) head_dim + (size_t) d;
                    const int row =
                        (table[(size_t) (cell / page_size)] * n_head_kv + h) * page_size + (cell % page_size);
                    const size_t at = (size_t) row * (size_t) head_dim + (size_t) d;
                    const bool bk = hk[at] != href_k[src];
                    const bool bv = hv[at] != href_v[src];
                    if (bk) ++pool_k_bad;
                    if (bv) ++pool_v_bad;
                    if (bk || bv) ++pool_bad;
                    const int64_t rl = ((int64_t) cell * n_head_kv + h) * page_size + (cell % page_size);
                    if (rl < 0 || rl >= rows_avail || hk[(size_t) rl * (size_t) head_dim + (size_t) d] != href_k[src])
                        ++pool_linear;
                }
            }
        }
        char wb[32];
        std::snprintf(wb, sizeof wb, "rel %.3e", worst);
        verdict("the block matches the reference", worst <= 1e-4, wb);
        verdict("the selected ids are exact", id_bad == 0, std::to_string(id_bad) + " differ");
        verdict("the pool matches the declared layout", pool_bad * 20 < n_pool,
                "k " + std::to_string(pool_k_bad) + ", v " + std::to_string(pool_v_bad) + " of " +
                    std::to_string(n_pool));
        verdict("ignoring the page table is observable", (double) pool_linear / (double) n_pool > 0.5,
                pct((double) pool_linear / (double) n_pool));
        verdict("leaving K unrotated is observable", rel_l1(last_attn, last_norot) > 0.05,
                pct(rel_l1(last_attn, last_norot)));
        verdict("the gate's first half is observable", rel_l1(last_gated, last_first) > 0.05,
                pct(rel_l1(last_gated, last_first)));
        verdict("kv head modulo is observable", rel_l1(last_attn, last_mod) > 0.05,
                pct(rel_l1(last_attn, last_mod)));

        if (refusals) {
            verdict("a negative position is refused",
                    throws_metal([&] {
                        Bytes x(16), o(16);
                        qsa_layer(g, w, st, sc, -1, 0, x, o);
                    }),
                    "");
            QsaLayerGeom bad = g;
            bad.n_embd = 200;
            verdict("an embedding that is not a multiple of 256 is refused",
                    throws_metal([&] { make_qsa_layer_scratch(bad); }), "");
        }
        std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-qsa-layer [--selftest]\n");
        return 2;
    }
    std::printf("qsa layer: one attention block, fp16 cache, synthetic weights\n\n");
    try {
        check_block("reduced", 256, 64, true);
        check_block("model width", 2560, 128, false);
        std::printf("qsa layer: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("qsa_layer_parity OK\n");
    return 0;
}
