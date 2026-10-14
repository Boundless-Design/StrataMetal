// metal/src/qsa_parity.cpp - paged split-K decode attention against the host reference.
//
// The properties are the ones in src/kernels/qsa_parity.cpp that this kernel can see: the query head
// reads KV head q/12, the score is divided by sqrt(256), an unselected or non-resident cell has weight
// zero, and a reversed page table is not the identity. The bound is that test's 1e-5, on the values the
// kernel actually loads. Indexer, top-k and the gate are not this kernel.
#include "strata/metal/bytes.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/qsa.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
using strata::kernels::QsaShapes;
using strata::kernels::kStepCount;
using strata::kernels::kStepNBid;
using strata::kernels::kStepNKv;
using strata::kernels::kStepPos;
using strata::kernels::kStepWidth;
using strata::metal::Bytes;
using strata::metal::QsaPools;

constexpr int HD = 256;
constexpr int NH = 24;
constexpr int NKV = 2;
constexpr int G = 12;
constexpr uint16_t kNan16 = 0x7E00;

int g_bad = 0;

std::string pct(double rel) {
    char b[40];
    std::snprintf(b, sizeof b, "(%.2f%% apart)", rel * 100.0);
    return b;
}

void verdict(const char* name, bool ok, const std::string& extra) {
    std::printf("  %-52s %-8s %s\n", name, ok ? "yes" : "*** NO ***", extra.c_str());
    if (!ok) ++g_bad;
}

void report(const char* name, double rel, double bound) {
    std::printf("  %-52s rel %.3e\n", name, rel);
    if (!(rel <= bound)) {
        std::printf("    *** over %.0e ***\n", bound);
        ++g_bad;
    }
}

template <class T>
Bytes upload(const std::vector<T>& v) {
    Bytes b(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
    return b;
}

std::vector<float> download(const Bytes& b, size_t n) {
    std::vector<float> v(n);
    std::memcpy(v.data(), b.data(), n * sizeof(float));
    return v;
}

void put_step(std::vector<int32_t>& step, int q, int n_kv, int width, const QsaShapes& s) {
    int32_t* p = step.data() + (size_t) q * kStepCount;
    p[kStepPos] = n_kv > 0 ? n_kv - 1 : 0;
    p[kStepNKv] = n_kv;
    p[kStepNBid] = s.idx_block > 0 ? n_kv / (int) s.idx_block : 0;
    p[kStepWidth] = width;
}

long row_of(int cell, int kv, const std::vector<int32_t>& table, int page_size) {
    const int page = table[(size_t) (cell / page_size)];
    if (page < 0) return -1;
    return ((long) page * NKV + kv) * page_size + (cell % page_size);
}

bool resident(int cell, const std::vector<int32_t>& table, int page_size) {
    if (cell < 0) return false;
    const int page = cell / page_size;
    if (page < 0 || page >= (int) table.size()) return false;
    return table[(size_t) page] >= 0;
}

// Global softmax in double over the f32 values the kernel loads. Masked cells are skipped.
std::vector<double> ref_attend(const std::vector<float>& q, const std::vector<float>& kdeq, const std::vector<float>& vdeq,
                               const std::vector<int32_t>& table, int page_size, const std::vector<int32_t>& ids, int width,
                               int q_base, bool scale, bool divide) {
    const double s = scale ? 1.0 / std::sqrt(256.0) : 1.0;
    std::vector<double> out((size_t) NH * HD, 0.0);
    const int n_cells = (int) (kdeq.size() / ((size_t) NKV * HD));
    for (int h = 0; h < NH; ++h) {
        const int kv = divide ? (h / G) : (h % NKV);
        std::vector<double> sc;
        std::vector<int> use;
        for (int i = 0; i < width; ++i) {
            const int cell = ids[(size_t) i];
            if (cell < 0 || cell >= n_cells || !resident(cell, table, page_size)) continue;
            double dot = 0;
            for (int d = 0; d < HD; ++d)
                dot += (double) kdeq[((size_t) cell * NKV + kv) * HD + d] * (double) q[(size_t) q_base + h * HD + d];
            sc.push_back(dot * s);
            use.push_back(cell);
        }
        if (sc.empty()) continue;
        double m = sc[0];
        for (double x : sc) m = std::max(m, x);
        double L = 0;
        for (double& x : sc) {
            x = std::exp(x - m);
            L += x;
        }
        for (int d = 0; d < HD; ++d) {
            double acc = 0;
            for (size_t j = 0; j < use.size(); ++j)
                acc += (sc[j] / L) * (double) vdeq[((size_t) use[j] * NKV + kv) * HD + d];
            out[(size_t) h * HD + d] = acc;
        }
    }
    return out;
}

double rel_l1(const std::vector<float>& got, const std::vector<double>& want) {
    double n = 0, d = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        n += std::fabs((double) got[i] - want[i]);
        d += std::fabs(want[i]);
    }
    return n / (d > 1e-30 ? d : 1e-30);
}

double rel_ff(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += std::fabs((double) a[i] - (double) b[i]);
        d += std::fabs((double) a[i]);
    }
    return n / (d > 1e-30 ? d : 1e-30);
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

std::vector<float> run(const std::vector<float>& q, const QsaPools& pools, const std::vector<int32_t>& ids,
                       const std::vector<int32_t>& step, int64_t cap, const QsaShapes& s, int64_t n_q,
                       const std::vector<float>* prefill) {
    Bytes dq = upload(q);
    Bytes dids = upload(ids);
    Bytes dstep = upload(step);
    const uint64_t scratch_n = (uint64_t) n_q * strata::metal::qsa_decode_attn_scratch_floats(cap, s);
    Bytes scratch(scratch_n * sizeof(float));
    std::memset(scratch.data(), 0, scratch.size());
    const size_t attn_n = (size_t) n_q * (size_t) s.n_head * HD;
    Bytes attn(attn_n * sizeof(float));
    if (prefill) std::memcpy(attn.data(), prefill->data(), attn_n * sizeof(float));
    else std::memset(attn.data(), 0, attn.size());
    strata::metal::qsa_decode_attn(dq, pools, dids, dstep, cap, s, scratch, attn, n_q);
    return download(attn, attn_n);
}

struct Logical {
    int n_cells = 0;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
};

Logical make_logical(int n_cells, std::mt19937& rng, float spread) {
    std::normal_distribution<float> gauss(0.0f, spread);
    Logical L;
    L.n_cells = n_cells;
    L.q.resize((size_t) NH * HD);
    L.k.resize((size_t) n_cells * NKV * HD);
    L.v.resize((size_t) n_cells * NKV * HD);
    for (float& x : L.q) x = gauss(rng);
    for (float& x : L.k) x = gauss(rng);
    for (float& x : L.v) x = gauss(rng);
    return L;
}

struct F16Pool {
    std::vector<int32_t> table;
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;
    std::vector<float> kdeq;
    std::vector<float> vdeq;
};

F16Pool pack_f16(const Logical& L, int page_size, const std::vector<int32_t>& table) {
    const int n_pages = (int) table.size();
    int n_phys = 1;
    for (int p : table)
        if (p + 1 > n_phys) n_phys = p + 1;
    F16Pool o;
    o.table = table;
    o.kdeq = L.k;
    o.vdeq = L.v;
    const size_t n = (size_t) n_phys * NKV * page_size * HD;
    o.k.assign(n, kNan16);
    o.v.assign(n, kNan16);
    for (int cell = 0; cell < L.n_cells; ++cell) {
        for (int kv = 0; kv < NKV; ++kv) {
            const long row = row_of(cell, kv, table, page_size);
            if (row < 0) continue;
            for (int d = 0; d < HD; ++d) {
                const size_t i = ((size_t) cell * NKV + kv) * HD + d;
                const uint16_t kb = f16_from_f32(L.k[i]);
                const uint16_t vb = f16_from_f32(L.v[i]);
                o.kdeq[i] = f32_from_f16(kb);
                o.vdeq[i] = f32_from_f16(vb);
                o.k[(size_t) row * HD + d] = kb;
                o.v[(size_t) row * HD + d] = vb;
            }
        }
    }
    return o;
}

std::vector<int32_t> reversed_table(int n_pages) {
    std::vector<int32_t> t((size_t) n_pages);
    for (int i = 0; i < n_pages; ++i) t[(size_t) i] = n_pages - 1 - i;
    return t;
}

std::vector<int32_t> identity_table(int n_pages) {
    std::vector<int32_t> t((size_t) n_pages);
    for (int i = 0; i < n_pages; ++i) t[(size_t) i] = i;
    return t;
}

int n_pages_for(int n_cells, int page_size) { return (n_cells + page_size - 1) / page_size; }

void quant_q8_row(const float* x, int8_t* codes, uint16_t* scales, float* decoded) {
    for (int g = 0; g < 4; ++g) {
        float amax = 0;
        for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(x[g * 64 + i]));
        const uint16_t bits = f16_from_f32(amax / 127.0f);
        const float sc = f32_from_f16(bits);
        scales[g] = bits;
        for (int i = 0; i < 64; ++i) {
            int code = 0;
            if (sc != 0.0f) {
                code = (int) std::nearbyint((double) x[g * 64 + i] / (double) sc);
                if (code < -127) code = -127;
                if (code > 127) code = 127;
            }
            codes[g * 64 + i] = (int8_t) code;
            decoded[g * 64 + i] = (float) code * sc;
        }
    }
}

float decode_q4(const uint8_t* row, int dim) {
    const int b = dim / 32;
    const int rem = dim - b * 32;
    const uint8_t* blk = row + b * 18;
    const uint16_t bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
    const float d = f32_from_f16(bits);
    const int j = rem < 16 ? rem : rem - 16;
    const uint8_t byte = blk[2 + j];
    const int nibble = rem < 16 ? (int) (byte & 0x0F) - 8 : (int) (byte >> 4) - 8;
    return (float) nibble * d;
}

void quant_q4_row(const float* x, uint8_t* dst, float* decoded) {
    for (int b = 0; b < 8; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const uint16_t bits = f16_from_f32(amax / 7.0f);
        const float dd = f32_from_f16(bits);
        dst[b * 18] = (uint8_t) (bits & 0xFF);
        dst[b * 18 + 1] = (uint8_t) (bits >> 8);
        uint8_t qs[16] = {};
        for (int i = 0; i < 32; ++i) {
            int n = 8;
            if (dd != 0.0f) {
                n = (int) std::nearbyint((double) x[b * 32 + i] / (double) dd) + 8;
                if (n < 0) n = 0;
                if (n > 15) n = 15;
            }
            if (i < 16) qs[i] = (uint8_t) n;
            else qs[i - 16] = (uint8_t) (qs[i - 16] | (uint8_t) (n << 4));
            decoded[b * 32 + i] = (float) (n - 8) * dd;
        }
        std::memcpy(dst + b * 18 + 2, qs, 16);
    }
}

// One resident cell. Output must be that cell's V, including across a split and a masked sibling.
void check_exact_v(const char* name, const std::vector<float>& got, const std::vector<float>& v_row, int sub_dim0,
                   int sub_dim1) {
    int normal_bad = 0, flushed = 0, sub_bad = 0;
    size_t at = 0;
    float gv = 0, wv = 0;
    for (int h = 0; h < NH; ++h) {
        const int kv = h / G;
        for (int d = 0; d < HD; ++d) {
            const float want = v_row[(size_t) kv * HD + d];
            const float g = got[(size_t) h * HD + d];
            const bool probe = (kv == 0 && d == sub_dim0) || (kv == 1 && d == sub_dim1);
            if (g == want) continue;
            if (probe && want != 0.0f && g == 0.0f) {
                ++flushed;
                continue;
            }
            if (probe) {
                ++sub_bad;
            } else {
                if (!normal_bad) {
                    at = (size_t) h * HD + d;
                    gv = g;
                    wv = want;
                }
                ++normal_bad;
            }
        }
    }
    if (normal_bad || sub_bad) {
        std::printf("  %-52s *** NO ***  %d normal, %d subnormal, first [%zu] got %.9g want %.9g\n", name, normal_bad,
                    sub_bad, at, gv, wv);
        ++g_bad;
    } else {
        std::printf("  %-52s exact%s\n", name, flushed ? "  (fp32 subnormals flushed to zero)" : "");
    }
}

QsaShapes shapes_with(int page_size) {
    QsaShapes s = strata::kernels::qsa_real_shapes();
    s.page_size = page_size;
    return s;
}

int run_tests(bool selftest) {
    std::printf("qsa_decode_attn: n_head %d  n_head_kv %d  head_dim %d  chunk 64\n\n", NH, NKV, HD);

    {
        Bytes b(sizeof(uint32_t));
        std::memset(b.data(), 0, b.size());
        const Bytes* bufs[] = {&b};
        strata::metal::launch("qsa_simd_width", bufs, 1, nullptr, 0, strata::metal::Grid{1, 1, 1, 32, 1, 1});
        uint32_t w = 0;
        std::memcpy(&w, b.data(), sizeof w);
        verdict("simdgroup is 32", w == 32, "width " + std::to_string(w));
        if (w != 32) return 1;
    }

    const QsaShapes real = strata::kernels::qsa_real_shapes();
    verdict("scratch for one chunk is n_head * 258 + 64",
            strata::metal::qsa_decode_attn_scratch_floats(64, real) == (uint64_t) NH * 258u + 64u, "");
    verdict("scratch grows by a chunk at 65 cells",
            strata::metal::qsa_decode_attn_scratch_floats(65, real) == (uint64_t) 2 * NH * 258u + 64u, "");

    {
        uint8_t blk[18] = {};
        blk[0] = 0x00;
        blk[1] = 0x3C;
        blk[2] = 0xA3;
        const bool ok = decode_q4(blk, 0) == -5.0f && decode_q4(blk, 16) == 2.0f && f32_from_f16(0x3C00) == 1.0f;
        verdict("q4 nibble layout (low 3 -> -5, high 10 -> 2)", ok, "");
    }

    std::mt19937 rng(20261014);

    // One live cell, physical page is not the logical page, slot inside the page is not 0.
    {
        const int page_size = 4;
        const int cap = 64;
        const QsaShapes s = shapes_with(page_size);
        std::vector<int32_t> table(2, -1);
        table[1] = 2;
        const int n_phys = 3;
        std::vector<uint16_t> kp((size_t) n_phys * NKV * page_size * HD, kNan16);
        std::vector<uint16_t> vp = kp;
        std::vector<float> vrow((size_t) NKV * HD);
        for (int kv = 0; kv < NKV; ++kv) {
            const long row = row_of(5, kv, table, page_size);
            for (int d = 0; d < HD; ++d) {
                uint16_t kb = f16_from_f32(0.5f);
                uint16_t vb;
                if (kv == 0 && d == 3) vb = 0x0001;
                else if (kv == 1 && d == 4) vb = 0x8001;
                else vb = f16_from_f32(kv == 0 ? (float) (d + 1) : (float) -(d + 1));
                kp[(size_t) row * HD + d] = kb;
                vp[(size_t) row * HD + d] = vb;
                vrow[(size_t) kv * HD + d] = f32_from_f16(vb);
            }
        }
        Bytes dk = upload(kp);
        Bytes dv = upload(vp);
        Bytes dt = upload(table);
        QsaPools pools;
        pools.k_f16 = &dk;
        pools.v_f16 = &dv;
        pools.page_table = &dt;
        std::vector<float> q((size_t) NH * HD, 0.5f);
        std::vector<int32_t> ids((size_t) cap, 0);
        ids[0] = 5;
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, 6, 1, s);
        const std::vector<float> got = run(q, pools, ids, step, cap, s, 1, nullptr);
        check_exact_v("one live cell, non-identity page", got, vrow, 3, 4);
    }

    // Cell 65 is the only resident cell. Chunk 0 is masked, chunk 1 holds it beside a masked sibling,
    // chunk 2 is past the live count.
    {
        const int page_size = 1;
        const int cap = 192;
        const int width = 66;
        const QsaShapes s = shapes_with(page_size);
        std::vector<int32_t> table((size_t) width, -1);
        table[65] = 0;
        std::vector<uint16_t> kp((size_t) NKV * HD, kNan16);
        std::vector<uint16_t> vp = kp;
        std::vector<float> vrow((size_t) NKV * HD);
        for (int kv = 0; kv < NKV; ++kv) {
            for (int d = 0; d < HD; ++d) {
                const uint16_t vb = f16_from_f32(kv == 0 ? (float) (1000 - d) : (float) (d - 500));
                kp[(size_t) kv * HD + d] = f16_from_f32(0.25f);
                vp[(size_t) kv * HD + d] = vb;
                vrow[(size_t) kv * HD + d] = f32_from_f16(vb);
            }
        }
        Bytes dk = upload(kp);
        Bytes dv = upload(vp);
        Bytes dt = upload(table);
        QsaPools pools;
        pools.k_f16 = &dk;
        pools.v_f16 = &dv;
        pools.page_table = &dt;
        std::vector<float> q((size_t) NH * HD, 0.25f);
        std::vector<int32_t> ids((size_t) cap);
        for (int i = 0; i < width; ++i) ids[(size_t) i] = i;
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, width, width, s);
        const std::vector<float> got = run(q, pools, ids, step, cap, s, 1, nullptr);
        check_exact_v("masked chunk, live cell in the next chunk", got, vrow, -1, -1);
    }

    {
        const int cap = 192;
        const QsaShapes s = shapes_with(4);
        std::vector<int32_t> table(1, -1);
        std::vector<uint16_t> pool((size_t) NKV * 4 * HD, kNan16);
        Bytes dk = upload(pool);
        Bytes dv = upload(pool);
        Bytes dt = upload(table);
        QsaPools pools;
        pools.k_f16 = &dk;
        pools.v_f16 = &dv;
        pools.page_table = &dt;
        std::vector<float> q((size_t) NH * HD, 1.0f);
        std::vector<int32_t> ids((size_t) cap, 0);
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, 0, 0, s);
        std::vector<float> pre((size_t) NH * HD, 7.0f);
        const std::vector<float> got = run(q, pools, ids, step, cap, s, 1, &pre);
        bool zero = true;
        for (float x : got)
            if (x != 0.0f) zero = false;
        verdict("an empty selection is zero, not the prefilled 7", zero, "");
    }

    // Loud cell, the numbers from qsa_parity: key 100 against 0, value 10 against 1, q dim 0 = 1.
    {
        const int n_cells = 8;
        const int page_size = 4;
        const int cap = 128;
        const QsaShapes s = shapes_with(page_size);
        Logical L;
        L.n_cells = n_cells;
        L.q.assign((size_t) NH * HD, 0.0f);
        L.k.assign((size_t) n_cells * NKV * HD, 0.0f);
        L.v.assign((size_t) n_cells * NKV * HD, 0.0f);
        for (int h = 0; h < NH; ++h) L.q[(size_t) h * HD] = 1.0f;
        for (int cell = 0; cell < n_cells; ++cell)
            for (int kv = 0; kv < NKV; ++kv) L.v[((size_t) cell * NKV + kv) * HD] = 1.0f;
        L.k[(size_t) 7 * NKV * HD] = 100.0f;
        L.v[(size_t) 7 * NKV * HD] = 10.0f;
        const std::vector<int32_t> rev = reversed_table(n_pages_for(n_cells, page_size));
        F16Pool pool = pack_f16(L, page_size, rev);
        Bytes dk = upload(pool.k);
        Bytes dv = upload(pool.v);
        Bytes dt = upload(pool.table);
        QsaPools pools;
        pools.k_f16 = &dk;
        pools.v_f16 = &dv;
        pools.page_table = &dt;
        std::vector<int32_t> ids((size_t) cap);
        for (int i = 0; i < n_cells; ++i) ids[(size_t) i] = i;
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, n_cells, n_cells, s);
        const std::vector<float> all = run(L.q, pools, ids, step, cap, s, 1, nullptr);
        const std::vector<double> want =
            ref_attend(L.q, pool.kdeq, pool.vdeq, pool.table, page_size, ids, n_cells, 0, true, true);
        verdict("with every cell selected the loud cell dominates head 0", all[0] > 5.0f,
                "out[0] = " + std::to_string(all[0]).substr(0, 6));
        report("loud cell vs the reference on the same fp16 cache", rel_l1(all, want), 1e-5);

        put_step(step, 0, n_cells, n_cells - 1, s);
        const std::vector<float> dropped = run(L.q, pools, ids, step, cap, s, 1, nullptr);
        const std::vector<double> want_drop =
            ref_attend(L.q, pool.kdeq, pool.vdeq, pool.table, page_size, ids, n_cells - 1, 0, true, true);
        verdict("with the loud cell unselected it contributes nothing", std::fabs(dropped[0] - 1.0f) < 1e-6f,
                "out[0] = " + std::to_string(dropped[0]).substr(0, 8));
        report("unselected loud cell vs the reference", rel_l1(dropped, want_drop), 1e-5);
        verdict("attending to the unselected cell is observable", rel_ff(all, dropped) > 0.05, pct(rel_ff(all, dropped)));
    }

    // Random fp16, several chunks, three page granules, a reversed table.
    {
        const int n_cells = 200;
        const int cap = 320;
        Logical L = make_logical(n_cells, rng, 1.0f);
        auto ids_of = [&] {
            std::vector<int32_t> ids((size_t) cap);
            for (int i = 0; i < n_cells; ++i) ids[(size_t) i] = i;
            return ids;
        };
        const std::vector<int32_t> ids = ids_of();

        std::vector<float> page_out[3];
        const int pages[3] = {1, 4, 64};
        double worst = 0;
        for (int pi = 0; pi < 3; ++pi) {
            const int page_size = pages[pi];
            const QsaShapes s = shapes_with(page_size);
            const std::vector<int32_t> rev = reversed_table(n_pages_for(n_cells, page_size));
            F16Pool pool = pack_f16(L, page_size, rev);
            if (page_size == 4) {
                const std::vector<double> right =
                    ref_attend(L.q, pool.kdeq, pool.vdeq, rev, page_size, ids, n_cells, 0, true, true);
                {
                    const std::vector<double> alt =
                        ref_attend(L.q, pool.kdeq, pool.vdeq, rev, page_size, ids, n_cells, 0, true, false);
                    double n = 0, d = 0;
                    for (size_t i = 0; i < right.size(); ++i) {
                        n += std::fabs(right[i] - alt[i]);
                        d += std::fabs(right[i]);
                    }
                    verdict("q / 12 vs q % 2 is observable", n / d > 0.05, pct(n / d));
                }
                {
                    const std::vector<double> alt =
                        ref_attend(L.q, pool.kdeq, pool.vdeq, rev, page_size, ids, n_cells, 0, false, true);
                    double n = 0, d = 0;
                    for (size_t i = 0; i < right.size(); ++i) {
                        n += std::fabs(right[i] - alt[i]);
                        d += std::fabs(right[i]);
                    }
                    verdict("omitting 1/sqrt(256) is observable", n / d > 0.05, pct(n / d));
                }
            }
            Bytes dk = upload(pool.k);
            Bytes dv = upload(pool.v);
            Bytes dt = upload(pool.table);
            QsaPools pools;
            pools.k_f16 = &dk;
            pools.v_f16 = &dv;
            pools.page_table = &dt;
            std::vector<int32_t> step(kStepCount);
            put_step(step, 0, n_cells, n_cells, s);
            page_out[pi] = run(L.q, pools, ids, step, cap, s, 1, nullptr);
            const std::vector<double> want =
                ref_attend(L.q, pool.kdeq, pool.vdeq, rev, page_size, ids, n_cells, 0, true, true);
            const double rel = rel_l1(page_out[pi], want);
            if (rel > worst) worst = rel;
            char label[80];
            std::snprintf(label, sizeof label, "fp16 page %d, reversed table, 200 cells", page_size);
            const bool finite = all_finite(page_out[pi]);
            if (!finite) verdict(label, false, "non-finite output");
            else report(label, rel, 1e-5);

            if (page_size == 4) {
                // A full selection hides a page permutation: the same pairs, another order. A short
                // selection does not. Identity addressing then reads different cells.
                const int width = 40;
                std::vector<int32_t> few((size_t) cap);
                for (int i = 0; i < width; ++i) few[(size_t) i] = i;
                std::vector<int32_t> step_few(kStepCount);
                put_step(step_few, 0, n_cells, width, s);
                const std::vector<float> right_few = run(L.q, pools, few, step_few, cap, s, 1, nullptr);
                const std::vector<double> want_few =
                    ref_attend(L.q, pool.kdeq, pool.vdeq, rev, page_size, few, width, 0, true, true);
                report("40-cell selection, reversed table", rel_l1(right_few, want_few), 1e-5);
                const std::vector<int32_t> ident = identity_table((int) rev.size());
                Bytes dit = upload(ident);
                pools.page_table = &dit;
                const std::vector<float> ignored = run(L.q, pools, few, step_few, cap, s, 1, nullptr);
                const double apart = rel_ff(right_few, ignored);
                verdict("a reversed page table is observable", apart > 0.05, pct(apart));
                pools.page_table = &dt;

                std::vector<int32_t> masked = rev;
                masked[3] = -1;
                Bytes dmask = upload(masked);
                pools.page_table = &dmask;
                const std::vector<float> mgot = run(L.q, pools, ids, step, cap, s, 1, nullptr);
                const std::vector<double> mwant =
                    ref_attend(L.q, pool.kdeq, pool.vdeq, masked, page_size, ids, n_cells, 0, true, true);
                if (!all_finite(mgot)) verdict("a page of -1 is dropped", false, "non-finite");
                else report("a page of -1 matches attention without those cells", rel_l1(mgot, mwant), 1e-5);
            }
        }
        const bool same = page_out[0] == page_out[1] && page_out[1] == page_out[2];
        verdict("page size 1, 4 and 64 agree", same, same ? "bitwise" : pct(rel_ff(page_out[0], page_out[1])));
        std::printf("  %-52s rel %.3e\n", "worst fp16 page vs the reference", worst);
    }

    auto numeric_quant = [&](const char* name, int mode, int n_cells) {
        const int page_size = 4;
        const int cap = 160;
        const QsaShapes s = shapes_with(page_size);
        Logical L = make_logical(n_cells, rng, 1.0f);
        const std::vector<int32_t> rev = reversed_table(n_pages_for(n_cells, page_size));
        int n_phys = 1;
        for (int p : rev)
            if (p + 1 > n_phys) n_phys = p + 1;
        const size_t rows = (size_t) n_phys * NKV * page_size;
        std::vector<float> kdeq((size_t) n_cells * NKV * HD, 0);
        std::vector<float> vdeq((size_t) n_cells * NKV * HD, 0);
        std::vector<int8_t> kq(rows * HD, 1), vq(rows * HD, 1);
        std::vector<uint16_t> ks(rows * 4, kNan16), vs(rows * 4, kNan16);
        std::vector<uint8_t> k4(rows * 144, 0), v4(rows * 144, 0);
        for (size_t r = 0; r < rows; ++r) {
            k4[r * 144] = 0x00;
            k4[r * 144 + 1] = 0x7E;
            v4[r * 144] = 0x00;
            v4[r * 144 + 1] = 0x7E;
        }
        for (int cell = 0; cell < n_cells; ++cell) {
            for (int kv = 0; kv < NKV; ++kv) {
                const long row = row_of(cell, kv, rev, page_size);
                float* kd = kdeq.data() + ((size_t) cell * NKV + kv) * HD;
                float* vd = vdeq.data() + ((size_t) cell * NKV + kv) * HD;
                const float* ksrc = L.k.data() + ((size_t) cell * NKV + kv) * HD;
                const float* vsrc = L.v.data() + ((size_t) cell * NKV + kv) * HD;
                if (mode == 1 || mode == 3) quant_q8_row(ksrc, kq.data() + row * HD, ks.data() + row * 4, kd);
                if (mode == 1) quant_q8_row(vsrc, vq.data() + row * HD, vs.data() + row * 4, vd);
                if (mode == 2) quant_q4_row(ksrc, k4.data() + row * 144, kd);
                if (mode == 2 || mode == 3) quant_q4_row(vsrc, v4.data() + row * 144, vd);
            }
        }
        if (mode == 2) {
            const long row = row_of(0, 0, rev, page_size);
            for (int d = 0; d < HD; ++d) {
                const float got = decode_q4(k4.data() + row * 144, d);
                if (got != kdeq[(size_t) d]) {
                    verdict("q4 quantizer matches the nibble decoder", false, "");
                    break;
                }
                if (d == HD - 1) verdict("q4 quantizer matches the nibble decoder", true, "");
            }
        }
        Bytes dk8 = upload(kq);
        Bytes dv8 = upload(vq);
        Bytes dks = upload(ks);
        Bytes dvs = upload(vs);
        Bytes dk4 = upload(k4);
        Bytes dv4 = upload(v4);
        Bytes dt = upload(rev);
        QsaPools pools;
        pools.page_table = &dt;
        if (mode == 1) {
            pools.k_q8 = &dk8;
            pools.v_q8 = &dv8;
            pools.k_scale = &dks;
            pools.v_scale = &dvs;
        } else if (mode == 2) {
            pools.k_q4 = &dk4;
            pools.v_q4 = &dv4;
        } else {
            pools.k_q8 = &dk8;
            pools.k_scale = &dks;
            pools.v_q4 = &dv4;
        }
        std::vector<int32_t> ids((size_t) cap);
        for (int i = 0; i < n_cells; ++i) ids[(size_t) i] = i;
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, n_cells, n_cells, s);
        const std::vector<float> got = run(L.q, pools, ids, step, cap, s, 1, nullptr);
        const std::vector<double> want = ref_attend(L.q, kdeq, vdeq, rev, page_size, ids, n_cells, 0, true, true);
        if (!all_finite(got)) verdict(name, false, "non-finite output");
        else report(name, rel_l1(got, want), 1e-5);
    };
    numeric_quant("int8 vs the reference on the dequantized values", 1, 96);
    numeric_quant("q4_0 vs the reference on the dequantized values", 2, 96);
    numeric_quant("hybrid int8 keys, q4_0 values", 3, 96);

    // One cell whose codes are exact, so the output is the decoded V bitwise.
    {
        const int page_size = 4;
        const int cap = 64;
        const QsaShapes s = shapes_with(page_size);
        std::vector<int32_t> table(2, -1);
        table[1] = 1;
        const int n_phys = 2;
        const size_t rows = (size_t) n_phys * NKV * page_size;
        std::vector<int8_t> kq(rows * HD, 1), vq(rows * HD, 1);
        std::vector<uint16_t> ks(rows * 4, kNan16), vs = ks;
        std::vector<uint8_t> v4(rows * 144, 0);
        for (size_t r = 0; r < rows; ++r) {
            v4[r * 144] = 0x00;
            v4[r * 144 + 1] = 0x7E;
        }
        std::vector<float> vq8((size_t) NKV * HD), vq4((size_t) NKV * HD);
        for (int kv = 0; kv < NKV; ++kv) {
            const long row = row_of(5, kv, table, page_size);
            for (int g = 0; g < 4; ++g) {
                const float sc = (float) (g + 1);
                ks[(size_t) row * 4 + g] = f16_from_f32(1.0f);
                vs[(size_t) row * 4 + g] = f16_from_f32(sc);
                for (int i = 0; i < 64; ++i) {
                    int code = (i - 32) * (kv == 0 ? 1 : -1);
                    kq[(size_t) row * HD + g * 64 + i] = (int8_t) (kv == 0 ? 1 : -1);
                    vq[(size_t) row * HD + g * 64 + i] = (int8_t) code;
                    vq8[(size_t) kv * HD + g * 64 + i] = (float) code * sc;
                }
            }
            for (int b = 0; b < 8; ++b) {
                uint8_t* blk = v4.data() + row * 144 + b * 18;
                const uint16_t bits = f16_from_f32((float) (b + 1));
                blk[0] = (uint8_t) (bits & 0xFF);
                blk[1] = (uint8_t) (bits >> 8);
                for (int j = 0; j < 16; ++j) {
                    const int low = kv == 0 ? j : (15 - j);
                    const int high = kv == 0 ? (15 - j) : j;
                    blk[2 + j] = (uint8_t) (low | (high << 4));
                }
            }
            for (int d = 0; d < HD; ++d) vq4[(size_t) kv * HD + d] = decode_q4(v4.data() + row * 144, d);
        }
        Bytes dk = upload(kq);
        Bytes dv = upload(vq);
        Bytes dks = upload(ks);
        Bytes dvs = upload(vs);
        Bytes dv4 = upload(v4);
        Bytes dt = upload(table);
        std::vector<float> q((size_t) NH * HD, 0.5f);
        std::vector<int32_t> ids((size_t) cap, 0);
        ids[0] = 5;
        std::vector<int32_t> step(kStepCount);
        put_step(step, 0, 6, 1, s);
        QsaPools pq;
        pq.k_q8 = &dk;
        pq.v_q8 = &dv;
        pq.k_scale = &dks;
        pq.v_scale = &dvs;
        pq.page_table = &dt;
        check_exact_v("one live cell, int8", run(q, pq, ids, step, cap, s, 1, nullptr), vq8, -1, -1);
        QsaPools p4;
        p4.k_q4 = nullptr;
        p4.k_q8 = &dk;
        p4.k_scale = &dks;
        p4.v_q4 = &dv4;
        p4.page_table = &dt;
        // Hybrid: int8 K, q4 V. One cell, so the output is the q4 V.
        check_exact_v("one live cell, hybrid", run(q, p4, ids, step, cap, s, 1, nullptr), vq4, -1, -1);
        QsaPools pq4;
        // q4 keys and values. Rebuild K as a copy of V's blocks so the score stays finite.
        std::vector<uint8_t> k4 = v4;
        Bytes dk4 = upload(k4);
        pq4.k_q4 = &dk4;
        pq4.v_q4 = &dv4;
        pq4.page_table = &dt;
        check_exact_v("one live cell, q4_0", run(q, pq4, ids, step, cap, s, 1, nullptr), vq4, -1, -1);
    }

    // Two queries, different widths and different id lists, one scratch.
    {
        const int n_cells = 80;
        const int cap = 128;
        const int page_size = 4;
        const int n_q = 2;
        const QsaShapes s = shapes_with(page_size);
        std::mt19937 rng2(7);
        Logical L = make_logical(n_cells, rng2, 1.0f);
        std::vector<float> q((size_t) n_q * NH * HD);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (int i = 0; i < n_q; ++i)
            for (int h = 0; h < NH * HD; ++h) q[(size_t) i * NH * HD + h] = gauss(rng2);
        // Query 0 also carries L.q so the two queries differ from each other and from a shared q.
        for (size_t i = 0; i < L.q.size(); ++i) q[i] = L.q[i];
        const std::vector<int32_t> rev = reversed_table(n_pages_for(n_cells, page_size));
        F16Pool pool = pack_f16(L, page_size, rev);
        Bytes dk = upload(pool.k);
        Bytes dv = upload(pool.v);
        Bytes dt = upload(pool.table);
        QsaPools pools;
        pools.k_f16 = &dk;
        pools.v_f16 = &dv;
        pools.page_table = &dt;
        std::vector<int32_t> ids((size_t) n_q * cap, 0);
        for (int i = 0; i < n_cells; ++i) ids[(size_t) i] = i;
        for (int i = 0; i < 40; ++i) ids[(size_t) cap + i] = 40 + i;
        std::vector<int32_t> step((size_t) n_q * kStepCount);
        put_step(step, 0, n_cells, n_cells, s);
        put_step(step, 1, n_cells, 40, s);
        const std::vector<float> got = run(q, pools, ids, step, cap, s, n_q, nullptr);
        double worst = 0;
        for (int i = 0; i < n_q; ++i) {
            const int width = i == 0 ? n_cells : 40;
            std::vector<int32_t> one((size_t) width);
            for (int c = 0; c < width; ++c) one[(size_t) c] = ids[(size_t) i * cap + c];
            const std::vector<double> want =
                ref_attend(q, pool.kdeq, pool.vdeq, rev, page_size, one, width, i * NH * HD, true, true);
            std::vector<float> slice(got.begin() + (size_t) i * NH * HD, got.begin() + (size_t) (i + 1) * NH * HD);
            if (!all_finite(slice)) {
                verdict(i == 0 ? "batch query 0" : "batch query 1", false, "non-finite");
            } else {
                const double rel = rel_l1(slice, want);
                if (rel > worst) worst = rel;
            }
        }
        report("batch of 2 queries, different selections", worst, 1e-5);
        const double apart = rel_ff(std::vector<float>(got.begin(), got.begin() + NH * HD),
                                     std::vector<float>(got.begin() + NH * HD, got.end()));
        verdict("the two queries do not share an output", apart > 0.05, pct(apart));
    }

    std::printf("\nqsa: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("qsa_parity OK\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else {
            std::fprintf(stderr, "usage: strata-metal-qsa [--selftest]\n");
            return 2;
        }
    }
    try {
        return run_tests(selftest);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-metal-qsa: %s\n", e.what());
        return 1;
    }
}
