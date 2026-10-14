// The GDN layer against an independent transcription of the same order. Bound 1e-5, the step's bound.
// Weights are synthetic S-form planes, not a GGUF. The reduced shape uses head ratio 2. The model ratio is 3,
// and a second case runs that head count with a narrow embedding. SiLU is elementwise, so the q and k copies
// exist because L2 reads from the start of its buffer.
#include "strata/metal/gdn_layer.hpp"

#include "strata/metal/device.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;
namespace bits = strata::kernels;

namespace {

constexpr double kBound = 1e-5;
int g_fail = 0;

struct Geom {
    int n_embd, S, h_k, h_v, d_conv, qk, V, C;
};

Geom geom_of(const GdnLayerGeom& g) {
    Geom d;
    d.n_embd = (int) g.n_embd;
    d.S = (int) g.S;
    d.h_k = (int) g.h_k;
    d.h_v = (int) g.h_v;
    d.d_conv = (int) g.d_conv;
    d.qk = d.S * d.h_k;
    d.V = d.S * d.h_v;
    d.C = d.qk + d.qk + d.V;
    return d;
}

struct Opt {
    bool scale_q = true;
    bool l2_v = false;
    bool silu = true;
    bool silu_qk_only = false;
    bool silu_before_conv = false;
    bool modulo = true;
    bool scale_k = false;
    bool softplus_branch = true;
    bool sigmoid_z = true;
};

struct Plane {
    SForm form;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<float> offs;
};

struct Fixture {
    std::vector<float> x;
    Plane qkv, gate, outp;
    std::vector<uint16_t> alpha_w, beta_w;
    std::vector<float> conv_w, ssm_norm, dt, ssm_a;
    std::vector<float> state0, conv0;
};

struct Result {
    std::vector<float> qkv, conv, mixed, q, k, v, alpha, gate, o, y, out, conv_state;
    std::vector<double> state;
    int over20 = 0;
    int over88 = 0;
};

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

void report(const char* name, double rel) {
    const bool ok = rel <= kBound && std::isfinite(rel);
    std::printf("  %-52s rel %.3e%s\n", name, rel, ok ? "" : "   *** NO ***");
    if (!ok) ++g_fail;
}

double rel_l1(const std::vector<float>& want, const std::vector<float>& got) {
    double d = 0, m = 0;
    const size_t n = want.size() < got.size() ? want.size() : got.size();
    for (size_t i = 0; i < n; ++i) {
        d += std::fabs((double) want[i] - (double) got[i]);
        m += std::fabs((double) want[i]);
    }
    if (!std::isfinite(d) || !std::isfinite(m)) return d;
    return d / (m > 1e-30 ? m : 1e-30);
}

double rel_state(const std::vector<double>& want, const std::vector<float>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d += std::fabs(want[i] - (double) got[i]);
        m += std::fabs(want[i]);
    }
    if (!std::isfinite(d) || !std::isfinite(m)) return d;
    return d / (m > 1e-30 ? m : 1e-30);
}

void observable(const char* name, const std::vector<float>& wrong, const std::vector<float>& right) {
    const double rel = rel_l1(right, wrong);
    const bool ok = !std::isfinite(rel) || rel > 0.05;
    if (!std::isfinite(rel)) std::printf("  %-52s %s (non-finite)\n", name, ok ? "yes" : "*** NO ***");
    else std::printf("  %-52s %s (%.2f%% apart)\n", name, ok ? "yes" : "*** NO ***", rel * 100.0);
    if (!ok) ++g_fail;
}

Bytes upload(const void* p, size_t n) {
    Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}

template <class T>
Bytes upload(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
}

std::vector<float> download(const Bytes& b, size_t n) {
    std::vector<float> v(n);
    std::memcpy(v.data(), b.data(), n * sizeof(float));
    return v;
}

int nearest_int_host(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

std::vector<uint8_t> quantize_ref(const float* x, int n) {
    std::vector<uint8_t> blocks((size_t) (n / 256) * 292, 0);
    for (int b = 0; b < n / 256; ++b) {
        const float* xb = x + b * 256;
        uint8_t* out = blocks.data() + (size_t) b * 292;
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) continue;
        const float iscale = -127.0f / max;
        int8_t qs[256];
        for (int j = 0; j < 256; ++j) {
            const int v = nearest_int_host(iscale * xb[j]);
            qs[j] = (int8_t) std::min(127, v);
            out[4 + j] = (uint8_t) qs[j];
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            const int16_t s = (int16_t) sum;
            out[260 + 2 * j] = (uint8_t) s;
            out[260 + 2 * j + 1] = (uint8_t) ((uint16_t) s >> 8);
        }
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
    }
    return blocks;
}

double q8k_at(const std::vector<uint8_t>& x, int i) {
    const uint8_t* blk = x.data() + (size_t) (i / 256) * 292;
    float d;
    std::memcpy(&d, blk, 4);
    const int8_t q = (int8_t) blk[4 + (i % 256)];
    return (double) d * (double) q;
}

double weight_at(const Plane& p, int n_in, int row, int i) {
    const int per_byte = 8 / p.form.code_bits;
    const int codes_per_row = n_in / per_byte;
    const int n_groups = n_in / p.form.group_elems;
    const uint8_t* c = p.codes.data() + (size_t) row * codes_per_row;
    const int code = (c[i / per_byte] >> ((i % per_byte) * p.form.code_bits)) & ((1 << p.form.code_bits) - 1);
    const int g = i / p.form.group_elems;
    double w = (double) (code + p.form.code_bias) * (double) p.scales[(size_t) row * n_groups + g];
    if (p.form.has_offset) w += (double) p.offs[(size_t) row * n_groups + g];
    return w;
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

std::vector<float> bf16_gemv_ref(const std::vector<float>& x, const std::vector<uint16_t>& w, int n_in, int n_out) {
    std::vector<uint16_t> xb((size_t) n_in);
    for (int i = 0; i < n_in; ++i) xb[(size_t) i] = bits::bf16_from_f32(x[(size_t) i]);
    std::vector<float> y((size_t) n_out);
    for (int row = 0; row < n_out; ++row) {
        double acc = 0;
        const uint16_t* wr = w.data() + (size_t) row * n_in;
        for (int i = 0; i < n_in; ++i)
            acc += (double) bits::f32_from_bf16(xb[(size_t) i]) * (double) bits::f32_from_bf16(wr[i]);
        y[(size_t) row] = (float) acc;
    }
    return y;
}

float silu_f(float v) { return v / (1.0f + std::exp(-v)); }

float sigmoid_f(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// The shader returns x above 20, before exp overflows. Metal has no log1p, so the small path is the
// same log(1 + y) form the glue kernel uses.
float softplus_f(float x, bool branch) {
    if (branch && x > 20.0f) return x;
    const float y = std::exp(x);
    const float u = 1.0f + y;
    if (u == 1.0f) return y;
    return std::log(u) * (y / (u - 1.0f));
}

void l2_rows(std::vector<float>& x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0;
        float* p = x.data() + (size_t) r * cols;
        for (int i = 0; i < cols; ++i) acc += (double) p[i] * (double) p[i];
        const float inv = (float) (1.0 / std::sqrt(acc + 1e-6));
        for (int i = 0; i < cols; ++i) p[i] *= inv;
    }
}

void scale_n(std::vector<float>& x, int n, float s) {
    for (int i = 0; i < n; ++i) x[(size_t) i] *= s;
}

SForm affine(int bits, int bias, bool offset) {
    SForm f;
    f.code_bits = bits;
    f.code_bias = bias;
    f.group_elems = 32;
    f.codebook = Codebook::Affine;
    f.has_offset = offset;
    return f;
}

Plane make_plane(SForm form, int n_in, int n_out, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    Plane p;
    p.form = form;
    const int per_byte = 8 / form.code_bits;
    p.codes.resize((size_t) n_out * (n_in / per_byte));
    for (uint8_t& v : p.codes) v = (uint8_t) (rng() & 0xFF);
    const int n_groups = n_in / form.group_elems;
    p.scales.resize((size_t) n_out * n_groups);
    p.offs.resize((size_t) n_out * n_groups);
    for (float& v : p.scales) v = 0.01f + 0.001f * (float) (rng() % 20);
    for (float& v : p.offs) v = 0.05f * g(rng);
    return p;
}

Fixture build(const Geom& d, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    Fixture f;
    f.x.resize((size_t) d.n_embd);
    for (float& v : f.x) v = g(rng);
    // attn_qkv is IQ4_XS, attn_gate is Q4_K, ssm_out is Q5_K. All three want a Q8_K activation.
    f.qkv = make_plane(affine(4, -8, false), d.n_embd, d.C, rng);
    f.gate = make_plane(affine(4, 0, true), d.n_embd, d.V, rng);
    f.outp = make_plane(affine(8, 0, true), d.V, d.n_embd, rng);

    f.alpha_w.resize((size_t) d.h_v * d.n_embd);
    f.beta_w.resize((size_t) d.h_v * d.n_embd);
    for (uint16_t& v : f.alpha_w) v = bits::bf16_from_f32(0.05f * g(rng));
    for (uint16_t& v : f.beta_w) v = bits::bf16_from_f32(0.05f * g(rng));

    f.conv_w.resize((size_t) d.C * d.d_conv);
    for (float& v : f.conv_w) v = 0.25f * g(rng);
    f.ssm_norm.resize((size_t) d.S);
    for (float& v : f.ssm_norm) v = 1.0f + 0.1f * g(rng);
    f.dt.assign((size_t) d.h_v, 0.0f);
    f.ssm_a.resize((size_t) d.h_v);
    // Head 0 sits on the softplus branch. Head 1 is past the point where exp overflows.
    f.dt[0] = 22.0f;
    if (d.h_v > 1) f.dt[1] = 120.0f;
    for (int h = 2; h < d.h_v; ++h) f.dt[(size_t) h] = 0.2f * g(rng);
    for (int h = 0; h < d.h_v; ++h) f.ssm_a[(size_t) h] = -(std::fabs(g(rng)) + 0.2f);

    f.state0.resize((size_t) d.S * d.h_v * d.S);
    for (float& v : f.state0) v = 0.02f * g(rng);
    f.conv0.resize((size_t) d.C * (d.d_conv - 1));
    for (float& v : f.conv0) v = 0.1f * g(rng);
    return f;
}

Result reference(const Fixture& f, const Geom& d, Opt opt, std::vector<double> state, std::vector<float> conv) {
    Result r;
    const std::vector<uint8_t> xq = quantize_ref(f.x.data(), d.n_embd);
    r.qkv = gemv_ref(f.qkv, xq, d.n_embd, d.C);
    if (opt.silu_before_conv)
        for (float& v : r.qkv) v = silu_f(v);

    r.conv.assign((size_t) d.C, 0.0f);
    for (int c = 0; c < d.C; ++c) {
        float* st = conv.data() + (size_t) c * (d.d_conv - 1);
        const float* w = f.conv_w.data() + (size_t) c * d.d_conv;
        double acc = 0;
        for (int i = 0; i < d.d_conv - 1; ++i) acc += (double) st[i] * (double) w[i];
        acc += (double) r.qkv[(size_t) c] * (double) w[d.d_conv - 1];
        r.conv[(size_t) c] = (float) acc;
        for (int i = 0; i < d.d_conv - 2; ++i) st[i] = st[i + 1];
        st[d.d_conv - 2] = r.qkv[(size_t) c];
    }
    r.conv_state = conv;

    std::vector<float> h = r.conv;
    if (opt.silu && !opt.silu_before_conv) {
        const int n = opt.silu_qk_only ? d.qk * 2 : d.C;
        for (int i = 0; i < n; ++i) h[(size_t) i] = silu_f(h[(size_t) i]);
    }
    r.mixed = h;
    r.q.assign(h.begin(), h.begin() + d.qk);
    r.k.assign(h.begin() + d.qk, h.begin() + 2 * d.qk);
    r.v.assign(h.begin() + 2 * d.qk, h.end());
    l2_rows(r.q, d.h_k, d.S);
    l2_rows(r.k, d.h_k, d.S);
    if (opt.l2_v) l2_rows(r.v, d.h_v, d.S);
    const float qs = 1.0f / std::sqrt((float) d.S);
    if (opt.scale_q) scale_n(r.q, d.qk, qs);
    if (opt.scale_k) scale_n(r.k, d.qk, qs);

    r.alpha = bf16_gemv_ref(f.x, f.alpha_w, d.n_embd, d.h_v);
    std::vector<float> beta = bf16_gemv_ref(f.x, f.beta_w, d.n_embd, d.h_v);
    for (float& v : beta) v = sigmoid_f(v);
    r.gate.resize((size_t) d.h_v);
    for (int h = 0; h < d.h_v; ++h) {
        const float x = r.alpha[(size_t) h] + f.dt[(size_t) h];
        if (x > 20.0f) ++r.over20;
        if (x > 88.0f) ++r.over88;
        r.gate[(size_t) h] = softplus_f(x, opt.softplus_branch) * f.ssm_a[(size_t) h];
    }

    r.o.assign((size_t) d.h_v * d.S, 0.0f);
    for (int h = 0; h < d.h_v; ++h) {
        const int src = opt.modulo ? (h % d.h_k) : (h / (d.h_v / d.h_k));
        const double dec = std::exp((double) r.gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < d.S; ++j) {
            auto at = [&](int i) -> double& { return state[(size_t) (i * d.h_v + h) * d.S + j]; };
            for (int i = 0; i < d.S; ++i) at(i) *= dec;
            double sk = 0;
            for (int i = 0; i < d.S; ++i) sk += at(i) * (double) r.k[(size_t) src * d.S + i];
            const double delta = ((double) r.v[(size_t) h * d.S + j] - sk) * b;
            double dot = 0;
            for (int i = 0; i < d.S; ++i) {
                at(i) += (double) r.k[(size_t) src * d.S + i] * delta;
                dot += at(i) * (double) r.q[(size_t) src * d.S + i];
            }
            r.o[(size_t) h * d.S + j] = (float) dot;
        }
    }
    r.state = std::move(state);

    const std::vector<float> z = gemv_ref(f.gate, xq, d.n_embd, d.V);
    r.y.assign((size_t) d.V, 0.0f);
    for (int h = 0; h < d.h_v; ++h) {
        const float* po = r.o.data() + (size_t) h * d.S;
        double acc = 0;
        for (int i = 0; i < d.S; ++i) acc += (double) po[i] * (double) po[i];
        const float inv = (float) (1.0 / std::sqrt(acc / d.S + 1e-6));
        for (int i = 0; i < d.S; ++i) {
            const float zz = z[(size_t) h * d.S + i];
            const float g = opt.sigmoid_z ? sigmoid_f(zz) : silu_f(zz);
            r.y[(size_t) h * d.S + i] = po[i] * inv * f.ssm_norm[(size_t) i] * g;
        }
    }
    const std::vector<uint8_t> yq = quantize_ref(r.y.data(), d.V);
    r.out = gemv_ref(f.outp, yq, d.V, d.n_embd);
    return r;
}

std::vector<double> state_as_double(const std::vector<float>& s) {
    return std::vector<double>(s.begin(), s.end());
}

void bind_plane(GdnPlane& dst, const Plane& src, const Bytes& codes, const Bytes& scales, const Bytes& offs) {
    dst.form = src.form;
    dst.codes = &codes;
    dst.scales = &scales;
    dst.offset = src.form.has_offset ? &offs : nullptr;
}

void check_case(const char* title, const GdnLayerGeom& lg, std::mt19937& rng, bool traps) {
    const Geom d = geom_of(lg);
    std::printf("%s  n_embd %d  S %d  h_k %d  h_v %d (ratio %d)  d_conv %d  C %d\n", title, d.n_embd, d.S, d.h_k,
                d.h_v, d.h_v / d.h_k, d.d_conv, d.C);
    const Fixture f = build(d, rng);
    const Result want = reference(f, d, Opt{}, state_as_double(f.state0), f.conv0);

    if (traps) {
        auto wrong = [&](Opt o) {
            return reference(f, d, o, state_as_double(f.state0), f.conv0);
        };
        Opt o;
        o.scale_q = false;
        observable("forgetting 1/sqrt(S) on q is observable", wrong(o).o, want.o);
        o = Opt{};
        o.l2_v = true;
        observable("L2-normalising v is observable", wrong(o).out, want.out);
        o = Opt{};
        o.silu = false;
        observable("no SiLU on the conv output is observable", wrong(o).out, want.out);
        o = Opt{};
        o.silu_qk_only = true;
        observable("SiLU on q and k only, not on v, is observable", wrong(o).out, want.out);
        o = Opt{};
        o.silu_before_conv = true;
        o.silu = false;
        observable("SiLU on the projection, before the conv, is observable", wrong(o).out, want.out);
        o = Opt{};
        o.modulo = false;
        observable("interleave head pairing is observable", wrong(o).out, want.out);
        o = Opt{};
        o.scale_k = true;
        observable("scaling k as well as q is observable", wrong(o).out, want.out);
        o = Opt{};
        o.softplus_branch = false;
        {
            const Result bad = wrong(o);
            char detail[96];
            std::snprintf(detail, sizeof detail, "(%d heads above 20, %d above 88)", want.over20, want.over88);
            const double rel = rel_l1(want.gate, bad.gate);
            const bool ok = !std::isfinite(rel) || rel > 0.05;
            if (!std::isfinite(rel))
                std::printf("  %-52s %s (non-finite)  %s\n", "omitting the softplus branch is observable",
                            ok ? "yes" : "*** NO ***", detail);
            else
                std::printf("  %-52s %s (%.2f%% apart)  %s\n", "omitting the softplus branch is observable",
                            ok ? "yes" : "*** NO ***", rel * 100.0, detail);
            if (!ok) ++g_fail;
        }
        o = Opt{};
        o.sigmoid_z = false;
        observable("sigmoid vs SiLU on z is observable", wrong(o).out, want.out);
    }

    Bytes x = upload(f.x);
    Bytes qkv_c = upload(f.qkv.codes), qkv_s = upload(f.qkv.scales), qkv_o = upload(f.qkv.offs);
    Bytes gate_c = upload(f.gate.codes), gate_s = upload(f.gate.scales), gate_o = upload(f.gate.offs);
    Bytes out_c = upload(f.outp.codes), out_s = upload(f.outp.scales), out_o = upload(f.outp.offs);
    Bytes alpha = upload(f.alpha_w), beta = upload(f.beta_w), conv = upload(f.conv_w);
    Bytes norm = upload(f.ssm_norm), dt = upload(f.dt), ssm_a = upload(f.ssm_a);
    Bytes state = upload(f.state0), conv_state = upload(f.conv0);
    Bytes y((size_t) d.n_embd * sizeof(float));

    GdnLayerWeights w;
    bind_plane(w.qkv, f.qkv, qkv_c, qkv_s, qkv_o);
    bind_plane(w.attn_gate, f.gate, gate_c, gate_s, gate_o);
    bind_plane(w.ssm_out, f.outp, out_c, out_s, out_o);
    w.alpha = &alpha;
    w.beta = &beta;
    w.conv = &conv;
    w.ssm_norm = &norm;
    w.dt = &dt;
    w.ssm_a = &ssm_a;
    GdnLayerState st{&state, &conv_state};
    GdnLayerScratch sc = make_gdn_layer_scratch(lg);
    gdn_layer(lg, w, st, sc, x, y);

    const std::vector<float> got = download(y, (size_t) d.n_embd);
    const std::vector<float> got_state = download(state, f.state0.size());
    const std::vector<float> got_conv = download(conv_state, f.conv0.size());
    report("qkv projection", rel_l1(want.qkv, download(sc.qkv, (size_t) d.C)));
    report("conv output, after SiLU", rel_l1(want.mixed, download(sc.conv_out, (size_t) d.C)));
    report("q, normalised and scaled", rel_l1(want.q, download(sc.q, (size_t) d.qk)));
    report("k, normalised", rel_l1(want.k, download(sc.k, (size_t) d.qk)));
    report("v, not normalised", rel_l1(want.v, download(sc.v, (size_t) d.V)));
    report("gate", rel_l1(want.gate, download(sc.gate, (size_t) d.h_v)));
    report("step output", rel_l1(want.o, download(sc.o, (size_t) d.V)));
    report("state", rel_state(want.state, got_state));
    report("output norm", rel_l1(want.y, download(sc.y, (size_t) d.V)));
    report("ssm_out", rel_l1(want.out, got));
    report("conv state", rel_l1(want.conv_state, got_conv));

    const float* cs = static_cast<const float*>(conv_state.data());
    const float* qkv = static_cast<const float*>(sc.qkv.data());
    const float* silu = static_cast<const float*>(sc.conv_out.data());
    int column = 0;
    std::vector<float> stored((size_t) d.C), after((size_t) d.C);
    for (int c = 0; c < d.C; ++c) {
        stored[(size_t) c] = cs[(size_t) c * (d.d_conv - 1) + (d.d_conv - 2)];
        after[(size_t) c] = silu[(size_t) c];
        if (std::memcmp(&stored[(size_t) c], qkv + c, sizeof(float)) != 0) ++column;
    }
    verdict("conv state appends the projection, before SiLU", column == 0,
            column ? std::to_string(column) + " channels differ" : "");
    observable("that stored column is not the SiLU output", after, stored);

    if (traps) {
        std::vector<float> flipped((size_t) d.C * d.d_conv);
        for (int c = 0; c < d.C; ++c)
            for (int i = 0; i < d.d_conv; ++i)
                flipped[(size_t) i * d.C + c] = f.conv_w[(size_t) c * d.d_conv + i];
        Bytes wrong = upload(flipped);
        std::memcpy(state.data(), f.state0.data(), f.state0.size() * sizeof(float));
        std::memcpy(conv_state.data(), f.conv0.data(), f.conv0.size() * sizeof(float));
        w.conv = &wrong;
        Bytes y2((size_t) d.n_embd * sizeof(float));
        gdn_layer(lg, w, st, sc, x, y2);
        observable("conv weight transposed on the GPU is observable", download(y2, (size_t) d.n_embd), got);
    }
    std::printf("\n");
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

int run() {
    std::printf("GDN layer, non-fused order, seed 20261014\n\n");
    std::mt19937 rng(20261014);
    check_case("reduced", GdnLayerGeom{256, 128, 2, 4, 4}, rng, true);
    check_case("model heads", GdnLayerGeom{256, 128, 16, 48, 4}, rng, false);
    verdict("S above 128 is refused",
            throws_metal([] {
                make_gdn_layer_scratch(GdnLayerGeom{256, 129, 2, 4, 4});
            }),
            "");
    verdict("a value dim that is not a multiple of 256 is refused",
            throws_metal([] {
                make_gdn_layer_scratch(GdnLayerGeom{256, 128, 1, 1, 4});
            }),
            "");
    std::printf("\ngdn_layer: %d failures\n", g_fail);
    if (g_fail) return 1;
    std::printf("gdn_layer_parity OK\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-gdn-layer [--selftest]\n");
        return 2;
    }
    try {
        return run();
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gdn_layer: %s\n", e.what());
        return 1;
    }
}
