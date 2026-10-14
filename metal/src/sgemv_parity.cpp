// The dense projection against the bytes it loads. Bound 1e-5.
// Q8_K is half-to-even with scale -127/max. Q8_0 divides in double, then stores an fp16 scale.
// A wrong codebook, a -128 scale, and an fp16 activation have to land visibly apart.
#include "strata/metal/sgemv.hpp"
#include "strata/metal/device.hpp"
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
namespace bits = strata::kernels;

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

void report(const char* name, double rel, double bound) {
    const bool ok = rel <= bound && std::isfinite(rel);
    std::printf("  %-52s rel %.3e%s\n", name, rel, ok ? "" : "   *** NO ***");
    if (!ok) ++g_fail;
}

std::string pct(double rel) {
    char b[64];
    std::snprintf(b, sizeof b, "(%.2f%% apart)", rel * 100.0);
    return b;
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

void fill_bytes(Bytes& b, uint8_t v) { std::memset(b.data(), v, (size_t) b.size()); }

int nearest_int_host(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

const int kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

double weight_at(const std::vector<uint8_t>& codes, const std::vector<float>& scales,
                 const std::vector<float>& offset, const SForm& f, long long n_in, long long o, long long i) {
    const int per_byte = 8 / f.code_bits;
    const long long codes_per_row = n_in / per_byte;
    const long long n_groups = n_in / f.group_elems;
    const uint8_t* c = codes.data() + o * codes_per_row;
    const int code = (c[i / per_byte] >> ((int) (i % per_byte) * f.code_bits)) & ((1 << f.code_bits) - 1);
    const double v = (f.codebook == Codebook::Iq4Nl) ? (double) kIq4Nl[code & 0x0F] : (double) (code + f.code_bias);
    const long long g = i / f.group_elems;
    double w = v * (double) scales[(size_t) (o * n_groups + g)];
    if (f.has_offset) w += (double) offset[(size_t) (o * n_groups + g)];
    return w;
}

double q8k_at(const std::vector<uint8_t>& x, long long i) {
    const uint8_t* blk = x.data() + (size_t) (i / 256) * 292;
    float d;
    std::memcpy(&d, blk, 4);
    const int8_t q = (int8_t) blk[4 + (i % 256)];
    return (double) d * (double) q;
}

double q8_0_at(const std::vector<uint8_t>& x, long long i) {
    const uint8_t* blk = x.data() + (size_t) (i / 32) * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const int8_t q = (int8_t) blk[2 + (i % 32)];
    return (double) bits::f32_from_f16(dbits) * (double) q;
}

void reference_q8_k(const float* x, long long n, std::vector<uint8_t>& blocks, std::vector<float>& back,
                    float scale_num) {
    blocks.assign((size_t) (n / 256) * 292, 0);
    back.assign((size_t) n, 0.0f);
    for (long long b = 0; b < n / 256; ++b) {
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
        const float iscale = scale_num / max;
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
        for (int j = 0; j < 256; ++j) back[(size_t) (b * 256 + j)] = (float) qs[j] * d;
    }
}

void reference_q8_0(const float* x, long long n, std::vector<uint8_t>& blocks, bool use_f16_divisor) {
    blocks.assign((size_t) (n / 32) * 34, 0);
    for (long long b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
        uint8_t* out = blocks.data() + (size_t) b * 34;
        if (amax == 0.0f) continue;
        const float d32 = amax / 127.0f;
        const uint16_t dbits = bits::f16_from_f32(d32);
        out[0] = (uint8_t) dbits;
        out[1] = (uint8_t) (dbits >> 8);
        const double div = use_f16_divisor ? (double) bits::f32_from_f16(dbits) : (double) d32;
        for (int i = 0; i < 32; ++i) {
            double q = std::rint((double) xb[i] / div);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            out[2 + i] = (uint8_t) (int8_t) q;
        }
    }
}

int byte_mismatch(const uint8_t* got, const uint8_t* want, size_t n, size_t* first) {
    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != want[i]) {
            if (bad == 0 && first) *first = i;
            ++bad;
        }
    }
    return bad;
}

int check_q8k(const char* name, const std::vector<float>& x) {
    const long long n = (long long) x.size();
    std::vector<uint8_t> ref;
    std::vector<float> back;
    reference_q8_k(x.data(), n, ref, back, -127.0f);
    Bytes dx = upload(x);
    Bytes db(q8k_bytes(n));
    fill_bytes(db, 0xA5);
    quantize_q8_k(dx, n, db);
    std::vector<uint8_t> got((size_t) db.size());
    std::memcpy(got.data(), db.data(), got.size());
    size_t first = 0;
    const int bad = byte_mismatch(got.data(), ref.data(), got.size(), &first);
    Bytes dy((size_t) n * 4);
    fill_bytes(dy, 0x5A);
    dequant_q8_k(db, n, dy);
    int val_bad = 0;
    const float* y = (const float*) dy.data();
    for (long long i = 0; i < n; ++i)
        if (std::memcmp(&y[i], &back[(size_t) i], 4) != 0) ++val_bad;
    const bool ok = bad == 0 && val_bad == 0;
    char detail[96];
    if (ok) std::snprintf(detail, sizeof detail, "byte-exact, round trip bit-exact");
    else std::snprintf(detail, sizeof detail, "%d bad bytes (first %zu), %d round-trip", bad, first, val_bad);
    verdict(name, ok, detail);
    return ok ? 0 : 1;
}

struct Case {
    const char* what;
    int bits;
    int bias;
    int group;
    Codebook cb;
    bool has_offset;
};

SForm form_of(const Case& cs) {
    SForm f;
    f.code_bits = cs.bits;
    f.code_bias = cs.bias;
    f.group_elems = cs.group;
    f.codebook = cs.cb;
    f.has_offset = cs.has_offset;
    return f;
}

struct Planes {
    std::vector<float> x;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<float> offs;
};

Planes make_planes(const Case& cs, long long n_in, long long n_out, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    Planes p;
    p.x.resize((size_t) n_in);
    for (float& v : p.x) v = g(rng);
    const int per_byte = 8 / cs.bits;
    p.codes.resize((size_t) (n_out * n_in / per_byte));
    for (uint8_t& v : p.codes) v = (uint8_t) (rng() & 0xFF);
    const long long n_groups = n_in / cs.group;
    p.scales.resize((size_t) (n_out * n_groups));
    p.offs.resize((size_t) (n_out * n_groups));
    for (float& v : p.scales) v = 0.01f + 0.001f * (float) (rng() % 20);
    for (float& v : p.offs) v = 0.2f * g(rng);
    return p;
}

std::vector<float> dot_ref(const Planes& p, const std::vector<uint8_t>& act, const SForm& f, long long n_in,
                           long long n_out, bool q8k) {
    std::vector<float> want((size_t) n_out);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i) {
            const double a = q8k ? q8k_at(act, i) : q8_0_at(act, i);
            acc += weight_at(p.codes, p.scales, p.offs, f, n_in, o, i) * a;
        }
        want[(size_t) o] = (float) acc;
    }
    return want;
}

std::vector<float> dot_f16_ref(const Planes& p, const SForm& f, long long n_in, long long n_out) {
    std::vector<float> want((size_t) n_out);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i) {
            const double a = (double) bits::f32_from_f16(bits::f16_from_f32(p.x[(size_t) i]));
            acc += weight_at(p.codes, p.scales, p.offs, f, n_in, o, i) * a;
        }
        want[(size_t) o] = (float) acc;
    }
    return want;
}

double rel_sum(const std::vector<float>& got, const std::vector<float>& want) {
    double m = 0, d = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        m += std::fabs((double) want[i]);
        d += std::fabs((double) want[i] - (double) got[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

std::vector<float> run_q8k(const std::vector<uint8_t>& act, const Planes& p, const SForm& f, long long n_in,
                           long long n_out, bool split) {
    Bytes x = upload(act);
    Bytes codes = upload(p.codes);
    Bytes scales = upload(p.scales);
    Bytes offs = upload(p.offs);
    Bytes y((size_t) n_out * 4);
    const Bytes* off = f.has_offset ? &offs : nullptr;
    if (split) s_gemv_q8k_split(x, codes, scales, off, y, n_in, n_out, f);
    else s_gemv_q8k(x, codes, scales, off, y, n_in, n_out, f);
    std::vector<float> got((size_t) n_out);
    std::memcpy(got.data(), y.data(), (size_t) n_out * 4);
    return got;
}

std::vector<float> run_q8_0(const std::vector<uint8_t>& act, const Planes& p, const SForm& f, long long n_in,
                            long long n_out) {
    Bytes x = upload(act);
    Bytes codes = upload(p.codes);
    Bytes scales = upload(p.scales);
    Bytes offs = upload(p.offs);
    Bytes y((size_t) n_out * 4);
    s_gemv_q8_0_split(x, codes, scales, f.has_offset ? &offs : nullptr, y, n_in, n_out, f);
    std::vector<float> got((size_t) n_out);
    std::memcpy(got.data(), y.data(), (size_t) n_out * 4);
    return got;
}

std::vector<float> run_f16(const Planes& p, const SForm& f, long long n_in, long long n_out) {
    std::vector<uint16_t> x16((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) x16[(size_t) i] = bits::f16_from_f32(p.x[(size_t) i]);
    Bytes x = upload(x16);
    Bytes codes = upload(p.codes);
    Bytes scales = upload(p.scales);
    Bytes offs = upload(p.offs);
    Bytes y((size_t) n_out * 4);
    s_gemv_f16(x, codes, scales, f.has_offset ? &offs : nullptr, y, n_in, n_out, f);
    std::vector<float> got((size_t) n_out);
    std::memcpy(got.data(), y.data(), (size_t) n_out * 4);
    return got;
}

std::vector<uint8_t> quant_k(const std::vector<float>& x) {
    Bytes dx = upload(x);
    Bytes db(q8k_bytes((int64_t) x.size()));
    quantize_q8_k(dx, (int64_t) x.size(), db);
    std::vector<uint8_t> out((size_t) db.size());
    std::memcpy(out.data(), db.data(), out.size());
    return out;
}

std::vector<uint8_t> quant_0(const std::vector<float>& x) {
    Bytes dx = upload(x);
    Bytes db(q8_0_bytes((int64_t) x.size()));
    fill_bytes(db, 0xA5);
    quantize_q8_0(dx, (int64_t) x.size(), db);
    std::vector<uint8_t> out((size_t) db.size());
    std::memcpy(out.data(), db.data(), out.size());
    return out;
}

void check_gemv_case(const Case& cs, long long n_in, long long n_out, uint32_t seed) {
    std::mt19937 rng(seed);
    const SForm f = form_of(cs);
    Planes p = make_planes(cs, n_in, n_out, rng);
    const std::vector<uint8_t> xk = quant_k(p.x);
    const std::vector<float> want = dot_ref(p, xk, f, n_in, n_out, true);
    const std::vector<float> naive = run_q8k(xk, p, f, n_in, n_out, false);
    const std::vector<float> warp = run_q8k(xk, p, f, n_in, n_out, true);
    char naive_name[80], warp_name[80];
    std::snprintf(naive_name, sizeof naive_name, "%s naive", cs.what);
    std::snprintf(warp_name, sizeof warp_name, "%s warp", cs.what);
    report(naive_name, rel_sum(naive, want), 1e-5);
    report(warp_name, rel_sum(warp, want), 1e-5);

    const std::vector<uint8_t> x0 = quant_0(p.x);
    const std::vector<float> want0 = dot_ref(p, x0, f, n_in, n_out, false);
    const std::vector<float> got0 = run_q8_0(x0, p, f, n_in, n_out);
    char q0_name[80];
    std::snprintf(q0_name, sizeof q0_name, "%s Q8_0", cs.what);
    report(q0_name, rel_sum(got0, want0), 1e-5);

    const std::vector<float> fp = run_f16(p, f, n_in, n_out);
    const double gap = rel_sum(fp, naive);
    char gap_name[80];
    std::snprintf(gap_name, sizeof gap_name, "%s Q8_K vs fp16", cs.what);
    verdict(gap_name, gap > 1e-3, pct(gap));

    if (cs.cb == Codebook::Iq4Nl) {
        SForm wrong = f;
        wrong.codebook = Codebook::Affine;
        const std::vector<float> affine = dot_ref(p, xk, wrong, n_in, n_out, true);
        verdict("Iq4Nl vs the affine codebook", rel_sum(warp, affine) > 0.05, pct(rel_sum(warp, affine)));
    }
}

int ragged(const Case& cs, long long n_out) {
    const long long n_in = 512;
    const long long guard = 64;
    std::mt19937 rng(77 + (unsigned) n_out);
    const SForm f = form_of(cs);
    Planes p = make_planes(cs, n_in, n_out, rng);
    const std::vector<uint8_t> xk = quant_k(p.x);
    const std::vector<uint8_t> x0 = quant_0(p.x);
    int bad = 0;
    for (int kind = 0; kind < 2; ++kind) {
        const std::vector<uint8_t>& act = kind == 0 ? xk : x0;
        std::vector<float> want((size_t) n_out);
        std::vector<double> mag((size_t) n_out);
        for (long long o = 0; o < n_out; ++o) {
            double acc = 0, am = 0;
            for (long long i = 0; i < n_in; ++i) {
                const double a = kind == 0 ? q8k_at(act, i) : q8_0_at(act, i);
                const double wa = weight_at(p.codes, p.scales, p.offs, f, n_in, o, i) * a;
                acc += wa;
                am += std::fabs(wa);
            }
            want[(size_t) o] = (float) acc;
            mag[(size_t) o] = am;
        }
        Bytes x = upload(act);
        Bytes codes = upload(p.codes);
        Bytes scales = upload(p.scales);
        Bytes offs = upload(p.offs);
        Bytes y((size_t) (n_out + guard) * 4);
        uint32_t* yp = (uint32_t*) y.data();
        for (long long i = 0; i < n_out + guard; ++i) yp[i] = 0x7FC0DEADu;
        if (kind == 0) s_gemv_q8k_split(x, codes, scales, f.has_offset ? &offs : nullptr, y, n_in, n_out, f);
        else s_gemv_q8_0_split(x, codes, scales, f.has_offset ? &offs : nullptr, y, n_in, n_out, f);
        double m = 0, d = 0;
        long long touched = 0;
        for (long long o = 0; o < n_out; ++o) {
            float v;
            std::memcpy(&v, &yp[o], 4);
            m += mag[(size_t) o];
            d += std::fabs((double) want[(size_t) o] - (double) v);
        }
        for (long long o = n_out; o < n_out + guard; ++o)
            if (yp[o] != 0x7FC0DEADu) ++touched;
        const double rel = d / (m > 1e-30 ? m : 1e-30);
        const bool ok = rel <= 1e-5 && touched == 0 && std::isfinite(rel);
        char name[96];
        std::snprintf(name, sizeof name, "%s %s n_out %lld", cs.what, kind == 0 ? "Q8_K" : "Q8_0", n_out);
        char detail[64];
        std::snprintf(detail, sizeof detail, "rel %.3e, guard %lld", rel, touched);
        verdict(name, ok, detail);
        if (!ok) ++bad;
    }
    return bad;
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

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-sgemv [--selftest]\n");
        return 2;
    }
    std::printf("S-form projection: Q8_K and Q8_0 activations, IQ4_XS Q4_K Q5_K IQ4_NL weights\n\n");
    try {
        std::mt19937 rng(7);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        const long long NK = 256 * 2048;
        std::vector<float> ka((size_t) NK);
        for (float& v : ka) v = gauss(rng);
        check_q8k("Q8_K random normal", ka);
        std::vector<float> kb((size_t) NK);
        for (float& v : kb) v = gauss(rng) * 1e-7f;
        check_q8k("Q8_K small magnitude (1e-7)", kb);
        check_q8k("Q8_K all zero", std::vector<float>((size_t) NK, 0.0f));

        std::vector<float> kd((size_t) NK, 0.0f);
        for (long long blk = 0; blk < NK / 256; ++blk) {
            for (int j = 0; j < 256; ++j) kd[(size_t) (blk * 256 + j)] = -(float) ((j % 100) + 0.5f);
            kd[(size_t) (blk * 256)] = 127.0f;
        }
        check_q8k("Q8_K exact .5 ties", kd);
        {
            long long ties = 0, differ = 0;
            for (long long i = 0; i < NK; ++i) {
                const float t = -kd[(size_t) i];
                if (t - std::floor(t) == 0.5f) {
                    ++ties;
                    if (nearest_int_host(t) != (int) std::round(t)) ++differ;
                }
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "%lld ties, %lld differ (%.1f%%)", ties, differ,
                          ties ? 100.0 * (double) differ / ties : 0.0);
            verdict("tie fixture sees half-to-even", differ >= ties / 3, detail);
        }

        {
            std::vector<float> one(256, 0.0f);
            one[0] = 127.0f;
            const std::vector<uint8_t> got = quant_k(one);
            const bool d = got[0] == 0x00 && got[1] == 0x00 && got[2] == 0x80 && got[3] == 0xBF;
            const bool q = got[4] == 0x81;
            bool rest = true;
            for (int i = 5; i < 260; ++i)
                if (got[i] != 0) rest = false;
            const bool bsum = got[260] == 0x81 && got[261] == 0xFF;
            bool btail = true;
            for (int i = 262; i < 292; ++i)
                if (got[i] != 0) btail = false;
            verdict("Q8_K [127, 0...] is d=-1, q=-127, bsum=-127", d && q && rest && bsum && btail, "");

            one[1] = -0.5f;
            const std::vector<uint8_t> half = quant_k(one);
            verdict("Q8_K nearest_int(0.5) is 0, not 1", half[5] == 0, "");

            one[1] = -2.5f;
            const std::vector<uint8_t> two = quant_k(one);
            verdict("Q8_K nearest_int(2.5) is 2, not 3", two[5] == 2, "");
        }

        {
            std::vector<uint8_t> r127, r128;
            std::vector<float> b127, b128;
            reference_q8_k(ka.data(), NK, r127, b127, -127.0f);
            reference_q8_k(ka.data(), NK, r128, b128, -128.0f);
            double d = 0, m = 0;
            long long qd = 0;
            for (long long i = 0; i < NK; ++i) {
                d += std::fabs((double) b127[(size_t) i] - (double) b128[(size_t) i]);
                m += std::fabs((double) b127[(size_t) i]);
            }
            for (size_t i = 4; i < r127.size(); ++i)
                if (r127[i] != r128[i]) ++qd;
            const double rel = d / m;
            char detail[80];
            std::snprintf(detail, sizeof detail, "(%.3f%% of magnitude, %lld bytes differ)", 100.0 * rel, qd);
            verdict("-127 vs -128 scale is observable", rel > 0.005, detail);
        }

        const long long N0 = 32 * 4096;
        std::vector<float> a((size_t) N0);
        for (float& v : a) v = gauss(rng);
        std::vector<uint8_t> ref0, rival;
        reference_q8_0(a.data(), N0, ref0, false);
        reference_q8_0(a.data(), N0, rival, true);
        const std::vector<uint8_t> got0 = quant_0(a);
        size_t first = 0;
        const int bad0 = byte_mismatch(got0.data(), ref0.data(), got0.size(), &first);
        int rival_bad = 0;
        for (size_t i = 0; i < ref0.size(); ++i)
            if (ref0[i] != rival[i]) ++rival_bad;
        char d0[96];
        std::snprintf(d0, sizeof d0, bad0 ? "%d bad bytes, first %zu" : "byte-exact", bad0, first);
        verdict("Q8_0 random normal", bad0 == 0, d0);
        verdict("dividing by the fp16 scale changes Q8_0 codes", rival_bad > 0, std::to_string(rival_bad) + " bytes");

        std::vector<float> small((size_t) N0);
        for (float& v : small) v = gauss(rng) * 1e-6f;
        std::vector<uint8_t> ref_s;
        reference_q8_0(small.data(), N0, ref_s, false);
        const std::vector<uint8_t> got_s = quant_0(small);
        const int bad_s = byte_mismatch(got_s.data(), ref_s.data(), got_s.size(), &first);
        verdict("Q8_0 small magnitude (1e-6)", bad_s == 0, bad_s ? std::to_string(bad_s) + " bad bytes" : "byte-exact");
        std::vector<uint8_t> zeros;
        reference_q8_0(std::vector<float>((size_t) N0, 0.0f).data(), N0, zeros, false);
        const std::vector<uint8_t> got_z = quant_0(std::vector<float>((size_t) N0, 0.0f));
        const int bad_z = byte_mismatch(got_z.data(), zeros.data(), got_z.size(), &first);
        verdict("Q8_0 all zero", bad_z == 0, bad_z ? "not zeroed" : "34 zero bytes per block");

        {
            std::vector<float> one(32, 0.0f);
            one[0] = 1.0f;
            const std::vector<uint8_t> blk = quant_0(one);
            const bool scale = blk[0] == 0x08 && blk[1] == 0x20;
            const bool code = blk[2] == 127;
            bool rest = true;
            for (int i = 3; i < 34; ++i)
                if (blk[i] != 0) rest = false;
            verdict("Q8_0 [1, 0...] is code 127, d=f16(1/127)", scale && code && rest, "");
        }

        std::printf("\n");
        const Case cases[] = {
            {"IQ4_XS S4 g32", 4, -8, 32, Codebook::Affine, false},
            {"Q4_K S4 g32 offset", 4, 0, 32, Codebook::Affine, true},
            {"Q5_K S8 g32 offset", 8, 0, 32, Codebook::Affine, true},
            {"IQ4_NL S4 g32", 4, 0, 32, Codebook::Iq4Nl, false},
        };
        for (const Case& cs : cases) check_gemv_case(cs, 512, 64, 2024);

        {
            const Case cs{"Q4_K S4 n_in 640", 4, 0, 32, Codebook::Affine, true};
            std::mt19937 rng640(2024);
            const SForm f = form_of(cs);
            Planes p = make_planes(cs, 640, 64, rng640);
            const std::vector<uint8_t> x0 = quant_0(p.x);
            const std::vector<float> want = dot_ref(p, x0, f, 640, 64, false);
            report("Q8_0 Q4_K n_in 640", rel_sum(run_q8_0(x0, p, f, 640, 64), want), 1e-5);
        }
        {
            const Case cs{"fp16 S2 g32", 2, 0, 32, Codebook::Affine, false};
            std::mt19937 rng2(2024);
            const SForm f = form_of(cs);
            Planes p = make_planes(cs, 32, 4, rng2);
            const std::vector<float> got = run_f16(p, f, 32, 4);
            report("fp16 S2 n_in 32", rel_sum(got, dot_f16_ref(p, f, 32, 4)), 1e-5);
        }

        std::printf("\n");
        for (const Case& cs : cases)
            for (long long n_out : {1LL, 7LL, 9LL, 61LL}) ragged(cs, n_out);

        std::printf("\n");
        verdict("Q8_K refuses code_bits 2",
                throws_metal([] {
                    SForm f;
                    f.code_bits = 2;
                    f.group_elems = 32;
                    Bytes b(16);
                    s_gemv_q8k(b, b, b, nullptr, b, 256, 1, f);
                }),
                "");
        verdict("Q8_0 refuses code_bits 2",
                throws_metal([] {
                    SForm f;
                    f.code_bits = 2;
                    f.group_elems = 32;
                    Bytes b(16);
                    s_gemv_q8_0_split(b, b, b, nullptr, b, 256, 1, f);
                }),
                "");
        verdict("Q8_K refuses n_in 640",
                throws_metal([] {
                    SForm f;
                    f.code_bits = 4;
                    f.group_elems = 32;
                    Bytes b(16);
                    s_gemv_q8k(b, b, b, nullptr, b, 640, 1, f);
                }),
                "");
        verdict("quantize_q8_k refuses n_in 640",
                throws_metal([] {
                    Bytes x(640 * 4), blocks(16);
                    quantize_q8_k(x, 640, blocks);
                }),
                "");
        verdict("split refuses a group that is not a power of two",
                throws_metal([] {
                    SForm f;
                    f.code_bits = 4;
                    f.group_elems = 24;
                    Bytes b(16);
                    s_gemv_q8_0_split(b, b, b, nullptr, b, 96, 1, f);
                }),
                "");
        verdict("split refuses a group that is not a multiple of 16",
                throws_metal([] {
                    SForm f;
                    f.code_bits = 4;
                    f.group_elems = 8;
                    Bytes b(16);
                    s_gemv_q8k_split(b, b, b, nullptr, b, 256, 1, f);
                }),
                "");
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("\nsgemv: %d failures\n", g_fail);
    if (g_fail) return 1;
    std::printf("sgemv_parity OK\n");
    return 0;
}
