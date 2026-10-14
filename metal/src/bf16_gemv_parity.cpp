// metal/src/bf16_gemv_parity.cpp - the Metal BF16 GEMV against the same host reference as
// src/kernels/bf16_gemv_parity.cpp, plus one real GGUF tensor read into a shared buffer.
//
// The reduction is what is under test: both sides read the same bf16 bits, in double on the host.
// The activation contract is the third check. A BF16 weight sees a BF16 activation. An fp16
// activation differs by about 2^-9, and the test requires that gap to be visible.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/bytes.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::metal::Bytes;

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(x[(size_t) i]) * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + 4);
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + 8);
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, (uint64_t) s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// One BF16 matrix, GGUF v3. Dim 0 is n_in, the contiguous weight row the kernel reads.
void write_gguf(const std::string& path, const std::vector<uint16_t>& w, uint64_t n_in, uint64_t n_out) {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747u);
    put_u32(b, 3);
    put_u64(b, 1);
    put_u64(b, 1);
    put_str(b, "general.architecture");
    put_u32(b, 8);   // STRING
    put_str(b, "qwen4exp");
    put_str(b, "blk.0.indexer.k_proj.weight");
    put_u32(b, 2);
    put_u64(b, n_in);
    put_u64(b, n_out);
    put_u32(b, 30);  // BF16
    put_u64(b, 0);
    while (b.size() % 32) b.push_back(0);
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(w.data());
    b.insert(b.end(), raw, raw + w.size() * sizeof(uint16_t));
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    if (!out) throw std::runtime_error("short write " + path);
}

std::vector<float> run_gemv(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in,
                            long long n_out, int mode) {
    Bytes dx(x.size() * sizeof(uint16_t));
    Bytes dw(w.size() * sizeof(uint16_t));
    Bytes dy((size_t) n_out * sizeof(float));
    std::memcpy(dx.data(), x.data(), dx.size());
    std::memcpy(dw.data(), w.data(), dw.size());
    if (mode == 0) strata::metal::bf16_gemv(dx, dw, dy, n_in, n_out);
    else if (mode == 32) strata::metal::bf16_gemv_split(dx, dw, dy, n_in, n_out, 32);
    else strata::metal::bf16_gemv_split(dx, dw, dy, n_in, n_out, 256);
    std::vector<float> y((size_t) n_out);
    std::memcpy(y.data(), dy.data(), y.size() * sizeof(float));
    return y;
}

int run(bool selftest) {
    int bad = 0;
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    std::vector<uint16_t> gguf_x, gguf_w;

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        for (auto& v : fw) v = g(rng) * 0.05f;

        std::vector<uint16_t> x((size_t) s.n_in), w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);
        const std::vector<float> naive = run_gemv(x, w, s.n_in, s.n_out, 0);
        const std::vector<float> warp = run_gemv(x, w, s.n_in, s.n_out, 32);
        const std::vector<float> split = run_gemv(x, w, s.n_in, s.n_out, 256);
        const double rn = rel_l1(want, naive), rw = rel_l1(want, warp), rs = rel_l1(want, split);
        std::printf("  %-34s naive %.2e  warp %.2e  split256 %.2e\n", s.what, rn, rw, rs);
        if (!(rn <= 1e-5 && rw <= 1e-5 && rs <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        {
            std::vector<float> fp16_as_f32((size_t) s.n_in);
            std::vector<uint16_t> x_fp((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) {
                const uint16_t h = strata::kernels::f16_from_f32(fx[i]);
                fp16_as_f32[i] = strata::kernels::f32_from_f16(h);
                x_fp[i] = bf16_from_f32(fp16_as_f32[i]);
            }
            const std::vector<float> rival = run_gemv(x_fp, w, s.n_in, s.n_out, 0);
            const double r = rel_l1(want, rival);
            const bool visible = r > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "bf16 vs fp16 activation",
                        visible ? "yes" : "*** NO ***", r * 100);
            if (!visible) ++bad;
        }

        // The bits of an fp16 activation, read as bf16, are a different number. The kernel must follow
        // the bf16 reading of those bits, and that reading must not be the bf16 activation above.
        if (s.n_out == 128) {
            std::vector<uint16_t> x16((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) x16[i] = strata::kernels::f16_from_f32(fx[i]);
            std::vector<float> as_bf16;
            reference(x16, w, s.n_in, s.n_out, as_bf16);
            const double apart = rel_l1(want, as_bf16);
            const std::vector<float> got = run_gemv(x16, w, s.n_in, s.n_out, 0);
            const double rel = rel_l1(as_bf16, got);
            std::printf("      %-30s %s (%.4f%% apart from bf16 activation), kernel rel %.2e\n",
                        "fp16 bits read as bf16", apart > 1e-3 ? "yes" : "*** NO ***", apart * 100, rel);
            if (!(apart > 1e-3 && rel <= 1e-5)) ++bad;
            gguf_x = x;
            gguf_w = w;
        }
    }

    {
        const char* path = "/tmp/strata-metal-bf16.gguf";
        const uint64_t n_in = 2560, n_out = 128;
        write_gguf(path, gguf_w, n_in, n_out);
        strata::GgufFile file(path);
        const strata::TensorInfo* t = file.find("blk.0.indexer.k_proj.weight");
        const strata::MetaValue* arch = file.get("general.architecture");
        if (!t || !arch || arch->s != "qwen4exp" || t->type != 30 || t->shape.size() != 2 ||
            t->shape[0] != n_in || t->shape[1] != n_out) {
            std::printf("  gguf header does not describe the tensor that was written\n");
            ++bad;
        } else {
            const uint64_t nbytes = gguf_w.size() * sizeof(uint16_t);
            const uint8_t* payload = file.tensor_data(*t);
            const bool in_file = file.data_start() + t->offset + nbytes <= file.file_size();
            const bool same = in_file && std::memcmp(payload, gguf_w.data(), (size_t) nbytes) == 0;
            std::vector<uint16_t> w_file(gguf_w.size());
            if (same) std::memcpy(w_file.data(), payload, (size_t) nbytes);
            std::vector<float> want;
            if (same) reference(gguf_x, w_file, (long long) n_in, (long long) n_out, want);
            const std::vector<float> got = same ? run_gemv(gguf_x, w_file, (long long) n_in, (long long) n_out, 0)
                                                : std::vector<float>();
            const double rel = same ? rel_l1(want, got) : 1.0;
            std::printf("  gguf %-28s %s, gemv rel %.2e\n", "blk.0.indexer.k_proj.weight",
                        same ? "bytes match" : "*** BYTES DIFFER ***", rel);
            if (!(same && rel <= 1e-5)) ++bad;
        }
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: strata-metal-bf16-gemv [--selftest]\n"); return 2; }
    }
    try {
        return run(selftest);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-metal-bf16-gemv: %s\n", e.what());
        return 1;
    }
}
