// metal/src/sgemv.mm - launch the dense projection.
#include "strata/metal/sgemv.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

struct SgDim {
    int n_in;
    int n_out;
    int bias;
    int codebook;
    int group_elems;
    int group_shift;
    int has_offset;
    int code_bits;
};
static_assert(sizeof(SgDim) == 32, "SgDim must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

int fit(const char* what, int64_t v) {
    if (v < 0 || v > 0x7fffffff) throw MetalError(std::string(what) + " does not fit in int");
    return (int) v;
}

const Bytes& dummy_offset() {
    static Bytes b(16);
    return b;
}

int per_byte(const SForm& form) {
    if (form.code_bits != 2 && form.code_bits != 4 && form.code_bits != 8)
        throw MetalError("sgemv: code_bits must be 2, 4 or 8");
    return 8 / form.code_bits;
}

void check_shape(const char* what, const SForm& form, int64_t n_in, int64_t n_out, bool q8k) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0)
        throw MetalError(std::string(what) + ": n_in is not a multiple of group_elems");
    if (q8k && n_in % 256 != 0) throw MetalError(std::string(what) + ": n_in is not a multiple of 256");
    if (!q8k && n_in % 32 != 0) throw MetalError(std::string(what) + ": n_in is not a multiple of 32");
    per_byte(form);
}

int shift_of(int group) {
    int s = 0;
    while (s < 30 && (1 << s) < group) ++s;
    if ((1 << s) != group) throw MetalError("sgemv: group_elems is not a power of two");
    if (group % 16 != 0) throw MetalError("sgemv: group_elems is not a multiple of 16");
    return s;
}

SgDim dim_of(const SForm& form, int64_t n_in, int64_t n_out, int shift) {
    return SgDim{fit("n_in", n_in), fit("n_out", n_out), form.code_bias, (int) form.codebook, form.group_elems,
                 shift, form.has_offset ? 1 : 0, form.code_bits};
}

void launch_gemv(const char* fn, const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset,
                 Bytes& y, const SForm& form, int64_t n_in, int64_t n_out, bool split, bool q8k, bool f16) {
    if (n_in == 0 || n_out == 0) return;
    if (n_in < 0 || n_out < 0) throw MetalError(std::string(fn) + ": negative size");
    check_shape(fn, form, n_in, n_out, q8k && !f16);
    if (!f16 && form.code_bits != 4 && form.code_bits != 8)
        throw MetalError(std::string(fn) + ": code_bits has no quantized-activation contract");
    if (form.has_offset && offset == nullptr) throw MetalError(std::string(fn) + ": offset is required");
    const int shift = split ? shift_of(form.group_elems) : 0;
    if (!split && form.group_elems <= 0) throw MetalError(std::string(fn) + ": bad group");
    need("sgemv x", x, f16 ? (uint64_t) n_in * 2 : (q8k ? q8k_bytes(n_in) : q8_0_bytes(n_in)));
    need("sgemv codes", codes, sgemv_code_bytes(form, n_in, n_out));
    need("sgemv scales", scales, sgemv_group_bytes(form, n_in, n_out));
    const Bytes& off = (form.has_offset && offset) ? *offset : dummy_offset();
    if (form.has_offset) need("sgemv offset", off, sgemv_group_bytes(form, n_in, n_out));
    need("sgemv y", y, (uint64_t) n_out * 4);
    SgDim dim = dim_of(form, n_in, n_out, shift);
    const Bytes* bufs[] = {&x, &codes, &scales, &off, &y};
    if (split) {
        const uint32_t groups = (uint32_t) ((n_out + 7) / 8);
        launch(fn, bufs, 5, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
    } else {
        const uint32_t groups = (uint32_t) ((n_out + 31) / 32);
        launch(fn, bufs, 5, &dim, sizeof dim, Grid{groups, 1, 1, 32, 1, 1});
    }
}

}  // namespace

size_t q8k_bytes(int64_t n) {
    if (n <= 0 || n % 256 != 0) throw MetalError("q8k_bytes: length must be a positive multiple of 256");
    return (size_t) (n / 256) * 292;
}

size_t q8_0_bytes(int64_t n) {
    if (n <= 0 || n % 32 != 0) throw MetalError("q8_0_bytes: length must be a positive multiple of 32");
    return (size_t) (n / 32) * 34;
}

size_t sgemv_code_bytes(const SForm& form, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) throw MetalError("sgemv_code_bytes: empty shape");
    const int pb = per_byte(form);
    if (n_in % pb != 0) throw MetalError("sgemv_code_bytes: n_in is not a multiple of the codes per byte");
    return (size_t) n_out * (size_t) (n_in / pb);
}

size_t sgemv_group_bytes(const SForm& form, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0 || form.group_elems <= 0 || n_in % form.group_elems != 0)
        throw MetalError("sgemv_group_bytes: bad shape");
    return (size_t) n_out * (size_t) (n_in / form.group_elems) * 4;
}

void quantize_q8_k(const Bytes& x, int64_t n, Bytes& blocks) {
    if (n <= 0) return;
    if (n % 256 != 0) throw MetalError("quantize_q8_k: length must be a multiple of 256");
    need("q8k x", x, (uint64_t) n * 4);
    need("q8k blocks", blocks, q8k_bytes(n));
    const int n_blocks = fit("quantize_q8_k", n / 256);
    const Bytes* bufs[] = {&x, &blocks};
    launch("sg_q8k", bufs, 2, &n_blocks, sizeof n_blocks,
           Grid{(uint32_t) ((n_blocks + 31) / 32), 1, 1, 32, 1, 1});
}

void dequant_q8_k(const Bytes& blocks, int64_t n, Bytes& x) {
    if (n <= 0) return;
    if (n % 256 != 0) throw MetalError("dequant_q8_k: length must be a multiple of 256");
    need("q8k blocks", blocks, q8k_bytes(n));
    need("q8k x", x, (uint64_t) n * 4);
    const int n_blocks = fit("dequant_q8_k", n / 256);
    const Bytes* bufs[] = {&blocks, &x};
    launch("sg_dequant_q8k", bufs, 2, &n_blocks, sizeof n_blocks,
           Grid{(uint32_t) ((n_blocks + 31) / 32), 1, 1, 32, 1, 1});
}

void quantize_q8_0(const Bytes& x, int64_t n, Bytes& blocks) {
    if (n <= 0) return;
    if (n % 32 != 0) throw MetalError("quantize_q8_0: length must be a multiple of 32");
    need("q8_0 x", x, (uint64_t) n * 4);
    need("q8_0 blocks", blocks, q8_0_bytes(n));
    const float* xf = (const float*) x.data();
    uint8_t* out = (uint8_t*) blocks.data();
    for (int64_t b = 0; b < n / 32; b++) {
        const float* xb = xf + b * 32;
        uint8_t* o = out + b * 34;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(xb[i]));
        if (amax == 0.0f) {
            std::memset(o, 0, 34);
            continue;
        }
        const float d32 = amax / 127.0f;
        const uint16_t d16 = strata::kernels::f16_from_f32(d32);
        o[0] = (uint8_t) d16;
        o[1] = (uint8_t) (d16 >> 8);
        for (int i = 0; i < 32; i++) {
            double q = std::rint((double) xb[i] / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            o[2 + i] = (uint8_t) (int8_t) q;
        }
    }
}

void s_gemv_q8k(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y, int64_t n_in,
                int64_t n_out, const SForm& form) {
    launch_gemv("sg_gemv_q8k", x, codes, scales, offset, y, form, n_in, n_out, false, true, false);
}

void s_gemv_q8k_split(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                      int64_t n_in, int64_t n_out, const SForm& form) {
    launch_gemv("sg_gemv_q8k_split", x, codes, scales, offset, y, form, n_in, n_out, true, true, false);
}

void s_gemv_q8_0_split(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                       int64_t n_in, int64_t n_out, const SForm& form) {
    launch_gemv("sg_gemv_q8_0_split", x, codes, scales, offset, y, form, n_in, n_out, true, false, false);
}

void s_gemv_f16(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y, int64_t n_in,
                int64_t n_out, const SForm& form) {
    launch_gemv("sg_gemv_f16", x, codes, scales, offset, y, form, n_in, n_out, false, false, true);
}

}  // namespace strata::metal
