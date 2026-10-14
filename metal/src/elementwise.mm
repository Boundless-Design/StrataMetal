// metal/src/elementwise.mm - launch the layer glue.
#include "strata/metal/elementwise.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct EwGate {
    int n_tokens;
    int h_v;
};
struct EwScale {
    int n;
    float s;
};
struct EwN {
    int n;
};
struct EwEmbed {
    int n;
    int code_bits;
    int code_bias;
    int group_elems;
    int has_offset;
    int row;
    int out_base;
};
struct EwRms {
    int rows;
    int cols;
    float eps;
    int has_w;
};
static_assert(sizeof(EwGate) == 8, "EwGate must match the shader");
static_assert(sizeof(EwScale) == 8, "EwScale must match the shader");
static_assert(sizeof(EwN) == 4, "EwN must match the shader");
static_assert(sizeof(EwEmbed) == 28, "EwEmbed must match the shader");
static_assert(sizeof(EwRms) == 16, "EwRms must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

int fit(const char* what, int64_t v) {
    if (v < 0 || v > 0x7fffffff) throw MetalError(std::string(what) + " does not fit in int");
    return (int) v;
}

uint32_t groups_for(int n) { return (uint32_t) ((n + 255) / 256); }

const Bytes& dummy_buf() {
    static Bytes b(16);
    return b;
}

int embed_per_byte(int code_bits) {
    if (code_bits != 2 && code_bits != 4 && code_bits != 8)
        throw MetalError("embedding_gather: code_bits must be 2, 4, or 8");
    return 8 / code_bits;
}

}  // namespace

void gdn_gate(const Bytes& alpha, const Bytes& dt, const Bytes& ssm_a, Bytes& gate, int64_t n_tokens, int64_t h_v) {
    if (n_tokens == 0 || h_v == 0) return;
    if (n_tokens < 0 || h_v < 0) throw MetalError("gdn_gate: negative size");
    if (h_v > 0x7fffffff / n_tokens) throw MetalError("gdn_gate: size does not fit");
    const int nt = fit("n_tokens", n_tokens);
    const int hv = fit("h_v", h_v);
    const int n = nt * hv;
    need("gate alpha", alpha, (uint64_t) n * 4);
    need("gate dt", dt, (uint64_t) hv * 4);
    need("gate ssm_a", ssm_a, (uint64_t) hv * 4);
    need("gate out", gate, (uint64_t) n * 4);
    const EwGate dim{nt, hv};
    const Bytes* bufs[] = {&alpha, &dt, &ssm_a, &gate};
    launch("ew_gate", bufs, 4, &dim, sizeof dim, Grid{groups_for(n), 1, 1, 256, 1, 1});
}

void scale_inplace(Bytes& x, int64_t n, float s) {
    if (n == 0) return;
    if (n < 0) throw MetalError("scale_inplace: negative size");
    const int nn = fit("scale n", n);
    need("scale x", x, (uint64_t) nn * 4);
    const EwScale dim{nn, s};
    const Bytes* bufs[] = {&x};
    launch("ew_scale", bufs, 1, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void silu_inplace(Bytes& x, int64_t n) {
    if (n == 0) return;
    if (n < 0) throw MetalError("silu_inplace: negative size");
    const int nn = fit("silu n", n);
    need("silu x", x, (uint64_t) nn * 4);
    const EwN dim{nn};
    const Bytes* bufs[] = {&x};
    launch("ew_silu", bufs, 1, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void f32_to_f16_bulk(const Bytes& x, Bytes& y, int64_t n) {
    if (n == 0) return;
    if (n < 0) throw MetalError("f32_to_f16_bulk: negative size");
    const int nn = fit("f16 n", n);
    need("f16 x", x, (uint64_t) nn * 4);
    need("f16 y", y, (uint64_t) nn * 2);
    const EwN dim{nn};
    const Bytes* bufs[] = {&x, &y};
    launch("ew_f32_to_f16", bufs, 2, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void f32_to_bf16_bulk(const Bytes& x, Bytes& y, int64_t n) {
    if (n == 0) return;
    if (n < 0) throw MetalError("f32_to_bf16_bulk: negative size");
    const int nn = fit("bf16 n", n);
    need("bf16 x", x, (uint64_t) nn * 4);
    need("bf16 y", y, (uint64_t) nn * 2);
    const EwN dim{nn};
    const Bytes* bufs[] = {&x, &y};
    launch("ew_f32_to_bf16", bufs, 2, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void embedding_gather(const Bytes& codes, const Bytes& scales, const Bytes* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, int64_t row, Bytes& out, int64_t out_base) {
    if (n == 0) return;
    if (n < 0 || row < 0 || out_base < 0) throw MetalError("embedding_gather: negative size");
    const int pb = embed_per_byte(code_bits);
    if (group_elems <= 0 || n % group_elems != 0)
        throw MetalError("embedding_gather: n is not a multiple of group_elems");
    if (n % pb != 0) throw MetalError("embedding_gather: n is not a multiple of the codes per byte");
    const int nn = fit("embedding n", n);
    const int rr = fit("embedding row", row);
    const int base = fit("embedding out_base", out_base);
    if (base > 0x7fffffff - nn) throw MetalError("embedding_gather: output index overflows");
    const uint64_t row_bytes = (uint64_t) nn / (uint64_t) pb;
    const uint64_t row_groups = (uint64_t) nn / (uint64_t) group_elems;
    need("embedding codes", codes, ((uint64_t) rr + 1) * row_bytes);
    need("embedding scales", scales, ((uint64_t) rr + 1) * row_groups * 4);
    const Bytes& off = offsets ? *offsets : dummy_buf();
    if (offsets) need("embedding offsets", off, ((uint64_t) rr + 1) * row_groups * 4);
    need("embedding out", out, ((uint64_t) base + (uint64_t) nn) * 4);
    const EwEmbed dim{nn, code_bits, code_bias, group_elems, offsets ? 1 : 0, rr, base};
    const Bytes* bufs[] = {&codes, &scales, &off, &out};
    launch("ew_embed", bufs, 4, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void add_inplace(Bytes& dst, const Bytes& src, int64_t n) {
    if (n == 0) return;
    if (n < 0) throw MetalError("add_inplace: negative size");
    const int nn = fit("add n", n);
    need("add dst", dst, (uint64_t) nn * 4);
    need("add src", src, (uint64_t) nn * 4);
    const EwN dim{nn};
    const Bytes* bufs[] = {&dst, &src};
    launch("ew_add", bufs, 2, &dim, sizeof dim, Grid{groups_for(nn), 1, 1, 256, 1, 1});
}

void rms_norm_weighted(Bytes& x, const Bytes* w, int64_t rows, int64_t cols, float eps) {
    if (rows == 0 || cols == 0) return;
    if (rows < 0 || cols < 0) throw MetalError("rms_norm_weighted: negative size");
    const int nr = fit("rms rows", rows);
    const int nc = fit("rms cols", cols);
    need("rms x", x, (uint64_t) nr * (uint64_t) nc * 4);
    const Bytes& weight = w ? *w : dummy_buf();
    if (w) need("rms w", weight, (uint64_t) nc * 4);
    const EwRms dim{nr, nc, eps, w ? 1 : 0};
    const Bytes* bufs[] = {&x, &weight};
    launch("ew_rms", bufs, 2, &dim, sizeof dim, Grid{(uint32_t) nr, 1, 1, 32, 1, 1});
}

}  // namespace strata::metal
