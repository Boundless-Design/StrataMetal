// metal/src/iq.mm - launch the Coder expert product.
#include "strata/metal/iq.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/iq_grids.hpp"
#include "strata/metal/launch.hpp"

#include <cstring>
#include <string>

namespace strata::metal {
namespace {

struct IqDim {
    int n_in;
    int n_out;
    int n_col;
    int row_bytes;
    int byte_off;
};
static_assert(sizeof(IqDim) == 20, "IqDim must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

int fit(const char* what, int64_t v) {
    if (v > 0x7fffffff) throw MetalError(std::string(what) + " does not fit in int");
    return (int) v;
}

const Bytes& grid_iq2s() {
    static Bytes b(sizeof(iq2s_grid));
    static bool once = false;
    if (!once) {
        std::memcpy(b.data(), iq2s_grid, sizeof(iq2s_grid));
        once = true;
    }
    return b;
}
const Bytes& grid_iq3xxs() {
    static Bytes b(sizeof(iq3xxs_grid));
    static bool once = false;
    if (!once) {
        std::memcpy(b.data(), iq3xxs_grid, sizeof(iq3xxs_grid));
        once = true;
    }
    return b;
}
const Bytes& grid_iq3s() {
    static Bytes b(sizeof(iq3s_grid));
    static bool once = false;
    if (!once) {
        std::memcpy(b.data(), iq3s_grid, sizeof(iq3s_grid));
        once = true;
    }
    return b;
}

const char* mmvq_kernel(int t) {
    switch (t) {
        case kIq2S: return "iq_mmvq_iq2_s";
        case kIq3Xxs: return "iq_mmvq_iq3_xxs";
        case kIq3S: return "iq_mmvq_iq3_s";
        case kIq4Nl: return "iq_mmvq_iq4_nl";
        case kIq4Xs: return "iq_mmvq_iq4_xs";
        case kQ2_0: return "iq_mmvq_q2_0";
        case kQ4_K: return "iq_mmvq_q4_k";
        case kQ5_K: return "iq_mmvq_q5_k";
        case kQ3_K: return "iq_mmvq_q3_k";
        case kQ5_0: return "iq_mmvq_q5_0";
        default: return nullptr;
    }
}

bool mmvq_has_grid(int t) { return t == kIq2S || t == kIq3Xxs || t == kIq3S; }

const Bytes& mmvq_grid(int t) {
    if (t == kIq2S) return grid_iq2s();
    if (t == kIq3Xxs) return grid_iq3xxs();
    return grid_iq3s();
}

}  // namespace

bool iq_type_supported(int ggml_type) noexcept {
    // These have a matrix-vector product. They are not a Coder expert gate or down.
    if (ggml_type == kQ4_K || ggml_type == kQ5_K || ggml_type == kQ3_K || ggml_type == kQ5_0) return false;
    return mmvq_kernel(ggml_type) != nullptr;
}

int iq_block_values(int ggml_type) {
    if (ggml_type == kIq4Nl || ggml_type == kQ5_0) return 32;
    if (ggml_type == kQ2_0) return 64;
    if (ggml_type == kQ4_K || ggml_type == kQ5_K || ggml_type == kQ3_K) return 256;
    if (iq_type_supported(ggml_type)) return 256;
    return 0;
}

size_t iq_block_bytes(int ggml_type) {
    switch (ggml_type) {
        case kIq3Xxs: return 98;
        case kIq4Nl: return 18;
        case kIq3S: return 110;
        case kIq2S: return 82;
        case kIq4Xs: return 136;
        case kQ2_0: return 18;
        case kQ4_K: return 144;
        case kQ5_K: return 176;
        case kQ3_K: return 110;
        case kQ5_0: return 22;
        default: return 0;
    }
}

size_t iq_row_bytes(int ggml_type, int64_t n) {
    const int values = iq_block_values(ggml_type);
    const size_t bytes = iq_block_bytes(ggml_type);
    if (values == 0 || n <= 0 || n % values != 0) throw MetalError("iq_row_bytes: bad type or length");
    return (size_t) (n / values) * bytes;
}

size_t iq_q8_1_bytes(int64_t n) {
    if (n <= 0 || n % 32 != 0) throw MetalError("iq_q8_1_bytes: length must be a positive multiple of 32");
    return (size_t) (n / 32) * 36;
}

void iq_quantize_q8_1(const Bytes& x, int64_t n, Bytes& y) {
    if (n <= 0) return;
    if (n % 32 != 0) throw MetalError("iq_quantize_q8_1: length must be a multiple of 32");
    need("q8 x", x, (uint64_t) n * 4);
    need("q8 y", y, iq_q8_1_bytes(n));
    const int count = fit("iq_quantize_q8_1", n);
    const Bytes* bufs[] = {&x, &y};
    launch("iq_q8_1", bufs, 2, &count, sizeof count, Grid{(uint32_t) (count / 32), 1, 1, 32, 1, 1});
}

void iq_mmvq(int ggml_type, const Bytes& w, int64_t byte_off, const Bytes& x_q8, int64_t n_in, int64_t n_out,
             int64_t n_col, Bytes& y) {
    if (n_in == 0 || n_out == 0 || n_col == 0) return;
    if (n_in < 0 || n_out < 0 || n_col < 0 || byte_off < 0) throw MetalError("iq_mmvq: negative size");
    const char* fn = mmvq_kernel(ggml_type);
    if (!fn) throw MetalError("iq_mmvq: type is not a Coder expert type");
    const size_t row = iq_row_bytes(ggml_type, n_in);
    if (n_in % 32 != 0) throw MetalError("iq_mmvq: n_in must be a multiple of 32");
    need("iq w", w, (uint64_t) byte_off + (uint64_t) n_out * row);
    need("iq x", x_q8, (uint64_t) n_col * iq_q8_1_bytes(n_in));
    need("iq y", y, (uint64_t) n_col * (uint64_t) n_out * 4);
    IqDim dim{fit("n_in", n_in), fit("n_out", n_out), fit("n_col", n_col), fit("row", (int64_t) row),
              fit("byte_off", byte_off)};
    if (mmvq_has_grid(ggml_type)) {
        const Bytes& g = mmvq_grid(ggml_type);
        const Bytes* bufs[] = {&w, &x_q8, &y, &g};
        launch(fn, bufs, 4, &dim, sizeof dim, Grid{(uint32_t) dim.n_out, 1, 1, 32, 1, 1});
    } else {
        const Bytes* bufs[] = {&w, &x_q8, &y};
        launch(fn, bufs, 3, &dim, sizeof dim, Grid{(uint32_t) dim.n_out, 1, 1, 32, 1, 1});
    }
}

size_t iq_expert_bytes(const IqExpert& e) {
    if (!iq_type_supported(e.gu_type) || !iq_type_supported(e.d_type))
        throw MetalError("iq_expert_bytes: type is not a Coder expert type");
    if (iq_block_values(e.gu_type) != 256) throw MetalError("iq_expert_bytes: gate type is a down quant");
    if (e.n_embd <= 0 || e.n_ff <= 0) throw MetalError("iq_expert_bytes: empty expert");
    const size_t gu = iq_row_bytes(e.gu_type, e.n_embd);
    const size_t d = iq_row_bytes(e.d_type, e.n_ff);
    if (e.n_ff % 32 != 0) throw MetalError("iq_expert_bytes: n_ff must be a multiple of 32");
    return 2 * (size_t) e.n_ff * gu + (size_t) e.n_embd * d;
}

void iq_expert(const Bytes& blob, const Bytes& x_q8, const IqExpert& e, int64_t n_tok, Bytes& y) {
    if (n_tok == 0) return;
    if (n_tok < 0) throw MetalError("iq_expert: negative token count");
    const size_t bytes = iq_expert_bytes(e);
    need("expert blob", blob, bytes);
    need("expert x", x_q8, (uint64_t) n_tok * iq_q8_1_bytes(e.n_embd));
    need("expert y", y, (uint64_t) n_tok * (uint64_t) e.n_embd * 4);
    const size_t gu_row = iq_row_bytes(e.gu_type, e.n_embd);
    const int64_t up_off = (int64_t) e.n_ff * (int64_t) gu_row;
    const int64_t down_off = 2 * up_off;
    Bytes gate((uint64_t) n_tok * (uint64_t) e.n_ff * 4);
    Bytes up(gate.size());
    Bytes hq((uint64_t) n_tok * iq_q8_1_bytes(e.n_ff));
    iq_mmvq(e.gu_type, blob, 0, x_q8, e.n_embd, e.n_ff, n_tok, gate);
    iq_mmvq(e.gu_type, blob, up_off, x_q8, e.n_embd, e.n_ff, n_tok, up);
    iq_swiglu_q8(gate, up, n_tok * e.n_ff, hq);
    iq_mmvq(e.d_type, blob, down_off, hq, e.n_ff, e.n_embd, n_tok, y);
}

void iq_swiglu_q8(const Bytes& gate, const Bytes& up, int64_t n, Bytes& y) {
    if (n == 0) return;
    if (n < 0 || n % 32 != 0) throw MetalError("iq_swiglu_q8: length must be a positive multiple of 32");
    need("swiglu gate", gate, (uint64_t) n * 4);
    need("swiglu up", up, (uint64_t) n * 4);
    need("swiglu y", y, iq_q8_1_bytes(n));
    const int count = fit("iq_swiglu_q8", n);
    const Bytes* bufs[] = {&gate, &up, &y};
    launch("iq_swiglu_q8", bufs, 3, &count, sizeof count, Grid{(uint32_t) (count / 32), 1, 1, 32, 1, 1});
}

}  // namespace strata::metal
