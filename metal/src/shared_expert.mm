// metal/src/shared_expert.mm - three 2-bit projections, SiLU on the gate, one scalar.
#include "strata/metal/shared_expert.hpp"

#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/s2_gemv.hpp"
#include "strata/metal/sgemv.hpp"

#include <string>

namespace strata::metal {
namespace {

struct SeN {
    int n;
    int pad0;
    int pad1;
    int pad2;
};
static_assert(sizeof(SeN) == 16, "SeN must match the shader");

constexpr int kThreads = 256;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

void launch_n(const char* fn, const Bytes* const* bufs, int nbuf, int n) {
    if (n <= 0) return;
    const SeN dim{n, 0, 0, 0};
    const uint32_t groups = (uint32_t) ((n + kThreads - 1) / kThreads);
    launch(fn, bufs, nbuf, &dim, sizeof dim, Grid{groups, 1, 1, (uint32_t) kThreads, 1, 1});
}

}  // namespace

void shared_expert(const Bytes& x_q8, const Bytes& x_bf16, const Bytes& gate_codes, const Bytes& gate_scales,
                   const Bytes& up_codes, const Bytes& up_scales, const Bytes& down_codes, const Bytes& down_scales,
                   const Bytes& gate_inp_bf16, Bytes& gate, Bytes& up, Bytes& h_q8, Bytes& g, Bytes& out,
                   int64_t n_embd, int64_t n_ff, int threads_per_row) {
    if (n_embd < 0 || n_ff < 0 || threads_per_row < 0) throw MetalError("shared_expert: negative size");
    if (n_embd == 0 || n_ff == 0) return;
    if (n_embd % 64 != 0 || n_ff % 64 != 0) throw MetalError("shared_expert: width is not a multiple of 64");

    need("x_q8", x_q8, q8_0_bytes(n_embd));
    need("x_bf16", x_bf16, (uint64_t) n_embd * 2u);
    need("gate_inp", gate_inp_bf16, (uint64_t) n_embd * 2u);
    need("gate scratch", gate, (uint64_t) n_ff * 4u);
    need("up scratch", up, (uint64_t) n_ff * 4u);
    need("h_q8", h_q8, q8_0_bytes(n_ff));
    need("g", g, 4);
    need("out", out, (uint64_t) n_embd * 4u);

    s2_gemv_q8(x_q8, gate_codes, gate_scales, gate, n_embd, n_ff, threads_per_row);
    s2_gemv_q8(x_q8, up_codes, up_scales, up, n_embd, n_ff, threads_per_row);
    const Bytes* sw[] = {&gate, &up, &gate};
    launch_n("se_swiglu", sw, 3, (int) n_ff);
    quantize_q8_0(gate, n_ff, h_q8);
    s2_gemv_q8(h_q8, down_codes, down_scales, out, n_ff, n_embd, threads_per_row);
    bf16_gemv(x_bf16, gate_inp_bf16, g, n_embd, 1);
    const Bytes* sc[] = {&out, &g};
    launch_n("se_sigmoid_scale", sc, 2, (int) n_embd);
}

}  // namespace strata::metal
