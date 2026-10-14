// metal/include/strata/metal/gr.hpp - the gated residual, one token.
//
// gr_read collapses the hc streams to the vector the mixer consumes, and writes one injection
// per stream. gr_write adds the mixer output back onto every stream.
//
// The norm is per stream, over n_embd. w_norm is the stored (1 + w), one value per element.
// lo = silu(dot / hc): the division is inside the silu. The gate is a sigmoid. mixed is the
// mean over streams of xn * sigmoid(gate), and xn is the unrounded norm output.
// The default activation is bf16. fp32_act keeps xn and lo in f32.
// A null w_inject is the final mixer: inject is not written.
// gr_write uses 2*sigmoid(inject/hc), so a zero injection adds the block output once.
//
// Native MMVF, the fused multi-token read, and CUDA graphs are not in this port.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

struct GrShapes {
    int64_t n_embd = 0;
    int64_t hc = 0;
    int64_t hc_lr = 0;
};

// Caller-owned scratch. Sized by make_gr_scratch.
struct GrScratch {
    Bytes xn;    // hc * n_embd floats, the unrounded norm output
    Bytes xq;    // hc * n_embd bf16, used when fp32_act is false
    Bytes proj;  // hc_lr floats, the down projection after silu
    Bytes lq;    // hc_lr bf16, used when fp32_act is false
    Bytes gate;  // hc * n_embd floats, then xn * sigmoid
};

GrScratch make_gr_scratch(const GrShapes& s);

// One token. R is (hc, n_embd), n_embd fastest.
// w_down is (hc_lr, hc*n_embd) bf16, row-major, no transpose.
// w_up is (hc*n_embd, hc_lr) bf16, row-major, no transpose.
// w_inject is (hc, hc*n_embd) bf16, or null.
// mixed is n_embd floats. inject is hc floats, left untouched when w_inject is null.
void gr_read(const GrShapes& s, const Bytes& R, const Bytes& w_norm, const Bytes& w_down, const Bytes& w_up,
             const Bytes* w_inject, float eps, bool fp32_act, GrScratch& sc, Bytes& mixed, Bytes& inject);

// R_out[c, d] = R[c, d] + block_out[d] * 2 * sigmoid(inject[c] / hc).
void gr_write(const GrShapes& s, const Bytes& R, const Bytes& block_out, const Bytes& inject, Bytes& R_out);

}  // namespace strata::metal
