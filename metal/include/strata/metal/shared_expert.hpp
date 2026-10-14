// metal/include/strata/metal/shared_expert.hpp - one shared expert, 2-bit weights.
//
// h = silu(x @ gate) * (x @ up), then h @ down, then times sigmoid(x · gate_inp).
// SiLU is on the gate projection. The scalar gate is one value per token, from the
// original hidden state, and it is not a per-element gate. The result is added plain
// by moe_combine. Native projections are not in this port.
//
// The three weights are 2-bit, group 64, bias -1, and the activation is Q8_0.
// A down weight in another code is not this function.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

// gate, up and the down input are scratch the caller owns. gate and up hold n_ff floats.
// h_q8 holds the Q8_0 image of the SwiGLU output. g holds the scalar dot, one float.
// out holds n_embd floats. A zero count returns. A negative count throws.
void shared_expert(const Bytes& x_q8, const Bytes& x_bf16, const Bytes& gate_codes, const Bytes& gate_scales,
                   const Bytes& up_codes, const Bytes& up_scales, const Bytes& down_codes, const Bytes& down_scales,
                   const Bytes& gate_inp_bf16, Bytes& gate, Bytes& up, Bytes& h_q8, Bytes& g, Bytes& out,
                   int64_t n_embd, int64_t n_ff, int threads_per_row);

}  // namespace strata::metal
