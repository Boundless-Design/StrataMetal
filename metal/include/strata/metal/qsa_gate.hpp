// metal/include/strata/metal/qsa_gate.hpp - the last multiply of a QSA head.
//
// Each head's query block is 2 * head_dim. The gate is the second half. The output is
// attn * sigmoid(gate). SiLU on that gate is a different function, and so is reading
// the first half. The fp16 result is that product rounded by the shared conversion.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// attn is (n_head, head_dim) f32. q_full is (n_head, 2 * head_dim) f32.
// out is (n_head, head_dim) f32. A zero count returns. A negative count throws.
void qsa_gate_apply_f32(const Bytes& attn, const Bytes& q_full, int n_head, int head_dim, Bytes& out);

// The same product, stored as fp16. out is n_head * head_dim ushorts.
void qsa_gate_apply(const Bytes& attn, const Bytes& q_full, int n_head, int head_dim, Bytes& out);

}  // namespace strata::metal
