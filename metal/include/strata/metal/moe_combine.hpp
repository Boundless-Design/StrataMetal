// metal/include/strata/metal/moe_combine.hpp - the MoE block's final sum.
//
// y = sum over the k routed experts of w[e] * parts[e] + shared.
// parts is (k, n_embd), one row per selected expert, in the router's order.
// The routed rows are weighted. The shared output is added plain, and it may be null.
// The weights are already renormalised. This kernel does not divide them again.
// k above 64 is refused. The native combine is not in this port.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// One token. A zero count returns. A negative count throws. k above 64 throws.
void moe_combine(const Bytes& parts, const Bytes& weights, const Bytes* shared, Bytes& y, int n_embd, int k);

}  // namespace strata::metal
