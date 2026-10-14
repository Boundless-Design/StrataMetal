// metal/include/strata/metal/rope.hpp - partial NEOX rotation.
//
// Pair i with i + n_rot/2. The tail past n_rot is copied through. The cos and sin table is built
// on the host in float64. Scaling is not in this call: an unscaled table is the whole contract.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// max_pos rows of n_rot/2 pairs. n_rot must be even and at least 2.
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab);

// x and out are (rows, head_dim). pos is one int per row. out may be x.
// A position outside the table throws. rows == 0 returns.
void rope_neox_apply(Bytes& out, const Bytes& x, int rows, int head_dim, int n_rot, const Bytes& cos_tab,
                     const Bytes& sin_tab, const Bytes& pos);

}  // namespace strata::metal
