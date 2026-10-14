// metal/include/strata/metal/qsa_index.hpp - one score per cached indexer cell.
//
// score[j] = bias[cb] + sum over heads of relu(dot(pooled[cb], q_idx[h])),
// cb = min(j / idx_block, n_bid). An incomplete tail cell also gets +1e9, even when bias is null.
// The relu is per head. relu of the sum is a different number. There is no softmax and no 1/sqrt(d).
// q_idx is already normed and rotated by the caller.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// pooled is (n_bid + 1, idx_dim) f32. q_idx is (idx_n_head, idx_dim) f32.
// bias is (n_bid + 1,) f32, or null. cell_scores is (n_kv,) f32.
// n_bid must be n_kv / idx_block. A negative count throws. n_kv <= 0 throws.
void qsa_index(const Bytes& pooled, int n_bid, const Bytes& q_idx, const Bytes* bias, int idx_n_head, int idx_dim,
               int idx_block, int n_kv, Bytes& cell_scores);

}  // namespace strata::metal
