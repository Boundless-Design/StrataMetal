// metal/include/strata/metal/topk.hpp - the largest indexer cells.
//
// The width is min(n_kv, idx_top_k + idx_block - 1). Ties keep the lower index.
// The ids come back ascending. n_kv above 32768 is refused: the selection is one
// threadgroup, which is what makes that tie rule one walk.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

inline constexpr int kTopkMaxCells = 32768;

// scores is (n_kv,) f32. ids is (cap,) i32 and cap must be at least the width.
// Writes exactly width ids. n_kv == 0 returns. A negative count throws.
void topk_512(const Bytes& scores, int n_kv, int idx_top_k, int idx_block, int cap, Bytes& ids);

}  // namespace strata::metal
