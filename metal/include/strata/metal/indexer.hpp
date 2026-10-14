// metal/include/strata/metal/indexer.hpp - one raw indexer key, and the pool on a block boundary.
//
// The raw key is never normed and never rotated. A cell that completes a block is not stored in the
// tail. The pooled row is the mean of the block, then the weighted RMS, then NEOX rotation at the
// block's first cell. The spare key is the weighted RMS of cell 0, rotated at position 0, and it
// moves to the row after the last complete block.
//
// This GPU has no fp64 in a shader. The mean and the sum of squares run on the CPU in IEEE double,
// in cell order and in dimension order, and are rounded to f32 once. The tail store and the
// rotation run on the GPU. A shader that did the sum in f32 would miss the spare key by a fraction
// of an ulp, which is the miss this comparison exists to see.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

struct IndexerBuf {
    Bytes& tail;       // (idx_block - 1, idx_dim) f32 raw
    Bytes& dead;       // (idx_dim,) f32 spare key
    Bytes& pooled;     // (n_blocks + 1, idx_dim) f32, spare at the last written row
    Bytes& block_pos;  // one i32, the first cell's position of the block just completed
};

// pos is one i32, the cell index. pos_base is the sequence's first cell position, constant
// for the sequence. Cell j is rotated at position pos_base + j. cos_tab and sin_tab are
// (rows, n_rot/2) f32, built on the host. A negative size throws. A negative cell index throws.
void indexer_key_append(const Bytes& raw, const Bytes& pos, int pos_base, const Bytes& w_k_norm, float eps,
                        IndexerBuf buf, int idx_dim, int idx_block, int n_rot, const Bytes& cos_tab,
                        const Bytes& sin_tab);

}  // namespace strata::metal
