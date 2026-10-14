// metal/include/strata/metal/kv.hpp - one cell into a paged fp16 cache, and a gather back out.
//
// The pool is [physical page][kv head][page slot][head dim]. The page table maps a logical
// page to a physical page and is not the identity. A negative physical page is not written.
// The conversion is the shared round-to-nearest-even fp16. K is already normed and rotated.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// kcur and vcur are (n_head_kv, head_dim) f32. Pools are fp16. A negative pos throws.
void kv_append(Bytes& k_pool, Bytes& v_pool, const Bytes& page_table, int pos, const Bytes& kcur, const Bytes& vcur,
               int n_head_kv, int head_dim, int page_size);

// ids are logical cells. Scratch is (n_ids, n_head_kv, head_dim) fp16. n_ids == 0 returns.
void kv_gather(const Bytes& k_pool, const Bytes& v_pool, const Bytes& page_table, const Bytes& ids, int n_ids,
               int n_head_kv, int head_dim, int page_size, Bytes& k_scratch, Bytes& v_scratch);

}  // namespace strata::metal
