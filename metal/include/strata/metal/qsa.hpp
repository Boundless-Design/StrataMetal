// metal/include/strata/metal/qsa.hpp - paged split-K decode attention.
//
// The contract is include/strata/kernels/qsa_decode_attn.hpp. Head dim 256, 12 query heads per KV
// head, 64 selected cells per chunk. The live count is step[kStepWidth]. cap sizes the grid.
#pragma once

#include "strata/metal/bytes.hpp"

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::metal {

// Which pointers are set selects the storage, in the same order as the CUDA entry point:
// k_q4 -> q4_0, k_q8 and v_q4 -> hybrid, k_q8 -> int8, otherwise fp16.
struct QsaPools {
    const Bytes* k_f16 = nullptr;
    const Bytes* v_f16 = nullptr;
    const Bytes* k_q8 = nullptr;
    const Bytes* v_q8 = nullptr;
    const Bytes* k_scale = nullptr;
    const Bytes* v_scale = nullptr;
    const Bytes* k_q4 = nullptr;
    const Bytes* v_q4 = nullptr;
    const Bytes* page_table = nullptr;
};

// Floats of scratch for one query: chunk partials, chunk max, chunk sum, and 64 of slack.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const strata::kernels::QsaShapes& s);

// q is [n_q, n_head, 256], ids [n_q, cap], step [n_q, kStepCount], attn [n_q, n_head, 256].
// scratch is n_q times qsa_decode_attn_scratch_floats.
void qsa_decode_attn(const Bytes& q, const QsaPools& pools, const Bytes& ids, const Bytes& step, int64_t cap,
                     const strata::kernels::QsaShapes& s, Bytes& scratch, Bytes& attn, int64_t n_q = 1);

}  // namespace strata::metal
