// metal/include/strata/metal/bf16_gemv.hpp - the BF16 GEMV, Metal side of include/strata/kernels/bf16_gemv.hpp.
//
// y[o] = sum_i f32(x[i]) * f32(w[o * n_in + i]). Both sides are bf16 bits, so each product is exact in
// f32 and only the summation order differs. The weight row is contiguous: shape [n_in, n_out], ne0 = n_in.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

struct GemvDim {
    uint32_t n_in;
    uint32_t n_out;
};
struct SplitDim {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t tpr;
};

// n_out >= 64 uses one simdgroup per row. Below that, one thread per row. Same threshold as the CUDA
// entry point, for the same reason: a 48-wide projection does not fill a machine of simdgroups.
void bf16_gemv(const Bytes& x, const Bytes& w, Bytes& y, int64_t n_in, int64_t n_out);

// threads_per_row == 32 selects the simdgroup path. Any other power of two up to 256 is the shared-memory
// tree, one threadgroup per row.
void bf16_gemv_split(const Bytes& x, const Bytes& w, Bytes& y, int64_t n_in, int64_t n_out, int threads_per_row);

}  // namespace strata::metal
