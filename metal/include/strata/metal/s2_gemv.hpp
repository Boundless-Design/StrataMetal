// metal/include/strata/metal/s2_gemv.hpp - a 2-bit row dotted with a Q8_0 activation.
//
// Q2_0's activation is Q8_0, 34 bytes per 32 values. The weight is 2 bits, group 64, bias -1:
// value = (code - 1) * scale. The fp16 activation product is a different kernel.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

// act is Q8_0. codes is n_out * (n_in/4) bytes. scales is n_out * (n_in/64) floats. y is n_out floats.
// n_in is a multiple of 64. threads_per_row is a power of two from 1 to 1024.
// A zero count returns. A negative count throws.
void s2_gemv_q8(const Bytes& act, const Bytes& codes, const Bytes& scales, Bytes& y, int64_t n_in, int64_t n_out,
                int threads_per_row);

}  // namespace strata::metal
