// metal/include/strata/metal/elementwise.hpp - the glue between the GDN projections.
//
// gate = softplus(alpha + dt) * ssm_a, and ssm_a is negative. softplus is x above 20.
// SiLU is the conv output, before the q|k|v split. scale is the extra 1/sqrt(S) on q.
// The bf16 conversion feeds ssm_alpha and ssm_beta. This GPU has no fp64, so SiLU uses the precise f32 exp.
// embedding_gather is one token row. add_inplace sums the two expert halves. rms_norm_weighted is the
// QSA norm: the mean of squares, times w. It is not the DeltaNet L2.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstdint>

namespace strata::metal {

// y[t, h] = softplus(alpha[t, h] + dt[h]) * ssm_a[h]. The real call is one token and 48 heads.
void gdn_gate(const Bytes& alpha, const Bytes& dt, const Bytes& ssm_a, Bytes& gate, int64_t n_tokens, int64_t h_v);

// x[i] *= s, in place. The layer passes the q length, not the whole conv output.
void scale_inplace(Bytes& x, int64_t n, float s);

// x[i] = x[i] / (1 + exp(-x[i])), in place.
void silu_inplace(Bytes& x, int64_t n);

// Round-to-nearest-even. The two conversions are not one routine with a flag.
void f32_to_f16_bulk(const Bytes& x, Bytes& y, int64_t n);
void f32_to_bf16_bulk(const Bytes& x, Bytes& y, int64_t n);

// One row of a packed embedding. Codes are low bits first, 2, 4, or 8 per byte. The multiply and the
// add round separately. `offsets` null means +0. `out_base` is the first float written.
void embedding_gather(const Bytes& codes, const Bytes& scales, const Bytes* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, int64_t row, Bytes& out, int64_t out_base);

// dst[i] += src[i]. The expert path adds the resident half to the pool half.
void add_inplace(Bytes& dst, const Bytes& src, int64_t n);

// In place. y = x / sqrt(mean(x^2) + eps) * w, over the last axis. `w` null skips the multiply.
void rms_norm_weighted(Bytes& x, const Bytes* w, int64_t rows, int64_t cols, float eps);

}  // namespace strata::metal
