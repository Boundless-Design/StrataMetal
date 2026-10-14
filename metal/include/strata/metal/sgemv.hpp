// metal/include/strata/metal/sgemv.hpp - the dense projection the GDN layer calls.
//
// The pack stores those rows in the canonical S form (docs/pack-format.md): a code, a scale, and sometimes
// an offset. attn_qkv is IQ4_XS (4-bit, group 32, no offset), attn_gate is Q4_K (4-bit, group 32, offset),
// ssm_out is Q5_K (8-bit, group 32, offset). The activation is the weight's vec_dot_type: Q8_K for those
// three, Q8_0 for a legacy block whose width is not a multiple of 256.
//
// value = codebook(code) * scale + offset, then times the dequantized activation. The offset is applied
// before that multiply. Q8_K is 292 bytes per 256 values: f32 scale, 256 int8, 16 int16 block sums. The
// dot does not read the sums. The scale is -127/max, and the rounding is half to even.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::metal {

enum class Codebook : int { Affine = 0, Iq4Nl = 1 };

struct SForm {
    int code_bits = 4;
    int code_bias = 0;
    int group_elems = 32;
    Codebook codebook = Codebook::Affine;
    bool has_offset = false;
};

// 292 bytes per 256 values. n is a positive multiple of 256.
size_t q8k_bytes(int64_t n);
// 34 bytes per 32 values: fp16 scale, 32 int8. n is a positive multiple of 32.
size_t q8_0_bytes(int64_t n);
size_t sgemv_code_bytes(const SForm& form, int64_t n_in, int64_t n_out);
size_t sgemv_group_bytes(const SForm& form, int64_t n_in, int64_t n_out);

// GPU. Half-to-even, scale -127/max, block sums written. A zero block is all zeros.
void quantize_q8_k(const Bytes& x, int64_t n, Bytes& blocks);
void dequant_q8_k(const Bytes& blocks, int64_t n, Bytes& x);

// Host. This GPU has no fp64, and the Q8_0 rule divides in double before rint. The stored scale is fp16.
void quantize_q8_0(const Bytes& x, int64_t n, Bytes& blocks);

// One thread per output row. y[row] = W[row] . x. x is Q8_K.
void s_gemv_q8k(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                int64_t n_in, int64_t n_out, const SForm& form);

// One simdgroup per output row. Same values, added in the warp order.
void s_gemv_q8k_split(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                      int64_t n_in, int64_t n_out, const SForm& form);

// The same warp kernel over a Q8_0 activation. n_in is a multiple of 32, not necessarily of 256.
void s_gemv_q8_0_split(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                       int64_t n_in, int64_t n_out, const SForm& form);

// fp16 activation, one ushort per value, one thread per row. The Q8_K product has to stay away from this one.
void s_gemv_f16(const Bytes& x, const Bytes& codes, const Bytes& scales, const Bytes* offset, Bytes& y,
                int64_t n_in, int64_t n_out, const SForm& form);

}  // namespace strata::metal
