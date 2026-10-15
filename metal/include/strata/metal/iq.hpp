// metal/include/strata/metal/iq.hpp - the Coder expert product.
//
// The pack is named IQ1_M. Its kept experts are stored like IQ3_S: gate and up rows are IQ2_S,
// IQ3_XXS, IQ3_S or IQ4_XS, and down rows are IQ4_NL or Q2_0. The activation is q8_1 (32 int8
// values, an fp16 scale, an fp16 sum). The dot is the integer product in iq_kernels.cu, then the
// fp16 scales. One expert is gate, up, SiLU(gate) * up, quantize that hidden row, then down.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::metal {

// ggml type ids the Coder expert uses.
constexpr int kIq3Xxs = 18;
constexpr int kIq4Nl = 20;
constexpr int kIq3S = 21;
constexpr int kIq2S = 22;
constexpr int kIq4Xs = 23;
constexpr int kQ2_0 = 42;
// ggml blocks that are not a Coder expert type. The dense gate is Q4_K and the dense output is Q5_K.
// The shared-expert up row is Q3_K and the shared-expert down row is Q5_0. iq_mmvq dispatches them.
// iq_type_supported stays false, so iq_expert still refuses them.
constexpr int kQ4_K = 12;
constexpr int kQ5_K = 13;
constexpr int kQ3_K = 11;
constexpr int kQ5_0 = 6;

bool iq_type_supported(int ggml_type) noexcept;
// Values in one block. IQ4_NL is 32, Q2_0 is 64, the others are 256.
int iq_block_values(int ggml_type);
size_t iq_block_bytes(int ggml_type);
// `n` values, a multiple of the block. Throws MetalError on an unknown type or a bad multiple.
size_t iq_row_bytes(int ggml_type, int64_t n);
// One q8_1 block is 36 bytes: fp16 scale, fp16 sum, 32 int8.
size_t iq_q8_1_bytes(int64_t n);

// Flat row, `n` a multiple of 32. `y` is n/32 blocks.
void iq_quantize_q8_1(const Bytes& x, int64_t n, Bytes& y);

// y[c * n_out + row] = W[row] . x[c]. W is `n_out` rows of `n_in` values. x is n_col q8_1 rows.
// `byte_off` is the first weight byte inside `w` (the up and down matrices sit behind the gate).
void iq_mmvq(int ggml_type, const Bytes& w, int64_t byte_off, const Bytes& x_q8, int64_t n_in, int64_t n_out,
             int64_t n_col, Bytes& y);

// SiLU(gate) * up, stored as q8_1. n is a positive multiple of 32.
void iq_swiglu_q8(const Bytes& gate, const Bytes& up, int64_t n, Bytes& y);

// Blob: [gate rows | up rows | down rows], each row `iq_row_bytes` of that side.
// x_q8 is n_tok rows of n_embd. y is n_tok rows of n_embd, floats.
struct IqExpert {
    int gu_type = 0;
    int d_type = 0;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
};
size_t iq_expert_bytes(const IqExpert& e);
void iq_expert(const Bytes& blob, const Bytes& x_q8, const IqExpert& e, int64_t n_tok, Bytes& y);

}  // namespace strata::metal
