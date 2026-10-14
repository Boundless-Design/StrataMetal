// metal/include/strata/metal/router.hpp - the MoE router, logits already in f32.
//
// One token, or a batch of them. Softmax over every expert, then the top k, ties to the
// lower index, then the weights divided by their sum. The sum is clamped at 2**-14.
// The BF16 product that fills the logits is a separate kernel.
// The native 512-by-10 router is not in this port. n_expert above 1024 is refused:
// the probabilities sit in threadgroup memory.
#pragma once

#include "strata/metal/bytes.hpp"

namespace strata::metal {

// logits is (n_tokens, n_expert) f32. ids is (n_tokens, k) i32. weights is (n_tokens, k) f32.
// k is at most 64. A zero count returns before launch. A negative count throws.
void router_top10(const Bytes& logits, int n_tokens, int n_expert, int k, Bytes& ids, Bytes& weights);

}  // namespace strata::metal
