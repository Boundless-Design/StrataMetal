// metal/include/strata/metal/sampler.hpp - one token, or a row per draft, from f32 logits.
//
// penalties, then top-k, then top-p, then min-p, then temperature, then one Philox draw.
// Temperature does not move the shortlist. A non-positive temperature, or greedy, is the argmax.
// Ties keep the lower index. The repeat penalty multiplies a non-positive logit and divides a
// positive one. The presence penalty is applied once, not once per occurrence.
// The penalty window is the tail of that row's history.
//
// The cluster argmax, the split top-k, and CUDA graphs are not in this port.
// A penalty bitmap fits 248320 logits, the model vocabulary. A wider row with penalties
// counts the window directly.
#pragma once

#include "strata/metal/bytes.hpp"

#include <algorithm>
#include <cstdint>

namespace strata::metal {

struct SamplerParams {
    int top_k = 20;              // 1..64 as given. 0 or anything wider keeps 64.
    float top_p = 0.95f;         // 1 disables the cut
    float min_p = 0.0f;          // 0 disables. Keeps a prefix with p >= min_p * p_max.
    float temperature = 1.0f;    // <= 0 is the argmax
    int min_keep = 1;
    int penalty_last_n = 0;      // 0 disables. Counts the tail of each history row.
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;
    uint64_t counter = 0;        // draw index of row 0. Row t uses counter + t.
    bool greedy = false;
};

// logits is (n_tokens, n_vocab) f32. out is n_tokens int32 ids.
// history is (n_tokens, history_len) int32, or null. Ids outside [0, n_vocab) are ignored.
// A zero count returns. A negative count throws. penalty_last_n > 0 without a history throws.
void sample_tokens(const Bytes& logits, int n_tokens, int n_vocab, const Bytes* history, int history_len,
                   const SamplerParams& p, Bytes& out);

// Row t is the last h tokens of tail, then window[0..t], most recent last. Unused front slots are -1.
inline void penalty_rows(const int32_t* tail, int64_t n_tail, const int32_t* window, int T, int h, int32_t* out) {
    for (int t = 0; t < T; ++t) {
        int32_t* row = out + (size_t) t * (size_t) h;
        const int64_t avail = n_tail + t + 1;
        const int take = (int) std::min<int64_t>(h, avail);
        std::fill(row, row + (h - take), -1);
        for (int j = 0; j < take; ++j) {
            const int64_t i = avail - take + j;
            row[h - take + j] = i < n_tail ? tail[i] : window[i - n_tail];
        }
    }
}

}  // namespace strata::metal
