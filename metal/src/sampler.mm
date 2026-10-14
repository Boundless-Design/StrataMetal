// metal/src/sampler.mm - one threadgroup per row.
#include "strata/metal/sampler.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct SamplerP {
    int n_vocab;
    int k;
    int min_keep;
    int hlen;
    int history_len;
    int greedy;
    int bits_words;
    float top_p;
    float min_p;
    float inv_t;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    uint32_t seed_lo;
    uint32_t seed_hi;
    uint32_t counter_lo;
    uint32_t counter_hi;
};
static_assert(sizeof(SamplerP) == 68, "SamplerP must match the shader");

constexpr int kMaxK = 64;
constexpr int kPenaltyVocab = 248320;
constexpr int kThreads = 512;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void sample_tokens(const Bytes& logits, int n_tokens, int n_vocab, const Bytes* history, int history_len,
                   const SamplerParams& p, Bytes& out) {
    if (n_tokens < 0 || n_vocab < 0 || history_len < 0) throw MetalError("sample_tokens: negative size");
    if (n_tokens == 0 || n_vocab == 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0))
        throw MetalError("sample_tokens: penalty_last_n needs a history");

    int hlen = 0;
    if (p.penalty_last_n > 0) hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
    int k = (p.top_k > 0 && p.top_k < kMaxK) ? p.top_k : kMaxK;
    if (k > n_vocab) k = n_vocab;
    const bool scan = hlen > 0 && n_vocab > kPenaltyVocab;

    SamplerP dim{};
    dim.n_vocab = n_vocab;
    dim.k = k;
    dim.min_keep = p.min_keep;
    dim.hlen = hlen;
    dim.history_len = history_len;
    dim.greedy = (p.greedy || p.temperature <= 0.0f) ? 1 : 0;
    dim.bits_words = scan || hlen == 0 ? 0 : (n_vocab + 31) / 32;
    dim.top_p = p.top_p;
    dim.min_p = p.min_p;
    dim.inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    dim.penalty_repeat = p.penalty_repeat;
    dim.penalty_freq = p.penalty_freq;
    dim.penalty_present = p.penalty_present;
    dim.seed_lo = (uint32_t) p.seed;
    dim.seed_hi = (uint32_t) (p.seed >> 32);
    dim.counter_lo = (uint32_t) p.counter;
    dim.counter_hi = (uint32_t) (p.counter >> 32);

    need("logits", logits, (uint64_t) n_tokens * (uint64_t) n_vocab * 4u);
    need("out", out, (uint64_t) n_tokens * 4u);
    Bytes dummy(16);
    const Bytes* hist = &dummy;
    if (hlen > 0) {
        need("history", *history, (uint64_t) n_tokens * (uint64_t) history_len * 4u);
        hist = history;
    }
    const Bytes* bufs[] = {&logits, hist, &out};
    launch(scan ? "sampler_scan" : "sampler_bits", bufs, 3, &dim, sizeof dim,
           Grid{(uint32_t) n_tokens, 1, 1, (uint32_t) kThreads, 1, 1});
}

}  // namespace strata::metal
