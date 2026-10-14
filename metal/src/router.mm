// metal/src/router.mm - one threadgroup per token, logits already f32.
#include "strata/metal/router.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct RouterP {
    int n_tokens;
    int n_expert;
    int k;
    int nt;
};
static_assert(sizeof(RouterP) == 16, "RouterP must match the shader");

constexpr int kMaxExpert = 1024;
constexpr int kMaxK = 64;
constexpr int kMaxThreads = 512;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void router_top10(const Bytes& logits, int n_tokens, int n_expert, int k, Bytes& ids, Bytes& weights) {
    if (n_tokens < 0 || n_expert < 0 || k < 0) throw MetalError("router_top10: negative size");
    if (n_tokens == 0 || n_expert == 0 || k == 0) return;
    if (k > kMaxK) throw MetalError("router_top10: k exceeds 64");
    if (n_expert > kMaxExpert) throw MetalError("router_top10: n_expert exceeds 1024");
    if (k > n_expert) throw MetalError("router_top10: k exceeds n_expert");

    need("logits", logits, (uint64_t) n_tokens * (uint64_t) n_expert * 4u);
    need("ids", ids, (uint64_t) n_tokens * (uint64_t) k * 4u);
    need("weights", weights, (uint64_t) n_tokens * (uint64_t) k * 4u);

    int threads = n_expert < kMaxThreads ? n_expert : kMaxThreads;
    threads = (threads + 31) & ~31;
    const RouterP dim{n_tokens, n_expert, k, threads};
    const Bytes* bufs[] = {&logits, &ids, &weights};
    launch("router_top10", bufs, 3, &dim, sizeof dim,
           Grid{(uint32_t) n_tokens, 1, 1, (uint32_t) threads, 1, 1});
}

}  // namespace strata::metal
