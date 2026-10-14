// metal/src/moe_combine.mm - one thread per output element. k is at most 64.
#include "strata/metal/moe_combine.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct MoeCombineP {
    int n_embd;
    int k;
    int has_shared;
    int pad;
};
static_assert(sizeof(MoeCombineP) == 16, "MoeCombineP must match the shader");

constexpr int kMaxK = 64;
constexpr int kThreads = 256;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void moe_combine(const Bytes& parts, const Bytes& weights, const Bytes* shared, Bytes& y, int n_embd, int k) {
    if (n_embd < 0 || k < 0) throw MetalError("moe_combine: negative size");
    if (n_embd == 0 || k == 0) return;
    if (k > kMaxK) throw MetalError("moe_combine: k exceeds 64");

    need("parts", parts, (uint64_t) k * (uint64_t) n_embd * 4u);
    need("weights", weights, (uint64_t) k * 4u);
    need("y", y, (uint64_t) n_embd * 4u);
    const Bytes dummy(16);
    const Bytes* shared_buf = shared;
    if (shared) need("shared", *shared, (uint64_t) n_embd * 4u);
    else shared_buf = &dummy;

    const MoeCombineP dim{n_embd, k, shared ? 1 : 0, 0};
    const uint32_t groups = (uint32_t) ((n_embd + kThreads - 1) / kThreads);
    const Bytes* bufs[] = {&parts, &weights, shared_buf, &y};
    launch("moe_combine", bufs, 4, &dim, sizeof dim, Grid{groups, 1, 1, (uint32_t) kThreads, 1, 1});
}

}  // namespace strata::metal
