// metal/src/qsa_gate.mm - sigmoid of the second half, then the multiply.
#include "strata/metal/qsa_gate.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct QgP {
    int n_head;
    int head_dim;
    int pad0;
    int pad1;
};
static_assert(sizeof(QgP) == 16, "QgP must match the shader");

constexpr int kThreads = 256;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void qsa_gate_apply_f32(const Bytes& attn, const Bytes& q_full, int n_head, int head_dim, Bytes& out) {
    if (n_head < 0 || head_dim < 0) throw MetalError("qsa_gate_apply: negative size");
    if (n_head == 0 || head_dim == 0) return;
    const uint64_t n = (uint64_t) n_head * (uint64_t) head_dim;
    need("attn", attn, n * 4u);
    need("q_full", q_full, n * 2u * 4u);
    need("out", out, n * 4u);
    const QgP dim{n_head, head_dim, 0, 0};
    const uint32_t groups = (uint32_t) ((n + kThreads - 1) / kThreads);
    const Bytes* bufs[] = {&attn, &q_full, &out};
    launch("qsa_gate_apply_f32", bufs, 3, &dim, sizeof dim, Grid{groups, 1, 1, (uint32_t) kThreads, 1, 1});
}

void qsa_gate_apply(const Bytes& attn, const Bytes& q_full, int n_head, int head_dim, Bytes& out) {
    if (n_head < 0 || head_dim < 0) throw MetalError("qsa_gate_apply: negative size");
    if (n_head == 0 || head_dim == 0) return;
    const uint64_t n = (uint64_t) n_head * (uint64_t) head_dim;
    need("out", out, n * 2u);
    Bytes f32(n * 4u);
    qsa_gate_apply_f32(attn, q_full, n_head, head_dim, f32);
    f32_to_f16_bulk(f32, out, (int64_t) n);
}

}  // namespace strata::metal
