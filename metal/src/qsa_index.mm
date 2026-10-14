// metal/src/qsa_index.mm - per-cell indexer scores.
#include "strata/metal/qsa_index.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct QiP {
    int idx_n_head;
    int idx_dim;
    int r;
    int n_bid;
    int n_kv;
    int has_bias;
    int pad0;
    int pad1;
};
static_assert(sizeof(QiP) == 32, "QiP must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

}  // namespace

void qsa_index(const Bytes& pooled, int n_bid, const Bytes& q_idx, const Bytes* bias, int idx_n_head, int idx_dim,
               int idx_block, int n_kv, Bytes& cell_scores) {
    if (idx_n_head <= 0 || idx_dim <= 0 || idx_block < 2) throw MetalError("qsa_index: geometry is not set up");
    if (idx_n_head > 32) throw MetalError("qsa_index: idx_n_head exceeds 32");
    if (n_bid < 0 || n_kv <= 0) throw MetalError("qsa_index: n_bid < 0 or n_kv <= 0");
    if ((n_kv / idx_block) != n_bid) throw MetalError("qsa_index: n_bid is not n_kv / idx_block");
    const uint64_t row = (uint64_t) idx_dim * 4u;
    need("pooled", pooled, (uint64_t) (n_bid + 1) * row);
    need("q_idx", q_idx, (uint64_t) idx_n_head * row);
    need("cell_scores", cell_scores, (uint64_t) n_kv * 4u);
    Bytes dummy(16);
    const bool has = bias != nullptr;
    if (has) need("bias", *bias, (uint64_t) (n_bid + 1) * 4u);
    const QiP dim{idx_n_head, idx_dim, idx_block, n_bid, n_kv, has ? 1 : 0, 0, 0};
    const Bytes* bufs[] = {&pooled, &q_idx, has ? bias : &dummy, &cell_scores};
    const uint32_t threads = (uint32_t) idx_n_head * 32u;
    launch("qsa_index_cells", bufs, 4, &dim, sizeof dim,
           Grid{(uint32_t) (n_bid + 1), 1, 1, threads, 1, 1});
}

}  // namespace strata::metal
