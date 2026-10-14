// metal/src/topk.mm - largest cells, ascending ids, lower index on a tie.
#include "strata/metal/topk.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

struct TkP {
    int n_kv;
    int width;
    int pad0;
    int pad1;
};
static_assert(sizeof(TkP) == 16, "TkP must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

int selection_width(int n_kv, int idx_top_k, int idx_block) {
    const int w = idx_top_k + idx_block - 1;
    return n_kv < w ? n_kv : w;
}

}  // namespace

void topk_512(const Bytes& scores, int n_kv, int idx_top_k, int idx_block, int cap, Bytes& ids) {
    if (n_kv < 0 || idx_top_k < 1 || idx_block < 2 || cap < 0) throw MetalError("topk_512: negative size");
    if (n_kv == 0) return;
    if (n_kv > kTopkMaxCells) throw MetalError("topk_512: n_kv exceeds 32768");
    const int width = selection_width(n_kv, idx_top_k, idx_block);
    if (cap < width) throw MetalError("topk_512: cap is below the selection width");
    need("scores", scores, (uint64_t) n_kv * 4u);
    need("ids", ids, (uint64_t) cap * 4u);
    const TkP dim{n_kv, width, 0, 0};
    const Bytes* bufs[] = {&scores, &ids};
    launch("topk_cells", bufs, 2, &dim, sizeof dim, Grid{1, 1, 1, 256, 1, 1});
}

}  // namespace strata::metal
