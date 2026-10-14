// metal/src/kv.mm - paged fp16 append and gather.
#include "strata/metal/kv.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

struct KvP {
    int n_head_kv;
    int head_dim;
    int page_size;
    int pos;
};
static_assert(sizeof(KvP) == 16, "KvP must match the shader");

struct KgP {
    int n_head_kv;
    int head_dim;
    int page_size;
    int n_ids;
};
static_assert(sizeof(KgP) == 16, "KgP must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

void geometry(int n_head_kv, int head_dim, int page_size) {
    if (n_head_kv < 1 || head_dim < 1 || page_size < 1) throw MetalError("kv: geometry is not set up");
    if (head_dim % 4 != 0) throw MetalError("kv: head_dim must be a multiple of 4");
}

int pages_in(const Bytes& table) { return (int) (table.size() / 4u); }

void check_page(const Bytes& table, int logical, const char* what) {
    if (logical < 0 || logical >= pages_in(table))
        throw MetalError(std::string(what) + ": page is outside the table");
}

Grid grid_for(int n) {
    const uint32_t threads = 256;
    const uint32_t groups = (uint32_t) ((n + 255) / 256);
    return Grid{groups == 0 ? 1u : groups, 1, 1, threads, 1, 1};
}

}  // namespace

void kv_append(Bytes& k_pool, Bytes& v_pool, const Bytes& page_table, int pos, const Bytes& kcur, const Bytes& vcur,
               int n_head_kv, int head_dim, int page_size) {
    geometry(n_head_kv, head_dim, page_size);
    if (pos < 0) throw MetalError("kv_append: pos < 0");
    const int logical = pos / page_size;
    check_page(page_table, logical, "kv_append");
    const int n = n_head_kv * head_dim;
    need("kcur", kcur, (uint64_t) n * 4u);
    need("vcur", vcur, (uint64_t) n * 4u);
    int page = 0;
    std::memcpy(&page, (const char*) page_table.data() + (size_t) logical * 4u, 4);
    if (page >= 0) {
        const uint64_t rows = ((uint64_t) page * (uint64_t) n_head_kv + (uint64_t) (n_head_kv - 1)) *
                                  (uint64_t) page_size +
                              (uint64_t) (pos % page_size) + 1u;
        const uint64_t bytes = rows * (uint64_t) head_dim * 2u;
        need("k_pool", k_pool, bytes);
        need("v_pool", v_pool, bytes);
    }
    const KvP dim{n_head_kv, head_dim, page_size, pos};
    const Bytes* bufs[] = {&k_pool, &v_pool, &page_table, &kcur, &vcur};
    launch("kv_append_cell", bufs, 5, &dim, sizeof dim, grid_for(n));
}

void kv_gather(const Bytes& k_pool, const Bytes& v_pool, const Bytes& page_table, const Bytes& ids, int n_ids,
               int n_head_kv, int head_dim, int page_size, Bytes& k_scratch, Bytes& v_scratch) {
    geometry(n_head_kv, head_dim, page_size);
    if (n_ids < 0) throw MetalError("kv_gather: n_ids < 0");
    if (n_ids == 0) return;
    need("ids", ids, (uint64_t) n_ids * 4u);
    const int* idp = (const int*) ids.data();
    int max_page = -1;
    for (int i = 0; i < n_ids; ++i) {
        if (idp[i] < 0) throw MetalError("kv_gather: negative cell");
        const int logical = idp[i] / page_size;
        check_page(page_table, logical, "kv_gather");
        int page = 0;
        std::memcpy(&page, (const char*) page_table.data() + (size_t) logical * 4u, 4);
        if (page < 0) throw MetalError("kv_gather: the cell's physical page is negative");
        if (page > max_page) max_page = page;
    }
    const uint64_t pool_rows = ((uint64_t) max_page * (uint64_t) n_head_kv + (uint64_t) (n_head_kv - 1)) *
                                   (uint64_t) page_size +
                               (uint64_t) page_size;
    need("k_pool", k_pool, pool_rows * (uint64_t) head_dim * 2u);
    need("v_pool", v_pool, pool_rows * (uint64_t) head_dim * 2u);
    const uint64_t scratch = (uint64_t) n_ids * (uint64_t) n_head_kv * (uint64_t) head_dim * 2u;
    need("k_scratch", k_scratch, scratch);
    need("v_scratch", v_scratch, scratch);
    const KgP dim{n_head_kv, head_dim, page_size, n_ids};
    const Bytes* bufs[] = {&k_pool, &v_pool, &page_table, &ids, &k_scratch, &v_scratch};
    launch("kv_gather_cells", bufs, 6, &dim, sizeof dim, grid_for(n_ids * n_head_kv * head_dim));
}

}  // namespace strata::metal
