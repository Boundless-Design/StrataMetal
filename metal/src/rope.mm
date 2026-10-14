// metal/src/rope.mm - host float64 table, one thread per row on the GPU.
#include "strata/metal/rope.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

struct RpP {
    int head_dim;
    int n_rot;
    int rows;
    int pad;
};
static_assert(sizeof(RpP) == 16, "RpP must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

Grid grid_for(int n) {
    const uint32_t threads = 256;
    const uint32_t groups = (uint32_t) ((n + 255) / 256);
    return Grid{groups == 0 ? 1u : groups, 1, 1, threads, 1, 1};
}

}  // namespace

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    if (n_rot < 2 || (n_rot % 2) != 0) throw MetalError("rope: n_rot must be even and at least 2");
    if (max_pos < 0) throw MetalError("rope: max_pos < 0");
    if (max_pos == 0) return;
    if (!cos_tab || !sin_tab) throw MetalError("rope: the table pointer is null");
    const int pairs = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < pairs; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * (size_t) pairs + (size_t) i] = (float) std::cos(ang);
            sin_tab[(size_t) p * (size_t) pairs + (size_t) i] = (float) std::sin(ang);
        }
    }
}

void rope_neox_apply(Bytes& out, const Bytes& x, int rows, int head_dim, int n_rot, const Bytes& cos_tab,
                     const Bytes& sin_tab, const Bytes& pos) {
    if (rows < 0) throw MetalError("rope: rows < 0");
    if (rows == 0) return;
    if (head_dim < 1) throw MetalError("rope: head_dim < 1");
    if (n_rot < 2 || (n_rot % 2) != 0 || n_rot > head_dim)
        throw MetalError("rope: n_rot must be even, at least 2, and no wider than head_dim");
    const int pairs = n_rot / 2;
    const uint64_t row_bytes = (uint64_t) rows * (uint64_t) head_dim * 4u;
    need("x", x, row_bytes);
    need("out", out, row_bytes);
    need("pos", pos, (uint64_t) rows * 4u);
    if (cos_tab.size() < 4u || sin_tab.size() < 4u) throw MetalError("rope: the table is empty");
    if (cos_tab.size() / 4u % (uint64_t) pairs != 0 || sin_tab.size() / 4u % (uint64_t) pairs != 0)
        throw MetalError("rope: the table length is not a whole number of positions");
    if (cos_tab.size() != sin_tab.size()) throw MetalError("rope: cos and sin tables differ in length");
    const int n_pos = (int) (cos_tab.size() / 4u / (uint64_t) pairs);
    const int* pp = (const int*) pos.data();
    for (int r = 0; r < rows; ++r) {
        if (pp[r] < 0 || pp[r] >= n_pos) throw MetalError("rope: a position is outside the table");
    }
    const RpP dim{head_dim, n_rot, rows, 0};
    const Bytes* bufs[] = {&out, &x, &cos_tab, &sin_tab, &pos};
    launch("rope_neox_rows", bufs, 5, &dim, sizeof dim, grid_for(rows));
}

}  // namespace strata::metal
