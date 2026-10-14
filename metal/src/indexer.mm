// metal/src/indexer.mm - append one raw indexer key.
// The tail store and the NEOX rotation are kernels. The mean and the RMS are IEEE double
// on the CPU, because a shader on this GPU has no fp64 and the spare key is compared bit for bit.
#include "strata/metal/indexer.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::metal {
namespace {

struct IxTailP {
    int idx_dim;
    int r;
    int pos;
    int pad;
};
static_assert(sizeof(IxTailP) == 16, "IxTailP must match the shader");

struct IxRotP {
    int idx_dim;
    int pairs;
    int row;
    int pos;
};
static_assert(sizeof(IxRotP) == 16, "IxRotP must match the shader");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

void write_spare(const float* raw, int n, const float* w, float eps, const float* cos_row, int n_rot,
                 float* dead, float* pooled0) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = (double) raw[i];
        const volatile double p = v * v;
        ss = ss + p;
    }
    const volatile double q = ss / (double) n;
    const volatile double arg = q + (double) eps;
    const volatile double root = std::sqrt(arg);
    const volatile double inv = 1.0 / root;
    const int half = n_rot / 2;
    for (int d = 0; d < n; ++d) {
        const volatile double a = (double) raw[d] * inv;
        const volatile double b = a * (double) w[d];
        float y = (float) b;
        if (d < n_rot) y *= cos_row[d % half];
        dead[d] = y;
        pooled0[d] = y;
    }
}

void write_block(const float* tail, const float* raw, int r, int n, const float* w, float eps, float* row) {
    std::vector<double> mean((size_t) n);
    for (int d = 0; d < n; ++d) {
        double m = 0.0;
        for (int j = 0; j < r - 1; ++j) {
            const volatile double v = (double) tail[(size_t) j * (size_t) n + (size_t) d];
            m = m + v;
        }
        const volatile double rawd = (double) raw[d];
        m = m + rawd;
        const volatile double div = m / (double) r;
        mean[(size_t) d] = div;
    }
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = mean[(size_t) i];
        const volatile double p = v * v;
        ss = ss + p;
    }
    const volatile double q = ss / (double) n;
    const volatile double arg = q + (double) eps;
    const volatile double root = std::sqrt(arg);
    const volatile double inv = 1.0 / root;
    for (int d = 0; d < n; ++d) {
        const volatile double a = mean[(size_t) d] * inv;
        const volatile double b = a * (double) w[d];
        row[d] = (float) b;
    }
}

}  // namespace

void indexer_key_append(const Bytes& raw, const Bytes& pos, int pos_base, const Bytes& w_k_norm, float eps,
                        IndexerBuf buf, int idx_dim, int idx_block, int n_rot, const Bytes& cos_tab,
                        const Bytes& sin_tab) {
    if (idx_dim <= 0 || idx_block < 2 || n_rot <= 0) throw MetalError("indexer_key_append: geometry is not set up");
    if (idx_dim > 1024) throw MetalError("indexer_key_append: idx_dim exceeds 1024");
    if ((n_rot % 2) != 0 || n_rot > idx_dim) throw MetalError("indexer_key_append: n_rot must be even and <= idx_dim");
    const uint64_t row_bytes = (uint64_t) idx_dim * 4u;
    need("raw", raw, row_bytes);
    need("pos", pos, 4u);
    need("w_k_norm", w_k_norm, row_bytes);
    need("tail", buf.tail, (uint64_t) (idx_block - 1) * row_bytes);
    need("dead", buf.dead, row_bytes);
    need("block_pos", buf.block_pos, 4u);
    int cell = 0;
    std::memcpy(&cell, pos.data(), 4);
    if (cell < 0) throw MetalError("indexer_key_append: pos < 0");
    const int slot = cell % idx_block;
    const int half = n_rot / 2;
    need("cos_tab", cos_tab, (uint64_t) half * 4u);
    need("sin_tab", sin_tab, (uint64_t) half * 4u);

    const IxTailP tail_dim{idx_dim, idx_block, cell, 0};
    const Bytes* tail_bufs[] = {&raw, &buf.tail};
    launch("ix_tail", tail_bufs, 2, &tail_dim, sizeof tail_dim,
           Grid{1, 1, 1, (uint32_t) idx_dim, 1, 1});

    const float* raw_h = (const float*) raw.data();
    const float* w = (const float*) w_k_norm.data();
    const float* cos_h = (const float*) cos_tab.data();
    float* dead = (float*) buf.dead.data();
    float* pooled = (float*) buf.pooled.data();
    if (cell == 0) {
        need("pooled", buf.pooled, row_bytes);
        write_spare(raw_h, idx_dim, w, eps, cos_h, n_rot, dead, pooled);
    }
    if (slot != idx_block - 1) return;

    const int b = cell / idx_block;
    const int rotary = pos_base + b * idx_block;
    if (rotary < 0) throw MetalError("indexer_key_append: rotation position is negative");
    const uint64_t need_tab = ((uint64_t) rotary + 1u) * (uint64_t) half * 4u;
    need("cos_tab", cos_tab, need_tab);
    need("sin_tab", sin_tab, need_tab);
    const uint64_t pooled_bytes = (uint64_t) (b + 2) * row_bytes;
    need("pooled", buf.pooled, pooled_bytes);

    const float* tail_h = (const float*) buf.tail.data();
    write_block(tail_h, raw_h, idx_block, idx_dim, w, eps, pooled + (size_t) b * (size_t) idx_dim);
    std::memcpy(pooled + (size_t) (b + 1) * (size_t) idx_dim, dead, (size_t) row_bytes);
    std::memcpy(buf.block_pos.data(), &rotary, 4);

    const IxRotP rot{idx_dim, half, b, rotary};
    const Bytes* rot_bufs[] = {&buf.pooled, &cos_tab, &sin_tab};
    launch("ix_rotate", rot_bufs, 3, &rot, sizeof rot, Grid{1, 1, 1, (uint32_t) half, 1, 1});
}

}  // namespace strata::metal
