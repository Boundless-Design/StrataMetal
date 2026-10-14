// metal/src/bf16_gemv.mm - launch the three BF16 GEMV kernels.
#include "strata/metal/bf16_gemv.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <string>

namespace strata::metal {
namespace {

void require_bytes(const char* what, const Bytes& buf, uint64_t need) {
    if (buf.size() < need)
        throw MetalError(std::string(what) + " buffer is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(need));
}

void gemv_warp(const Bytes& x, const Bytes& w, Bytes& y, uint32_t n_in, uint32_t n_out) {
    const GemvDim dim{n_in, n_out};
    const Bytes* bufs[] = {&x, &w, &y};
    const uint32_t groups = (n_out + 8u - 1u) / 8u;
    launch("bf16_gemv_warp", bufs, 3, &dim, sizeof(dim), Grid{groups, 1, 1, 256, 1, 1});
}

void gemv_naive(const Bytes& x, const Bytes& w, Bytes& y, uint32_t n_in, uint32_t n_out) {
    const GemvDim dim{n_in, n_out};
    const Bytes* bufs[] = {&x, &w, &y};
    const uint32_t groups = (n_out + 256u - 1u) / 256u;
    launch("bf16_gemv_naive", bufs, 3, &dim, sizeof(dim), Grid{groups, 1, 1, 256, 1, 1});
}

}  // namespace

void bf16_gemv(const Bytes& x, const Bytes& w, Bytes& y, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in > 0x7fffffff || n_out > 0x7fffffff) throw MetalError("bf16_gemv: dimension does not fit in uint32");
    require_bytes("bf16 x", x, (uint64_t) n_in * 2);
    require_bytes("bf16 w", w, (uint64_t) n_in * (uint64_t) n_out * 2);
    require_bytes("bf16 y", y, (uint64_t) n_out * 4);
    if (n_out >= 64) gemv_warp(x, w, y, (uint32_t) n_in, (uint32_t) n_out);
    else gemv_naive(x, w, y, (uint32_t) n_in, (uint32_t) n_out);
}

void bf16_gemv_split(const Bytes& x, const Bytes& w, Bytes& y, int64_t n_in, int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in > 0x7fffffff || n_out > 0x7fffffff) throw MetalError("bf16_gemv_split: dimension does not fit in uint32");
    require_bytes("bf16 x", x, (uint64_t) n_in * 2);
    require_bytes("bf16 w", w, (uint64_t) n_in * (uint64_t) n_out * 2);
    require_bytes("bf16 y", y, (uint64_t) n_out * 4);
    if (threads_per_row == 32) {
        gemv_warp(x, w, y, (uint32_t) n_in, (uint32_t) n_out);
        return;
    }
    if (threads_per_row <= 0 || threads_per_row > 256 || (threads_per_row & (threads_per_row - 1)) != 0)
        throw MetalError("bf16_gemv_split: threads_per_row must be a power of two, at most 256 (32 selects the simdgroup path)");
    const SplitDim dim{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) threads_per_row};
    const Bytes* bufs[] = {&x, &w, &y};
    launch("bf16_gemv_split", bufs, 3, &dim, sizeof(dim),
           Grid{(uint32_t) n_out, 1, 1, (uint32_t) threads_per_row, 1, 1});
}

}  // namespace strata::metal
