// metal/src/s2_gemv.mm - one threadgroup per output row.
#include "strata/metal/s2_gemv.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/sgemv.hpp"

#include <string>

namespace strata::metal {
namespace {

struct S2P {
    int n_in;
    int n_out;
    int tpr;
    int pad;
};
static_assert(sizeof(S2P) == 16, "S2P must match the shader");

constexpr int kMaxThreads = 1024;

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

bool power_of_two(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace

void s2_gemv_q8(const Bytes& act, const Bytes& codes, const Bytes& scales, Bytes& y, int64_t n_in, int64_t n_out,
                int threads_per_row) {
    if (n_in < 0 || n_out < 0 || threads_per_row < 0) throw MetalError("s2_gemv_q8: negative size");
    if (n_in == 0 || n_out == 0) return;
    if (n_in % 64 != 0) throw MetalError("s2_gemv_q8: n_in is not a multiple of 64");
    if (!power_of_two(threads_per_row) || threads_per_row > kMaxThreads)
        throw MetalError("s2_gemv_q8: threads_per_row must be a power of two from 1 to 1024");
    if (n_in > 0x7fffffff || n_out > 0x7fffffff) throw MetalError("s2_gemv_q8: dimension does not fit in int");

    need("act", act, q8_0_bytes(n_in));
    need("codes", codes, (uint64_t) n_out * (uint64_t) (n_in / 4));
    need("scales", scales, (uint64_t) n_out * (uint64_t) (n_in / 64) * 4u);
    need("y", y, (uint64_t) n_out * 4u);

    const S2P dim{(int) n_in, (int) n_out, threads_per_row, 0};
    const Bytes* bufs[] = {&act, &codes, &scales, &y};
    launch("s2_gemv_q8", bufs, 4, &dim, sizeof dim,
           Grid{(uint32_t) n_out, 1, 1, (uint32_t) threads_per_row, 1, 1});
}

}  // namespace strata::metal
