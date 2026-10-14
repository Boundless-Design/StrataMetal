// metal/include/strata/metal/device.hpp - the GPU this Mac actually has.
//
// On a discrete card, cudaMemGetInfo is the number the planner spends. Here the CPU and the GPU share one
// pool, and the number Metal will let a process wire is recommendedMaxWorkingSetSize, not the RAM sticker
// and not whatever is free this second. The fit printer spends that number. A plan that used the sticker
// would promise a model the GPU is not allowed to wire.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::metal {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    bool unified = false;
    bool metal3 = false;
    bool metal4 = false;
    int apple_family = 0;                 // highest MTLGPUFamilyAppleN this GPU reports, else 0
    uint64_t physical_bytes = 0;          // hw.memsize, the RAM sticker
    uint64_t working_set_bytes = 0;       // recommendedMaxWorkingSetSize
    uint64_t max_buffer_bytes = 0;
    uint64_t allocated_bytes = 0;         // this process, at query time
};

class MetalError : public std::runtime_error {
public:
    explicit MetalError(const std::string& what) : std::runtime_error(what) {}
};

// How many GPUs Metal enumerates. 0 when there is no device.
int device_count();

// false when `ordinal` is out of range. `err` says why.
bool device_summary(int ordinal, DeviceInfo& out, std::string& err);

// Throws MetalError when there is no device, or `ordinal` is out of range.
DeviceInfo device_info(int ordinal = 0);

// Copy one shared buffer onto another through a blit command buffer and read it back.
// "" when the bytes match. A kernel is not involved: this checks the device, the shared
// storage mode, and the command queue, which is what a later kernel launch sits on.
std::string selftest(int ordinal = 0);

}  // namespace strata::metal
