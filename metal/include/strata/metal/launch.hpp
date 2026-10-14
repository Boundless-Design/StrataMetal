// metal/include/strata/metal/launch.hpp - compile the shaders once and run one kernel to completion.
#pragma once

#include "strata/metal/bytes.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::metal {

struct Grid {
    uint32_t groups_x = 1, groups_y = 1, groups_z = 1;
    uint32_t threads_x = 1, threads_y = 1, threads_z = 1;
};

// `buffers` are bound in order at buffer(0).. ; `params` is the next buffer, as raw bytes.
// Waits until the command buffer completes. Throws MetalError on a compiler, launch, or GPU error.
void launch(const char* function, const Bytes* const* buffers, int n_buffers, const void* params, size_t param_bytes,
            Grid grid);

}  // namespace strata::metal
