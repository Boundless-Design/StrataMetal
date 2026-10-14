// metal/include/strata/metal/fit.hpp - published model sizes against this GPU's working set.
#pragma once

#include "strata/metal/device.hpp"

namespace strata::metal {

// Prints one block to stdout. Expert sizes are setup.py's arena_gb (decimal GB). KV bytes per token and the
// GDN recurrent state come from include/strata/plan/plan.hpp, so a geometry change shows up here without a
// second copy of the formula. Dense weights and the draft layer are added only where a doc states a figure.
void print_fit(const DeviceInfo& device);

}  // namespace strata::metal
