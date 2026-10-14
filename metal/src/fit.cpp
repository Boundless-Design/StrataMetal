// metal/src/fit.cpp - does a published size fit this GPU's working set?
//
// The discrete engine's planner spends a VRAM pool (about 5.94 GB on a 12 GB card) on the KV cache and an
// expert cache, and puts every expert in RAM for the CPU. That split is the whole design, and it does not
// exist here: one working set has to hold the experts, the dense weights, the KV cache and the recurrent
// state together. Experts that do not fit are a later SSD read, which this binary does not do.
#include "strata/metal/fit.hpp"

#include "strata/plan/plan.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace strata::metal {
namespace {

// setup.py MODELS[*]["arena_gb"]. Decimal GB, the unit that table is written in.
struct Size {
    const char* name;
    double arena_gb;
    bool coder;  // the one size whose dense copy is published
};

constexpr Size kSizes[] = {
    {"Q2_0", 34.0, false},
    {"IQ2_XS", 35.5, false},
    {"IQ3_XXS", 42.9, false},
    {"IQ3_S", 50.3, false},
    {"Coder IQ1_M", 23.4, true},
    {"UD-IQ4_XS", 59.5, false},
    {"UD-Q4_K_XL", 77.0, false},
};

// docs/MULTI_GPU.md: "a copy of the dense weights (~3.4 GB for the Coder)" and "the output head and the
// draft layer (~0.8 GB)". The tilde is the doc's. Other sizes do not publish a dense figure; their routed
// experts differ and their dense stack is not this number, so it is not added.
constexpr double kCoderDenseGb = 3.4;
constexpr double kDraftGb = 0.8;

uint64_t decimal_gb(double gb) {
    return (uint64_t) std::llround(gb * 1e9);
}

std::string gb_of(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GB", (double) bytes / 1e9);
    return buf;
}

void print_row(const Size& size, uint64_t kv, uint64_t state, uint64_t working_set, uint64_t max_buffer) {
    uint64_t sum = decimal_gb(size.arena_gb) + kv + state + decimal_gb(kDraftGb);
    const bool split_arena = max_buffer != 0 && decimal_gb(size.arena_gb) > max_buffer;
    std::printf("  %-14s experts %5.1f GB", size.name, size.arena_gb);
    if (size.coder) {
        sum += decimal_gb(kCoderDenseGb);
        std::printf(" + dense ~%.1f GB", kCoderDenseGb);
    }
    std::printf(" + KV %s + state %s + draft ~%.1f GB", gb_of(kv).c_str(), gb_of(state).c_str(), kDraftGb);
    std::printf("  = %s", gb_of(sum).c_str());
    if (!size.coder) std::printf("  (dense weights not in this sum)");
    if (working_set == 0) {
        std::printf("  (no working-set figure)\n");
        return;
    }
    if (sum <= working_set) {
        std::printf("  fits (%s of working set left)", gb_of(working_set - sum).c_str());
    } else {
        std::printf("  does not fit (over by %s)", gb_of(sum - working_set).c_str());
    }
    if (split_arena) std::printf("; expert arena is larger than one buffer");
    std::printf("\n");
}

}  // namespace

void print_fit(const DeviceInfo& device) {
    const strata::plan::Geometry geometry;
    const uint64_t per_token = strata::plan::kv_bytes_per_token(geometry);
    const uint64_t state = strata::plan::state_bytes(geometry);

    std::printf("\n");
    std::printf("resident plan. One pool: experts + dense weights + KV + GDN state + draft layer.\n");
    std::printf("Expert GB are setup.py arena_gb (decimal). KV is INT8, %llu B/token, from plan.hpp\n",
                (unsigned long long) per_token);
    std::printf("(%d QSA layers, %d KV heads, head dim %d). GDN state is %s (%d layers).\n",
                geometry.n_qsa_layers, geometry.n_kv_heads, geometry.head_dim, gb_of(state).c_str(),
                geometry.n_gdn_layers());
    std::printf("The 28.8 GB n-gram table stays on the SSD (docs/DETAILS.md) and is not in the sum.\n");
    std::printf("Prompt workspace is not in the sum. A \"fits\" line is this arithmetic against\n");
    std::printf("recommendedMaxWorkingSetSize, not a model that has been loaded on this Mac.\n");
    std::printf("Longest single buffer: %s. An expert arena larger than that is more than one allocation.\n",
                gb_of(device.max_buffer_bytes).c_str());

    const uint64_t contexts[] = {32768ull, 262144ull};
    for (uint64_t ctx : contexts) {
        const uint64_t kv = per_token * ctx;
        std::printf("\ncontext %llu, KV %s:\n", (unsigned long long) ctx, gb_of(kv).c_str());
        for (const Size& size : kSizes)
            print_row(size, kv, state, device.working_set_bytes, device.max_buffer_bytes);
    }
}

}  // namespace strata::metal
