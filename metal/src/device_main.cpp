// metal/src/device_main.cpp - `strata-metal-device`: the GPU, then the resident-memory plan.
#include "strata/metal/device.hpp"
#include "strata/metal/fit.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

std::string gib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

void print_device(const strata::metal::DeviceInfo& d) {
    std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
    std::printf("  unified memory      %s\n", d.unified ? "yes" : "no");
    std::printf("  GPU family          Apple%d, Metal3 %s, Metal4 %s\n", d.apple_family,
                d.metal3 ? "yes" : "no", d.metal4 ? "yes" : "no");
    std::printf("  physical memory     %s\n", gib(d.physical_bytes).c_str());
    std::printf("  GPU working set     %s  (recommendedMaxWorkingSetSize)\n", gib(d.working_set_bytes).c_str());
    std::printf("  longest buffer      %s\n", gib(d.max_buffer_bytes).c_str());
    std::printf("  allocated now       %s\n", gib(d.allocated_bytes).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    bool do_selftest = false;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) do_selftest = true;
        else if (std::strcmp(argv[i], "--list-devices") == 0) list = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: strata-metal-device [--selftest] [--list-devices]\n"
                        "  names the GPU and prints which published model sizes fit its working set.\n"
                        "  --selftest  copy 1 MiB through a shared-buffer blit and check the bytes.\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    try {
        if (list) {
            const int n = strata::metal::device_count();
            if (n == 0) std::printf("(no Metal device)\n");
            for (int i = 0; i < n; ++i) {
                strata::metal::DeviceInfo d;
                std::string err;
                if (!strata::metal::device_summary(i, d, err)) {
                    std::printf("device %d: %s\n", i, err.c_str());
                    continue;
                }
                print_device(d);
            }
            return 0;
        }

        const strata::metal::DeviceInfo d = strata::metal::device_info(0);
        print_device(d);
        strata::metal::print_fit(d);
        if (do_selftest) {
            const std::string err = strata::metal::selftest(0);
            if (!err.empty()) {
                std::fprintf(stderr, "selftest: FAIL: %s\n", err.c_str());
                return 1;
            }
            std::printf("\nselftest: PASS (1 MiB shared buffer, blit, readback)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-metal-device: %s\n", e.what());
        return 1;
    }
    return 0;
}
