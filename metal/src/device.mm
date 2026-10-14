// metal/src/device.mm - Metal's view of this Mac. No shader, so the Metal toolchain is not required.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "strata/metal/device.hpp"

#include <sys/sysctl.h>

namespace strata::metal {
namespace {

uint64_t physical_bytes() {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) != 0) return 0;
    return bytes;
}

int highest_apple_family(id<MTLDevice> device) {
    // MTLGPUFamilyApple1 .. Apple11 are 1001 .. 1011, contiguous in the SDK header.
    int highest = 0;
    for (int n = 1; n <= 11; ++n) {
        const MTLGPUFamily family = (MTLGPUFamily) (MTLGPUFamilyApple1 + (n - 1));
        if ([device supportsFamily:family]) highest = n;
    }
    return highest;
}

DeviceInfo describe(id<MTLDevice> device, int ordinal) {
    DeviceInfo d;
    d.ordinal = ordinal;
    d.name = device.name.UTF8String ? device.name.UTF8String : "(unnamed)";
    d.unified = device.hasUnifiedMemory;
    d.metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
    if (@available(macOS 26.0, *)) {
        d.metal4 = [device supportsFamily:MTLGPUFamilyMetal4];
    }
    d.apple_family = highest_apple_family(device);
    d.physical_bytes = physical_bytes();
    d.working_set_bytes = device.recommendedMaxWorkingSetSize;
    d.max_buffer_bytes = device.maxBufferLength;
    d.allocated_bytes = device.currentAllocatedSize;
    return d;
}

}  // namespace

int device_count() {
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    return all ? (int) all.count : 0;
}

bool device_summary(int ordinal, DeviceInfo& out, std::string& err) {
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    if (!all || ordinal < 0 || ordinal >= (int) all.count) {
        err = "no Metal device " + std::to_string(ordinal);
        return false;
    }
    out = describe(all[(NSUInteger) ordinal], ordinal);
    return true;
}

DeviceInfo device_info(int ordinal) {
    DeviceInfo out;
    std::string err;
    if (!device_summary(ordinal, out, err)) throw MetalError(err);
    return out;
}

std::string selftest(int ordinal) {
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    if (!all || ordinal < 0 || ordinal >= (int) all.count)
        return "no Metal device " + std::to_string(ordinal);
    id<MTLDevice> device = all[(NSUInteger) ordinal];
    if (!device.hasUnifiedMemory)
        return "this GPU reports no unified memory; the Metal engine addresses one shared pool";

    constexpr NSUInteger n = 1u << 20;
    id<MTLBuffer> src = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!src || !dst) return "newBufferWithLength failed for a 1 MiB shared buffer";

    auto* s = static_cast<uint8_t*>(src.contents);
    auto* d = static_cast<uint8_t*>(dst.contents);
    for (NSUInteger i = 0; i < n; ++i) {
        s[i] = (uint8_t) (i * 131u + 17u);
        d[i] = 0;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return "newCommandQueue failed";
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    if (!buffer) return "commandBuffer failed";
    id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
    if (!blit) return "blitCommandEncoder failed";
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:n];
    [blit endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    if (buffer.status != MTLCommandBufferStatusCompleted) {
        const char* why = buffer.error.localizedDescription.UTF8String;
        return std::string("blit did not complete: ") + (why ? why : "no description");
    }
    for (NSUInteger i = 0; i < n; ++i) {
        const uint8_t want = (uint8_t) (i * 131u + 17u);
        if (d[i] != want)
            return "shared-buffer blit mismatch at byte " + std::to_string((unsigned long long) i);
    }
    return "";
}

}  // namespace strata::metal
