// metal/src/runtime.mm - one device, one queue, the shaders compiled as Metal 4.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "strata/metal/bytes.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace strata::metal {
namespace {

id<MTLDevice> device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) throw MetalError("no Metal device");
    return dev;
}

std::string shader_source() {
#ifndef STRATA_METAL_SHADER_DIR
#error "STRATA_METAL_SHADER_DIR is the shaders directory"
#endif
    const char* names[] = {"bf16_gemv.metal", "gdn.metal", "qsa.metal", "iq.metal", "sgemv.metal",
                           "elementwise.metal", "gr.metal", "router.metal", "sampler.metal",
                           "moe_combine.metal", "s2_gemv.metal", "shared_expert.metal", "qsa_gate.metal",
                           "indexer.metal", "qsa_index.metal", "topk.metal", "kv.metal", "rope.metal"};
    std::string src;
    for (const char* name : names) {
        const std::string path = std::string(STRATA_METAL_SHADER_DIR) + "/" + name;
        std::ifstream in(path);
        if (!in) throw MetalError("cannot read shader " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        src += ss.str();
        src += "\n";
    }
    return src;
}

std::string ns_error(NSError* err) {
    if (!err) return "unknown Metal error";
    NSString* text = [NSString stringWithFormat:@"%@", err];
    const char* utf = text.UTF8String;
    return utf ? utf : "Metal error";
}

id<MTLLibrary> library() {
    static id<MTLLibrary> lib = nil;
    static bool tried = false;
    if (lib) return lib;
    if (tried) throw MetalError("Metal shaders failed to compile earlier in this process");
    tried = true;
    NSError* err = nil;
    MTLCompileOptions* opt = [MTLCompileOptions new];
    // Safe math, and the precise exp/sqrt. The default function set is the fast one, which would
    // move the DeltaNet recurrence off the reference the parity test measures.
    if (@available(macOS 15.0, *)) {
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    if (@available(macOS 27.0, *)) opt.languageVersion = MTLLanguageVersion4_1;
    else if (@available(macOS 26.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    else opt.languageVersion = MTLLanguageVersion3_0;
    const std::string src = shader_source();
    NSString* text = [[NSString alloc] initWithBytes:src.data() length:src.size() encoding:NSUTF8StringEncoding];
    lib = [device() newLibraryWithSource:text options:opt error:&err];
    if (!lib) throw MetalError("Metal shader compile failed: " + ns_error(err));
    return lib;
}

id<MTLCommandQueue> queue() {
    static id<MTLCommandQueue> q = [device() newCommandQueue];
    if (!q) throw MetalError("newCommandQueue failed");
    return q;
}

id<MTLComputePipelineState> pipeline(const char* name) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cache = [NSMutableDictionary new]; });
    NSString* key = [NSString stringWithUTF8String:name];
    if (!key) throw MetalError("bad kernel name");
    id<MTLComputePipelineState> found = cache[key];
    if (found) return found;
    NSError* err = nil;
    id<MTLFunction> fn = [library() newFunctionWithName:key];
    if (!fn) throw MetalError(std::string("shader has no kernel ") + name);
    id<MTLComputePipelineState> made = [device() newComputePipelineStateWithFunction:fn error:&err];
    if (!made) throw MetalError(std::string(name) + " pipeline failed: " + ns_error(err));
    cache[key] = made;
    return made;
}

}  // namespace

struct Bytes::Impl {
    id<MTLBuffer> buf;
};

Bytes::Bytes(uint64_t bytes) : impl_(new Impl) {
    impl_->buf = [device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!impl_->buf) {
        delete impl_;
        impl_ = nullptr;
        throw MetalError("shared buffer allocation failed (" + std::to_string(bytes) + " bytes)");
    }
}

Bytes::Bytes(Bytes&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }

Bytes& Bytes::operator=(Bytes&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

Bytes::~Bytes() { delete impl_; }

void* Bytes::data() const { return impl_->buf.contents; }

uint64_t Bytes::size() const { return impl_->buf.length; }

void* Bytes::handle() const { return (__bridge void*) impl_->buf; }

void launch(const char* function, const Bytes* const* buffers, int n_buffers, const void* params, size_t param_bytes,
            Grid grid) {
    if (grid.groups_x == 0 || grid.groups_y == 0 || grid.groups_z == 0)
        throw MetalError(std::string(function) + " launched with an empty grid");
    id<MTLComputePipelineState> pipe = pipeline(function);
    id<MTLCommandBuffer> buffer = [queue() commandBuffer];
    if (!buffer) throw MetalError(std::string(function) + ": commandBuffer failed");
    id<MTLComputeCommandEncoder> enc = [buffer computeCommandEncoder];
    if (!enc) throw MetalError(std::string(function) + ": encoder failed");
    [enc setComputePipelineState:pipe];
    for (int i = 0; i < n_buffers; ++i) {
        if (!buffers[i]) throw MetalError(std::string(function) + " missing buffer");
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>) buffers[i]->handle();
        [enc setBuffer:buf offset:0 atIndex:(NSUInteger) i];
    }
    if (param_bytes) [enc setBytes:params length:param_bytes atIndex:(NSUInteger) n_buffers];
    [enc dispatchThreadgroups:MTLSizeMake(grid.groups_x, grid.groups_y, grid.groups_z)
        threadsPerThreadgroup:MTLSizeMake(grid.threads_x, grid.threads_y, grid.threads_z)];
    [enc endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    if (buffer.status != MTLCommandBufferStatusCompleted) {
        const char* why = buffer.error.localizedDescription.UTF8String;
        throw MetalError(std::string(function) + " did not complete: " + (why ? why : "no description"));
    }
}

}  // namespace strata::metal
