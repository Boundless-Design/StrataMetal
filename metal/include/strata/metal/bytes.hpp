// metal/include/strata/metal/bytes.hpp - one shared MTLBuffer.
//
// Shared storage is the whole point of this backend: the CPU writes the bytes, the GPU reads the same
// bytes, and nothing is copied across a bus. `data()` is that mapping. The buffer lives until `Bytes`
// does, so a pointer from `data()` is only good for the life of the object.
#pragma once

#include <cstdint>

namespace strata::metal {

class Bytes {
public:
    explicit Bytes(uint64_t bytes);
    Bytes(const Bytes&) = delete;
    Bytes& operator=(const Bytes&) = delete;
    Bytes(Bytes&& other) noexcept;
    Bytes& operator=(Bytes&& other) noexcept;
    ~Bytes();

    void* data() const;
    uint64_t size() const;
    // The MTLBuffer, as a void* so this header stays C++. Valid while this object is.
    void* handle() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace strata::metal
