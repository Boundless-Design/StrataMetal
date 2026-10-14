// Indexer tail and the rotation of one pooled row.
// The mean and the RMS are IEEE double on the CPU. This GPU has no fp64 in a shader.
// The rotation is NEOX: pair i with i + n_rot/2. The tail of the row is left alone.
#include <metal_stdlib>
using namespace metal;

struct IxTailP {
    int idx_dim;
    int r;
    int pos;
    int pad;
};

kernel void ix_tail(const device float* raw [[buffer(0)]], device float* tail [[buffer(1)]],
                    constant IxTailP& dim [[buffer(2)]], uint d [[thread_position_in_grid]]) {
    if ((int) d >= dim.idx_dim) return;
    const int slot = dim.pos % dim.r;
    if (slot < dim.r - 1) tail[(size_t) slot * (size_t) dim.idx_dim + d] = raw[d];
}

struct IxRotP {
    int idx_dim;
    int pairs;
    int row;
    int pos;
};

kernel void ix_rotate(device float* pooled [[buffer(0)]], const device float* cos_tab [[buffer(1)]],
                      const device float* sin_tab [[buffer(2)]], constant IxRotP& dim [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if ((int) i >= dim.pairs) return;
    device float* rowp = pooled + (size_t) dim.row * (size_t) dim.idx_dim;
    const int toff = dim.pos * dim.pairs + (int) i;
    const float a = rowp[i];
    const float b = rowp[dim.pairs + i];
    const float c = cos_tab[toff];
    const float s = sin_tab[toff];
    const float ac = a * c;
    const float bs = b * s;
    const float as_ = a * s;
    const float bc = b * c;
    rowp[i] = ac - bs;
    rowp[dim.pairs + i] = as_ + bc;
}
