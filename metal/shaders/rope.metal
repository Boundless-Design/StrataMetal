// Partial NEOX rotation. Pair i with i + pairs. The tail past n_rot is copied.
// One thread owns a whole row, so the tail copy and the pair writes do not race.
#include <metal_stdlib>
using namespace metal;

struct RpP {
    int head_dim;
    int n_rot;
    int rows;
    int pad;
};

kernel void rope_neox_rows(device float* out [[buffer(0)]], const device float* x [[buffer(1)]],
                           const device float* cos_tab [[buffer(2)]], const device float* sin_tab [[buffer(3)]],
                           const device int* pos [[buffer(4)]], constant RpP& dim [[buffer(5)]],
                           uint r [[thread_position_in_grid]]) {
    if ((int) r >= dim.rows) return;
    const int pairs = dim.n_rot / 2;
    const device float* xr = x + (size_t) r * (size_t) dim.head_dim;
    device float* orow = out + (size_t) r * (size_t) dim.head_dim;
    for (int d = dim.n_rot; d < dim.head_dim; ++d) orow[d] = xr[d];
    const int p = pos[r];
    const device float* ctab = cos_tab + (size_t) p * (size_t) pairs;
    const device float* stab = sin_tab + (size_t) p * (size_t) pairs;
    for (int i = 0; i < pairs; ++i) {
        const float a = xr[i];
        const float b = xr[pairs + i];
        const float c = ctab[i];
        const float s = stab[i];
        const float ac = a * c;
        const float bs = b * s;
        const float as_ = a * s;
        const float bc = b * c;
        orow[i] = ac - bs;
        orow[pairs + i] = as_ + bc;
    }
}
