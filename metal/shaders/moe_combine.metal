// MoE combine. One thread per output element.
// y[j] = sum_e w[e] * parts[e, j] + shared[j].
// The shared term is added plain. A null shared omits it.
// This GPU has no fp64. Each product is the exact f32 product, kept as a pair of
// floats, and the sum is that pair added in expert order. The stored value is the
// pair rounded back to f32, which is what a double sum casts to.
#include <metal_stdlib>
using namespace metal;

struct MoeCombineP {
    int n_embd;
    int k;
    int has_shared;
    int pad;
};

struct McDS { float hi; float lo; };

inline McDS mc_add(McDS a, float b) {
    float s = a.hi + b;
    float z = s - a.hi;
    float e = (a.hi - (s - z)) + (b - z) + a.lo;
    float z2 = s + e;
    McDS r;
    r.hi = z2;
    r.lo = e - (z2 - s);
    return r;
}

kernel void moe_combine(device const float* parts [[buffer(0)]],
                        device const float* weights [[buffer(1)]],
                        device const float* shared [[buffer(2)]],
                        device float* y [[buffer(3)]],
                        constant MoeCombineP& dim [[buffer(4)]],
                        uint j [[thread_position_in_grid]]) {
    if (j >= (uint) dim.n_embd) return;
    const int n = dim.n_embd;
    McDS acc;
    acc.hi = 0.0f;
    acc.lo = 0.0f;
    for (int e = 0; e < dim.k; ++e) {
        const float w = weights[e];
        const float x = parts[(size_t) e * (size_t) n + j];
        const float prod = w * x;
        const float err = fma(w, x, -prod);
        acc = mc_add(acc, prod);
        acc = mc_add(acc, err);
    }
    if (dim.has_shared) acc = mc_add(acc, shared[j]);
    y[j] = acc.hi + acc.lo;
}
