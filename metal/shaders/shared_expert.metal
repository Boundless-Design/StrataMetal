// Shared-expert glue. The projections are other kernels.
// SwiGLU is silu(gate) * up. The scalar multiplies every output by sigmoid of one dot.
#include <metal_stdlib>
using namespace metal;

struct SeN {
    int n;
    int pad0;
    int pad1;
    int pad2;
};

kernel void se_swiglu(device const float* gate [[buffer(0)]],
                      device const float* up [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant SeN& dim [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= (uint) dim.n) return;
    const float x = gate[i];
    const float s = x / (1.0f + exp(-x));
    out[i] = s * up[i];
}

kernel void se_sigmoid_scale(device float* out [[buffer(0)]],
                             device const float* g [[buffer(1)]],
                             constant SeN& dim [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= (uint) dim.n) return;
    const float z = g[0];
    const float s = 1.0f / (1.0f + exp(-z));
    out[i] = out[i] * s;
}
