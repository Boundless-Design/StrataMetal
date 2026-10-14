// QSA gate. One thread per output element.
// q_full is (n_head, 2 * head_dim). The gate is the second head_dim of each head.
// out = attn * sigmoid(gate). This GPU has no fp64. The sigmoid is the precise f32 exp.
#include <metal_stdlib>
using namespace metal;

struct QgP {
    int n_head;
    int head_dim;
    int pad0;
    int pad1;
};

kernel void qsa_gate_apply_f32(device const float* attn [[buffer(0)]],
                               device const float* q_full [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant QgP& dim [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    const int n = dim.n_head * dim.head_dim;
    if ((int) i >= n) return;
    const int head_dim = dim.head_dim;
    const int h = (int) i / head_dim;
    const int d = (int) i - h * head_dim;
    const float g = q_full[(size_t) h * (size_t) (head_dim * 2) + (size_t) head_dim + (size_t) d];
    const float s = 1.0f / (1.0f + exp(-g));
    out[i] = attn[i] * s;
}
