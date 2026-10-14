// Paged fp16 K and V. The layout is [page][kv head][slot][head dim].
// A negative physical page writes nothing. The fp16 round matches the shared conversion.
#include <metal_stdlib>
using namespace metal;

inline ushort kv_f16_from_f32(float f) {
    uint x = as_type<uint>(f);
    uint sign = (x >> 16) & 0x8000u;
    uint rawexp = (x >> 23) & 0xFFu;
    int exp = (int) rawexp - 127 + 15;
    uint man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (ushort) (sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (ushort) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (ushort) sign;
        man |= 0x800000u;
        uint sh = (uint) (14 - exp);
        uint h = (man >> sh) & 0x3FFu;
        uint rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) h++;
        return (ushort) (sign | h);
    }
    ushort h = (ushort) (sign | ((uint) exp << 10) | (man >> 13));
    uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return h;
}

struct KvP {
    int n_head_kv;
    int head_dim;
    int page_size;
    int pos;
};

kernel void kv_append_cell(device ushort* k_pool [[buffer(0)]], device ushort* v_pool [[buffer(1)]],
                           const device int* table [[buffer(2)]], const device float* kcur [[buffer(3)]],
                           const device float* vcur [[buffer(4)]], constant KvP& dim [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    const int n = dim.n_head_kv * dim.head_dim;
    if ((int) i >= n) return;
    const int h = (int) i / dim.head_dim;
    const int d = (int) i - h * dim.head_dim;
    const int page = table[dim.pos / dim.page_size];
    if (page < 0) return;
    const int row = (page * dim.n_head_kv + h) * dim.page_size + (dim.pos % dim.page_size);
    const int at = row * dim.head_dim + d;
    k_pool[at] = kv_f16_from_f32(kcur[i]);
    v_pool[at] = kv_f16_from_f32(vcur[i]);
}

struct KgP {
    int n_head_kv;
    int head_dim;
    int page_size;
    int n_ids;
};

kernel void kv_gather_cells(const device ushort* k_pool [[buffer(0)]], const device ushort* v_pool [[buffer(1)]],
                            const device int* table [[buffer(2)]], const device int* ids [[buffer(3)]],
                            device ushort* k_scratch [[buffer(4)]], device ushort* v_scratch [[buffer(5)]],
                            constant KgP& dim [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    const int per = dim.n_head_kv * dim.head_dim;
    const int total = dim.n_ids * per;
    if ((int) i >= total) return;
    const int id = (int) i / per;
    const int rem = (int) i - id * per;
    const int h = rem / dim.head_dim;
    const int d = rem - h * dim.head_dim;
    const int cell = ids[id];
    const int page = table[cell / dim.page_size];
    const int src = ((page * dim.n_head_kv + h) * dim.page_size + (cell % dim.page_size)) * dim.head_dim + d;
    const int dst = (id * dim.n_head_kv + h) * dim.head_dim + d;
    k_scratch[dst] = k_pool[src];
    v_scratch[dst] = v_pool[src];
}
