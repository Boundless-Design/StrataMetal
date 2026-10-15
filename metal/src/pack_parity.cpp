// The pack index, loaded into the kernels that already pass. One stack token is drawn from a pack.
#include "strata/metal/pack.hpp"
#include "strata/metal/moe_combine.hpp"
#include "strata/metal/router.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/qsa_layer.hpp"
#include "strata/metal/rope.hpp"
#include "strata/metal/s2_gemv.hpp"
#include "strata/metal/sgemv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-52s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("      %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

std::string hex16(uint16_t h) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%04x", h);
    return b;
}

bool threw(const char* needle, const std::function<void()>& fn, std::string& got) {
    try {
        fn();
        got = "it loaded";
        return false;
    } catch (const strata::metal::MetalError& e) {
        got = e.what();
        return got.find(needle) != std::string::npos;
    }
}

struct Tmp {
    std::filesystem::path path;
    bool armed = false;
    Tmp() {
        std::string s = (std::filesystem::temp_directory_path() / "strata-metal-pack-XXXXXX").string();
        std::vector<char> buf(s.begin(), s.end());
        buf.push_back('\0');
        if (!mkdtemp(buf.data())) throw strata::metal::MetalError("mkdtemp failed");
        path = buf.data();
        armed = true;
    }
    ~Tmp() {
        if (!armed) return;
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    Tmp(const Tmp&) = delete;
    Tmp& operator=(const Tmp&) = delete;
};

void write_bytes(const std::filesystem::path& p, const void* data, size_t n) {
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw strata::metal::MetalError("cannot write " + p.string());
    if (n && std::fwrite(data, 1, n, f) != n) {
        std::fclose(f);
        throw strata::metal::MetalError("short write " + p.string());
    }
    std::fclose(f);
}

struct Row {
    std::string name;
    int file = 0;
    int kind = 0;
    uint64_t src_off = 0, src_bytes = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0, act_kind = 0;
};

struct Built {
    std::vector<uint8_t> bin[4];
    std::vector<Row> rows;
};

void add_src(Built& b, Row row, const std::vector<uint8_t>& src) {
    if (row.file < 0 || row.file > 3) throw strata::metal::MetalError("bad file id");
    row.src_off = b.bin[row.file].size();
    row.src_bytes = src.size();
    b.bin[row.file].insert(b.bin[row.file].end(), src.begin(), src.end());
    b.rows.push_back(std::move(row));
}

void add_f32(Built& b, const std::string& name, int64_t ne0, int64_t ne1, uint8_t fill, int last) {
    const uint64_t n = (uint64_t) ne0 * (uint64_t) ne1 * 4;
    std::vector<uint8_t> src(n, fill);
    if (last >= 0 && n) src[n - 1] = (uint8_t) last;
    Row row;
    row.name = name;
    row.kind = 2;
    row.ne0 = ne0;
    row.ne1 = ne1;
    row.dst_bytes = n;
    add_src(b, row, src);
}

void add_f32_fill(Built& b, const std::string& name, int64_t ne0, int64_t ne1, float value) {
    const uint64_t elems = (uint64_t) ne0 * (uint64_t) ne1;
    std::vector<uint8_t> src(elems * 4);
    for (uint64_t i = 0; i < elems; ++i) std::memcpy(src.data() + i * 4, &value, 4);
    Row row;
    row.name = name;
    row.kind = 2;
    row.ne0 = ne0;
    row.ne1 = ne1;
    row.dst_bytes = elems * 4;
    add_src(b, row, src);
}

void add_bf16(Built& b, const std::string& name, int64_t ne0, int64_t ne1, uint32_t word) {
    const uint64_t elems = (uint64_t) ne0 * (uint64_t) ne1;
    std::vector<uint8_t> src(elems * 4);
    for (uint64_t i = 0; i < elems; ++i) std::memcpy(src.data() + i * 4, &word, 4);
    Row row;
    row.name = name;
    row.kind = 1;
    row.ne0 = ne0;
    row.ne1 = ne1;
    row.dst_bytes = elems * 2;
    add_src(b, row, src);
}

void add_quant(Built& b, const std::string& name, int64_t ne0, int64_t ne1, int bits, int bias, int has_offset,
               uint8_t code0, uint16_t scale = 0x3c00, float offset = 0.25f) {
    if (bits != 2 && bits != 4 && bits != 8) throw strata::metal::MetalError("bad code bits in the fixture");
    const int per = 8 / bits;
    const uint64_t codes_bytes = (uint64_t) ne1 * (uint64_t) (ne0 / per);
    const uint64_t groups = (uint64_t) ne1 * (uint64_t) (ne0 / 32);
    const uint64_t scale_src = groups * 2;
    const uint64_t scale_dst = groups * 4;
    const uint64_t off_bytes = has_offset ? groups * 4 : 0;
    std::vector<uint8_t> src(codes_bytes + scale_src + off_bytes);
    for (uint64_t i = 0; i < codes_bytes; ++i) src[i] = (uint8_t) (0x31 + (i * 17));
    if (codes_bytes) src[0] = code0;
    for (uint64_t i = 0; i < groups; ++i) std::memcpy(src.data() + codes_bytes + i * 2, &scale, 2);
    for (uint64_t i = 0; i < groups && has_offset; ++i)
        std::memcpy(src.data() + codes_bytes + scale_src + i * 4, &offset, 4);
    Row row;
    row.name = name;
    row.kind = 0;
    row.ne0 = ne0;
    row.ne1 = ne1;
    row.code_bits = bits;
    row.code_bias = bias;
    row.group = 32;
    row.has_offset = has_offset;
    row.codes_bytes = codes_bytes;
    row.scales_bytes = scale_dst;
    row.offset_bytes = off_bytes;
    row.scales_fp16 = 1;
    row.act_kind = 1;
    row.dst_bytes = codes_bytes + scale_dst + off_bytes;
    add_src(b, std::move(row), src);
}

struct SOpt {
    int bits = 4;
    int bias = -8;
    int group = 32;
    int has_offset = 0;
    int codebook = 0;
    int act_kind = 1;
    uint8_t code0 = 0x21;
    uint16_t scale = 0x3c00;
};

void add_sform(Built& b, const std::string& name, int64_t ne0, int64_t ne1, const SOpt& o) {
    if (o.bits != 2 && o.bits != 4 && o.bits != 8) throw strata::metal::MetalError("bad code bits in the fixture");
    if (o.group <= 0 || ne0 % o.group != 0 || ne0 % (8 / o.bits) != 0)
        throw strata::metal::MetalError("fixture shape does not match the group");
    const int per = 8 / o.bits;
    const uint64_t codes_bytes = (uint64_t) ne1 * (uint64_t) (ne0 / per);
    const uint64_t groups = (uint64_t) ne1 * (uint64_t) (ne0 / o.group);
    const uint64_t scale_src = groups * 2;
    const uint64_t scale_dst = groups * 4;
    const uint64_t off_bytes = o.has_offset ? groups * 4 : 0;
    std::vector<uint8_t> src(codes_bytes + scale_src + off_bytes);
    for (uint64_t i = 0; i < codes_bytes; ++i) src[i] = (uint8_t) (0x31 + (i * 17));
    if (codes_bytes) src[0] = o.code0;
    for (uint64_t i = 0; i < groups; ++i) std::memcpy(src.data() + codes_bytes + i * 2, &o.scale, 2);
    for (uint64_t i = 0; i < groups && o.has_offset; ++i) {
        const float off = 0.25f;
        std::memcpy(src.data() + codes_bytes + scale_src + i * 4, &off, 4);
    }
    Row row;
    row.name = name;
    row.kind = 0;
    row.ne0 = ne0;
    row.ne1 = ne1;
    row.code_bits = o.bits;
    row.code_bias = o.bias;
    row.group = o.group;
    row.codebook = o.codebook;
    row.has_offset = o.has_offset;
    row.codes_bytes = codes_bytes;
    row.scales_bytes = scale_dst;
    row.offset_bytes = off_bytes;
    row.scales_fp16 = 1;
    row.act_kind = o.act_kind;
    row.dst_bytes = codes_bytes + scale_dst + off_bytes;
    add_src(b, std::move(row), src);
}

void write_pack(const Tmp& dir, const Built& b) {
    uint64_t pool = 0;
    for (const Row& r : b.rows) pool += r.dst_bytes;
    if (pool == 0) pool = 1;
    std::string index;
    char head[128];
    std::snprintf(head, sizeof head, "# strata pack index v3\n# align 64 pool %llu tensors %zu\n",
                  (unsigned long long) pool, b.rows.size());
    index = head;
    for (const Row& r : b.rows) {
        char line[640];
        std::snprintf(line, sizeof line, "%s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d\n",
                      r.name.c_str(), r.file, r.kind, (unsigned long long) r.src_off, (unsigned long long) r.src_bytes,
                      0ull, (unsigned long long) r.dst_bytes, (long long) r.ne0, (long long) r.ne1, r.code_bits,
                      r.code_bias, r.group, r.codebook, r.has_offset, (unsigned long long) r.codes_bytes,
                      (unsigned long long) r.scales_bytes, (unsigned long long) r.offset_bytes, r.scales_fp16,
                      r.act_kind);
        index += line;
    }
    write_bytes(dir.path / "index.txt", index.data(), index.size());
    const char* names[] = {"dense.bin", "embd.bin", "experts.bin", "extra.bin"};
    for (int i = 0; i < 4; ++i) {
        if (!b.bin[i].empty()) write_bytes(dir.path / names[i], b.bin[i].data(), b.bin[i].size());
    }
}

void write_text(const Tmp& dir, const std::string& index) {
    write_bytes(dir.path / "index.txt", index.data(), index.size());
}

std::string tn(int layer, const char* suffix) { return "blk." + std::to_string(layer) + "." + suffix; }

strata::core::ModelGeometry tiny_geom() {
    strata::core::ModelGeometry g;
    g.n_embd = 8;
    g.n_layers = 4;
    g.qsa_interval = 4;
    g.ssm_state_size = 4;
    g.ssm_k_heads = 1;
    g.ssm_v_heads = 2;
    g.ssm_d_conv = 2;
    g.ssm_conv_channels = 4;
    g.ssm_value_dim = 4;
    g.n_head = 2;
    g.n_head_kv = 1;
    g.head_dim = 4;
    g.idx_q_heads = 1;
    g.idx_key_dim = 4;
    g.hc = 2;
    g.hc_lr = 4;
    g.n_expert = 3;
    g.n_ff = 4;
    return g;
}

strata::core::ModelGeometry bind_geom() {
    strata::core::ModelGeometry g;
    g.n_embd = 256;
    g.n_layers = 4;
    g.qsa_interval = 4;
    g.ssm_state_size = 128;
    g.ssm_k_heads = 2;
    g.ssm_v_heads = 4;
    g.ssm_d_conv = 4;
    g.ssm_conv_channels = 1024;
    g.ssm_value_dim = 512;
    g.n_head = 24;
    g.n_head_kv = 2;
    g.head_dim = 256;
    g.idx_q_heads = 4;
    g.idx_key_dim = 64;
    g.hc = 2;
    g.hc_lr = 4;
    g.n_expert = 4;
    g.n_ff = 32;
    return g;
}

struct ShapeOpt {
    const char* skip = nullptr;
    int skip_layer = -1;
    const char* ne0_name = nullptr;
    int64_t ne0_override = 0;
    bool shexp_f32 = false;
};

void add_layer_shape(Built& b, const strata::core::ModelGeometry& g, int layer, bool qsa, const ShapeOpt& opt) {
    const int64_t hc = g.hc_dim();
    auto keep = [&](const std::string& name) {
        return !(opt.skip && opt.skip_layer == layer && name == tn(layer, opt.skip));
    };
    auto f32 = [&](const char* suffix, int64_t ne0, int64_t ne1, int last) {
        const std::string name = tn(layer, suffix);
        if (!keep(name)) return;
        int64_t n0 = ne0;
        if (opt.ne0_name && name == tn(layer, opt.ne0_name)) n0 = opt.ne0_override;
        // The bytes stay the real shape. Only the index ne0 changes, which is what the check reads.
        if (n0 != ne0) {
            const uint64_t n = (uint64_t) ne0 * (uint64_t) ne1 * 4;
            std::vector<uint8_t> src(n, 0x11);
            Row row;
            row.name = name;
            row.kind = 2;
            row.ne0 = n0;
            row.ne1 = ne1;
            row.dst_bytes = n;
            add_src(b, row, src);
            return;
        }
        add_f32(b, name, ne0, ne1, 0x11, last);
    };
    auto f32_1 = [&](const char* suffix, int64_t n, int last) { f32(suffix, n, 1, last); };
    f32("hc_attn_down.weight", hc, g.hc_lr, -1);
    f32("hc_attn_up.weight", g.hc_lr, hc, -1);
    f32("hc_attn_inject.weight", hc, g.hc, -1);
    f32("hc_ffn_down.weight", hc, g.hc_lr, -1);
    f32("hc_ffn_up.weight", g.hc_lr, hc, -1);
    f32("hc_ffn_inject.weight", hc, g.hc, -1);
    f32("ffn_gate_inp.weight", g.n_embd, g.n_expert, -1);
    f32("ffn_gate_shexp.weight", g.n_embd, g.n_ff, -1);
    f32("ffn_up_shexp.weight", g.n_embd, g.n_ff, -1);
    f32("ffn_down_shexp.weight", g.n_ff, g.n_embd, -1);
    f32_1("hc_attn_norm.weight", hc, -1);
    f32_1("hc_ffn_norm.weight", hc, -1);
    if (keep(tn(layer, "ffn_gate_inp_shexp.weight"))) {
        if (opt.shexp_f32) add_f32(b, tn(layer, "ffn_gate_inp_shexp.weight"), g.n_embd, 1, 0x22, -1);
        else add_bf16(b, tn(layer, "ffn_gate_inp_shexp.weight"), g.n_embd, 1, 0x3f800001u);
    }
    if (!qsa) {
        f32("attn_qkv.weight", g.n_embd, g.ssm_conv_channels, -1);
        f32("attn_gate.weight", g.n_embd, g.ssm_value_dim, -1);
        f32("ssm_out.weight", g.ssm_value_dim, g.n_embd, -1);
        f32("ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, -1);
        f32("ssm_alpha.weight", g.n_embd, g.ssm_v_heads, -1);
        f32("ssm_beta.weight", g.n_embd, g.ssm_v_heads, -1);
        f32_1("ssm_a", g.ssm_v_heads, -1);
        f32_1("ssm_dt.bias", g.ssm_v_heads, -1);
        f32_1("ssm_norm.weight", g.ssm_state_size, layer == 0 ? 0x5A : -1);
    } else {
        f32("attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, layer == 3 ? 0xA5 : -1);
        f32("attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, -1);
        f32("attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, -1);
        f32("attn_output.weight", g.n_head * g.head_dim, g.n_embd, -1);
        f32("indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, -1);
        f32("indexer.k_proj.weight", g.n_embd, g.idx_key_dim, -1);
        f32_1("attn_q_norm.weight", g.head_dim, -1);
        f32_1("attn_k_norm.weight", g.head_dim, -1);
        f32_1("indexer.q_norm.weight", g.idx_key_dim, -1);
        f32_1("indexer.k_norm.weight", g.idx_key_dim, -1);
    }
}

void add_bind_layer(Built& b, const strata::core::ModelGeometry& g, int layer, bool qsa, int qkv_bits, int q_bits) {
    const int64_t hc = g.hc_dim();
    add_bf16(b, tn(layer, "hc_attn_down.weight"), hc, g.hc_lr, 0xA5A50001u);
    add_bf16(b, tn(layer, "hc_attn_up.weight"), g.hc_lr, hc, 0x3f800001u);
    add_bf16(b, tn(layer, "hc_attn_inject.weight"), hc, g.hc, 0x3f800001u);
    add_bf16(b, tn(layer, "hc_ffn_down.weight"), hc, g.hc_lr, 0x3f800001u);
    add_bf16(b, tn(layer, "hc_ffn_up.weight"), g.hc_lr, hc, 0x3f800001u);
    add_bf16(b, tn(layer, "hc_ffn_inject.weight"), hc, g.hc, 0x3f800001u);
    add_bf16(b, tn(layer, "ffn_gate_inp.weight"), g.n_embd, g.n_expert, 0xA5A50001u);
    add_f32(b, tn(layer, "ffn_gate_shexp.weight"), g.n_embd, g.n_ff, 0x44, -1);
    add_f32(b, tn(layer, "ffn_up_shexp.weight"), g.n_embd, g.n_ff, 0x44, -1);
    add_f32(b, tn(layer, "ffn_down_shexp.weight"), g.n_ff, g.n_embd, 0x5C, -1);
    add_f32(b, tn(layer, "hc_attn_norm.weight"), hc, 1, 0x11, -1);
    add_f32(b, tn(layer, "hc_ffn_norm.weight"), hc, 1, 0x11, -1);
    add_bf16(b, tn(layer, "ffn_gate_inp_shexp.weight"), g.n_embd, 1, 0x3f800001u);
    if (!qsa) {
        add_quant(b, tn(layer, "attn_qkv.weight"), g.n_embd, g.ssm_conv_channels, qkv_bits, -8, 0, 0xA5);
        add_quant(b, tn(layer, "attn_gate.weight"), g.n_embd, g.ssm_value_dim, 4, -8, 1, 0x22);
        add_quant(b, tn(layer, "ssm_out.weight"), g.ssm_value_dim, g.n_embd, 8, -16, 1, 0x33);
        add_f32(b, tn(layer, "ssm_conv1d.weight"), g.ssm_d_conv, g.ssm_conv_channels, 0x11, -1);
        add_bf16(b, tn(layer, "ssm_alpha.weight"), g.n_embd, g.ssm_v_heads, 0x3f800001u);
        add_bf16(b, tn(layer, "ssm_beta.weight"), g.n_embd, g.ssm_v_heads, 0x3f800001u);
        add_f32(b, tn(layer, "ssm_a"), g.ssm_v_heads, 1, 0x11, -1);
        add_f32(b, tn(layer, "ssm_dt.bias"), g.ssm_v_heads, 1, 0x11, -1);
        add_f32(b, tn(layer, "ssm_norm.weight"), g.ssm_state_size, 1, 0x11, -1);
    } else {
        add_quant(b, tn(layer, "attn_q.weight"), g.n_embd, 2 * g.n_head * g.head_dim, q_bits, -8, 0, 0x5A);
        add_quant(b, tn(layer, "attn_k.weight"), g.n_embd, g.n_head_kv * g.head_dim, 4, -8, 0, 0x21);
        add_quant(b, tn(layer, "attn_v.weight"), g.n_embd, g.n_head_kv * g.head_dim, 4, -8, 0, 0x21);
        add_quant(b, tn(layer, "attn_output.weight"), g.n_head * g.head_dim, g.n_embd, 4, -8, 0, 0x21);
        add_bf16(b, tn(layer, "indexer.q_proj.weight"), g.n_embd, g.idx_q_heads * g.idx_key_dim, 0x3f800001u);
        add_bf16(b, tn(layer, "indexer.k_proj.weight"), g.n_embd, g.idx_key_dim, 0x3f800001u);
        add_f32(b, tn(layer, "attn_q_norm.weight"), g.head_dim, 1, 0x11, -1);
        add_f32(b, tn(layer, "attn_k_norm.weight"), g.head_dim, 1, 0x11, -1);
        add_f32(b, tn(layer, "indexer.q_norm.weight"), g.idx_key_dim, 1, 0x11, -1);
        add_f32(b, tn(layer, "indexer.k_norm.weight"), g.idx_key_dim, 1, 0x11, -1);
    }
}

// The last row must be the tail of file 0. The embedding file is file 1.
void move_tail_to_embd(Built& b) {
    Row& row = b.rows.back();
    const uint64_t off = row.src_off;
    std::vector<uint8_t> bytes(b.bin[0].begin() + (std::ptrdiff_t) off, b.bin[0].end());
    b.bin[0].resize((size_t) off);
    row.file = 1;
    row.src_off = b.bin[1].size();
    b.bin[1].insert(b.bin[1].end(), bytes.begin(), bytes.end());
}

void add_embd(Built& b, int64_t ne0, int64_t ne1, int bits, int bias, int has_offset, uint8_t code0) {
    add_quant(b, "token_embd.weight", ne0, ne1, bits, bias, has_offset, code0);
    move_tail_to_embd(b);
}

void add_side_files(Built& b) {
    add_bf16(b, "token_embd.weight", 4, 1, 0x3f800001u);
    b.rows.back().file = 1;
    // add_bf16 appended to file 0. Move those bytes onto embd.bin.
    const uint64_t n = b.rows.back().src_bytes;
    const uint64_t off = b.rows.back().src_off;
    std::vector<uint8_t> bytes(b.bin[0].begin() + (std::ptrdiff_t) off, b.bin[0].end());
    b.bin[0].resize((size_t) off);
    b.rows.back().src_off = 0;
    b.bin[1] = std::move(bytes);
    (void) n;
    std::vector<uint8_t> blob(32, 0xEE);
    Row ex;
    ex.name = "blk.0.ffn_gate_exps.weight";
    ex.file = 2;
    ex.kind = 0;
    ex.ne0 = 32;
    ex.ne1 = 1;
    ex.dst_bytes = 32;
    add_src(b, ex, blob);
}

using namespace strata::metal;

int differ_floats(const float* a, const float* b, int n) {
    int d = 0;
    for (int i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, 4) != 0) ++d;
    return d;
}

bool still_seven(const float* a, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != 7.f) return false;
    return true;
}

void collect(const PackLayer& L, std::vector<const void*>& p) {
    auto add = [&](const Bytes* b) {
        if (b) p.push_back(b->data());
    };
    add(L.attn.norm);
    add(L.attn.down);
    add(L.attn.up);
    add(L.attn.inject);
    add(L.ffn.norm);
    add(L.ffn.down);
    add(L.ffn.up);
    add(L.ffn.inject);
    add(L.router);
    auto gplane = [&](const GdnPlane& g) {
        add(g.codes);
        add(g.scales);
        add(g.offset);
    };
    gplane(L.gdn.qkv);
    gplane(L.gdn.attn_gate);
    gplane(L.gdn.ssm_out);
    add(L.gdn.alpha);
    add(L.gdn.beta);
    add(L.gdn.conv);
    add(L.gdn.ssm_norm);
    add(L.gdn.dt);
    add(L.gdn.ssm_a);
    auto qplane = [&](const QsaPlane& q) {
        add(q.codes);
        add(q.scales);
        add(q.offset);
    };
    qplane(L.qsa.q);
    qplane(L.qsa.k);
    qplane(L.qsa.v);
    qplane(L.qsa.out);
    add(L.qsa.idx_k);
    add(L.qsa.idx_q);
    add(L.qsa.k_norm);
    add(L.qsa.q_norm);
    add(L.qsa.idx_k_norm);
    add(L.qsa.idx_q_norm);
}

bool aliased(const void* data, const std::vector<const void*>& bound) {
    if (!data) return false;
    for (const void* p : bound)
        if (p == data) return true;
    return false;
}

void run_conversions() {
    const uint32_t words[8] = {0x3f800001u, 0x40000001u, 0x3f818000u, 0x00000001u,
                               0x3f800001u, 0x3f808000u, 0x3f800001u, 0x3f81ffffu};
    const uint16_t raw_bf[4] = {0x3f80, 0x4000, 0x3f81, 0x0001};
    const uint16_t raw_f[2] = {0x3c00, 0x0001};
    const float promoted[2] = {1.f, 65520.f};

    Built b;
    {
        std::vector<uint8_t> src(sizeof words);
        std::memcpy(src.data(), words, sizeof words);
        Row row;
        row.name = "gemv.bf16";
        row.kind = 1;
        row.ne0 = 4;
        row.ne1 = 2;
        row.dst_bytes = 16;
        add_src(b, row, src);
    }
    {
        const int64_t n_in = 256, n_out = 2;
        const uint64_t codes_bytes = (uint64_t) n_out * (n_in / 2);
        const uint64_t groups = (uint64_t) n_out * (n_in / 32);
        const uint64_t scale_src = groups * 2;
        const uint64_t off_bytes = groups * 4;
        std::vector<uint8_t> src(codes_bytes + scale_src + off_bytes);
        for (uint64_t i = 0; i < codes_bytes; ++i) src[i] = (uint8_t) (0x40 + (i * 3));
        for (uint64_t i = 0; i < groups; ++i) {
            const uint16_t h = (i == 1) ? 0x0001 : 0x3c00;
            std::memcpy(src.data() + codes_bytes + i * 2, &h, 2);
        }
        for (uint64_t i = 0; i < groups; ++i) {
            const float o = 0.25f * (float) ((int) i - 3);
            std::memcpy(src.data() + codes_bytes + scale_src + i * 4, &o, 4);
        }
        Row row;
        row.name = "gemv.sform";
        row.kind = 0;
        row.ne0 = n_in;
        row.ne1 = n_out;
        row.code_bits = 4;
        row.code_bias = -8;
        row.group = 32;
        row.has_offset = 1;
        row.codes_bytes = codes_bytes;
        row.scales_bytes = groups * 4;
        row.offset_bytes = off_bytes;
        row.scales_fp16 = 1;
        row.act_kind = 1;
        row.dst_bytes = codes_bytes + groups * 4 + off_bytes;
        add_src(b, row, src);
    }
    {
        std::vector<uint8_t> src(sizeof raw_bf);
        std::memcpy(src.data(), raw_bf, sizeof raw_bf);
        Row row;
        row.name = "raw.bf16";
        row.kind = 4;
        row.ne0 = 4;
        row.ne1 = 1;
        row.dst_bytes = sizeof raw_bf;
        add_src(b, row, src);
    }
    {
        std::vector<uint8_t> src(sizeof raw_f);
        std::memcpy(src.data(), raw_f, sizeof raw_f);
        Row row;
        row.name = "raw.f16";
        row.kind = 5;
        row.ne0 = 2;
        row.ne1 = 1;
        row.dst_bytes = sizeof raw_f;
        add_src(b, row, src);
    }
    {
        std::vector<uint8_t> src(sizeof promoted);
        std::memcpy(src.data(), promoted, sizeof promoted);
        Row row;
        row.name = "promoted.f16";
        row.kind = 3;
        row.ne0 = 2;
        row.ne1 = 1;
        row.dst_bytes = 4;
        add_src(b, row, src);
    }

    Tmp dir;
    write_pack(dir, b);
    PackTable table = PackTable::load(dir.path.string());

    const PackTensor* bf = table.find("gemv.bf16");
    bool high_ok = bf && bf->values && bf->values->size() == 16 && bf->kind == strata::core::WeightKind::Bf16InF32;
    const uint16_t* loaded = high_ok ? static_cast<const uint16_t*>(bf->values->data()) : nullptr;
    if (loaded) {
        high_ok = loaded[0] == 0x3f80 && loaded[1] == 0x4000 && loaded[3] == 0x0000 && loaded[5] == 0x3f80;
    }
    verdict("the high 16 bits of a dirty bf16 are exact", high_ok, high_ok ? "" : "bits");

    uint32_t tie_bits = 0x3f818000u;
    float tie_f = 0;
    std::memcpy(&tie_f, &tie_bits, 4);
    const uint16_t rounded = strata::kernels::bf16_from_f32(tie_f);
    const bool tie_ok = loaded && loaded[2] == 0x3f81 && rounded == 0x3f82;
    uint32_t tail_bits = 0x3f81ffffu;
    float tail_f = 0;
    std::memcpy(&tail_f, &tail_bits, 4);
    const bool tail_ok = loaded && loaded[7] == 0x3f81 && strata::kernels::bf16_from_f32(tail_f) == 0x3f82;
    verdict("a rounding conversion would change it", tie_ok && tail_ok,
            tie_ok && tail_ok ? "" : "loaded " + (loaded ? hex16(loaded[2]) : std::string("missing")));

    const PackTensor* sf = table.find("gemv.sform");
    bool planes = sf && sf->codes && sf->scales && sf->offset && sf->codes->data() != sf->scales->data() &&
                  sf->scales->data() != sf->offset->data() && sf->codes->size() == sf->codes_bytes &&
                  sf->scales->size() == sf->scales_bytes && sf->offset->size() == sf->offset_bytes && !sf->values;
    if (planes) {
        const uint64_t codes_bytes = sf->codes_bytes;
        const uint64_t groups = 16;
        planes = std::memcmp(sf->codes->data(), b.bin[0].data() + sf->src_off, (size_t) codes_bytes) == 0;
        const auto* sc = static_cast<const float*>(sf->scales->data());
        const float one = strata::kernels::f32_from_f16(0x3c00);
        const float sub = strata::kernels::f32_from_f16(0x0001);
        if (sc[0] != one || sc[0] != 1.f || sc[1] != sub || sc[1] == 0.f) planes = false;
        const uint8_t* off_src = b.bin[0].data() + sf->src_off + codes_bytes + groups * 2;
        if (std::memcmp(sf->offset->data(), off_src, (size_t) sf->offset_bytes) != 0) planes = false;
    }
    verdict("fp16 1.0 widens to 1", planes, "");
    verdict("a subnormal scale widens exactly", planes, "");
    verdict("verbatim planes stay in their own buffers", planes, "");

    const PackTensor* rawb = table.find("raw.bf16");
    const bool raw_bf_ok = rawb && rawb->raw16 && rawb->kind == strata::core::WeightKind::Bf16InF32 && rawb->values &&
                           rawb->values->size() == sizeof raw_bf &&
                           std::memcmp(rawb->values->data(), raw_bf, sizeof raw_bf) == 0;
    verdict("a raw 16-bit bf16 is copied", raw_bf_ok, "");

    const PackTensor* rawf = table.find("raw.f16");
    const bool raw_f_ok = rawf && rawf->raw16 && rawf->kind == strata::core::WeightKind::F16InF32 && rawf->values &&
                          std::memcmp(rawf->values->data(), raw_f, sizeof raw_f) == 0;
    verdict("a raw 16-bit f16 is copied", raw_f_ok, "");

    const PackTensor* pf = table.find("promoted.f16");
    bool pf_ok = pf && pf->values && pf->values->size() == 4 && pf->kind == strata::core::WeightKind::F16InF32;
    if (pf_ok) {
        const auto* h = static_cast<const uint16_t*>(pf->values->data());
        const uint16_t want0 = strata::kernels::f16_from_f32(1.f);
        const uint16_t want1 = strata::kernels::f16_from_f32(65520.f);
        pf_ok = h[0] == want0 && h[0] == 0x3c00 && h[1] == want1 && h[1] == 0x7C00 && h[1] != 0x477f;
    }
    verdict("an f32 promoted to f16 saturates", pf_ok, "");

    // The same loaded bytes, through the kernels, against a direct upload of the engine form.
    if (high_ok && loaded) {
        std::vector<uint16_t> direct_w(8);
        for (int i = 0; i < 8; ++i) direct_w[i] = (uint16_t) (words[i] >> 16);
        Bytes wdir(direct_w.size() * 2);
        std::memcpy(wdir.data(), direct_w.data(), direct_w.size() * 2);
        const uint16_t ones[4] = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
        Bytes x(8);
        std::memcpy(x.data(), ones, 8);
        Bytes y0(8), y1(8);
        std::fill_n(static_cast<float*>(y0.data()), 2, 7.f);
        std::fill_n(static_cast<float*>(y1.data()), 2, 7.f);
        bf16_gemv(x, *bf->values, y0, 4, 2);
        bf16_gemv(x, wdir, y1, 4, 2);
        const int d = differ_floats(static_cast<float*>(y0.data()), static_cast<float*>(y1.data()), 2);
        const bool moved = !still_seven(static_cast<float*>(y0.data()), 2);
        verdict("a loaded bf16 matrix matches a direct upload", d == 0 && moved, std::to_string(d) + " differ");
    } else {
        verdict("a loaded bf16 matrix matches a direct upload", false, "no matrix");
    }

    if (planes && sf) {
        const int64_t n_in = 256, n_out = 2;
        Bytes codes(sf->codes->size());
        Bytes scales(sf->scales->size());
        Bytes offset(sf->offset->size());
        std::memcpy(codes.data(), sf->codes->data(), (size_t) sf->codes->size());
        std::memcpy(scales.data(), sf->scales->data(), (size_t) sf->scales->size());
        std::memcpy(offset.data(), sf->offset->data(), (size_t) sf->offset->size());
        std::vector<float> hx(256);
        for (int i = 0; i < 256; ++i) hx[i] = 0.01f * (float) (i + 1);
        Bytes x(256 * 4);
        std::memcpy(x.data(), hx.data(), hx.size() * 4);
        Bytes xq(q8k_bytes(256));
        quantize_q8_k(x, 256, xq);
        SForm form;
        form.code_bits = 4;
        form.code_bias = -8;
        form.group_elems = 32;
        form.has_offset = true;
        Bytes y0((uint64_t) n_out * 4), y1((uint64_t) n_out * 4);
        std::fill_n(static_cast<float*>(y0.data()), (size_t) n_out, 7.f);
        std::fill_n(static_cast<float*>(y1.data()), (size_t) n_out, 7.f);
        s_gemv_q8k(xq, *sf->codes, *sf->scales, sf->offset.get(), y0, n_in, n_out, form);
        s_gemv_q8k(xq, codes, scales, &offset, y1, n_in, n_out, form);
        const int d = differ_floats(static_cast<float*>(y0.data()), static_cast<float*>(y1.data()), (int) n_out);
        const bool moved = !still_seven(static_cast<float*>(y0.data()), (int) n_out);
        verdict("a loaded S-form matches a direct upload", d == 0 && moved, std::to_string(d) + " differ");
    } else {
        verdict("a loaded S-form matches a direct upload", false, "no plane");
    }
}

void run_refusals() {
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 16 tensors 1\n"
                        "bad 0 2 0 16 0 16 4 1 0 0 0 0 0 0 0 0 0\n");
        std::string got;
        verdict("an 18-field row is refused", threw("18 of 19", [&] { PackTable::load(dir.path.string()); }, got), got);
    }
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 10 tensors 1\n"
                        "bad 0 0 0 10 0 10 8 1 4 0 32 0 0 4 4 0 0 1\n");
        std::string got;
        verdict("a plane sum that disagrees is refused",
                threw("segments cover", [&] { PackTable::load(dir.path.string()); }, got), got);
    }
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 4 tensors 1\n"
                        "bad 0 1 0 4 0 4 2 1 0 0 0 0 0 0 0 0 0 0\n");
        std::string got;
        verdict("a promoted tensor of the wrong size is refused",
                threw("promoted tensor", [&] { PackTable::load(dir.path.string()); }, got), got);
    }
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 16 tensors 1\n"
                        "bad 0 2 0 16 0 16 4 1 0 0 0 0 0 0 0 0 0 0\n");
        std::string got;
        verdict("a missing bin is refused", threw("dense.bin", [&] { PackTable::load(dir.path.string()); }, got), got);
    }
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 16 tensors 1\n"
                        "short 0 2 0 16 0 16 4 1 0 0 0 0 0 0 0 0 0 0\n");
        const uint8_t few[4] = {1, 2, 3, 4};
        write_bytes(dir.path / "dense.bin", few, 4);
        std::string got;
        verdict("a short read is refused", threw("short read", [&] { PackTable::load(dir.path.string()); }, got), got);
    }
    {
        Tmp dir;
        write_text(dir, "# align 64 pool 1 tensors 1\n"
                        "gonly 0 0 0 0 0 0 4 1 4 0 32 0 1 0 0 0 0 1\n");
        std::string got;
        verdict("a GGUF-only row is refused", threw("GGUF form", [&] { PackTable::load(dir.path.string()); }, got),
                got);
    }
}

void run_shapes() {
    const auto g = tiny_geom();
    Tmp good;
    {
        Built b;
        ShapeOpt opt;
        add_layer_shape(b, g, 0, false, opt);
        add_layer_shape(b, g, 3, true, opt);
        write_pack(good, b);
    }
    PackTable table = PackTable::load(good.path.string());
    std::string err0, err3;
    const bool ok0 = check_pack_layer(table, g, 0, err0);
    const bool ok3 = check_pack_layer(table, g, 3, err3);
    const PackTensor* norm = table.find(tn(0, "ssm_norm.weight"));
    const PackTensor* aq = table.find(tn(3, "attn_q.weight"));
    bool sent = ok0 && ok3 && norm && norm->values && aq && aq->values;
    if (sent) {
        const auto* nb = static_cast<const uint8_t*>(norm->values->data());
        const auto* qb = static_cast<const uint8_t*>(aq->values->data());
        sent = nb[norm->values->size() - 1] == 0x5A && qb[aq->values->size() - 1] == 0xA5;
    }
    const PackTensor* gate = table.find(tn(0, "ffn_gate_inp_shexp.weight"));
    bool gate_bits = gate && gate->values && gate->values->size() >= 2;
    if (gate_bits) {
        uint16_t h = 0;
        std::memcpy(&h, gate->values->data(), 2);
        gate_bits = h == 0x3f80;
    }
    verdict("a complete layer keeps its sentinel", sent && gate_bits, sent && gate_bits ? "2 bytes" : err0 + err3);
    verdict("a GDN layer has no QSA tensor", ok0 && table.find(tn(0, "attn_q.weight")) == nullptr, "");
    verdict("a full-attention layer has no GDN tensor", ok3 && table.find(tn(3, "attn_qkv.weight")) == nullptr, "");

    auto z = g;
    z.qsa_interval = 0;
    std::string zerr;
    verdict("a zero interval is refused", !check_pack_layer(table, z, 0, zerr) && zerr.find("positive") != std::string::npos,
            zerr);
    std::string outside;
    const bool out_hi = !check_pack_layer(table, g, 9, outside) && outside.find("outside") != std::string::npos;
    std::string neg;
    const bool out_lo = !check_pack_layer(table, g, -1, neg) && neg.find("outside") != std::string::npos;
    verdict("a layer outside the model is refused", out_hi && out_lo, out_hi ? neg : outside);

    {
        Tmp dir;
        Built b;
        ShapeOpt opt;
        opt.skip = "attn_qkv.weight";
        opt.skip_layer = 0;
        add_layer_shape(b, g, 0, false, opt);
        add_layer_shape(b, g, 3, true, ShapeOpt{});
        write_pack(dir, b);
        PackTable t = PackTable::load(dir.path.string());
        std::string err;
        const bool refused = !check_pack_layer(t, g, 0, err) && err.find("missing blk.0.attn_qkv.weight") != std::string::npos;
        verdict("a missing GDN tensor is refused", refused, err);
    }
    {
        Tmp dir;
        Built b;
        ShapeOpt opt;
        opt.skip = "attn_q.weight";
        opt.skip_layer = 3;
        add_layer_shape(b, g, 0, false, ShapeOpt{});
        add_layer_shape(b, g, 3, true, opt);
        write_pack(dir, b);
        PackTable t = PackTable::load(dir.path.string());
        std::string err;
        const bool refused = !check_pack_layer(t, g, 3, err) && err.find("missing blk.3.attn_q.weight") != std::string::npos;
        verdict("a missing QSA tensor is refused", refused, err);
    }
    {
        Tmp dir;
        Built b;
        ShapeOpt opt;
        opt.ne0_name = "attn_qkv.weight";
        opt.ne0_override = 9;
        add_layer_shape(b, g, 0, false, opt);
        write_pack(dir, b);
        PackTable t = PackTable::load(dir.path.string());
        std::string err;
        const bool refused = !check_pack_layer(t, g, 0, err) && err.find("ne0 is 9") != std::string::npos;
        verdict("a wrong ne0 is refused", refused, err);
    }
    {
        Tmp dir;
        Built b;
        ShapeOpt opt;
        opt.shexp_f32 = true;
        add_layer_shape(b, g, 0, false, opt);
        write_pack(dir, b);
        PackTable t = PackTable::load(dir.path.string());
        std::string err;
        const bool refused = !check_pack_layer(t, g, 0, err) && err.find("engine form 2") != std::string::npos;
        verdict("a bf16 gate stored as f32 is refused", refused, err);
    }
}

void run_bind() {
    const auto g = bind_geom();
    Tmp good;
    {
        Built b;
        add_bind_layer(b, g, 0, false, 4, 4);
        add_bind_layer(b, g, 3, true, 4, 4);
        add_side_files(b);
        write_pack(good, b);
    }
    PackTable table = PackTable::load(good.path.string());
    std::string err;
    const bool named = check_pack_layer(table, g, 0, err) && check_pack_layer(table, g, 3, err);
    verdict("the bound layers pass the name check", named, named ? "" : err);
    if (!named) {
        verdict("a 4-bit GDN plane binds", false, "names");
        verdict("a 4-bit QSA plane binds", false, "names");
        verdict("full attention is not bound as GDN", false, "names");
        verdict("a GDN layer is not bound as QSA", false, "names");
        verdict("the shared expert stays unbound", false, "names");
        verdict("the routed experts stay unbound", false, "names");
        verdict("the embedding stays unbound", false, "names");
    } else {
        PackLayer gdn = bind_layer(table, g, 0);
        PackLayer qsa = bind_layer(table, g, 3);
        const bool gdn_ok = !gdn.qsa_layer && gdn.gdn.qkv.form.code_bits == 4 && gdn.gdn.qkv.form.code_bias == -8 &&
                            gdn.gdn.qkv.form.group_elems == 32 && !gdn.gdn.qkv.form.has_offset &&
                            gdn.gdn.qkv.offset == nullptr && gdn.gdn.qkv.codes && gdn.gdn.qkv.scales &&
                            gdn.gdn.qkv.codes != gdn.gdn.qkv.scales &&
                            static_cast<const uint8_t*>(gdn.gdn.qkv.codes->data())[0] == 0xA5 &&
                            gdn.gdn.attn_gate.form.has_offset && gdn.gdn.attn_gate.offset != nullptr &&
                            gdn.gdn.ssm_out.form.code_bits == 8 && gdn.gdn.ssm_out.form.code_bias == -16 &&
                            gdn.gdn.ssm_out.form.has_offset && gdn.router && gdn.attn.down;
        uint16_t router_h = 0, down_h = 0, alpha_h = 0;
        if (gdn_ok) {
            std::memcpy(&router_h, gdn.router->data(), 2);
            std::memcpy(&down_h, gdn.attn.down->data(), 2);
            std::memcpy(&alpha_h, gdn.gdn.alpha->data(), 2);
        }
        verdict("a 4-bit GDN plane binds", gdn_ok && router_h == 0xA5A5 && down_h == 0xA5A5 && alpha_h == 0x3f80,
                "bias -8, offset on the gate");

        const bool qsa_ok = qsa.qsa_layer && qsa.qsa.cos_tab == nullptr && qsa.qsa.sin_tab == nullptr &&
                            qsa.qsa.q.form.code_bits == 4 && qsa.qsa.q.codes &&
                            static_cast<const uint8_t*>(qsa.qsa.q.codes->data())[0] == 0x5A && qsa.qsa.idx_k &&
                            qsa.gdn.qkv.codes == nullptr;
        uint16_t idx_h = 0;
        if (qsa_ok) std::memcpy(&idx_h, qsa.qsa.idx_k->data(), 2);
        verdict("a 4-bit QSA plane binds", qsa_ok && idx_h == 0x3f80, "rope table left empty");

        std::string cross;
        const bool not_gdn = threw("is full attention", [&] { bind_gdn_layer(table, g, 3); }, cross);
        verdict("full attention is not bound as GDN", not_gdn, cross);
        const bool not_qsa = threw("not full attention", [&] { bind_qsa_layer(table, g, 0); }, cross);
        verdict("a GDN layer is not bound as QSA", not_qsa, cross);

        std::vector<const void*> bound;
        collect(gdn, bound);
        collect(qsa, bound);
        const PackTensor* shexp = table.find(tn(0, "ffn_down_shexp.weight"));
        const PackTensor* exps = table.find("blk.0.ffn_gate_exps.weight");
        const PackTensor* emb = table.find("token_embd.weight");
        const bool shexp_ok = shexp && shexp->values && shexp->code_bits == 0 &&
                              static_cast<const uint8_t*>(shexp->values->data())[0] == 0x5C &&
                              !aliased(shexp->values->data(), bound);
        const bool exps_ok = exps && exps->values && static_cast<const uint8_t*>(exps->values->data())[0] == 0xEE &&
                             exps->file_id == 2 && !aliased(exps->values->data(), bound);
        const bool emb_ok = emb && emb->values && emb->file_id == 1 && emb->values->size() == 8 &&
                            !aliased(emb->values->data(), bound);
        uint16_t emb_h = 0;
        if (emb_ok) std::memcpy(&emb_h, emb->values->data(), 2);
        verdict("the shared expert stays unbound", shexp_ok, "");
        verdict("the routed experts stay unbound", exps_ok, "");
        verdict("the embedding stays unbound", emb_ok && emb_h == 0x3f80, "");
    }

    Tmp bits2;
    {
        Built b;
        add_bind_layer(b, g, 0, false, 2, 4);
        add_bind_layer(b, g, 3, true, 4, 2);
        write_pack(bits2, b);
    }
    PackTable narrow = PackTable::load(bits2.path.string());
    std::string nerr;
    const bool names = check_pack_layer(narrow, g, 0, nerr) && check_pack_layer(narrow, g, 3, nerr);
    std::string gerr, qerr;
    const bool gref = threw("code_bits 2", [&] { bind_gdn_layer(narrow, g, 0); }, gerr);
    const bool qref = threw("code_bits 2", [&] { bind_qsa_layer(narrow, g, 3); }, qerr);
    verdict("a 2-bit Q8_K plane is refused", names && gref && qref, names ? gerr : nerr);
}

bool one_embd_row(int bits, int bias, int has_offset, std::string& detail) {
    const int64_t n = 32;
    const int64_t vocab = 3;
    const int64_t token = 1;
    Tmp dir;
    Built b;
    add_embd(b, n, vocab, bits, bias, has_offset, 0xA5);
    write_pack(dir, b);
    PackTable table = PackTable::load(dir.path.string());
    const PackTensor* w = table.find("token_embd.weight");
    if (!w || !w->codes || !w->scales || w->file_id != 1 || w->code_bits != bits || w->group_elems != 32) {
        detail = std::to_string(bits) + "-bit row did not load";
        return false;
    }
    const float* loaded_scales = static_cast<const float*>(w->scales->data());
    if (loaded_scales[token] != 1.f) {
        detail = std::to_string(bits) + "-bit scale did not widen to 1";
        return false;
    }
    std::unique_ptr<Bytes> off_copy;
    if (has_offset) {
        if (!w->offset) {
            detail = std::to_string(bits) + "-bit offset missing";
            return false;
        }
        const float* loaded_off = static_cast<const float*>(w->offset->data());
        if (loaded_off[token] != 0.25f) {
            detail = std::to_string(bits) + "-bit offset was not 0.25";
            return false;
        }
        off_copy = std::make_unique<Bytes>(w->offset->size());
        std::memcpy(off_copy->data(), w->offset->data(), w->offset->size());
    }
    Bytes codes(w->codes->size());
    Bytes scales(w->scales->size());
    std::memcpy(codes.data(), w->codes->data(), w->codes->size());
    std::memcpy(scales.data(), w->scales->data(), w->scales->size());
    if (codes.data() == w->codes->data()) {
        detail = std::to_string(bits) + "-bit copy aliased the pack";
        return false;
    }
    const int guard = 1;
    Bytes got((uint64_t) (n + guard) * 4);
    Bytes ref((uint64_t) (n + guard) * 4);
    Bytes row0((uint64_t) (n + guard) * 4);
    std::fill_n(static_cast<float*>(got.data()), (size_t) (n + guard), 7.f);
    std::fill_n(static_cast<float*>(ref.data()), (size_t) (n + guard), 7.f);
    std::fill_n(static_cast<float*>(row0.data()), (size_t) (n + guard), 7.f);
    strata::core::ModelGeometry g;
    g.n_embd = n;
    gather_token_embd(table, g, token, got);
    embedding_gather(codes, scales, off_copy ? off_copy.get() : nullptr, n, bits, bias, w->group_elems, token, ref,
                     0);
    gather_token_embd(table, g, 0, row0);
    const float* gp = static_cast<const float*>(got.data());
    const float* rp = static_cast<const float*>(ref.data());
    const int diff = differ_floats(gp, rp, (int) n);
    const int apart = differ_floats(gp, static_cast<const float*>(row0.data()), (int) n);
    const bool guard_held = gp[n] == 7.f && rp[n] == 7.f;
    if (diff != 0 || apart == 0 || still_seven(gp, (int) n) || !guard_held) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "%d-bit %d differ, rows apart %d, guard %d", bits, diff, apart,
                      guard_held ? 1 : 0);
        detail = buf;
        return false;
    }
    return true;
}

void run_embed() {
    std::string d2, d4, d8;
    const bool s2 = one_embd_row(2, -1, 0, d2);
    const bool s4 = one_embd_row(4, -8, 1, d4);
    const bool s8 = one_embd_row(8, -16, 0, d8);
    std::string detail = "0 differ, scale widened to 1";
    if (!s2) detail = d2;
    else if (!s4) detail = d4;
    else if (!s8) detail = d8;
    verdict("an S2, S4, and S8 embedding row matches", s2 && s4 && s8, detail);

    strata::core::ModelGeometry g;
    g.n_embd = 32;
    {
        Tmp dir;
        Built b;
        add_f32(b, "other.weight", 4, 1, 0x11, -1);
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out(16);
        std::string got;
        const bool refused = threw("is missing", [&] { gather_token_embd(table, g, 0, out); }, got);
        verdict("a missing embedding is refused", refused, got);
    }
    {
        Tmp dir;
        Built b;
        add_bf16(b, "token_embd.weight", 32, 3, 0x3f800001u);
        move_tail_to_embd(b);
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out((uint64_t) 33 * 4);
        std::fill_n(static_cast<float*>(out.data()), 33, 7.f);
        std::string got;
        const bool refused = threw("not an S2/S4/S8", [&] { gather_token_embd(table, g, 1, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), 33);
        verdict("a bf16 embedding is refused", refused && held, got);
    }
    {
        Tmp dir;
        Built b;
        add_embd(b, 32, 3, 4, -8, 1, 0xA5);
        b.rows.back().codebook = 1;
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out((uint64_t) 33 * 4);
        std::fill_n(static_cast<float*>(out.data()), 33, 7.f);
        std::string got;
        const bool refused = threw("IQ4NL", [&] { gather_token_embd(table, g, 1, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), 33);
        verdict("an IQ4NL embedding is refused", refused && held, got);
    }
    {
        Tmp dir;
        Built b;
        add_embd(b, 32, 3, 4, -8, 0, 0xA5);
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out(16);
        strata::core::ModelGeometry wide = g;
        wide.n_embd = 64;
        std::string got;
        const bool refused =
            threw("dimensions or group size", [&] { gather_token_embd(table, wide, 1, out); }, got);
        verdict("a row length other than n_embd is refused", refused, got);
    }
    {
        Tmp dir;
        Built b;
        add_embd(b, 32, 3, 4, -8, 0, 0xA5);
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out((uint64_t) 33 * 4);
        std::fill_n(static_cast<float*>(out.data()), 33, 7.f);
        std::string past, neg;
        const bool hi = threw("outside", [&] { gather_token_embd(table, g, 3, out); }, past);
        const bool lo = threw("outside", [&] { gather_token_embd(table, g, -1, out); }, neg);
        const bool held = still_seven(static_cast<const float*>(out.data()), 33);
        verdict("a token outside the table is refused", hi && lo && held, hi ? (lo ? past : neg) : past);
    }
    {
        Tmp dir;
        Built b;
        add_embd(b, 32, 3, 4, -8, 0, 0xA5);
        b.rows.back().group = 24;
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out(16);
        std::string got;
        const bool refused =
            threw("dimensions or group size", [&] { gather_token_embd(table, g, 1, out); }, got);
        verdict("a group that does not divide the row is refused", refused, got);
    }
    {
        Tmp dir;
        Built b;
        std::vector<uint8_t> src(18, 0x11);
        const uint16_t h = 0x3c00;
        std::memcpy(src.data() + 16, &h, 2);
        Row row;
        row.name = "token_embd.weight";
        row.file = 1;
        row.kind = 0;
        row.ne0 = 32;
        row.ne1 = 2;
        row.code_bits = 4;
        row.code_bias = -8;
        row.group = 32;
        row.codes_bytes = 16;
        row.scales_bytes = 4;
        row.scales_fp16 = 1;
        row.act_kind = 1;
        row.dst_bytes = 20;
        add_src(b, row, src);
        write_pack(dir, b);
        PackTable table = PackTable::load(dir.path.string());
        Bytes out((uint64_t) 33 * 4);
        std::fill_n(static_cast<float*>(out.data()), 33, 7.f);
        std::string got;
        const bool refused = threw("planes do not match", [&] { gather_token_embd(table, g, 0, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), 33);
        verdict("embedding planes that miss the shape are refused", refused && held, got);
    }
}

strata::core::ModelGeometry shexp_geom(int64_t n_embd, int64_t n_ff) {
    strata::core::ModelGeometry g;
    g.n_embd = n_embd;
    g.n_ff = n_ff;
    g.n_layers = 1;
    return g;
}

void add_shexp_up_down(Built& b, int64_t n_embd, int64_t n_ff, int down_act) {
    SOpt up;
    up.bits = 4;
    up.bias = -8;
    up.group = 32;
    up.has_offset = 1;
    up.act_kind = 0;
    up.code0 = 0x5A;
    SOpt down;
    down.bits = 4;
    down.bias = -8;
    down.group = 32;
    down.has_offset = 1;
    down.codebook = down_act == 1 ? 0 : 1;
    down.act_kind = down_act;
    down.code0 = 0x3C;
    add_sform(b, tn(0, "ffn_up_shexp.weight"), n_embd, n_ff, up);
    add_sform(b, tn(0, "ffn_down_shexp.weight"), n_ff, n_embd, down);
    add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), n_embd, 1, 0x3f800001u);
}

Bytes shexp_x(int64_t n) {
    Bytes x((uint64_t) n * 4);
    auto* p = static_cast<float*>(x.data());
    for (int64_t i = 0; i < n; ++i) p[i] = (float) ((i * 17) % 29) * 0.02f - 0.2f;
    return x;
}

void run_shexp() {
    const int64_t n_embd = 256;
    const int64_t n_ff = 640;
    const auto g = shexp_geom(n_embd, n_ff);
    Tmp dir;
    {
        Built b;
        SOpt gate;
        gate.bits = 2;
        gate.bias = -1;
        gate.group = 64;
        gate.has_offset = 0;
        gate.act_kind = 0;
        gate.code0 = 0xA5;
        SOpt up;
        up.bits = 4;
        up.bias = -8;
        up.group = 32;
        up.has_offset = 1;
        up.act_kind = 1;
        up.code0 = 0x5A;
        SOpt down;
        down.bits = 4;
        down.bias = -8;
        down.group = 32;
        down.has_offset = 1;
        down.codebook = 1;
        down.act_kind = 0;
        down.code0 = 0x3C;
        add_sform(b, tn(0, "ffn_gate_shexp.weight"), n_embd, n_ff, gate);
        add_sform(b, tn(0, "ffn_up_shexp.weight"), n_embd, n_ff, up);
        add_sform(b, tn(0, "ffn_down_shexp.weight"), n_ff, n_embd, down);
        add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), n_embd, 1, 0x3f800001u);
        write_pack(dir, b);
    }
    PackTable table = PackTable::load(dir.path.string());
    const PackTensor* gate = table.find(tn(0, "ffn_gate_shexp.weight"));
    const PackTensor* up = table.find(tn(0, "ffn_up_shexp.weight"));
    const PackTensor* down = table.find(tn(0, "ffn_down_shexp.weight"));
    const PackTensor* ginp = table.find(tn(0, "ffn_gate_inp_shexp.weight"));
    const bool fields = gate && up && down && ginp && gate->code_bits == 2 && gate->code_bias == -1 &&
                        gate->group_elems == 64 && !gate->has_offset && !gate->codebook_iq4nl && gate->act_kind == 0 &&
                        gate->file_id == 0 && up->code_bits == 4 && up->code_bias == -8 && up->group_elems == 32 &&
                        up->has_offset && up->act_kind == 1 && !up->codebook_iq4nl && down->code_bits == 4 &&
                        down->codebook_iq4nl && down->act_kind == 0 && down->has_offset && down->group_elems == 32 &&
                        gate->scales && up->offset && down->offset && ginp->kind == strata::core::WeightKind::Bf16InF32 &&
                        ginp->values;
    float scale = 0.f, offset = 0.f;
    uint16_t inp = 0;
    if (fields) {
        std::memcpy(&scale, gate->scales->data(), 4);
        std::memcpy(&offset, down->offset->data(), 4);
        std::memcpy(&inp, ginp->values->data(), 2);
    }
    const bool widened = scale == 1.f && offset == 0.25f && inp == 0x3f80;
    std::string detail = "fields";
    bool matched = false;
    int affine_diff = 0;
    if (fields && widened) {
        struct SeN {
            int n;
            int pad0;
            int pad1;
            int pad2;
        };
        static_assert(sizeof(SeN) == 16, "SeN must match the shader");
        auto launch_se = [](const char* fn, const Bytes* const* bufs, int nbuf, int n) {
            const SeN dim{n, 0, 0, 0};
            const uint32_t groups = (uint32_t) ((n + 255) / 256);
            launch(fn, bufs, nbuf, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
        };
        Bytes x = shexp_x(n_embd);
        auto once = [&](bool affine, Bytes& y) {
            Bytes x_bf16((uint64_t) n_embd * 2);
            Bytes x80(q8_0_bytes(n_embd));
            Bytes xk(q8k_bytes(n_embd));
            quantize_q8_0(x, n_embd, x80);
            quantize_q8_k(x, n_embd, xk);
            f32_to_bf16_bulk(x, x_bf16, n_embd);
            Bytes gate_y((uint64_t) n_ff * 4);
            Bytes up_y((uint64_t) n_ff * 4);
            s2_gemv_q8(x80, *gate->codes, *gate->scales, gate_y, n_embd, n_ff, 32);
            SForm up_form;
            up_form.code_bits = up->code_bits;
            up_form.code_bias = up->code_bias;
            up_form.group_elems = up->group_elems;
            up_form.has_offset = up->has_offset;
            s_gemv_q8k_split(xk, *up->codes, *up->scales, up->offset.get(), up_y, n_embd, n_ff, up_form);
            const Bytes* sw[] = {&gate_y, &up_y, &gate_y};
            launch_se("se_swiglu", sw, 3, (int) n_ff);
            Bytes h(q8_0_bytes(n_ff));
            quantize_q8_0(gate_y, n_ff, h);
            SForm down_form;
            down_form.code_bits = down->code_bits;
            down_form.code_bias = down->code_bias;
            down_form.group_elems = down->group_elems;
            down_form.codebook = affine ? Codebook::Affine : Codebook::Iq4Nl;
            down_form.has_offset = down->has_offset;
            s_gemv_q8_0_split(h, *down->codes, *down->scales, down->offset.get(), y, n_ff, n_embd, down_form);
            Bytes scalar(4);
            bf16_gemv(x_bf16, *ginp->values, scalar, n_embd, 1);
            const Bytes* sc[] = {&y, &scalar};
            launch_se("se_sigmoid_scale", sc, 2, (int) n_embd);
        };
        const int guard = 1;
        Bytes got((uint64_t) (n_embd + guard) * 4);
        Bytes ref((uint64_t) (n_embd + guard) * 4);
        Bytes aff((uint64_t) (n_embd + guard) * 4);
        std::fill_n(static_cast<float*>(got.data()), (size_t) (n_embd + guard), 7.f);
        std::fill_n(static_cast<float*>(ref.data()), (size_t) (n_embd + guard), 7.f);
        std::fill_n(static_cast<float*>(aff.data()), (size_t) (n_embd + guard), 7.f);
        run_shared_expert(table, g, 0, x, got);
        once(false, ref);
        once(true, aff);
        const float* gp = static_cast<const float*>(got.data());
        const int diff = differ_floats(gp, static_cast<const float*>(ref.data()), (int) n_embd);
        affine_diff = differ_floats(gp, static_cast<const float*>(aff.data()), (int) n_embd);
        const bool guard_held = gp[n_embd] == 7.f && static_cast<const float*>(ref.data())[n_embd] == 7.f;
        matched = diff == 0 && affine_diff > 0 && !still_seven(gp, (int) n_embd) && guard_held;
        char buf[96];
        std::snprintf(buf, sizeof buf, "%d differ, affine down differed on %d", diff, affine_diff);
        detail = buf;
    }
    verdict("a mixed shared expert matches the three products", fields && widened && matched, detail);

    auto refuse = [&](const char* name, const char* needle, const std::function<void(Built&)>& build, int64_t ne,
                      int64_t nf) {
        Tmp d;
        Built b;
        build(b);
        write_pack(d, b);
        PackTable t = PackTable::load(d.path.string());
        const auto geom = shexp_geom(ne, nf);
        Bytes x = shexp_x(ne);
        Bytes out((uint64_t) (ne + 1) * 4);
        std::fill_n(static_cast<float*>(out.data()), (size_t) (ne + 1), 7.f);
        std::string got;
        const bool thrown = threw(needle, [&] { run_shared_expert(t, geom, 0, x, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), (int) ne + 1);
        verdict(name, thrown && held, got);
    };

    refuse("a missing shared-expert weight is refused", "ffn_up_shexp.weight is missing",
           [](Built& b) {
               SOpt gate;
               gate.bits = 2;
               gate.bias = -1;
               gate.group = 64;
               gate.act_kind = 0;
               add_sform(b, tn(0, "ffn_gate_shexp.weight"), 64, 64, gate);
               SOpt down;
               down.act_kind = 0;
               add_sform(b, tn(0, "ffn_down_shexp.weight"), 64, 64, down);
               add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), 64, 1, 0x3f800001u);
           },
           64, 64);
    refuse("a shared-expert gate stored as f32 is refused", "reads it as bf16",
           [](Built& b) {
               add_f32(b, tn(0, "ffn_gate_shexp.weight"), 32, 32, 0x11, -1);
               add_f32(b, tn(0, "ffn_up_shexp.weight"), 32, 32, 0x11, -1);
               add_f32(b, tn(0, "ffn_down_shexp.weight"), 32, 32, 0x11, -1);
               add_f32(b, tn(0, "ffn_gate_inp_shexp.weight"), 32, 1, 0x11, -1);
           },
           32, 32);
    refuse("a shared-expert row of the wrong length is refused", "ne0 is 8",
           [](Built& b) {
               add_f32(b, tn(0, "ffn_gate_shexp.weight"), 8, 32, 0x11, -1);
               add_f32(b, tn(0, "ffn_up_shexp.weight"), 32, 32, 0x11, -1);
               add_f32(b, tn(0, "ffn_down_shexp.weight"), 32, 32, 0x11, -1);
               add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), 32, 1, 0x3f800001u);
           },
           32, 32);
    refuse("a 2-bit shared expert with the wrong bias is refused", "bias -1",
           [](Built& b) {
               SOpt gate;
               gate.bits = 2;
               gate.bias = -8;
               gate.group = 64;
               gate.act_kind = 0;
               add_sform(b, tn(0, "ffn_gate_shexp.weight"), 64, 64, gate);
               add_shexp_up_down(b, 64, 64, 0);
           },
           64, 64);
    refuse("a Q8_K down weight is refused", "not a multiple of 256",
           [](Built& b) {
               SOpt gate;
               gate.bits = 2;
               gate.bias = -1;
               gate.group = 64;
               gate.act_kind = 0;
               add_sform(b, tn(0, "ffn_gate_shexp.weight"), 256, 640, gate);
               SOpt up;
               up.act_kind = 0;
               add_sform(b, tn(0, "ffn_up_shexp.weight"), 256, 640, up);
               SOpt down;
               down.act_kind = 1;
               add_sform(b, tn(0, "ffn_down_shexp.weight"), 640, 256, down);
               add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), 256, 1, 0x3f800001u);
           },
           256, 640);
    refuse("a shared-expert group of 24 is refused", "group_elems 24",
           [](Built& b) {
               SOpt bad;
               bad.group = 24;
               bad.act_kind = 0;
               add_sform(b, tn(0, "ffn_gate_shexp.weight"), 48, 48, bad);
               add_sform(b, tn(0, "ffn_up_shexp.weight"), 48, 48, bad);
               add_sform(b, tn(0, "ffn_down_shexp.weight"), 48, 48, bad);
               add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), 48, 1, 0x3f800001u);
           },
           48, 48);
    refuse("an f32 shared-expert weight is refused", "not an S2/S4/S8",
           [](Built& b) {
               add_f32(b, tn(0, "ffn_gate_shexp.weight"), 32, 32, 0x11, -1);
               SOpt up;
               up.act_kind = 0;
               add_sform(b, tn(0, "ffn_up_shexp.weight"), 32, 32, up);
               add_sform(b, tn(0, "ffn_down_shexp.weight"), 32, 32, up);
               add_bf16(b, tn(0, "ffn_gate_inp_shexp.weight"), 32, 1, 0x3f800001u);
           },
           32, 32);
}

void put_f16_le(uint8_t* p, uint16_t h) {
    p[0] = (uint8_t) h;
    p[1] = (uint8_t) (h >> 8);
}

std::vector<uint8_t> iq_rows(int type, int rows, int n_in, uint32_t seed) {
    const size_t rb = iq_row_bytes(type, n_in);
    const size_t bs = iq_block_bytes(type);
    if (bs < 2) throw MetalError("fixture block");
    std::mt19937 rng(seed);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    std::uniform_int_distribution<int> ex(10, 14);
    std::uniform_int_distribution<int> man(0, 1023);
    std::uniform_int_distribution<int> sgn(0, 3);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        put_f16_le(w.data() + o, h);
    }
    return w;
}

std::vector<uint8_t> expert_blob(const IqExpert& e, uint32_t seed) {
    auto blob = iq_rows(e.gu_type, (int) (2 * e.n_ff), (int) e.n_embd, seed);
    auto down = iq_rows(e.d_type, (int) e.n_embd, (int) e.n_ff, seed + 17);
    blob.insert(blob.end(), down.begin(), down.end());
    if (blob.size() != iq_expert_bytes(e)) throw MetalError("fixture blob size");
    return blob;
}

// A full expert blob with a finite fp16 scale on every block. Bytes 2 and 3 of the
// gate carry the seed, so two experts are not the same file.
std::vector<uint8_t> model_expert_blob(const IqExpert& e, uint32_t seed) {
    auto fill = [](int type, int rows, int n_in, uint32_t s) {
        const size_t rb = iq_row_bytes(type, n_in);
        const size_t bs = iq_block_bytes(type);
        if (bs < 4) throw MetalError("fixture block");
        std::vector<uint8_t> w((size_t) rows * rb);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (s + (uint32_t) i * 17u);
        const uint16_t h = 0x3c00;
        for (size_t o = 0; o < w.size(); o += bs) {
            w[o] = (uint8_t) h;
            w[o + 1] = (uint8_t) (h >> 8);
        }
        w[2] = (uint8_t) s;
        w[3] = (uint8_t) (s >> 8);
        return w;
    };
    auto blob = fill(e.gu_type, (int) (2 * e.n_ff), (int) e.n_embd, seed);
    auto down = fill(e.d_type, (int) e.n_embd, (int) e.n_ff, seed + 17);
    blob.insert(blob.end(), down.begin(), down.end());
    if (blob.size() != iq_expert_bytes(e)) throw MetalError("fixture blob size");
    return blob;
}

void add_role_span(Built& b, const std::vector<uint8_t>& bytes) {
    Row row;
    row.name = "blk.0.ffn_gate_exps.weight";
    row.file = 3;
    row.kind = 0;
    row.ne0 = (int64_t) bytes.size();
    row.ne1 = 1;
    row.dst_bytes = bytes.size();
    add_src(b, row, bytes);
}

void add_pad(Built& b) { add_f32(b, "pad", 1, 1, 0x11, -1); }

std::string native_v3(int64_t n_expert, uint64_t total, int gu, int d, uint64_t blob) {
    char buf[640];
    std::snprintf(buf, sizeof buf,
                  "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                  "[shard] (n_expert %lld, total %llu)\n"
                  "0 %d %d 0 %llu 0 0 0\n",
                  (long long) n_expert, (unsigned long long) total, gu, d, (unsigned long long) blob);
    return buf;
}

void run_experts() {
    const int64_t n_embd = 256;
    const int64_t n_ff = 32;
    const int file_experts = 2;
    IqExpert shape{kIq4Xs, kIq4Nl, n_embd, n_ff};
    const size_t blob_n = iq_expert_bytes(shape);
    const std::vector<uint8_t> blob0 = expert_blob(shape, 11);
    const std::vector<uint8_t> blob1 = expert_blob(shape, 29);
    const std::vector<uint8_t> trap = expert_blob(shape, 47);
    if (blob0 == blob1 || blob0 == trap || blob1 == trap) throw MetalError("fixture blobs collided");

    strata::core::ModelGeometry g;
    g.n_embd = n_embd;
    g.n_ff = n_ff;
    g.n_layers = 1;

    Tmp dir;
    {
        Built built;
        add_role_span(built, trap);
        write_pack(dir, built);
    }
    std::vector<uint8_t> file;
    file.insert(file.end(), blob0.begin(), blob0.end());
    file.insert(file.end(), blob1.begin(), blob1.end());
    write_bytes(dir.path / "experts.bin", file.data(), file.size());
    const std::string native = native_v3(file_experts, (uint64_t) file_experts * blob_n, kIq4Xs, kIq4Nl, blob_n);
    write_bytes(dir.path / "native_experts.txt", native.data(), native.size());

    PackTable table = PackTable::load(dir.path.string());
    const PackTensor* role = table.find("blk.0.ffn_gate_exps.weight");
    const bool role_loaded = role && role->values && role->file_id == 3 && role->values->size() == trap.size() &&
                             std::memcmp(role->values->data(), trap.data(), trap.size()) == 0;

    Bytes x((uint64_t) n_embd * 4);
    auto* xp = static_cast<float*>(x.data());
    for (int64_t i = 0; i < n_embd; ++i) xp[i] = (float) ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes xq(iq_q8_1_bytes(n_embd));
    iq_quantize_q8_1(x, n_embd, xq);

    const int guard = 1;
    auto filled = [&]() {
        Bytes y((uint64_t) (n_embd + guard) * 4);
        std::fill_n(static_cast<float*>(y.data()), (size_t) (n_embd + guard), 7.f);
        return y;
    };
    auto oracle = [&](const std::vector<uint8_t>& src, Bytes& y) {
        Bytes copied(src.size());
        std::memcpy(copied.data(), src.data(), src.size());
        iq_expert(copied, xq, shape, 1, y);
    };
    Bytes got0 = filled();
    Bytes got1 = filled();
    Bytes ref0 = filled();
    Bytes ref1 = filled();
    Bytes via_role = filled();
    run_routed_expert(table, dir.path.string(), g, 0, 0, xq, got0);
    run_routed_expert(table, dir.path.string(), g, 0, 1, xq, got1);
    oracle(blob0, ref0);
    oracle(blob1, ref1);
    oracle(trap, via_role);
    const float* p0 = static_cast<const float*>(got0.data());
    const float* p1 = static_cast<const float*>(got1.data());
    const int d0 = differ_floats(p0, static_cast<const float*>(ref0.data()), (int) n_embd);
    const int d1 = differ_floats(p1, static_cast<const float*>(ref1.data()), (int) n_embd);
    const int between = differ_floats(p0, p1, (int) n_embd);
    const int role_diff = differ_floats(p0, static_cast<const float*>(via_role.data()), (int) n_embd);
    const bool guard_held = p0[n_embd] == 7.f && p1[n_embd] == 7.f &&
                            static_cast<const float*>(ref0.data())[n_embd] == 7.f &&
                            static_cast<const float*>(ref1.data())[n_embd] == 7.f;
    const bool live = !still_seven(p0, (int) n_embd) && !still_seven(p1, (int) n_embd);
    const bool role_untouched = role && role->values &&
                                std::memcmp(role->values->data(), trap.data(), trap.size()) == 0;
    const bool matched = role_loaded && d0 == 0 && d1 == 0 && between > 0 && role_diff > 0 && guard_held && live &&
                         role_untouched && g.n_expert == 512;
    char detail[192];
    std::snprintf(detail, sizeof detail, "blob %zu, %d and %d differ, experts differed on %d, role row differed on %d",
                  blob_n, d0, d1, between, role_diff);
    verdict("a routed expert blob matches the copied bytes", matched, detail);

    {
        Bytes out = filled();
        std::string got;
        const bool thrown =
            threw("outside", [&] { run_routed_expert(table, dir.path.string(), g, 0, file_experts, xq, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), (int) n_embd + guard);
        verdict("an expert past the file count is refused", thrown && held, got);
    }

    auto refuse = [&](const char* name, const char* needle, const std::function<void(Built&)>& index,
                      const std::string& text, const std::vector<uint8_t>& experts_bin) {
        Tmp d;
        Built b;
        index(b);
        write_pack(d, b);
        if (!text.empty()) write_bytes(d.path / "native_experts.txt", text.data(), text.size());
        if (!experts_bin.empty()) write_bytes(d.path / "experts.bin", experts_bin.data(), experts_bin.size());
        PackTable t = PackTable::load(d.path.string());
        Bytes out = filled();
        std::string got;
        const bool thrown = threw(needle, [&] { run_routed_expert(t, d.path.string(), g, 0, 0, xq, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), (int) n_embd + guard);
        verdict(name, thrown && held, got);
    };
    refuse("a 3-D role span is not the expert blob", "not the concatenated",
           [](Built& b) {
               std::vector<uint8_t> row(32, 0xEE);
               add_role_span(b, row);
           },
           "", {});
    refuse("a short expert blob is refused", "but its formats make", [](Built& b) { add_pad(b); },
           native_v3(1, 32, kIq4Xs, kIq4Nl, 32), {});
    refuse("an unknown expert type is refused", "not a Coder expert type", [](Built& b) { add_pad(b); },
           native_v3(1, 32, 29, kIq4Nl, 32), {});
    refuse("a down quant used as the gate is refused", "gate type is a down quant", [](Built& b) { add_pad(b); },
           native_v3(1, 32, kIq4Nl, kIq4Nl, 32), {});
    refuse("a missing experts.bin is refused", "GGUF shard", [](Built& b) { add_pad(b); },
           native_v3(1, blob_n, kIq4Xs, kIq4Nl, blob_n), {});
}

SOpt attn_s2() {
    SOpt q;
    q.bits = 2;
    q.bias = -1;
    q.group = 64;
    q.has_offset = 0;
    q.codebook = 0;
    q.act_kind = 0;
    q.code0 = 0xA5;
    return q;
}

Bytes attn_x(int64_t n) {
    Bytes x((uint64_t) n * 4);
    auto* p = static_cast<float*>(x.data());
    for (int64_t i = 0; i < n; ++i) p[i] = (float) ((i * 17) % 29) * 0.05f - 0.4f;
    if (n >= 4) {
        p[0] = 1.f;
        p[1] = 0.5f;
        p[2] = 0.25f;
        p[3] = -0.125f;
    }
    return x;
}

void run_attn() {
    const auto g = bind_geom();
    const int64_t n_embd = g.n_embd;
    const int64_t n_out = 2 * g.n_head * g.head_dim;
    Tmp dir;
    {
        Built b;
        add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, attn_s2());
        write_pack(dir, b);
    }
    PackTable table = PackTable::load(dir.path.string());
    const PackTensor* q = table.find(tn(3, "attn_q.weight"));
    float scale = 0.f;
    const bool fields = q && q->codes && q->scales && !q->offset && q->code_bits == 2 && q->code_bias == -1 &&
                        q->group_elems == 64 && !q->has_offset && !q->codebook_iq4nl && q->act_kind == 0 &&
                        q->ne0 == n_embd && q->ne1 == n_out && q->file_id == 0;
    if (fields) std::memcpy(&scale, q->scales->data(), 4);
    const bool widened = scale == 1.f;
    std::string detail = "fields";
    bool matched = false;
    if (fields && widened) {
        Bytes x = attn_x(n_embd);
        const int guard = 1;
        Bytes got((uint64_t) (n_out + guard) * 4);
        Bytes ref((uint64_t) (n_out + guard) * 4);
        Bytes bad((uint64_t) (n_out + guard) * 4);
        std::fill_n(static_cast<float*>(got.data()), (size_t) (n_out + guard), 7.f);
        std::fill_n(static_cast<float*>(ref.data()), (size_t) (n_out + guard), 7.f);
        std::fill_n(static_cast<float*>(bad.data()), (size_t) (n_out + guard), 7.f);
        run_attn_q(table, g, 3, x, got);
        Bytes act(q8_0_bytes(n_embd));
        quantize_q8_0(x, n_embd, act);
        s2_gemv_q8(act, *q->codes, *q->scales, ref, n_embd, n_out, 32);
        Bytes flipped(q->codes->size());
        std::memcpy(flipped.data(), q->codes->data(), q->codes->size());
        static_cast<uint8_t*>(flipped.data())[0] ^= 0xFFu;
        s2_gemv_q8(act, flipped, *q->scales, bad, n_embd, n_out, 32);
        const float* gp = static_cast<const float*>(got.data());
        const int diff = differ_floats(gp, static_cast<const float*>(ref.data()), (int) n_out);
        const int flipped_diff = differ_floats(gp, static_cast<const float*>(bad.data()), (int) n_out);
        const bool guard_held = gp[n_out] == 7.f && static_cast<const float*>(ref.data())[n_out] == 7.f &&
                                static_cast<const float*>(bad.data())[n_out] == 7.f;
        const bool codes_held = static_cast<const uint8_t*>(q->codes->data())[0] == 0xA5;
        matched = diff == 0 && flipped_diff > 0 && !still_seven(gp, (int) n_out) && guard_held && codes_held;
        char buf[96];
        std::snprintf(buf, sizeof buf, "%d differ, flipped code differed on %d of %lld", diff, flipped_diff,
                      (long long) n_out);
        detail = buf;
    }
    verdict("a 2-bit query matches the S2 product", fields && widened && matched, detail);

    auto refuse = [&](const char* name, const char* needle, strata::core::ModelGeometry geom, int64_t layer,
                      const std::function<void(Built&)>& build) {
        Tmp d;
        Built b;
        build(b);
        write_pack(d, b);
        PackTable t = PackTable::load(d.path.string());
        const int64_t xn = geom.n_embd > 0 ? geom.n_embd : 1;
        Bytes x = attn_x(xn);
        int64_t yn = 8;
        if (geom.n_head > 0 && geom.head_dim > 0) yn = 2 * geom.n_head * geom.head_dim + 1;
        Bytes out((uint64_t) yn * 4);
        std::fill_n(static_cast<float*>(out.data()), (size_t) yn, 7.f);
        std::string got;
        const bool thrown = threw(needle, [&] { run_attn_q(t, geom, layer, x, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), (int) yn);
        verdict(name, thrown && held, got);
    };

    refuse("a query on a GDN layer is refused", "not full attention", g, 0, [](Built& b) { add_pad(b); });
    refuse("a missing query is refused", "is missing", g, 3, [](Built& b) { add_pad(b); });
    {
        auto z = g;
        z.qsa_interval = 0;
        refuse("a zero query interval is refused", "qsa_interval must be positive", z, 3,
               [](Built& b) { add_pad(b); });
    }
    refuse(
        "a 4-bit query is not this call", "not an S2", g, 3,
        [&](Built& b) {
            SOpt q;
            q.bits = 4;
            q.bias = -8;
            q.group = 32;
            q.act_kind = 1;
            add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, q);
        });
    refuse(
        "a 2-bit query with an offset is refused", "no offset plane", g, 3,
        [&](Built& b) {
            SOpt q = attn_s2();
            q.has_offset = 1;
            add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, q);
        });
    refuse(
        "a 2-bit query with bias -8 is refused", "bias -1", g, 3,
        [&](Built& b) {
            SOpt q = attn_s2();
            q.bias = -8;
            add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, q);
        });
    refuse(
        "a 2-bit query with group 32 is refused", "group 64", g, 3,
        [&](Built& b) {
            SOpt q = attn_s2();
            q.group = 32;
            add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, q);
        });
    refuse(
        "an IQ4NL query is refused", "affine", g, 3,
        [&](Built& b) {
            SOpt q = attn_s2();
            q.codebook = 1;
            add_sform(b, tn(3, "attn_q.weight"), n_embd, n_out, q);
        });
    refuse(
        "a query whose ne1 is the K width is refused", "ne1 is 512", g, 3,
        [&](Built& b) { add_sform(b, tn(3, "attn_q.weight"), n_embd, g.n_head_kv * g.head_dim, attn_s2()); });
    {
        auto narrow = g;
        narrow.n_embd = 32;
        narrow.n_head = 1;
        narrow.head_dim = 64;
        const int64_t rows = 2 * narrow.n_head * narrow.head_dim;
        refuse(
            "a query whose n_in is 32 is refused", "not a multiple of 64", narrow, 3,
            [&](Built& b) {
                SOpt q = attn_s2();
                q.group = 32;
                add_sform(b, tn(3, "attn_q.weight"), narrow.n_embd, rows, q);
                b.rows.back().group = 64;
            });
    }
    {
        auto empty = g;
        empty.n_head = 0;
        refuse("an empty query shape is refused", "empty shape", empty, 3,
               [&](Built& b) { add_sform(b, tn(3, "attn_q.weight"), n_embd, 64, attn_s2()); });
    }

    {
        Bytes x(4);
        Bytes out((uint64_t) (n_out + 1) * 4);
        std::fill_n(static_cast<float*>(out.data()), (size_t) (n_out + 1), 7.f);
        std::string got;
        const bool thrown = threw("activation is short", [&] { run_attn_q(table, g, 3, x, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), (int) n_out + 1);
        verdict("a short query activation is refused", thrown && held, got);
    }
    {
        Bytes x = attn_x(n_embd);
        Bytes out(4);
        std::fill_n(static_cast<float*>(out.data()), 1, 7.f);
        std::string got;
        const bool thrown = threw("output is short", [&] { run_attn_q(table, g, 3, x, out); }, got);
        const bool held = still_seven(static_cast<const float*>(out.data()), 1);
        verdict("a short query output is refused", thrown && held, got);
    }
}

uint32_t bf_word(float v) { return (uint32_t) strata::kernels::bf16_from_f32(v) << 16; }

struct FeedGeom {
    strata::core::ModelGeometry model;
    StackGeom stack;
};

void mirror_stack(FeedGeom& p, int64_t n_vocab, int n_rot, int64_t max_cells, int64_t idx_block, int64_t idx_top,
                  int64_t page) {
    const strata::core::ModelGeometry& g = p.model;
    StackGeom& s = p.stack;
    s.n_layers = g.n_layers;
    s.qsa_interval = g.qsa_interval;
    s.token.n_embd = g.n_embd;
    s.token.hc = g.hc;
    s.token.hc_lr = g.hc_lr;
    s.token.n_expert = (int) g.n_expert;
    s.token.k = (int) g.n_expert;
    s.token.n_ff_shared = g.n_ff;
    s.token.shared_tpr = 32;
    s.token.n_vocab = (int) n_vocab;
    s.token.emb_code_bits = 4;
    s.token.emb_code_bias = -8;
    s.token.emb_group = 32;
    s.token.qsa.n_embd = g.n_embd;
    s.token.qsa.max_cells = max_cells;
    s.token.qsa.qsa.n_head = g.n_head;
    s.token.qsa.qsa.n_head_kv = g.n_head_kv;
    s.token.qsa.qsa.head_dim = g.head_dim;
    s.token.qsa.qsa.n_rot = n_rot;
    s.token.qsa.qsa.idx_n_head = g.idx_q_heads;
    s.token.qsa.qsa.idx_dim = g.idx_key_dim;
    s.token.qsa.qsa.idx_block = idx_block;
    s.token.qsa.qsa.idx_top_k = idx_top;
    s.token.qsa.qsa.page_size = page;
    s.token.expert = IqExpert{kIq4Xs, kQ2_0, g.n_embd, g.n_ff};
    s.gdn = GdnLayerGeom{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
}

FeedGeom live_feed() {
    FeedGeom p;
    auto& g = p.model;
    g.n_embd = 256;
    g.n_layers = 4;
    g.qsa_interval = 4;
    g.ssm_state_size = 128;
    g.ssm_k_heads = 2;
    g.ssm_v_heads = 4;
    g.ssm_d_conv = 4;
    g.ssm_conv_channels = 1024;
    g.ssm_value_dim = 512;
    g.n_head = 24;
    g.n_head_kv = 2;
    g.head_dim = 256;
    g.idx_q_heads = 4;
    g.idx_key_dim = 64;
    g.hc = 4;
    g.hc_lr = 32;
    g.n_expert = 2;
    g.n_ff = 64;
    mirror_stack(p, 32, 64, 4, 4, 2, 2);
    return p;
}

FeedGeom tiny_feed(int64_t n_layers, int64_t interval) {
    FeedGeom p;
    auto& g = p.model;
    g.n_embd = 64;
    g.n_layers = n_layers;
    g.qsa_interval = interval;
    g.ssm_state_size = 4;
    g.ssm_k_heads = 1;
    g.ssm_v_heads = 2;
    g.ssm_d_conv = 2;
    g.ssm_conv_channels = 16;
    g.ssm_value_dim = 8;
    g.n_head = 2;
    g.n_head_kv = 2;
    g.head_dim = 64;
    g.idx_q_heads = 1;
    g.idx_key_dim = 4;
    g.hc = 2;
    g.hc_lr = 4;
    g.n_expert = 2;
    g.n_ff = 64;
    mirror_stack(p, 4, 64, 4, 4, 2, 2);
    return p;
}

FeedGeom gdn_refuse_feed() {
    FeedGeom p = live_feed();
    p.model.n_layers = 1;
    p.model.hc = 2;
    p.model.hc_lr = 4;
    p.model.n_head = 2;
    mirror_stack(p, 4, 64, 4, 4, 2, 2);
    return p;
}

FeedGeom qsa_refuse_feed() {
    FeedGeom p = gdn_refuse_feed();
    p.model.qsa_interval = 1;
    p.model.head_dim = 256;
    p.model.idx_key_dim = 64;
    mirror_stack(p, 4, 64, 4, 4, 2, 2);
    return p;
}

void add_gr_router(Built& b, const strata::core::ModelGeometry& g, int layer, uint32_t word) {
    const int64_t hc = g.hc_dim();
    add_bf16(b, tn(layer, "hc_attn_down.weight"), hc, g.hc_lr, word);
    add_bf16(b, tn(layer, "hc_attn_up.weight"), g.hc_lr, hc, word);
    add_bf16(b, tn(layer, "hc_attn_inject.weight"), hc, g.hc, word);
    add_bf16(b, tn(layer, "hc_ffn_down.weight"), hc, g.hc_lr, word);
    add_bf16(b, tn(layer, "hc_ffn_up.weight"), g.hc_lr, hc, word);
    add_bf16(b, tn(layer, "hc_ffn_inject.weight"), hc, g.hc, word);
    add_bf16(b, tn(layer, "ffn_gate_inp.weight"), g.n_embd, g.n_expert, word);
    add_f32_fill(b, tn(layer, "hc_attn_norm.weight"), hc, 1, 1.f);
    add_f32_fill(b, tn(layer, "hc_ffn_norm.weight"), hc, 1, 1.f);
}

void add_shexp2(Built& b, const strata::core::ModelGeometry& g, int layer, uint16_t scale, uint32_t word) {
    SOpt s;
    s.bits = 2;
    s.bias = -1;
    s.group = 64;
    s.has_offset = 0;
    s.act_kind = 0;
    s.scale = scale;
    s.code0 = 0xA5;
    add_sform(b, tn(layer, "ffn_gate_shexp.weight"), g.n_embd, g.n_ff, s);
    s.code0 = 0x5A;
    add_sform(b, tn(layer, "ffn_up_shexp.weight"), g.n_embd, g.n_ff, s);
    s.code0 = 0x3C;
    add_sform(b, tn(layer, "ffn_down_shexp.weight"), g.n_ff, g.n_embd, s);
    add_bf16(b, tn(layer, "ffn_gate_inp_shexp.weight"), g.n_embd, 1, word);
}

void add_shexp_plain(Built& b, const strata::core::ModelGeometry& g, int layer) {
    add_f32(b, tn(layer, "ffn_gate_shexp.weight"), g.n_embd, g.n_ff, 0x44, -1);
    add_f32(b, tn(layer, "ffn_up_shexp.weight"), g.n_embd, g.n_ff, 0x44, -1);
    add_f32(b, tn(layer, "ffn_down_shexp.weight"), g.n_ff, g.n_embd, 0x44, -1);
    add_bf16(b, tn(layer, "ffn_gate_inp_shexp.weight"), g.n_embd, 1, 0x3f800000u);
}

void add_gdn_planes(Built& b, const strata::core::ModelGeometry& g, int layer, uint16_t scale, uint32_t word) {
    add_quant(b, tn(layer, "attn_qkv.weight"), g.n_embd, g.ssm_conv_channels, 4, -8, 0, 0xA5, scale);
    add_quant(b, tn(layer, "attn_gate.weight"), g.n_embd, g.ssm_value_dim, 4, -8, 0, 0x22, scale);
    add_quant(b, tn(layer, "ssm_out.weight"), g.ssm_value_dim, g.n_embd, 8, -16, 0, 0x33, scale);
    add_f32_fill(b, tn(layer, "ssm_conv1d.weight"), g.ssm_d_conv, g.ssm_conv_channels, 0.01f);
    add_bf16(b, tn(layer, "ssm_alpha.weight"), g.n_embd, g.ssm_v_heads, word);
    add_bf16(b, tn(layer, "ssm_beta.weight"), g.n_embd, g.ssm_v_heads, word);
    add_f32_fill(b, tn(layer, "ssm_a"), g.ssm_v_heads, 1, -0.5f);
    add_f32_fill(b, tn(layer, "ssm_dt.bias"), g.ssm_v_heads, 1, -2.f);
    add_f32_fill(b, tn(layer, "ssm_norm.weight"), g.ssm_state_size, 1, 1.f);
}

void add_qsa_planes(Built& b, const strata::core::ModelGeometry& g, int layer, uint16_t scale, uint32_t word,
                    bool query_2bit) {
    const int64_t n_out = 2 * g.n_head * g.head_dim;
    if (query_2bit) {
        SOpt q = attn_s2();
        q.scale = scale;
        add_sform(b, tn(layer, "attn_q.weight"), g.n_embd, n_out, q);
    } else {
        add_quant(b, tn(layer, "attn_q.weight"), g.n_embd, n_out, 4, -8, 0, 0x5A, scale);
    }
    add_quant(b, tn(layer, "attn_k.weight"), g.n_embd, g.n_head_kv * g.head_dim, 4, -8, 0, 0x21, scale);
    add_quant(b, tn(layer, "attn_v.weight"), g.n_embd, g.n_head_kv * g.head_dim, 4, -8, 0, 0x21, scale);
    add_quant(b, tn(layer, "attn_output.weight"), g.n_head * g.head_dim, g.n_embd, 4, -8, 0, 0x21, scale);
    add_bf16(b, tn(layer, "indexer.q_proj.weight"), g.n_embd, g.idx_q_heads * g.idx_key_dim, word);
    add_bf16(b, tn(layer, "indexer.k_proj.weight"), g.n_embd, g.idx_key_dim, word);
    add_f32_fill(b, tn(layer, "attn_q_norm.weight"), g.head_dim, 1, 1.f);
    add_f32_fill(b, tn(layer, "attn_k_norm.weight"), g.head_dim, 1, 1.f);
    add_f32_fill(b, tn(layer, "indexer.q_norm.weight"), g.idx_key_dim, 1, 1.f);
    add_f32_fill(b, tn(layer, "indexer.k_norm.weight"), g.idx_key_dim, 1, 1.f);
}

void add_output_hc(Built& b, const strata::core::ModelGeometry& g, uint32_t word) {
    const int64_t hc = g.hc_dim();
    add_f32_fill(b, "output_hc_norm.weight", hc, 1, 1.f);
    add_bf16(b, "output_hc_down.weight", hc, g.hc_lr, word);
    add_bf16(b, "output_hc_up.weight", g.hc_lr, hc, word);
}

void add_output_weight(Built& b, int64_t n_embd, int64_t n_vocab, int bits, uint16_t scale) {
    if (bits == 2) {
        SOpt o;
        o.bits = 2;
        o.bias = -1;
        o.group = 64;
        o.act_kind = 0;
        o.scale = scale;
        o.code0 = 0xA5;
        add_sform(b, "output.weight", n_embd, n_vocab, o);
        return;
    }
    add_quant(b, "output.weight", n_embd, n_vocab, bits, -8, 0, 0x21, scale);
}

void add_stack_embd(Built& b, int64_t ne0, int64_t ne1, uint16_t scale, int codebook, float offset = 0.25f) {
    if (codebook) {
        SOpt e;
        e.bits = 4;
        e.bias = -8;
        e.group = 32;
        e.has_offset = 1;
        e.codebook = 1;
        e.act_kind = 1;
        e.scale = scale;
        e.code0 = 0xA5;
        add_sform(b, "token_embd.weight", ne0, ne1, e);
    } else {
        add_quant(b, "token_embd.weight", ne0, ne1, 4, -8, 1, 0xA5, scale, offset);
    }
    move_tail_to_embd(b);
}

// Two rows with different shapes. A shared offset, or two copies of one pattern, becomes the
// same direction once the residual is normalized, and the logits do not move.
void stamp_embd_rows(Built& b, int64_t n_embd) {
    const size_t row = (size_t) n_embd / 2;
    if (b.bin[1].size() < row * 4) throw MetalError("fixture embedding is short");
    uint8_t* p = b.bin[1].data();
    std::memset(p, 0x00, row);
    std::memset(p + 3 * row, 0x88, row);
    std::memset(p + 3 * row, 0xFF, 16);
}

std::string native_layers(int layers, int64_t n_expert, int gu, int d, uint64_t blob) {
    const uint64_t total = (uint64_t) layers * (uint64_t) n_expert * blob;
    char head[320];
    std::snprintf(head, sizeof head,
                  "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                  "[shard] (n_expert %lld, total %llu)\n",
                  (long long) n_expert, (unsigned long long) total);
    std::string text = head;
    for (int layer = 0; layer < layers; ++layer) {
        const uint64_t off = (uint64_t) layer * (uint64_t) n_expert * blob;
        char line[128];
        std::snprintf(line, sizeof line, "%d %d %d %llu %llu 0 0 0\n", layer, gu, d, (unsigned long long) off,
                      (unsigned long long) blob);
        text += line;
    }
    return text;
}

struct StBuf {
    Bytes gdn_state;
    Bytes gdn_conv;
    Bytes k_pool;
    Bytes v_pool;
    Bytes idx_tail;
    Bytes idx_dead;
    Bytes idx_pooled;
    Bytes block_pos;
};

StBuf make_st(const StackGeom& g) {
    const auto& q = g.token.qsa.qsa;
    const int64_t pages = (g.token.qsa.max_cells + q.page_size - 1) / q.page_size;
    const uint64_t pool =
        (uint64_t) pages * (uint64_t) q.n_head_kv * (uint64_t) q.page_size * (uint64_t) q.head_dim * 2u;
    const int64_t S = g.gdn.S;
    const int64_t C = S * (2 * g.gdn.h_k + g.gdn.h_v);
    return StBuf{Bytes((uint64_t) S * (uint64_t) g.gdn.h_v * (uint64_t) S * 4u),
                 Bytes((uint64_t) C * (uint64_t) (g.gdn.d_conv - 1) * 4u),
                 Bytes(pool),
                 Bytes(pool),
                 Bytes((uint64_t) (q.idx_block - 1) * (uint64_t) q.idx_dim * 4u),
                 Bytes((uint64_t) q.idx_dim * 4u),
                 Bytes((uint64_t) (g.token.qsa.max_cells / q.idx_block + 1) * (uint64_t) q.idx_dim * 4u),
                 Bytes(4)};
}

void zero_buf(Bytes& b) { std::memset(b.data(), 0, (size_t) b.size()); }

struct Draw {
    int id = -1;
    std::vector<float> logits;
    bool finite = false;
    bool guard = false;
    bool live = false;
};

Draw draw_token(const StackGeom& g, const PackStack& bound, const Bytes& page, int token) {
    const int64_t n = g.n_layers;
    std::vector<StBuf> bufs;
    bufs.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) bufs.push_back(make_st(g));
    std::vector<StackLayerState> views;
    views.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        zero_buf(bufs[(size_t) i].gdn_state);
        zero_buf(bufs[(size_t) i].gdn_conv);
        zero_buf(bufs[(size_t) i].k_pool);
        zero_buf(bufs[(size_t) i].v_pool);
        zero_buf(bufs[(size_t) i].idx_tail);
        zero_buf(bufs[(size_t) i].idx_dead);
        zero_buf(bufs[(size_t) i].idx_pooled);
        zero_buf(bufs[(size_t) i].block_pos);
        StackLayerState s;
        s.gdn.state = &bufs[(size_t) i].gdn_state;
        s.gdn.conv_state = &bufs[(size_t) i].gdn_conv;
        s.qsa.k_pool = &bufs[(size_t) i].k_pool;
        s.qsa.v_pool = &bufs[(size_t) i].v_pool;
        s.qsa.page_table = &page;
        s.qsa.idx_tail = &bufs[(size_t) i].idx_tail;
        s.qsa.idx_dead = &bufs[(size_t) i].idx_dead;
        s.qsa.idx_pooled = &bufs[(size_t) i].idx_pooled;
        s.qsa.idx_block_pos = &bufs[(size_t) i].block_pos;
        views.push_back(s);
    }
    Bytes R((uint64_t) g.token.hc * (uint64_t) g.token.n_embd * 4u);
    zero_buf(R);
    StackState st{&R, views.data()};
    StackScratch sc = make_stack_scratch(g);
    const int vocab = g.token.n_vocab;
    const int guard = 16;
    std::fill_n(static_cast<float*>(sc.tok.logits.data()), (size_t) (vocab + guard), 7.f);
    SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.f;
    Draw d;
    d.id = generate_stack_token(g, bound.weights, st, sc, token, 0, 0, sp, nullptr, 0);
    d.logits.assign(static_cast<const float*>(sc.tok.logits.data()),
                    static_cast<const float*>(sc.tok.logits.data()) + vocab + guard);
    d.finite = true;
    for (int i = 0; i < vocab; ++i)
        if (!std::isfinite(d.logits[(size_t) i])) d.finite = false;
    d.guard = true;
    for (int i = 0; i < guard; ++i)
        if (d.logits[(size_t) vocab + i] != 7.f) d.guard = false;
    d.live = !still_seven(d.logits.data(), vocab);
    return d;
}

int argmax_of(const std::vector<float>& logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (logits[(size_t) i] > logits[(size_t) best]) best = i;
    return best;
}

void run_stack() {
    const uint16_t scale = strata::kernels::f16_from_f32(0.02f);
    const uint32_t word = bf_word(0.01f);
    const FeedGeom live = live_feed();
    const IqExpert shape = live.stack.token.expert;
    const size_t blob_n = iq_expert_bytes(shape);
    const int layers = (int) live.model.n_layers;
    const int n_expert = (int) live.model.n_expert;
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<uint8_t> file;
    for (int layer = 0; layer < layers; ++layer) {
        for (int e = 0; e < n_expert; ++e) {
            blobs.push_back(expert_blob(shape, (uint32_t) (1000 + layer * 16 + e)));
            file.insert(file.end(), blobs.back().begin(), blobs.back().end());
        }
    }
    std::vector<uint8_t> role(32, 0xEE);

    Tmp dir;
    {
        Built b;
        for (int layer = 0; layer < layers; ++layer) {
            const bool qsa = strata::core::is_qsa_layer(live.model, layer);
            add_gr_router(b, live.model, layer, word);
            if (qsa) add_qsa_planes(b, live.model, layer, scale, word, false);
            else add_gdn_planes(b, live.model, layer, scale, word);
            add_shexp2(b, live.model, layer, scale, word);
        }
        add_output_hc(b, live.model, word);
        add_output_weight(b, live.model.n_embd, live.stack.token.n_vocab, 4, scale);
        add_stack_embd(b, live.model.n_embd, live.stack.token.n_vocab, scale, 0, 0.f);
        stamp_embd_rows(b, live.model.n_embd);
        add_role_span(b, role);
        write_pack(dir, b);
    }
    write_bytes(dir.path / "experts.bin", file.data(), file.size());
    const std::string native = native_layers(layers, n_expert, kIq4Xs, kQ2_0, blob_n);
    write_bytes(dir.path / "native_experts.txt", native.data(), native.size());

    PackTable table = PackTable::load(dir.path.string());
    const int pairs = (int) live.stack.token.qsa.qsa.n_rot / 2;
    const int cells = (int) live.stack.token.qsa.max_cells;
    std::vector<float> cos_h((size_t) cells * (size_t) pairs), sin_h(cos_h.size());
    build_rope_table((int) live.stack.token.qsa.qsa.n_rot, 100.0, cells, cos_h.data(), sin_h.data());
    Bytes cos_tab(cos_h.size() * 4);
    Bytes sin_tab(sin_h.size() * 4);
    std::memcpy(cos_tab.data(), cos_h.data(), (size_t) cos_tab.size());
    std::memcpy(sin_tab.data(), sin_h.data(), (size_t) sin_tab.size());
    const int pages = (cells + (int) live.stack.token.qsa.qsa.page_size - 1) / (int) live.stack.token.qsa.qsa.page_size;
    std::vector<int> rev((size_t) pages);
    for (int i = 0; i < pages; ++i) rev[(size_t) i] = pages - 1 - i;
    Bytes page((uint64_t) rev.size() * 4u);
    std::memcpy(page.data(), rev.data(), (size_t) page.size());

    PackStack bound;
    bind_pack_stack(table, dir.path.string(), live.model, live.stack, &cos_tab, &sin_tab, bound);

    const PackTensor* emb = table.find("token_embd.weight");
    const PackTensor* qkv = table.find(tn(0, "attn_qkv.weight"));
    const PackTensor* query = table.find(tn(3, "attn_q.weight"));
    const PackTensor* shexp = table.find(tn(0, "ffn_gate_shexp.weight"));
    const PackTensor* outw = table.find("output.weight");
    const PackTensor* role_row = table.find("blk.0.ffn_gate_exps.weight");
    bool planes = emb && qkv && query && shexp && outw && role_row && bound.weights.layers &&
                  bound.weights.emb_codes && bound.weights.out.codes && bound.layers.size() == 4 &&
                  bound.experts.size() == 4 && bound.experts[0].size() == 2 && bound.experts[3].size() == 2;
    if (planes) {
        planes = bound.weights.emb_codes->data() == emb->codes->data() &&
                 bound.weights.layers[0].gdn.qkv.codes->data() == qkv->codes->data() &&
                 bound.weights.layers[3].qsa.q.codes->data() == query->codes->data() &&
                 bound.weights.layers[0].sh_gate_codes->data() == shexp->codes->data() &&
                 bound.weights.out.codes->data() == outw->codes->data() &&
                 bound.weights.layers[3].qsa.cos_tab == &cos_tab && bound.weights.layers[3].qsa.sin_tab == &sin_tab &&
                 bound.weights.layers == bound.layers.data() &&
                 bound.layers[0].experts == bound.experts[0].data() && query->code_bits == 4 &&
                 shexp->code_bits == 2 && outw->code_bits == 4 && outw->act_kind == 1 && emb->code_bias == -8 &&
                 emb->has_offset && role_row->file_id == 3;
        const Bytes& e0 = bound.experts[0][0];
        const Bytes& e1 = bound.experts[3][1];
        const size_t last = blobs.size() - 1;
        planes = planes && e0.size() == blob_n && e1.size() == blob_n &&
                 std::memcmp(e0.data(), blobs[0].data(), blob_n) == 0 &&
                 std::memcmp(e1.data(), blobs[last].data(), blob_n) == 0 &&
                 std::memcmp(e0.data(), blobs[1].data(), blob_n) != 0 && role_row->values &&
                 role_row->values->data() != e0.data() && role_row->values->size() == role.size() &&
                 std::memcmp(role_row->values->data(), role.data(), role.size()) == 0 &&
                 std::memcmp(e0.data(), role.data(), role.size()) != 0;
    }
    char id_detail[96];
    std::snprintf(id_detail, sizeof id_detail, "blob %zu", blob_n);
    verdict("the stack reads the pack's own planes", planes, id_detail);

    Draw first;
    Draw second;
    Draw other;
    bool drew = false;
    std::string draw_err;
    if (planes) {
        try {
            first = draw_token(live.stack, bound, page, 0);
            second = draw_token(live.stack, bound, page, 0);
            other = draw_token(live.stack, bound, page, 3);
            drew = true;
        } catch (const MetalError& e) {
            draw_err = e.what();
        }
    }
    const int vocab = live.stack.token.n_vocab;
    const int again = drew ? differ_floats(first.logits.data(), second.logits.data(), vocab) : -1;
    const int moved = drew ? differ_floats(first.logits.data(), other.logits.data(), vocab) : -1;
    const int best = drew ? argmax_of(first.logits, vocab) : -1;
    const bool token_ok = drew && first.finite && second.finite && other.finite && first.guard && second.guard &&
                          other.guard && first.live && first.id == best && again == 0 && second.id == first.id &&
                          moved > 0;
    char draw_detail[192];
    if (!draw_err.empty()) {
        std::snprintf(draw_detail, sizeof draw_detail, "%s", draw_err.c_str());
    } else {
        std::snprintf(draw_detail, sizeof draw_detail, "id %d, second %d differ, token 3 differed on %d of %d", first.id,
                      again, moved, vocab);
    }
    verdict("one pack token matches a second draw", token_ok, draw_detail);

    auto refuse = [&](const char* name, const char* needle, FeedGeom fg, const Bytes* cos, const Bytes* sin,
                      const std::function<void(const Tmp&)>& write) {
        Tmp d;
        write(d);
        std::string got = "it loaded";
        bool thrown = false;
        try {
            PackTable t = PackTable::load(d.path.string());
            PackStack held;
            thrown = threw(needle, [&] { bind_pack_stack(t, d.path.string(), fg.model, fg.stack, cos, sin, held); },
                           got);
        } catch (const MetalError& e) {
            got = e.what();
        }
        verdict(name, thrown, got);
    };

    {
        FeedGeom fg = tiny_feed(1, 4);
        fg.stack.n_layers = 2;
        refuse("a stack geometry mismatch is refused", "does not match the pack", fg, nullptr, nullptr,
               [](const Tmp& d) {
                   Built b;
                   add_pad(b);
                   write_pack(d, b);
               });
    }
    refuse("a missing rope table is refused", "rope table", tiny_feed(1, 1), nullptr, nullptr, [](const Tmp& d) {
        Built b;
        add_pad(b);
        write_pack(d, b);
    });
    refuse("a missing embedding is refused", "token_embd.weight is missing", tiny_feed(1, 4), nullptr, nullptr,
           [](const Tmp& d) {
               Built b;
               add_pad(b);
               write_pack(d, b);
           });
    refuse(
        "an IQ4NL embedding is refused", "IQ4NL", tiny_feed(1, 4), nullptr, nullptr,
        [&](const Tmp& d) {
            Built b;
            add_stack_embd(b, 64, 4, scale, 1);
            write_pack(d, b);
        });
    refuse(
        "a missing output mix is refused", "output_hc_", tiny_feed(1, 4), nullptr, nullptr,
        [&](const Tmp& d) {
            Built b;
            add_stack_embd(b, 64, 4, scale, 0);
            write_pack(d, b);
        });
    {
        Tmp d;
        Built b;
        const FeedGeom fg = tiny_feed(1, 4);
        add_output_hc(b, fg.model, 0x3f800000u);
        add_stack_embd(b, 64, 4, scale, 0);
        write_pack(d, b);
        PackTable t = PackTable::load(d.path.string());
        PackStack held;
        std::string got;
        const bool thrown =
            threw("output.weight is missing",
                  [&] { bind_pack_stack(t, d.path.string(), fg.model, fg.stack, nullptr, nullptr, held); }, got);
        verdict("a missing output weight is refused", thrown, got);
    }
    {
        Tmp d;
        Built b;
        const FeedGeom fg = tiny_feed(1, 4);
        add_output_hc(b, fg.model, 0x3f800000u);
        add_output_weight(b, fg.model.n_embd, fg.stack.token.n_vocab, 2, scale);
        add_stack_embd(b, 64, 4, scale, 0);
        write_pack(d, b);
        PackTable t = PackTable::load(d.path.string());
        PackStack held;
        std::string got;
        const bool thrown = threw("no Q8_K contract",
                                  [&] { bind_pack_stack(t, d.path.string(), fg.model, fg.stack, nullptr, nullptr, held); },
                                  got);
        const bool named = got.find("output.weight") != std::string::npos;
        verdict("a 2-bit output weight is refused", thrown && named, got);
    }

    const FeedGeom gdn = gdn_refuse_feed();
    auto plant_gdn = [&](Built& b, bool shexp_2bit) {
        add_gr_router(b, gdn.model, 0, word);
        add_gdn_planes(b, gdn.model, 0, scale, word);
        if (shexp_2bit) add_shexp2(b, gdn.model, 0, scale, word);
        else add_shexp_plain(b, gdn.model, 0);
        add_output_hc(b, gdn.model, word);
        add_output_weight(b, gdn.model.n_embd, gdn.stack.token.n_vocab, 4, scale);
        add_stack_embd(b, gdn.model.n_embd, gdn.stack.token.n_vocab, scale, 0);
    };
    refuse(
        "a shared expert that is not 2-bit is refused", "stack's shared expert is 2-bit", gdn, nullptr, nullptr,
        [&](const Tmp& d) {
            Built b;
            plant_gdn(b, false);
            write_pack(d, b);
        });
    refuse(
        "a missing native expert file is refused", "GGUF shard", gdn, nullptr, nullptr,
        [&](const Tmp& d) {
            Built b;
            plant_gdn(b, true);
            add_role_span(b, role);
            write_pack(d, b);
        });
    {
        Tmp d;
        Built b;
        plant_gdn(b, true);
        write_pack(d, b);
        const std::string one = native_v3(gdn.model.n_expert, (uint64_t) gdn.model.n_expert * blob_n, kIq4Xs, kQ2_0,
                                           blob_n);
        write_bytes(d.path / "native_experts.txt", one.data(), one.size());
        PackTable t = PackTable::load(d.path.string());
        PackStack held;
        std::string got;
        const bool thrown =
            threw("GGUF shard", [&] { bind_pack_stack(t, d.path.string(), gdn.model, gdn.stack, nullptr, nullptr, held); },
                  got);
        const bool named = got.find("experts.bin") != std::string::npos;
        verdict("a missing experts.bin is refused", thrown && named, got);
    }

    const FeedGeom qref = qsa_refuse_feed();
    Bytes rope_stub(16);
    zero_buf(rope_stub);
    refuse(
        "a 2-bit stack query is refused", "no Q8_K contract", qref, &rope_stub, &rope_stub,
        [&](const Tmp& d) {
            Built b;
            add_gr_router(b, qref.model, 0, word);
            add_qsa_planes(b, qref.model, 0, scale, word, true);
            add_shexp2(b, qref.model, 0, scale, word);
            add_output_hc(b, qref.model, word);
            add_output_weight(b, qref.model.n_embd, qref.stack.token.n_vocab, 4, scale);
            add_stack_embd(b, qref.model.n_embd, qref.stack.token.n_vocab, scale, 0);
            write_pack(d, b);
        });
}

double rel_l1(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        num += std::fabs((double) a[i] - (double) b[i]);
        den += std::fabs((double) a[i]);
    }
    if (den == 0) return num == 0 ? 0 : 1;
    return num / den;
}

bool all_finite(const float* a, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(a[i])) return false;
    return true;
}

std::vector<uint8_t> read_at_path(const std::filesystem::path& path, uint64_t off, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw MetalError("cannot open " + path.string());
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
        std::fclose(f);
        throw MetalError("seek failed in " + path.string());
    }
    std::vector<uint8_t> got(n);
    const size_t nread = n ? std::fread(got.data(), 1, n, f) : 0;
    std::fclose(f);
    if (nread != n) throw MetalError("short read in " + path.string());
    return got;
}

// One GDN layer at the model width. This check writes the index and the bins, then
// loads them. n_layers is 1. The other 47 layers are not written. experts.bin is the
// 512 blobs. The 3-D role span is a different file and is not passed to the expert.
void run_model_layer() {
    strata::core::ModelGeometry g;
    if (g.n_embd != 2560 || g.n_expert != 512 || g.n_ff != 640 || g.hc != 4 || g.hc_lr != 320 ||
        g.qsa_interval != 4 || g.n_layers != 48 || g.ssm_state_size != 128 || g.ssm_k_heads != 16 ||
        g.ssm_v_heads != 48 || g.ssm_d_conv != 4 || g.ssm_conv_channels != 10240 || g.ssm_value_dim != 6144) {
        throw MetalError("model geometry defaults moved");
    }
    g.n_layers = 1;
    const IqExpert shape{kIq4Xs, kQ2_0, g.n_embd, g.n_ff};
    const size_t blob_n = iq_expert_bytes(shape);
    if (blob_n != 2201600) throw MetalError("model expert blob is not 2201600");
    const int n_expert = (int) g.n_expert;
    const uint16_t scale = strata::kernels::f16_from_f32(0.02f);
    const uint32_t word = bf_word(0.01f);

    std::printf("one model layer from a pack index\n");
    Tmp dir;
    std::vector<uint8_t> role = model_expert_blob(shape, 9001);
    {
        Built b;
        add_gr_router(b, g, 0, word);
        add_gdn_planes(b, g, 0, scale, word);
        add_shexp2(b, g, 0, scale, word);
        add_role_span(b, role);
        write_pack(dir, b);
    }
    const auto experts_path = dir.path / "experts.bin";
    {
        std::FILE* f = std::fopen(experts_path.c_str(), "wb");
        if (!f) throw MetalError("cannot write experts.bin");
        std::vector<char> buf(8u << 20);
        std::setvbuf(f, buf.data(), _IOFBF, buf.size());
        for (int e = 0; e < n_expert; ++e) {
            const std::vector<uint8_t> blob = model_expert_blob(shape, (uint32_t) (1000 + e));
            if (std::fwrite(blob.data(), 1, blob.size(), f) != blob.size()) {
                std::fclose(f);
                throw MetalError("short expert write");
            }
        }
        if (std::fclose(f) != 0) throw MetalError("experts.bin did not close");
    }
    const uint64_t file_n = (uint64_t) n_expert * (uint64_t) blob_n;
    const std::string native = native_v3(n_expert, file_n, kIq4Xs, kQ2_0, blob_n);
    write_bytes(dir.path / "native_experts.txt", native.data(), native.size());

    PackTable table = PackTable::load(dir.path.string());
    const PackTensor* qkv = table.find(tn(0, "attn_qkv.weight"));
    const PackTensor* gate = table.find(tn(0, "attn_gate.weight"));
    const PackTensor* ssm = table.find(tn(0, "ssm_out.weight"));
    const PackTensor* router = table.find(tn(0, "ffn_gate_inp.weight"));
    const PackTensor* shexp = table.find(tn(0, "ffn_down_shexp.weight"));
    const PackTensor* role_row = table.find("blk.0.ffn_gate_exps.weight");
    std::string err;
    const bool schedule = check_pack_layer(table, g, 0, err);
    const bool shapes = schedule && qkv && gate && ssm && router && shexp && role_row && qkv->codes && ssm->codes &&
                        shexp->codes && router->values && role_row->values && qkv->ne0 == g.n_embd &&
                        qkv->ne1 == g.ssm_conv_channels && qkv->code_bits == 4 && qkv->act_kind == 1 &&
                        gate->ne0 == g.n_embd && gate->ne1 == g.ssm_value_dim && gate->code_bits == 4 &&
                        ssm->ne0 == g.ssm_value_dim && ssm->ne1 == g.n_embd && ssm->code_bits == 8 &&
                        router->kind == strata::core::WeightKind::Bf16InF32 && router->ne0 == g.n_embd &&
                        router->ne1 == g.n_expert && router->values->size() == (uint64_t) g.n_embd * (uint64_t) g.n_expert * 2 &&
                        shexp->code_bits == 2 && shexp->group_elems == 64 && shexp->code_bias == -1 && !shexp->has_offset &&
                        shexp->ne0 == g.n_ff && shexp->ne1 == g.n_embd && role_row->file_id == 3 &&
                        role_row->values->size() == role.size() &&
                        std::memcmp(role_row->values->data(), role.data(), role.size()) == 0 &&
                        !strata::core::is_qsa_layer(g, 0);
    char shape_detail[160];
    std::snprintf(shape_detail, sizeof shape_detail, "tensors %zu, n_embd %lld, experts %d", table.size(),
                  (long long) g.n_embd, n_expert);
    verdict("one pack layer passes the name check", shapes, shapes ? shape_detail : err);

    std::string outside;
    const bool second = !check_pack_layer(table, g, 1, outside) && outside.find("outside") != std::string::npos;
    verdict("a second layer is outside this pack", second, outside);

    PackLayer layer{};
    std::string bind_err = err;
    bool bound = false;
    if (shapes) {
        try {
            layer = bind_layer(table, g, 0);
            std::vector<const void*> ptrs;
            collect(layer, ptrs);
            bound = !layer.qsa_layer && layer.gdn.qkv.codes && layer.gdn.qkv.codes->data() == qkv->codes->data() &&
                    layer.gdn.qkv.form.code_bits == 4 && layer.gdn.ssm_out.codes &&
                    layer.gdn.ssm_out.codes->data() == ssm->codes->data() && layer.gdn.ssm_out.form.code_bits == 8 &&
                    layer.router && layer.router->data() == router->values->data() && layer.gdn.alpha &&
                    layer.attn.down && layer.ffn.down && layer.qsa.q.codes == nullptr &&
                    !aliased(shexp->codes->data(), ptrs) && !aliased(role_row->values->data(), ptrs);
        } catch (const MetalError& e) {
            bind_err = e.what();
        }
    }
    verdict("one pack layer binds", bound, bound ? "GDN planes, router, residual" : bind_err);

    const int n = (int) g.n_embd;
    const int guard = 1;
    bool repeat = false;
    bool moved = false;
    std::string gdn_detail = bind_err;
    std::string move_detail = bind_err;
    if (bound) {
        try {
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4), state_b(state_n * 4), conv_b(conv_n * 4);
            Bytes state_z(state_n * 4), conv_z(conv_n * 4);
            std::memset(state.data(), 0, (size_t) state.size());
            std::memset(conv.data(), 0, (size_t) conv.size());
            std::memset(state_b.data(), 0, (size_t) state_b.size());
            std::memset(conv_b.data(), 0, (size_t) conv_b.size());
            std::memset(state_z.data(), 0, (size_t) state_z.size());
            std::memset(conv_z.data(), 0, (size_t) conv_z.size());
            Bytes x((uint64_t) n * 4);
            auto* xp = static_cast<float*>(x.data());
            for (int i = 0; i < n; ++i) xp[i] = (float) ((i * 17) % 29) * 0.01f - 0.1f;
            Bytes y((uint64_t) (n + guard) * 4);
            Bytes y2((uint64_t) (n + guard) * 4);
            Bytes yz((uint64_t) (n + guard) * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) (n + guard), 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) (n + guard), 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) (n + guard), 7.f);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            GdnLayerState st{&state, &conv};
            GdnLayerState st_b{&state_b, &conv_b};
            GdnLayerState st_z{&state_z, &conv_z};
            gdn_layer(lg, layer.gdn, st, sc, x, y);
            gdn_layer(lg, layer.gdn, st_b, sc, x, y2);
            Bytes zero((uint64_t) n * 4);
            std::memset(zero.data(), 0, (size_t) zero.size());
            gdn_layer(lg, layer.gdn, st_z, sc, zero, yz);
            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const int again = differ_floats(yp, y2p, n);
            const int apart = differ_floats(yp, yzp, n);
            const double rel = rel_l1(yp, yzp, n);
            const bool guard_held = yp[n] == 7.f && y2p[n] == 7.f && yzp[n] == 7.f;
            const bool live = all_finite(yp, n) && all_finite(y2p, n) && all_finite(yzp, n) && !still_seven(yp, n);
            repeat = again == 0 && live && guard_held;
            moved = apart > 0 && live && guard_held && std::isfinite(rel);
            char buf[192];
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            gdn_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n, rel * 100.0);
            move_detail = buf;
        } catch (const MetalError& e) {
            gdn_detail = e.what();
            move_detail = e.what();
        }
    }
    verdict("one pack GDN step repeats", repeat, gdn_detail);
    verdict("a zero activation moves that step", moved, move_detail);

    const uint64_t on_disk = std::filesystem::file_size(experts_path);
    int stamps = 0;
    {
        std::FILE* f = std::fopen(experts_path.c_str(), "rb");
        if (!f) throw MetalError("cannot reopen experts.bin");
        for (int e = 0; e < n_expert; ++e) {
            if (fseeko(f, (off_t) e * (off_t) blob_n, SEEK_SET) != 0) {
                std::fclose(f);
                throw MetalError("experts.bin seek failed");
            }
            uint8_t b[4] = {};
            if (std::fread(b, 1, 4, f) != 4) {
                std::fclose(f);
                throw MetalError("experts.bin stamp read failed");
            }
            const uint32_t s = (uint32_t) (1000 + e);
            if (b[0] != 0x00 || b[1] != 0x3c || b[2] != (uint8_t) s || b[3] != (uint8_t) (s >> 8)) ++stamps;
        }
        std::fclose(f);
    }

    Bytes x((uint64_t) n * 4);
    auto* xp = static_cast<float*>(x.data());
    for (int i = 0; i < n; ++i) xp[i] = (float) ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes xq(iq_q8_1_bytes(g.n_embd));
    iq_quantize_q8_1(x, g.n_embd, xq);
    auto filled = [&]() {
        Bytes y((uint64_t) (n + guard) * 4);
        std::fill_n(static_cast<float*>(y.data()), (size_t) (n + guard), 7.f);
        return y;
    };
    const int sample[] = {0, 1, 256, 511};
    int sample_diff = 0;
    bool sample_guard = true;
    std::vector<float> y0((size_t) n, 0.f), y511((size_t) n, 0.f);
    bool experts_ran = false;
    std::string expert_err;
    try {
        for (int id : sample) {
            const std::vector<uint8_t> file_blob = read_at_path(experts_path, (uint64_t) id * blob_n, blob_n);
            const std::vector<uint8_t> made = model_expert_blob(shape, (uint32_t) (1000 + id));
            if (file_blob != made) throw MetalError("expert " + std::to_string(id) + " bytes are not the written blob");
            Bytes got = filled();
            Bytes ref = filled();
            run_routed_expert(table, dir.path.string(), g, 0, id, xq, got);
            Bytes copied(file_blob.size());
            std::memcpy(copied.data(), file_blob.data(), file_blob.size());
            iq_expert(copied, xq, shape, 1, ref);
            const float* gp = static_cast<const float*>(got.data());
            sample_diff += differ_floats(gp, static_cast<const float*>(ref.data()), n);
            if (gp[n] != 7.f || static_cast<const float*>(ref.data())[n] != 7.f) sample_guard = false;
            if (!all_finite(gp, n) || still_seven(gp, n)) throw MetalError("expert " + std::to_string(id) + " was not live");
            if (id == 0) std::memcpy(y0.data(), gp, (size_t) n * 4);
            if (id == 511) std::memcpy(y511.data(), gp, (size_t) n * 4);
        }
        experts_ran = true;
    } catch (const MetalError& e) {
        expert_err = e.what();
    }
    const int between = experts_ran ? differ_floats(y0.data(), y511.data(), n) : 0;
    const double between_rel = experts_ran ? rel_l1(y0.data(), y511.data(), n) : 0;
    Bytes via_role = filled();
    {
        Bytes copied(role.size());
        std::memcpy(copied.data(), role.data(), role.size());
        iq_expert(copied, xq, shape, 1, via_role);
    }
    const int role_diff = experts_ran ? differ_floats(y0.data(), static_cast<const float*>(via_role.data()), n) : 0;
    const double role_rel = experts_ran ? rel_l1(y0.data(), static_cast<const float*>(via_role.data()), n) : 0;
    const bool role_held = role_row && role_row->values &&
                           std::memcmp(role_row->values->data(), role.data(), role.size()) == 0;
    const bool blobs_ok = experts_ran && on_disk == file_n && stamps == 0 && sample_diff == 0 && sample_guard &&
                          between > 0 && role_diff > 0 && role_held;
    char blob_detail[320];
    if (!expert_err.empty()) {
        std::snprintf(blob_detail, sizeof blob_detail, "%s", expert_err.c_str());
    } else {
        std::snprintf(blob_detail, sizeof blob_detail,
                      "blob %zu, file %llu, stamps %d, %d differ, expert 511 %.2f%% apart", blob_n,
                      (unsigned long long) on_disk, stamps, sample_diff, between_rel * 100.0);
    }
    verdict("512 expert blobs match the copied bytes", blobs_ok, blob_detail);

    char role_detail[160];
    std::snprintf(role_detail, sizeof role_detail, "%d of %d differ, %.2f%% apart", role_diff, n, role_rel * 100.0);
    verdict("the role span is not that expert", blobs_ok && role_diff > 0, experts_ran ? role_detail : expert_err);

    {
        Bytes out = filled();
        std::string got;
        const bool thrown = threw("outside", [&] { run_routed_expert(table, dir.path.string(), g, 0, n_expert, xq, out); },
                                  got);
        const bool held = still_seven(static_cast<const float*>(out.data()), n + guard);
        verdict("an expert past 511 is refused", thrown && held, got);
    }
}

int q2_dp4a(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; ++i) {
        int sa = (a >> (8 * i)) & 0xFF;
        int sb = (b >> (8 * i)) & 0xFF;
        if (sa >= 128) sa -= 256;
        if (sb >= 128) sb -= 256;
        s += sa * sb;
    }
    return s;
}

int q2_pack_byte(unsigned byte) {
    int r = 0;
    for (int c = 0; c < 4; ++c) {
        const int sym = (int) ((byte >> (2 * c)) & 3u) - 1;
        r |= (sym & 0xFF) << (8 * c);
    }
    return r;
}

int q2_ld(const uint8_t* p) {
    const uint32_t u = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
    return (int) u;
}

float q2_partial(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 2;
    const uint8_t* x = x0 + iqs * 36;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const unsigned b0 = qs[8 * iqs + 2 * j];
        const unsigned b1 = qs[8 * iqs + 2 * j + 1];
        sumi = q2_dp4a(q2_pack_byte(b0), q2_ld(x + 4 + 4 * (j * 2)), sumi);
        sumi = q2_dp4a(q2_pack_byte(b1), q2_ld(x + 4 + 4 * (j * 2 + 1)), sumi);
    }
    uint16_t hs = 0, hd = 0;
    std::memcpy(&hs, blk, 2);
    std::memcpy(&hd, x, 2);
    const float d = strata::kernels::f32_from_f16(hs) * strata::kernels::f32_from_f16(hd);
    return d * (float) sumi;
}

float q2_row(const uint8_t* row, const uint8_t* col, int n_in) {
    float lane[32] = {};
    const int npart = (n_in / 64) * 2;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 2;
        const int iqs = k % 2;
        lane[k & 31] += q2_partial(row + (size_t) kbx * 18, col + (size_t) kbx * 2 * 36, iqs);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void q2_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    const size_t row = (size_t) (n_in / 64) * 18;
    for (int r = 0; r < n_out; ++r) y[r] = q2_row(w + (size_t) r * row, x, n_in);
}

// IQ4_XS block, 136 bytes and 256 values. Same integer dot as iq_mmvq_iq4_xs.
const int kIq4Code[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void xs_nibbles(int aux, int& vx, int& vy) {
    vx = vy = 0;
    for (int b = 0; b < 4; ++b) {
        const unsigned byte = (unsigned) (aux >> (8 * b)) & 0xFFu;
        vx |= (int) (uint8_t) kIq4Code[byte & 0xFu] << (8 * b);
        vy |= (int) (uint8_t) kIq4Code[byte >> 4] << (8 * b);
    }
}

float xs_partial(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 8;
    const uint8_t* x = x0 + (iqs / 4) * 36;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        int vx = 0, vy = 0;
        xs_nibbles(q2_ld(qs + 4 * (iqs + j)), vx, vy);
        sumi = q2_dp4a(vx, q2_ld(x + 4 + 4 * j), sumi);
        sumi = q2_dp4a(vy, q2_ld(x + 4 + 4 * (j + 4)), sumi);
    }
    const int sh = (int) blk[2] | ((int) blk[3] << 8);
    const int ls = (((blk + 4)[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((sh >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const uint16_t hs = (uint16_t) (blk[0] | (blk[1] << 8));
    const uint16_t hd = (uint16_t) (x[0] | (x[1] << 8));
    const float d = strata::kernels::f32_from_f16(hs) * strata::kernels::f32_from_f16(hd);
    return d * (float) sumi;
}

float xs_row(const uint8_t* row, const uint8_t* col, int n_in) {
    float lane[32] = {};
    const int npart = (n_in / 256) * 8;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 8;
        const int iqs = 4 * (k % 8);
        lane[k & 31] += xs_partial(row + (size_t) kbx * 136, col + (size_t) kbx * 8 * 36, iqs);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void xs_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    const size_t row = (size_t) (n_in / 256) * 136;
    for (int r = 0; r < n_out; ++r) y[r] = xs_row(w + (size_t) r * row, x, n_in);
}

// Q4_K is 144 bytes and Q5_K is 176. Both are 256 values. The dot is vec_dot_q4_K_q8_1 /
// vec_dot_q5_K_q8_1: d * sum(scale * q * a) - dmin * sum(min * a), with a the q8_1 values.
int k_u16(const uint8_t* p) { return (int) p[0] | ((int) p[1] << 8); }

void k_scale_min(const uint8_t* scales8, int bq8_offset, int sc[2], int mn[2]) {
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const unsigned s0 = (unsigned) k_u16(scales8 + 2 * (jm + 0));
    const unsigned s2 = (unsigned) k_u16(scales8 + 2 * (jm + 2));
    const unsigned s4 = (unsigned) k_u16(scales8 + 2 * (jm + 4));
    const unsigned hi = j >= 2 ? 0xFFFFFFFFu : 0u;
    const unsigned a0 = ((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi);
    const unsigned a1 = ((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi);
    sc[0] = (int) (a0 & 0xFFu);
    sc[1] = (int) ((a0 >> 8) & 0xFFu);
    mn[0] = (int) (a1 & 0xFFu);
    mn[1] = (int) ((a1 >> 8) & 0xFFu);
}

float k_partial(const uint8_t* blk, const uint8_t* x0, int iqs, bool q5) {
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int byte = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    const uint8_t* ql = q5 ? blk + 48 : blk + 16;
    const int v0 = q2_ld(ql + byte);
    const int v1 = q2_ld(ql + byte + 16);
    int vh0 = 0, vh1 = 0;
    if (q5) {
        const int qhb = 4 * ((iqs / 2) % 4);
        vh0 = q2_ld(blk + 16 + qhb) >> bq8_offset;
        vh1 = q2_ld(blk + 32 + qhb) >> bq8_offset;
    }
    int sc[2], mn[2];
    k_scale_min(blk + 4, bq8_offset, sc, mn);
    float sum_d = 0, sum_m = 0;
    for (int i = 0; i < 2; ++i) {
        const uint8_t* bq8 = x0 + (size_t) (bq8_offset + i) * 36;
        const float d8 = strata::kernels::f32_from_f16((uint16_t) k_u16(bq8));
        const int qoff = (iqs / 2) % 4;
        const int u0 = q2_ld(bq8 + 4 + 4 * qoff);
        const int u1 = q2_ld(bq8 + 4 + 4 * (qoff + 4));
        int p0 = (v0 >> (4 * i)) & 0x0F0F0F0F;
        int p1 = (v1 >> (4 * i)) & 0x0F0F0F0F;
        if (q5) {
            p0 |= ((vh0 >> i) << 4) & 0x10101010;
            p1 |= ((vh1 >> i) << 4) & 0x10101010;
        }
        const int dot1 = q5 ? q2_dp4a(p0, u0, q2_dp4a(p1, u1, 0)) : q2_dp4a(p1, u1, q2_dp4a(p0, u0, 0));
        const int dot2 = q5 ? q2_dp4a(0x01010101, u0, q2_dp4a(0x01010101, u1, 0))
                            : q2_dp4a(0x01010101, u1, q2_dp4a(0x01010101, u0, 0));
        sum_d += d8 * (float) (dot1 * sc[i]);
        sum_m += d8 * (float) (dot2 * mn[i]);
    }
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(blk));
    const float dmin = strata::kernels::f32_from_f16((uint16_t) k_u16(blk + 2));
    return d * sum_d - dmin * sum_m;
}

float k_row(const uint8_t* row, const uint8_t* col, int n_in, bool q5) {
    float lane[32] = {};
    const int block = q5 ? 176 : 144;
    const int npart = (n_in / 256) * 16;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 16;
        const int iqs = 2 * (k % 16);
        lane[k & 31] += k_partial(row + (size_t) kbx * block, col + (size_t) kbx * 8 * 36, iqs, q5);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void k_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, bool q5, float* y) {
    const size_t row = (size_t) (n_in / 256) * (q5 ? 176 : 144);
    for (int r = 0; r < n_out; ++r) y[r] = k_row(w + (size_t) r * row, x, n_in, q5);
}

void k_scale_byte(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void dequant_q4(const uint8_t* block, float* out) {
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(block));
    const float mn = strata::kernels::f32_from_f16((uint16_t) k_u16(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* q = block + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int sc = 0, m = 0;
        k_scale_byte(is, scales, sc, m);
        const float d1 = d * (float) sc;
        const float m1 = mn * (float) m;
        k_scale_byte(is + 1, scales, sc, m);
        const float d2 = d * (float) sc;
        const float m2 = mn * (float) m;
        for (int l = 0; l < 32; ++l) *out++ = d1 * (float) (q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *out++ = d2 * (float) (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

void dequant_q5(const uint8_t* block, float* out) {
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(block));
    const float mn = strata::kernels::f32_from_f16((uint16_t) k_u16(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* ql = block + 48;
    int is = 0;
    int u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        int sc = 0, m = 0;
        k_scale_byte(is, scales, sc, m);
        const float d1 = d * (float) sc;
        const float m1 = mn * (float) m;
        k_scale_byte(is + 1, scales, sc, m);
        const float d2 = d * (float) sc;
        const float m2 = mn * (float) m;
        for (int l = 0; l < 32; ++l)
            *out++ = d1 * (float) ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l)
            *out++ = d2 * (float) ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

void q8_dequant(const uint8_t* x, int n, float* a) {
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = x + (size_t) b * 36;
        const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(blk));
        for (int i = 0; i < 32; ++i) {
            int q = blk[4 + i];
            if (q >= 128) q -= 256;
            a[(size_t) b * 32 + i] = d * (float) q;
        }
    }
}

void scalar_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, bool q5, float* y) {
    std::vector<float> act((size_t) n_in), row((size_t) n_in);
    q8_dequant(x, n_in, act.data());
    const size_t bytes = (size_t) (n_in / 256) * (q5 ? 176 : 144);
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* src = w + (size_t) r * bytes;
        for (int b = 0; b < n_in / 256; ++b) {
            if (q5) dequant_q5(src + (size_t) b * 176, row.data() + (size_t) b * 256);
            else dequant_q4(src + (size_t) b * 144, row.data() + (size_t) b * 256);
        }
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += (double) row[(size_t) i] * (double) act[(size_t) i];
        y[r] = (float) acc;
    }
}

bool k_scales_match(const uint8_t* block) {
    const uint8_t* scales = block + 4;
    for (int pair = 0; pair < 4; ++pair) {
        int sc[2], mn[2];
        k_scale_min(scales, pair * 2, sc, mn);
        for (int t = 0; t < 2; ++t) {
            int d = 0, m = 0;
            k_scale_byte(pair * 2 + t, scales, d, m);
            if (sc[t] != d || mn[t] != m) return false;
        }
    }
    return true;
}

void check_k(const char* match_name, const char* move_name, const uint8_t* w, uint64_t nbytes, int n_in, int n_out,
             bool q5, int flip_at) {
    const size_t row = (size_t) (n_in / 256) * (q5 ? 176 : 144);
    const bool layout = w && nbytes == (uint64_t) n_out * row && n_in % 256 == 0;
    bool scales = false;
    if (layout) {
        scales = k_scales_match(w) && k_scales_match(w + (size_t) (n_out - 1) * row);
        if (n_in >= 512) scales = scales && k_scales_match(w + (q5 ? 176 : 144));
    }
    std::vector<float> x((size_t) n_in);
    for (int i = 0; i < n_in; ++i) x[(size_t) i] = ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes xf((uint64_t) n_in * 4);
    std::memcpy(xf.data(), x.data(), xf.size());
    Bytes xq(iq_q8_1_bytes(n_in));
    iq_quantize_q8_1(xf, n_in, xq);
    std::vector<uint8_t> xqb((size_t) xq.size());
    std::memcpy(xqb.data(), xq.data(), xqb.size());
    std::vector<float> got((size_t) n_out), host((size_t) n_out), scalar((size_t) n_out);
    double rel = 1, irel = 1;
    bool ok = false;
    if (layout) {
        Bytes dw(nbytes);
        std::memcpy(dw.data(), w, nbytes);
        Bytes dy((uint64_t) n_out * 4);
        iq_mmvq(q5 ? kQ5_K : kQ4_K, dw, 0, xq, n_in, n_out, 1, dy);
        std::memcpy(got.data(), dy.data(), got.size() * 4);
        k_ref(w, xqb.data(), n_in, n_out, q5, host.data());
        scalar_ref(w, xqb.data(), n_in, n_out, q5, scalar.data());
        rel = rel_l1(scalar.data(), got.data(), n_out);
        irel = rel_l1(host.data(), got.data(), n_out);
        ok = scales && all_finite(got.data(), n_out) && rel <= 1e-5 && irel <= 1e-5;
    }
    char detail[120];
    std::snprintf(detail, sizeof detail, "rel %.3e, integer %.3e, %d rows", rel, irel, n_out);
    verdict(match_name, layout && ok, layout ? detail : "bytes");

    int moved = 0;
    bool zero_clear = false;
    int from_zero = 0;
    if (ok) {
        Bytes dw(nbytes);
        std::memcpy(dw.data(), w, nbytes);
        static_cast<uint8_t*>(dw.data())[flip_at] ^= 1u;
        Bytes dy((uint64_t) n_out * 4);
        iq_mmvq(q5 ? kQ5_K : kQ4_K, dw, 0, xq, n_in, n_out, 1, dy);
        const float* yp = static_cast<const float*>(dy.data());
        for (int i = 0; i < n_out; ++i)
            if (yp[i] != got[(size_t) i]) ++moved;
        std::memcpy(dw.data(), w, nbytes);
        Bytes zx(xq.size());
        std::memset(zx.data(), 0, zx.size());
        iq_mmvq(q5 ? kQ5_K : kQ4_K, dw, 0, zx, n_in, n_out, 1, dy);
        const float* zp = static_cast<const float*>(dy.data());
        zero_clear = true;
        for (int i = 0; i < n_out; ++i) {
            if (zp[i] != 0.f) zero_clear = false;
            if (got[(size_t) i] != 0.f) ++from_zero;
        }
    }
    char move_detail[80];
    std::snprintf(move_detail, sizeof move_detail, "%d of %d differ, zero %d", moved, n_out, from_zero);
    verdict(move_name, ok && moved > 0 && zero_clear && from_zero == n_out, move_detail);
}

int q3_vsub4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        const int ba = (a >> (8 * i)) & 0xFF;
        const int bb = (b >> (8 * i)) & 0xFF;
        r |= ((ba - bb) & 0xFF) << (8 * i);
    }
    return r;
}

// Q3_K is 110 bytes and 256 values. Same integer dot as iq_mmvq_q3_k.
float q3_partial(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - (iqs % 8) + (iqs % 8) / 4;
    const uint8_t* scales = blk + 96;
    const int vl = q2_ld(blk + 32 + 4 * iqs);
    const uint32_t vh = (~(uint32_t) q2_ld(blk + 4 * (iqs % 8))) >> bq8_offset;
    float sumf = 0;
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((scales[8 + (isc % 4)] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = (int) (((vh >> i) << 2) & 0x04040404u);
        const uint8_t* bq8 = x0 + (size_t) (bq8_offset + i) * 36;
        const float d8 = strata::kernels::f32_from_f16((uint16_t) k_u16(bq8));
        sumf += d8 * (float) (q2_dp4a(q3_vsub4(vil, vih), q2_ld(bq8 + 4 + 4 * (iqs % 8)), 0) * sc);
    }
    return strata::kernels::f32_from_f16((uint16_t) k_u16(blk + 108)) * sumf;
}

float q3_row(const uint8_t* row, const uint8_t* col, int n_in) {
    float lane[32] = {};
    const int npart = (n_in / 256) * 16;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 16;
        lane[k & 31] += q3_partial(row + (size_t) kbx * 110, col + (size_t) kbx * 8 * 36, k % 16);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void q3_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    const size_t row = (size_t) (n_in / 256) * 110;
    for (int r = 0; r < n_out; ++r) y[r] = q3_row(w + (size_t) r * row, x, n_in);
}

void dequant_q3(const uint8_t* block, float* out) {
    const uint8_t* hm = block;
    const uint8_t* qq = block + 32;
    const float d_all = strata::kernels::f32_from_f16((uint16_t) k_u16(block + 108));
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, block + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    float* y = out;
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            const float d0 = d_all * (float) (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = d0 * (float) ((int) ((qq[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            const float d1 = d_all * (float) (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = d1 * (float) ((int) ((qq[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t) (m << 1);
        }
        qq += 32;
    }
}

void q3_scalar(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    std::vector<float> act((size_t) n_in), rowv((size_t) n_in);
    q8_dequant(x, n_in, act.data());
    const size_t bytes = (size_t) (n_in / 256) * 110;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* src = w + (size_t) r * bytes;
        for (int b = 0; b < n_in / 256; ++b) dequant_q3(src + (size_t) b * 110, rowv.data() + (size_t) b * 256);
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += (double) rowv[(size_t) i] * (double) act[(size_t) i];
        y[r] = (float) acc;
    }
}

// Q5_0 is 22 bytes and 32 values. Same integer dot as iq_mmvq_q5_0.
float q50_partial(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 6;
    int sumi = 0;
    int suma = 0;
    for (int i = 0; i < 2; ++i) {
        const int vl = q2_ld(qs + 4 * (iqs + i));
        const uint32_t vh = (uint32_t) q2_ld(blk + 2) >> (4 * (iqs + i));
        const int u0 = q2_ld(x0 + 4 + 4 * (iqs + i));
        const int u1 = q2_ld(x0 + 4 + 4 * (iqs + i + 4));
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (int) ((vh << 4) & 0x00000010u);
        vi0 |= (int) ((vh << 11) & 0x00001000u);
        vi0 |= (int) ((vh << 18) & 0x00100000u);
        vi0 |= (int) ((vh << 25) & 0x10000000u);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (int) ((vh >> 12) & 0x00000010u);
        vi1 |= (int) ((vh >> 5) & 0x00001000u);
        vi1 |= (int) ((vh << 2) & 0x00100000u);
        vi1 |= (int) ((vh << 9) & 0x10000000u);
        sumi = q2_dp4a(vi0, u0, sumi);
        sumi = q2_dp4a(vi1, u1, sumi);
        suma = q2_dp4a(0x01010101, u0, suma);
        suma = q2_dp4a(0x01010101, u1, suma);
    }
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(blk)) *
                    strata::kernels::f32_from_f16((uint16_t) k_u16(x0));
    return d * (float) (sumi - 16 * suma);
}

float q50_row(const uint8_t* row, const uint8_t* col, int n_in) {
    float lane[32] = {};
    const int npart = (n_in / 32) * 2;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 2;
        lane[k & 31] += q50_partial(row + (size_t) kbx * 22, col + (size_t) kbx * 36, 2 * (k % 2));
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void q50_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    const size_t row = (size_t) (n_in / 32) * 22;
    for (int r = 0; r < n_out; ++r) y[r] = q50_row(w + (size_t) r * row, x, n_in);
}

void dequant_q50(const uint8_t* block, float* out) {
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(block));
    const uint32_t h = (uint32_t) q2_ld(block + 2);
    const uint8_t* qs = block + 6;
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int) (qs[j] & 0x0F) | (int) (((h >> j) & 1u) << 4);
        const int x1 = (int) (qs[j] >> 4) | (int) (((h >> (j + 16)) & 1u) << 4);
        out[j] = (float) (x0 - 16) * d;
        out[j + 16] = (float) (x1 - 16) * d;
    }
}

void q50_scalar(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    std::vector<float> act((size_t) n_in), rowv((size_t) n_in);
    q8_dequant(x, n_in, act.data());
    const size_t bytes = (size_t) (n_in / 32) * 22;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* src = w + (size_t) r * bytes;
        for (int b = 0; b < n_in / 32; ++b) dequant_q50(src + (size_t) b * 22, rowv.data() + (size_t) b * 32);
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += (double) rowv[(size_t) i] * (double) act[(size_t) i];
        y[r] = (float) acc;
    }
}

// Q4_0 is 18 bytes and 32 values. Same integer dot as iq_mmvq_q4_0. Bias is 8, and the
// nibbles start at byte 2. There is no high-bit word.
float q40_partial(const uint8_t* blk, const uint8_t* x0, int iqs) {
    const uint8_t* qs = blk + 2;
    int sumi = 0;
    int suma = 0;
    for (int i = 0; i < 2; ++i) {
        const int vl = q2_ld(qs + 4 * (iqs + i));
        const int u0 = q2_ld(x0 + 4 + 4 * (iqs + i));
        const int u1 = q2_ld(x0 + 4 + 4 * (iqs + i + 4));
        const int vi0 = vl & 0x0F0F0F0F;
        const int vi1 = (vl >> 4) & 0x0F0F0F0F;
        sumi = q2_dp4a(vi0, u0, sumi);
        sumi = q2_dp4a(vi1, u1, sumi);
        suma = q2_dp4a(0x01010101, u0, suma);
        suma = q2_dp4a(0x01010101, u1, suma);
    }
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(blk)) *
                    strata::kernels::f32_from_f16((uint16_t) k_u16(x0));
    return d * (float) (sumi - 8 * suma);
}

float q40_row(const uint8_t* row, const uint8_t* col, int n_in) {
    float lane[32] = {};
    const int npart = (n_in / 32) * 2;
    for (int k = 0; k < npart; ++k) {
        const int kbx = k / 2;
        lane[k & 31] += q40_partial(row + (size_t) kbx * 18, col + (size_t) kbx * 36, 2 * (k % 2));
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        std::memcpy(lane, nxt, sizeof lane);
    }
    return lane[0];
}

void q40_ref(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    const size_t row = (size_t) (n_in / 32) * 18;
    for (int r = 0; r < n_out; ++r) y[r] = q40_row(w + (size_t) r * row, x, n_in);
}

void dequant_q40(const uint8_t* block, float* out) {
    const float d = strata::kernels::f32_from_f16((uint16_t) k_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = (float) ((int) (qs[j] & 0x0F) - 8) * d;
        out[j + 16] = (float) ((int) (qs[j] >> 4) - 8) * d;
    }
}

void q40_scalar(const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
    std::vector<float> act((size_t) n_in), rowv((size_t) n_in);
    q8_dequant(x, n_in, act.data());
    const size_t bytes = (size_t) (n_in / 32) * 18;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* src = w + (size_t) r * bytes;
        for (int b = 0; b < n_in / 32; ++b) dequant_q40(src + (size_t) b * 18, rowv.data() + (size_t) b * 32);
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += (double) rowv[(size_t) i] * (double) act[(size_t) i];
        y[r] = (float) acc;
    }
}

using RowRef = void (*)(const uint8_t*, const uint8_t*, int, int, float*);

void check_rows(const char* match_name, const char* move_name, const uint8_t* w, uint64_t nbytes, int n_in, int n_out,
                int ggml, int flip_at, RowRef iref, RowRef sref) {
    size_t row = 0;
    bool layout = false;
    if (w && n_in > 0 && n_out > 0) {
        row = iq_row_bytes(ggml, n_in);
        layout = nbytes == (uint64_t) n_out * row;
    }
    std::vector<float> x((size_t) n_in);
    for (int i = 0; i < n_in; ++i) x[(size_t) i] = ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes xf((uint64_t) n_in * 4);
    if (n_in > 0) std::memcpy(xf.data(), x.data(), xf.size());
    Bytes xq(n_in > 0 ? iq_q8_1_bytes(n_in) : 0);
    if (n_in > 0) iq_quantize_q8_1(xf, n_in, xq);
    std::vector<uint8_t> xqb((size_t) xq.size());
    if (!xqb.empty()) std::memcpy(xqb.data(), xq.data(), xqb.size());
    std::vector<float> got((size_t) n_out), host((size_t) n_out), scalar((size_t) n_out);
    double rel = 1, irel = 1;
    bool ok = false;
    if (layout) {
        Bytes dw(nbytes);
        std::memcpy(dw.data(), w, nbytes);
        Bytes dy((uint64_t) n_out * 4);
        iq_mmvq(ggml, dw, 0, xq, n_in, n_out, 1, dy);
        std::memcpy(got.data(), dy.data(), got.size() * 4);
        iref(w, xqb.data(), n_in, n_out, host.data());
        irel = rel_l1(host.data(), got.data(), n_out);
        if (sref) {
            sref(w, xqb.data(), n_in, n_out, scalar.data());
            rel = rel_l1(scalar.data(), got.data(), n_out);
        }
        ok = all_finite(got.data(), n_out) && irel <= 1e-5 && (!sref || rel <= 1e-5);
    }
    char detail[120];
    if (sref) std::snprintf(detail, sizeof detail, "rel %.3e, integer %.3e, %d rows", rel, irel, n_out);
    else std::snprintf(detail, sizeof detail, "rel %.3e, %d rows", irel, n_out);
    verdict(match_name, layout && ok, layout ? detail : "bytes");

    int moved = 0;
    bool zero_clear = false;
    int from_zero = 0;
    if (ok) {
        Bytes dw(nbytes);
        std::memcpy(dw.data(), w, nbytes);
        static_cast<uint8_t*>(dw.data())[flip_at] ^= 1u;
        Bytes dy((uint64_t) n_out * 4);
        iq_mmvq(ggml, dw, 0, xq, n_in, n_out, 1, dy);
        const float* yp = static_cast<const float*>(dy.data());
        for (int i = 0; i < n_out; ++i)
            if (yp[i] != got[(size_t) i]) ++moved;
        Bytes zx(xq.size());
        std::memset(zx.data(), 0, zx.size());
        iq_mmvq(ggml, dw, 0, zx, n_in, n_out, 1, dy);
        const float* zp = static_cast<const float*>(dy.data());
        zero_clear = true;
        for (int i = 0; i < n_out; ++i) {
            if (zp[i] != 0.f) zero_clear = false;
            if (got[(size_t) i] != 0.f) ++from_zero;
        }
    }
    char move_detail[80];
    std::snprintf(move_detail, sizeof move_detail, "%d of %d differ, zero %d", moved, n_out, from_zero);
    verdict(move_name, ok && moved > 0 && zero_clear && from_zero == n_out, move_detail);
}

void run_kquant() {
    std::printf("\nQ4_K and Q5_K ggml blocks\n");
    std::mt19937 rng(20261014);
    const int n_in = 512;
    const int n_out = 8;
    std::vector<uint8_t> q4((size_t) n_out * (n_in / 256) * 144);
    std::vector<uint8_t> q5((size_t) n_out * (n_in / 256) * 176);
    for (uint8_t& b : q4) b = (uint8_t) rng();
    for (uint8_t& b : q5) b = (uint8_t) rng();
    auto finite_scales = [](std::vector<uint8_t>& raw, int block) {
        for (size_t off = 0; off < raw.size(); off += (size_t) block) {
            raw[off + 0] = 0x00;
            raw[off + 1] = 0x3C;
            raw[off + 2] = 0x00;
            raw[off + 3] = 0x3C;
            for (int s = 0; s < 12; ++s) raw[off + 4 + s] = 0x3F;
        }
    };
    finite_scales(q4, 144);
    finite_scales(q5, 176);
    check_k("a Q4_K block matches the dequant", "a changed Q4_K code moves that block", q4.data(), q4.size(), n_in,
            n_out, false, 16);
    check_k("a Q5_K block matches the dequant", "a changed Q5_K code moves that block", q5.data(), q5.size(), n_in,
            n_out, true, 48);

    std::printf("\nQ3_K, Q5_0, and Q4_0 ggml blocks\n");
    const int q3_in = 512;
    const int q3_out = 8;
    const int q50_in = 128;
    const int q50_out = 8;
    std::vector<uint8_t> q3((size_t) q3_out * (q3_in / 256) * 110);
    std::vector<uint8_t> q50((size_t) q50_out * (q50_in / 32) * 22);
    for (uint8_t& b : q3) b = (uint8_t) rng();
    for (uint8_t& b : q50) b = (uint8_t) rng();
    for (size_t off = 0; off < q3.size(); off += 110) {
        q3[off + 108] = 0x00;
        q3[off + 109] = 0x3C;
    }
    for (size_t off = 0; off < q50.size(); off += 22) {
        q50[off + 0] = 0x00;
        q50[off + 1] = 0x3C;
    }
    check_rows("a Q3_K block matches the dequant", "a changed Q3_K code moves that block", q3.data(), q3.size(), q3_in,
               q3_out, kQ3_K, 32, q3_ref, q3_scalar);
    check_rows("a Q5_0 block matches the dequant", "a changed Q5_0 code moves that block", q50.data(), q50.size(),
               q50_in, q50_out, kQ5_0, 6, q50_ref, q50_scalar);

    const int q40_in = 128;
    const int q40_out = 8;
    std::vector<uint8_t> q40((size_t) q40_out * (q40_in / 32) * 18);
    for (uint8_t& b : q40) b = (uint8_t) rng();
    for (size_t off = 0; off < q40.size(); off += 18) {
        q40[off + 0] = 0x00;
        q40[off + 1] = 0x3C;
    }
    check_rows("a Q4_0 block matches the dequant", "a changed Q4_0 code moves that block", q40.data(), q40.size(),
               q40_in, q40_out, kQ4_0, 2, q40_ref, q40_scalar);
    std::string q40got;
    const bool q40_refused =
        threw("not a Coder expert type", [&] { iq_expert_bytes(IqExpert{kIq4Xs, kQ4_0, 2560, 640}); }, q40got);
    verdict("a Q4_0 down is refused", q40_refused, q40got);
}

// One GDN step from a zero state. The projections are the ggml dots. The middle is the
// same transcription as the S-form layer: conv, SiLU, L2, 1/sqrt(S) on q, sigmoid beta,
// softplus gate, head h % h_k, and sigmoid on z.
float gdn_silu(float v) { return v / (1.0f + std::exp(-v)); }

float gdn_sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

float gdn_softplus(float x) {
    if (x > 20.0f) return x;
    const float y = std::exp(x);
    const float u = 1.0f + y;
    if (u == 1.0f) return y;
    return std::log(u) * (y / (u - 1.0f));
}

void gdn_l2_rows(std::vector<float>& x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0;
        float* p = x.data() + (size_t) r * cols;
        for (int i = 0; i < cols; ++i) acc += (double) p[i] * (double) p[i];
        const float inv = (float) (1.0 / std::sqrt(acc + 1e-6));
        for (int i = 0; i < cols; ++i) p[i] *= inv;
    }
}

void gdn_bf16_gemv(const float* x, const uint16_t* w, int n_in, int n_out, float* y) {
    std::vector<uint16_t> xb((size_t) n_in);
    for (int i = 0; i < n_in; ++i) xb[(size_t) i] = strata::kernels::bf16_from_f32(x[i]);
    for (int row = 0; row < n_out; ++row) {
        double acc = 0;
        const uint16_t* wr = w + (size_t) row * n_in;
        for (int i = 0; i < n_in; ++i)
            acc += (double) strata::kernels::f32_from_bf16(xb[(size_t) i]) *
                   (double) strata::kernels::f32_from_bf16(wr[i]);
        y[row] = (float) acc;
    }
}

std::vector<float> gdn_iq_host(const float* x, const uint8_t* qkv_w, const uint8_t* gate_w, const uint8_t* out_w,
                               const uint16_t* alpha_w, const uint16_t* beta_w, const float* conv_w,
                               const float* norm, const float* dt, const float* ssm_a, int n_embd, int S, int h_k,
                               int h_v, int d_conv, std::vector<float>* conv_io = nullptr,
                               std::vector<float>* state_io = nullptr, int gate_type = kQ4_K,
                               int out_type = kQ5_K) {
    const int qk = S * h_k;
    const int V = S * h_v;
    const int C = qk + qk + V;
    Bytes xf((uint64_t) n_embd * 4);
    std::memcpy(xf.data(), x, xf.size());
    Bytes xq(iq_q8_1_bytes(n_embd));
    iq_quantize_q8_1(xf, n_embd, xq);
    std::vector<uint8_t> xqb((size_t) xq.size());
    std::memcpy(xqb.data(), xq.data(), xqb.size());

    std::vector<float> qkv((size_t) C);
    xs_ref(qkv_w, xqb.data(), n_embd, C, qkv.data());
    std::vector<float> conv_state((size_t) C * (d_conv - 1), 0.f);
    if (conv_io) {
        if (conv_io->size() != conv_state.size()) throw MetalError("gdn conv state has the wrong length");
        conv_state = *conv_io;
    }
    std::vector<float> mixed((size_t) C, 0.f);
    for (int c = 0; c < C; ++c) {
        float* st = conv_state.data() + (size_t) c * (d_conv - 1);
        const float* cw = conv_w + (size_t) c * d_conv;
        double acc = 0;
        for (int i = 0; i < d_conv - 1; ++i) acc += (double) st[i] * (double) cw[i];
        acc += (double) qkv[(size_t) c] * (double) cw[d_conv - 1];
        mixed[(size_t) c] = (float) acc;
        for (int i = 0; i < d_conv - 2; ++i) st[i] = st[i + 1];
        st[d_conv - 2] = qkv[(size_t) c];
    }
    for (float& v : mixed) v = gdn_silu(v);

    std::vector<float> q(mixed.begin(), mixed.begin() + qk);
    std::vector<float> k(mixed.begin() + qk, mixed.begin() + 2 * qk);
    std::vector<float> vv(mixed.begin() + 2 * qk, mixed.end());
    gdn_l2_rows(q, h_k, S);
    gdn_l2_rows(k, h_k, S);
    const float qs = 1.0f / std::sqrt((float) S);
    for (float& v : q) v *= qs;

    std::vector<float> alpha((size_t) h_v), beta((size_t) h_v), gate((size_t) h_v);
    gdn_bf16_gemv(x, alpha_w, n_embd, h_v, alpha.data());
    gdn_bf16_gemv(x, beta_w, n_embd, h_v, beta.data());
    for (float& v : beta) v = gdn_sigmoid(v);
    for (int h = 0; h < h_v; ++h) gate[(size_t) h] = gdn_softplus(alpha[(size_t) h] + dt[h]) * ssm_a[h];

    std::vector<double> state((size_t) S * h_v * S, 0.0);
    if (state_io) {
        if (state_io->size() != state.size()) throw MetalError("gdn state has the wrong length");
        for (size_t i = 0; i < state.size(); ++i) state[i] = (double) (*state_io)[i];
    }
    std::vector<float> o((size_t) V, 0.f);
    for (int h = 0; h < h_v; ++h) {
        const int src = h % h_k;
        const double dec = std::exp((double) gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            auto at = [&](int i) -> double& { return state[(size_t) (i * h_v + h) * S + j]; };
            for (int i = 0; i < S; ++i) at(i) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i) * (double) k[(size_t) src * S + i];
            const double delta = ((double) vv[(size_t) h * S + j] - sk) * b;
            double dot = 0;
            for (int i = 0; i < S; ++i) {
                at(i) += (double) k[(size_t) src * S + i] * delta;
                dot += at(i) * (double) q[(size_t) src * S + i];
            }
            o[(size_t) h * S + j] = (float) dot;
        }
    }

    auto project_rows = [&](int type, const uint8_t* w, const uint8_t* x, int n_in, int n_out, float* y) {
        if (type == kIq4Xs) xs_ref(w, x, n_in, n_out, y);
        else if (type == kQ4_K) scalar_ref(w, x, n_in, n_out, false, y);
        else if (type == kQ5_K) scalar_ref(w, x, n_in, n_out, true, y);
        else throw MetalError("gdn host quant is not one of these");
    };
    std::vector<float> z((size_t) V);
    project_rows(gate_type, gate_w, xqb.data(), n_embd, V, z.data());
    std::vector<float> y((size_t) V, 0.f);
    for (int h = 0; h < h_v; ++h) {
        const float* po = o.data() + (size_t) h * S;
        double acc = 0;
        for (int i = 0; i < S; ++i) acc += (double) po[i] * (double) po[i];
        const float inv = (float) (1.0 / std::sqrt(acc / (double) S + 1e-6));
        for (int i = 0; i < S; ++i)
            y[(size_t) h * S + i] = po[i] * inv * norm[i] * gdn_sigmoid(z[(size_t) h * S + i]);
    }
    Bytes yf((uint64_t) V * 4);
    std::memcpy(yf.data(), y.data(), yf.size());
    Bytes yq(iq_q8_1_bytes(V));
    iq_quantize_q8_1(yf, V, yq);
    std::vector<uint8_t> yqb((size_t) yq.size());
    std::memcpy(yqb.data(), yq.data(), yqb.size());
    std::vector<float> out((size_t) n_embd);
    project_rows(out_type, out_w, yqb.data(), V, n_embd, out.data());
    if (conv_io) *conv_io = conv_state;
    if (state_io) {
        for (size_t i = 0; i < state.size(); ++i) (*state_io)[i] = (float) state[i];
    }
    return out;
}

float round_bf16(float f) { return strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(f)); }

float bf16_dot(const float* x, const uint16_t* row, int n) {
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += (double) x[i] * (double) strata::kernels::f32_from_bf16(row[i]);
    return (float) acc;
}

// The attention half of one layer-0 token. gr_read, the ggml GDN step, gr_write.
// Activations round to bf16. The norm weight is used as stored. S-form planes are not read.
struct AttnHalf {
    std::vector<float> mixed;
    std::vector<float> block;
    std::vector<float> out;
};

AttnHalf layer0_attn_host(const float* R, const float* norm_w, const uint16_t* down_w, const uint16_t* up_w,
                          const uint16_t* inject_w, const uint8_t* qkv_w, const uint8_t* gate_w, const uint8_t* out_w,
                          const uint16_t* alpha_w, const uint16_t* beta_w, const float* conv_w, const float* ssm_norm,
                          const float* dt, const float* ssm_a, int n_embd, int hc, int hc_lr, int S, int h_k, int h_v,
                          int d_conv) {
    const int hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    for (int c = 0; c < hc; ++c) {
        double ms = 0;
        for (int d = 0; d < n_embd; ++d) {
            const double v = R[(size_t) c * n_embd + d];
            ms += v * v;
        }
        ms /= (double) n_embd;
        const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
        for (int d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) c * n_embd + d;
            xn[i] = R[i] * rs * norm_w[i];
        }
    }
    std::vector<float> act((size_t) hc_dim);
    for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);

    std::vector<float> lo((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) {
        const float p = bf16_dot(act.data(), down_w + (size_t) k * hc_dim, hc_dim);
        lo[(size_t) k] = gdn_silu(p / (float) hc);
    }
    std::vector<float> lq((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) lq[(size_t) k] = round_bf16(lo[(size_t) k]);

    std::vector<float> gated((size_t) hc_dim);
    AttnHalf half;
    half.mixed.assign((size_t) n_embd, 0.f);
    for (int d = 0; d < n_embd; ++d) {
        float m = 0.f;
        for (int c = 0; c < hc; ++c) {
            const int i = c * n_embd + d;
            const float g = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
            gated[(size_t) i] = xn[(size_t) i] * gdn_sigmoid(g);
            m += gated[(size_t) i];
        }
        half.mixed[(size_t) d] = m / (float) hc;
    }
    std::vector<float> inject((size_t) hc);
    for (int c = 0; c < hc; ++c) inject[(size_t) c] = bf16_dot(act.data(), inject_w + (size_t) c * hc_dim, hc_dim);

    half.block = gdn_iq_host(half.mixed.data(), qkv_w, gate_w, out_w, alpha_w, beta_w, conv_w, ssm_norm, dt, ssm_a,
                             n_embd, S, h_k, h_v, d_conv);
    half.out.assign((size_t) hc_dim, 0.f);
    for (int c = 0; c < hc; ++c) {
        const float scale = 2.0f * gdn_sigmoid(inject[(size_t) c] / (float) hc);
        for (int d = 0; d < n_embd; ++d)
            half.out[(size_t) c * n_embd + d] = R[(size_t) c * n_embd + d] + half.block[(size_t) d] * scale;
    }
    return half;
}

// The feed-forward half of one layer-0 token. gr_read, the router, expert 0, gr_write.
// The block written back is expert 0. The router logits are not a scale on that block.
// The shared-expert rows are not read. S-form planes are not read.
struct FfnHalf {
    std::vector<float> mixed;
    std::vector<float> router;
    std::vector<float> block;
    std::vector<float> out;
};

FfnHalf layer0_ffn_host(const float* R, const float* norm_w, const uint16_t* down_w, const uint16_t* up_w,
                        const uint16_t* inject_w, const uint16_t* router_w, const uint8_t* expert_w, int n_embd,
                        int n_ff, int n_expert, int hc, int hc_lr) {
    const int hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    for (int c = 0; c < hc; ++c) {
        double ms = 0;
        for (int d = 0; d < n_embd; ++d) {
            const double v = R[(size_t) c * n_embd + d];
            ms += v * v;
        }
        ms /= (double) n_embd;
        const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
        for (int d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) c * n_embd + d;
            xn[i] = R[i] * rs * norm_w[i];
        }
    }
    std::vector<float> act((size_t) hc_dim);
    for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);

    std::vector<float> lo((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) {
        const float p = bf16_dot(act.data(), down_w + (size_t) k * hc_dim, hc_dim);
        lo[(size_t) k] = gdn_silu(p / (float) hc);
    }
    std::vector<float> lq((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) lq[(size_t) k] = round_bf16(lo[(size_t) k]);

    FfnHalf half;
    half.mixed.assign((size_t) n_embd, 0.f);
    for (int d = 0; d < n_embd; ++d) {
        float m = 0.f;
        for (int c = 0; c < hc; ++c) {
            const int i = c * n_embd + d;
            const float g = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
            m += xn[(size_t) i] * gdn_sigmoid(g);
        }
        half.mixed[(size_t) d] = m / (float) hc;
    }
    std::vector<float> inject((size_t) hc);
    for (int c = 0; c < hc; ++c) inject[(size_t) c] = bf16_dot(act.data(), inject_w + (size_t) c * hc_dim, hc_dim);

    std::vector<float> rounded((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(half.mixed[(size_t) i]);
    half.router.resize((size_t) n_expert);
    for (int o = 0; o < n_expert; ++o)
        half.router[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);

    Bytes xf((uint64_t) n_embd * 4);
    std::memcpy(xf.data(), half.mixed.data(), xf.size());
    Bytes xq(iq_q8_1_bytes(n_embd));
    iq_quantize_q8_1(xf, n_embd, xq);
    std::vector<uint8_t> xqb((size_t) xq.size());
    std::memcpy(xqb.data(), xq.data(), xqb.size());
    const size_t gu = iq_row_bytes(kQ2_0, n_embd);
    std::vector<float> gate((size_t) n_ff), up((size_t) n_ff), hidden((size_t) n_ff);
    q2_ref(expert_w, xqb.data(), n_embd, n_ff, gate.data());
    q2_ref(expert_w + (size_t) n_ff * gu, xqb.data(), n_embd, n_ff, up.data());
    for (int i = 0; i < n_ff; ++i) hidden[(size_t) i] = gdn_silu(gate[(size_t) i]) * up[(size_t) i];
    Bytes hf((uint64_t) n_ff * 4);
    std::memcpy(hf.data(), hidden.data(), hf.size());
    Bytes hq(iq_q8_1_bytes(n_ff));
    iq_quantize_q8_1(hf, n_ff, hq);
    std::vector<uint8_t> hqb((size_t) hq.size());
    std::memcpy(hqb.data(), hq.data(), hqb.size());
    half.block.resize((size_t) n_embd);
    q2_ref(expert_w + (size_t) n_ff * gu * 2, hqb.data(), n_ff, n_embd, half.block.data());

    half.out.assign((size_t) hc_dim, 0.f);
    for (int c = 0; c < hc; ++c) {
        const float scale = 2.0f * gdn_sigmoid(inject[(size_t) c] / (float) hc);
        for (int d = 0; d < n_embd; ++d)
            half.out[(size_t) c * n_embd + d] = R[(size_t) c * n_embd + d] + half.block[(size_t) d] * scale;
    }
    return half;
}

// Layer 0 of the Q2_0 extract. The other 47 layers are not in the directory.
// The loader does not open the GGUF. A missing directory skips this section.
// Gate is IQ4_XS, up is Q3_K, down is Q5_0, and the scalar is one bf16 row.
// The layer-0 pass does not read these rows. The S-form runner still refuses code_bits 0.
void run_shared_rows(const PackTable& table, const strata::core::ModelGeometry& g) {
    std::printf("\nshared-expert rows from the extracted pack\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* gate = plane("blk.0.ffn_gate_shexp.weight", 870400, n_embd, n_ff);
    const PackTensor* up = plane("blk.0.ffn_up_shexp.weight", 704000, n_embd, n_ff);
    const PackTensor* down = plane("blk.0.ffn_down_shexp.weight", 1126400, n_ff, n_embd);
    const PackTensor* ginp = table.find("blk.0.ffn_gate_inp_shexp.weight");
    const bool ginp_ok = ginp && ginp->values && ginp->kind == strata::core::WeightKind::Bf16InF32 &&
                         ginp->code_bits == 0 && ginp->ne0 == n_embd && ginp->ne1 == 1 &&
                         ginp->values->size() == (uint64_t) n_embd * 2;
    check_rows("the IQ4_XS shared gate rows match", "a changed IQ4_XS code moves the shared gate",
               gate ? static_cast<const uint8_t*>(gate->values->data()) : nullptr, gate ? gate->values->size() : 0,
               n_embd, n_ff, kIq4Xs, 8, xs_ref, nullptr);
    check_rows("the Q3_K shared up rows match", "a changed Q3_K code moves the shared up",
               up ? static_cast<const uint8_t*>(up->values->data()) : nullptr, up ? up->values->size() : 0, n_embd,
               n_ff, kQ3_K, 32, q3_ref, q3_scalar);
    check_rows("the Q5_0 shared down rows match", "a changed Q5_0 code moves the shared down",
               down ? static_cast<const uint8_t*>(down->values->data()) : nullptr, down ? down->values->size() : 0,
               n_ff, n_embd, kQ5_0, 6, q50_ref, q50_scalar);

    Bytes x((uint64_t) n_embd * 4);
    auto* xp = static_cast<float*>(x.data());
    for (int i = 0; i < n_embd; ++i) xp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
    std::string sgot = "bytes";
    bool srefused = false;
    bool sheld = false;
    if (gate && up && down && ginp_ok) {
        Bytes out((uint64_t) (n_embd + 1) * 4);
        std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + 1), 7.f);
        srefused = threw("S2/S4/S8", [&] { run_shared_expert(table, g, 0, x, out); }, sgot);
        sheld = still_seven(static_cast<const float*>(out.data()), n_embd + 1);
    }
    verdict("the S-form shared expert refuses these rows", srefused && sheld, sgot);

    std::string egot;
    const bool erefused = threw(
        "not a Coder expert type", [&] { iq_expert_bytes(IqExpert{kQ3_K, kQ5_0, n_embd, n_ff}); }, egot);
    verdict("a Q3_K gate is refused", erefused, egot);

    const bool rows = gate && up && down && ginp_ok;
    bool ok = false;
    bool repeat = false;
    bool cleared = false;
    bool flipped = false;
    std::string detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string zero_detail = "bytes";
    std::string flip_detail = "bytes";
    if (rows) {
        try {
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            const int guard = 1;
            Bytes gate_y((uint64_t) n_ff * 4), up_y((uint64_t) n_ff * 4), down_y((uint64_t) n_embd * 4);
            Bytes hq(iq_q8_1_bytes(n_ff)), scalar(4);
            Bytes y((uint64_t) (n_embd + guard) * 4), y2((uint64_t) (n_embd + guard) * 4);
            Bytes yz((uint64_t) (n_embd + guard) * 4), yf((uint64_t) (n_embd + guard) * 4);
            auto forward = [&](const Bytes& act, const Bytes& down_w, Bytes& out) {
                Bytes xq(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, xq);
                iq_mmvq(kIq4Xs, *gate->values, 0, xq, n_embd, n_ff, 1, gate_y);
                iq_mmvq(kQ3_K, *up->values, 0, xq, n_embd, n_ff, 1, up_y);
                iq_swiglu_q8(gate_y, up_y, n_ff, hq);
                iq_mmvq(kQ5_0, down_w, 0, hq, n_ff, n_embd, 1, down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(act, x_bf16, n_embd);
                bf16_gemv(x_bf16, *ginp->values, scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* sc[] = {&out, &scalar};
                launch("se_sigmoid_scale", sc, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            forward(x, *down->values, y);
            std::vector<float> got_gate((size_t) n_ff), got_up((size_t) n_ff), got_down((size_t) n_embd);
            std::vector<float> got_y((size_t) n_embd);
            std::vector<uint8_t> got_hq((size_t) hq.size());
            std::memcpy(got_gate.data(), gate_y.data(), got_gate.size() * 4);
            std::memcpy(got_up.data(), up_y.data(), got_up.size() * 4);
            std::memcpy(got_down.data(), down_y.data(), got_down.size() * 4);
            std::memcpy(got_y.data(), y.data(), got_y.size() * 4);
            std::memcpy(got_hq.data(), hq.data(), got_hq.size());
            const float got_s = static_cast<const float*>(scalar.data())[0];

            Bytes xq(iq_q8_1_bytes(n_embd));
            iq_quantize_q8_1(x, n_embd, xq);
            std::vector<uint8_t> xqb((size_t) xq.size());
            std::memcpy(xqb.data(), xq.data(), xqb.size());
            const uint8_t* gate_b = static_cast<const uint8_t*>(gate->values->data());
            const uint8_t* up_b = static_cast<const uint8_t*>(up->values->data());
            const uint8_t* down_b = static_cast<const uint8_t*>(down->values->data());
            std::vector<float> host_gate((size_t) n_ff), host_up((size_t) n_ff), host_down((size_t) n_embd);
            xs_ref(gate_b, xqb.data(), n_embd, n_ff, host_gate.data());
            q3_ref(up_b, xqb.data(), n_embd, n_ff, host_up.data());
            q50_ref(down_b, got_hq.data(), n_ff, n_embd, host_down.data());
            std::vector<float> rounded((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(xp[i]);
            const float host_s = bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
            const float sig = gdn_sigmoid(host_s);
            std::vector<float> host_y((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) host_y[(size_t) i] = host_down[(size_t) i] * sig;

            std::vector<float> hidden((size_t) n_ff);
            for (int i = 0; i < n_ff; ++i) hidden[(size_t) i] = gdn_silu(host_gate[(size_t) i]) * host_up[(size_t) i];
            Bytes hf((uint64_t) n_ff * 4);
            std::memcpy(hf.data(), hidden.data(), hf.size());
            Bytes hqq(iq_q8_1_bytes(n_ff));
            iq_quantize_q8_1(hf, n_ff, hqq);
            std::vector<uint8_t> hqb((size_t) hqq.size());
            std::memcpy(hqb.data(), hqq.data(), hqb.size());
            std::vector<float> chain_down((size_t) n_embd), chain((size_t) n_embd);
            q50_ref(down_b, hqb.data(), n_ff, n_embd, chain_down.data());
            for (int i = 0; i < n_embd; ++i) chain[(size_t) i] = chain_down[(size_t) i] * sig;

            const double gate_rel = rel_l1(host_gate.data(), got_gate.data(), n_ff);
            const double up_rel = rel_l1(host_up.data(), got_up.data(), n_ff);
            const double down_rel = rel_l1(host_down.data(), got_down.data(), n_embd);
            const double scale_rel = rel_l1(&host_s, &got_s, 1);
            const double write_rel = rel_l1(host_y.data(), got_y.data(), n_embd);
            const double chain_rel = rel_l1(chain.data(), got_y.data(), n_embd);

            forward(x, *down->values, y2);
            Bytes zero_x((uint64_t) n_embd * 4);
            std::memset(zero_x.data(), 0, (size_t) zero_x.size());
            forward(zero_x, *down->values, yz);

            int flip_at = 6;
            Bytes down_flip(down->values->size());
            std::memcpy(down_flip.data(), down->values->data(), down_flip.size());
            static_cast<uint8_t*>(down_flip.data())[flip_at] ^= 1u;
            forward(x, down_flip, yf);
            int flipped_n = differ_floats(got_y.data(), static_cast<const float*>(yf.data()), n_embd);
            if (flipped_n == 0) {
                const size_t drow = iq_row_bytes(kQ5_0, n_ff);
                std::vector<uint8_t> one(drow);
                bool found = false;
                for (int r = 0; r < n_embd && !found; ++r) {
                    const uint8_t* src = down_b + (size_t) r * drow;
                    const float base = q50_row(src, got_hq.data(), n_ff);
                    for (int b = 0; b < (int) drow; ++b) {
                        if (b % 22 < 6) continue;
                        std::memcpy(one.data(), src, drow);
                        one[(size_t) b] ^= 1u;
                        if (q50_row(one.data(), got_hq.data(), n_ff) != base) {
                            flip_at = r * (int) drow + b;
                            found = true;
                            break;
                        }
                    }
                }
                if (found) {
                    std::memcpy(down_flip.data(), down->values->data(), down_flip.size());
                    static_cast<uint8_t*>(down_flip.data())[flip_at] ^= 1u;
                    forward(x, down_flip, yf);
                    flipped_n = differ_floats(got_y.data(), static_cast<const float*>(yf.data()), n_embd);
                }
            }

            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = static_cast<const float*>(y.data())[n_embd] == 7.f && y2p[n_embd] == 7.f &&
                                    yzp[n_embd] == 7.f && yfp[n_embd] == 7.f;
            int zeros = 0;
            for (int i = 0; i < n_embd; ++i)
                if (yzp[i] == 0.f) ++zeros;
            const int again = differ_floats(got_y.data(), y2p, n_embd);
            const int apart = differ_floats(got_y.data(), yzp, n_embd);
            const double apart_rel = rel_l1(got_y.data(), yzp, n_embd);
            const bool live = all_finite(got_gate.data(), n_ff) && all_finite(got_up.data(), n_ff) &&
                              all_finite(got_down.data(), n_embd) && all_finite(got_y.data(), n_embd) &&
                              std::isfinite(got_s) && std::isfinite(host_s) && all_finite(host_y.data(), n_embd) &&
                              all_finite(chain.data(), n_embd) && all_finite(y2p, n_embd) && all_finite(yzp, n_embd) &&
                              all_finite(yfp, n_embd) && !still_seven(got_y.data(), n_embd);
            ok = live && guard_held && gate_rel <= 1e-5 && up_rel <= 1e-5 && down_rel <= 1e-5 && scale_rel <= 1e-4 &&
                 write_rel <= 1e-4;
            repeat = live && guard_held && again == 0;
            cleared = live && guard_held && zeros == n_embd && apart > 0 && std::isfinite(apart_rel);
            flipped = live && guard_held && flipped_n > 0;
            char buf[240];
            std::snprintf(buf, sizeof buf,
                          "gate %.3e, up %.3e, down %.3e, scale %.3e, write %.3e, chain %.3e", gate_rel, up_rel,
                          down_rel, scale_rel, write_rel, chain_rel);
            detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d are zero, %d differ, %.2f%% apart", zeros, n_embd, apart,
                          apart_rel * 100.0);
            zero_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_embd, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            detail = e.what();
            repeat_detail = e.what();
            zero_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the shared expert matches", rows && ok, detail);
    verdict("that shared expert repeats", rows && repeat, repeat_detail);
    verdict("a zero activation clears the shared expert", rows && cleared, zero_detail);
    verdict("a changed down code moves the shared expert", rows && flipped, flip_detail);
}

// The vocabulary rows live in embd.bin. The embedding is Q3_K and the output weight is Q5_K.
// The layer-0 pass does not read them. gather_token_embd still refuses code_bits 0.
// Comparing every row is the product. This is not a token.
void run_vocab_rows(const PackTable& table, const strata::core::ModelGeometry& g) {
    std::printf("\ntoken embedding and output weight from the extracted pack\n");
    const int n_embd = (int) g.n_embd;
    const int n_vocab = 248320;
    auto plane = [&](const char* name, uint64_t nbytes) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != n_embd || t->ne1 != n_vocab ||
            t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* emb = plane("token_embd.weight", 273152000);
    const PackTensor* outw = plane("output.weight", 437043200);
    check_rows("the token embedding rows match", "a changed embedding code moves one row",
               emb ? static_cast<const uint8_t*>(emb->values->data()) : nullptr, emb ? emb->values->size() : 0,
               n_embd, n_vocab, kQ3_K, 32, q3_ref, q3_scalar);
    check_k("the output weight rows match", "a changed output-weight code moves one row",
            outw ? static_cast<const uint8_t*>(outw->values->data()) : nullptr, outw ? outw->values->size() : 0,
            n_embd, n_vocab, true, 48);

    std::string got = "bytes";
    bool refused = false;
    bool held = false;
    if (emb) {
        Bytes y((uint64_t) (n_embd + 1) * 4);
        std::fill_n(static_cast<float*>(y.data()), (size_t) (n_embd + 1), 7.f);
        refused = threw("S2/S4/S8", [&] { gather_token_embd(table, g, 0, y); }, got);
        held = still_seven(static_cast<const float*>(y.data()), n_embd + 1);
    }
    verdict("the embedding gather refuses these rows", refused && held, got);
}

// The output mixer is the residual read with no inject weight. The norm is used as stored.
// Same read as the layer-0 residual. The layer-0 pass does not read these rows.
// live_i is the down-row element with the largest rounded activation.
std::vector<float> output_mix_host(const float* R, const float* norm_w, const uint16_t* down_w, const uint16_t* up_w,
                                   int n_embd, int hc, int hc_lr, int* live_i) {
    const int hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    for (int c = 0; c < hc; ++c) {
        double ms = 0;
        for (int d = 0; d < n_embd; ++d) {
            const double v = R[(size_t) c * n_embd + d];
            ms += v * v;
        }
        ms /= (double) n_embd;
        const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
        for (int d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) c * n_embd + d;
            xn[i] = R[i] * rs * norm_w[i];
        }
    }
    std::vector<float> act((size_t) hc_dim);
    int best_i = 0;
    float best = 0.f;
    for (int i = 0; i < hc_dim; ++i) {
        act[(size_t) i] = round_bf16(xn[(size_t) i]);
        const float a = std::fabs(act[(size_t) i]);
        if (a > best) {
            best = a;
            best_i = i;
        }
    }
    if (live_i) *live_i = best_i;
    std::vector<float> lo((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) {
        const float p = bf16_dot(act.data(), down_w + (size_t) k * hc_dim, hc_dim);
        lo[(size_t) k] = gdn_silu(p / (float) hc);
    }
    std::vector<float> lq((size_t) hc_lr);
    for (int k = 0; k < hc_lr; ++k) lq[(size_t) k] = round_bf16(lo[(size_t) k]);
    std::vector<float> mixed((size_t) n_embd, 0.f);
    for (int d = 0; d < n_embd; ++d) {
        float m = 0.f;
        for (int c = 0; c < hc; ++c) {
            const int i = c * n_embd + d;
            const float g = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
            m += xn[(size_t) i] * gdn_sigmoid(g);
        }
        mixed[(size_t) d] = m / (float) hc;
    }
    return mixed;
}

// gr_read of output_hc_down, output_hc_norm, and output_hc_up. No inject, no residual write,
// no output projection. This is not added into the layer-0 residual and it does not draw a token.
void run_output_mix(const PackTable& table, const strata::core::ModelGeometry& g) {
    std::printf("\noutput mixer from the extracted pack\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const PackTensor* down = table.find("output_hc_down.weight");
    const PackTensor* up = table.find("output_hc_up.weight");
    const PackTensor* norm = table.find("output_hc_norm.weight");
    const bool bytes =
        down && down->values && down->kind == strata::core::WeightKind::Bf16InF32 && down->code_bits == 0 &&
        down->ne0 == hc_dim && down->ne1 == hc_lr && down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 &&
        up && up->values && up->kind == strata::core::WeightKind::Bf16InF32 && up->code_bits == 0 &&
        up->ne0 == hc_lr && up->ne1 == hc_dim && up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 &&
        norm && norm->values && norm->kind == strata::core::WeightKind::F32 && norm->code_bits == 0 &&
        norm->ne0 == hc_dim && norm->ne1 == 1 && norm->values->size() == (uint64_t) hc_dim * 4;
    bool ok = false;
    bool repeat = false;
    bool cleared = false;
    bool flipped = false;
    std::string detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string zero_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int i = 0; i < hc_dim; ++i) Rp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes y((uint64_t) (n_embd + guard) * 4), y2((uint64_t) (n_embd + guard) * 4);
            Bytes yz((uint64_t) (n_embd + guard) * 4), yf((uint64_t) (n_embd + guard) * 4);
            Bytes inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(y);
            fill7(y2);
            fill7(yz);
            fill7(yf);
            fill7(inject);
            GrScratch gr = make_gr_scratch(gs);
            bool inject_held = true;
            auto read = [&](const Bytes& residual, const Bytes& down_w, Bytes& out) {
                std::fill_n(static_cast<float*>(inject.data()), (size_t) (hc + guard), 7.f);
                gr_read(gs, residual, *norm->values, down_w, *up->values, nullptr, 1e-6f, false, gr, out, inject);
                if (!still_seven(static_cast<const float*>(inject.data()), hc + guard)) inject_held = false;
            };
            read(R, *down->values, y);
            int live_i = 0;
            const std::vector<float> host = output_mix_host(
                Rp, static_cast<const float*>(norm->values->data()),
                static_cast<const uint16_t*>(down->values->data()),
                static_cast<const uint16_t*>(up->values->data()), n_embd, hc, hc_lr, &live_i);
            read(R, *down->values, y2);
            read(zero_r, *down->values, yz);
            const int flip_at = live_i * 2 + 1;
            Bytes down_flip(down->values->size());
            std::memcpy(down_flip.data(), down->values->data(), down_flip.size());
            static_cast<uint8_t*>(down_flip.data())[flip_at] ^= 1u;
            read(R, down_flip, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = yp[n_embd] == 7.f && y2p[n_embd] == 7.f && yzp[n_embd] == 7.f && yfp[n_embd] == 7.f;
            const bool live = all_finite(yp, n_embd) && all_finite(y2p, n_embd) && all_finite(yzp, n_embd) &&
                              all_finite(yfp, n_embd) && all_finite(host.data(), n_embd) && !still_seven(yp, n_embd);
            const double rel = rel_l1(host.data(), yp, n_embd);
            const int again = differ_floats(yp, y2p, n_embd);
            int zeros = 0;
            for (int i = 0; i < n_embd; ++i)
                if (yzp[i] == 0.f) ++zeros;
            const int apart = differ_floats(yp, yzp, n_embd);
            const double apart_rel = rel_l1(yp, yzp, n_embd);
            const int flipped_n = differ_floats(yp, yfp, n_embd);
            ok = live && guard_held && inject_held && rel <= 1e-4;
            repeat = live && guard_held && inject_held && again == 0;
            cleared = live && guard_held && inject_held && zeros == n_embd && apart > 0 && std::isfinite(apart_rel);
            flipped = live && guard_held && inject_held && flipped_n > 0;
            char buf[180];
            std::snprintf(buf, sizeof buf, "rel %.3e, %d rows, inject %d", rel, n_embd, inject_held ? 1 : 0);
            detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d are zero, %d differ, %.2f%% apart", zeros, n_embd, apart,
                          apart_rel * 100.0);
            zero_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_embd, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            detail = e.what();
            repeat_detail = e.what();
            zero_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the output mixer matches", bytes && ok, detail);
    verdict("that output mixer repeats", bytes && repeat, repeat_detail);
    verdict("a zero residual clears the output mixer", bytes && cleared, zero_detail);
    verdict("a changed down byte moves the mixer", bytes && flipped, flip_detail);
}

// One Q3_K embedding row, copied into each residual stream, then the output mixer, then the
// Q5_K output weight. The row is the dequantized embedding. The product is iq_mmvq.
// gather_token_embd is not called. The Q8_K head is not called. This is not added into the
// layer-0 residual and it does not draw a token.
void run_head(const PackTable& table, const strata::core::ModelGeometry& g) {
    std::printf("\none embedding row through the output mixer\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const PackTensor* emb = table.find("token_embd.weight");
    const PackTensor* outw = table.find("output.weight");
    const PackTensor* down = table.find("output_hc_down.weight");
    const PackTensor* up = table.find("output_hc_up.weight");
    const PackTensor* norm = table.find("output_hc_norm.weight");
    const size_t emb_row = (size_t) (n_embd / 256) * 110;
    const size_t out_row = (size_t) (n_embd / 256) * 176;
    const bool bytes =
        emb && emb->values && emb->code_bits == 0 && emb->ne0 == n_embd && emb->ne1 == n_vocab &&
        emb->values->size() == (uint64_t) n_vocab * emb_row &&
        outw && outw->values && outw->code_bits == 0 && outw->ne0 == n_embd && outw->ne1 == n_vocab &&
        outw->values->size() == (uint64_t) n_vocab * out_row &&
        down && down->values && down->kind == strata::core::WeightKind::Bf16InF32 && down->code_bits == 0 &&
        down->ne0 == hc_dim && down->ne1 == hc_lr && down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 &&
        up && up->values && up->kind == strata::core::WeightKind::Bf16InF32 && up->code_bits == 0 &&
        up->ne0 == hc_lr && up->ne1 == hc_dim && up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 &&
        norm && norm->values && norm->kind == strata::core::WeightKind::F32 && norm->code_bits == 0 &&
        norm->ne0 == hc_dim && norm->ne1 == 1 && norm->values->size() == (uint64_t) hc_dim * 4;
    bool mix_ok = false;
    bool logits_ok = false;
    bool repeat = false;
    bool cleared = false;
    bool flipped = false;
    std::string mix_detail = "bytes";
    std::string logits_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string zero_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            const uint8_t* emb_w = static_cast<const uint8_t*>(emb->values->data());
            const float* norm_w = static_cast<const float*>(norm->values->data());
            const uint16_t* down_w = static_cast<const uint16_t*>(down->values->data());
            const uint16_t* up_w = static_cast<const uint16_t*>(up->values->data());
            std::vector<float> row((size_t) n_embd);
            int token = -1;
            double best_ms = 0;
            for (int r = 0; r < n_vocab; ++r) {
                const uint8_t* src = emb_w + (size_t) r * emb_row;
                for (int b = 0; b < n_embd / 256; ++b) dequant_q3(src + (size_t) b * 110, row.data() + (size_t) b * 256);
                double ms = 0;
                bool finite = true;
                for (int i = 0; i < n_embd; ++i) {
                    const double v = row[(size_t) i];
                    if (!std::isfinite(v)) {
                        finite = false;
                        break;
                    }
                    ms += v * v;
                }
                if (finite && ms > best_ms) {
                    best_ms = ms;
                    token = r;
                }
            }
            if (token < 0) throw MetalError("no finite embedding row");
            const uint8_t* src = emb_w + (size_t) token * emb_row;
            for (int b = 0; b < n_embd / 256; ++b) dequant_q3(src + (size_t) b * 110, row.data() + (size_t) b * 256);

            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes y((uint64_t) (n_embd + guard) * 4), y2((uint64_t) (n_embd + guard) * 4);
            Bytes yz((uint64_t) (n_embd + guard) * 4), yf((uint64_t) (n_embd + guard) * 4);
            Bytes logits((uint64_t) (n_vocab + guard) * 4), logits2((uint64_t) (n_vocab + guard) * 4);
            Bytes logits_z((uint64_t) (n_vocab + guard) * 4), logits_f((uint64_t) (n_vocab + guard) * 4);
            Bytes inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(y);
            fill7(y2);
            fill7(yz);
            fill7(yf);
            fill7(inject);
            GrScratch gr = make_gr_scratch(gs);
            bool inject_held = true;
            auto read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(inject.data()), (size_t) (hc + guard), 7.f);
                gr_read(gs, residual, *norm->values, *down->values, *up->values, nullptr, 1e-6f, false, gr, out, inject);
                if (!still_seven(static_cast<const float*>(inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mixed, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mixed.data(), act.size());
                Bytes xq(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, xq);
                fill7(out);
                iq_mmvq(kQ5_K, *outw->values, 0, xq, n_embd, n_vocab, 1, out);
                return xq;
            };
            read(R, y);
            const Bytes xq = project(y, logits);
            const std::vector<float> host_mix =
                output_mix_host(Rp, norm_w, down_w, up_w, n_embd, hc, hc_lr, nullptr);
            read(R, y2);
            const Bytes xq2 = project(y2, logits2);
            read(zero_r, yz);
            project(yz, logits_z);

            std::vector<uint8_t> row_copy(emb_row);
            std::memcpy(row_copy.data(), src, emb_row);
            auto host_of_row = [&](const uint8_t* raw) {
                std::vector<float> e((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b) dequant_q3(raw + (size_t) b * 110, e.data() + (size_t) b * 256);
                std::vector<float> tiled((size_t) hc_dim);
                for (int c = 0; c < hc; ++c)
                    std::memcpy(tiled.data() + (size_t) c * n_embd, e.data(), (size_t) n_embd * sizeof(float));
                return output_mix_host(tiled.data(), norm_w, down_w, up_w, n_embd, hc, hc_lr, nullptr);
            };
            int flip_local = -1;
            auto moves_mix = [&](int local) {
                row_copy[(size_t) local] ^= 1u;
                const std::vector<float> moved = host_of_row(row_copy.data());
                const bool changed = differ_floats(host_mix.data(), moved.data(), n_embd) > 0;
                row_copy[(size_t) local] ^= 1u;
                return changed;
            };
            if (moves_mix(32)) flip_local = 32;
            else {
                for (int i = 0; i < (int) emb_row && flip_local < 0; ++i) {
                    if (i == 32) continue;
                    if (moves_mix(i)) flip_local = i;
                }
            }
            const int flip_at = flip_local < 0 ? -1 : (int) ((size_t) token * emb_row + (size_t) flip_local);
            if (flip_local >= 0) {
                row_copy[(size_t) flip_local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                auto* Rfp = static_cast<float*>(Rf.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
                read(Rf, yf);
                project(yf, logits_f);
            } else {
                fill7(yf);
                fill7(logits_f);
            }

            std::vector<uint8_t> xqb((size_t) xq.size());
            std::memcpy(xqb.data(), xq.data(), xqb.size());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(static_cast<const uint8_t*>(outw->values->data()), xqb.data(), n_embd, n_vocab, true,
                  host_logits.data());
            scalar_ref(static_cast<const uint8_t*>(outw->values->data()), xqb.data(), n_embd, n_vocab, true,
                       scalar_logits.data());

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            const bool guard_held = yp[n_embd] == 7.f && y2p[n_embd] == 7.f && yzp[n_embd] == 7.f && yfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f && lfp[n_vocab] == 7.f;
            const bool xq_same = xq.size() == xq2.size() && std::memcmp(xq.data(), xq2.data(), xq.size()) == 0;
            const bool live = all_finite(yp, n_embd) && all_finite(y2p, n_embd) && all_finite(yzp, n_embd) &&
                              all_finite(yfp, n_embd) && all_finite(host_mix.data(), n_embd) &&
                              all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                              all_finite(lfp, n_vocab) && all_finite(host_logits.data(), n_vocab) &&
                              all_finite(scalar_logits.data(), n_vocab) && !still_seven(yp, n_embd) &&
                              !still_seven(lp, n_vocab);
            const double mix_rel = rel_l1(host_mix.data(), yp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(yp, y2p, n_embd);
            int zeros = 0;
            int mix_zeros = 0;
            for (int i = 0; i < n_vocab; ++i)
                if (lzp[i] == 0.f) ++zeros;
            for (int i = 0; i < n_embd; ++i)
                if (yzp[i] == 0.f) ++mix_zeros;
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int flipped_n = flip_local < 0 ? 0 : differ_floats(lp, lfp, n_vocab);
            mix_ok = live && guard_held && inject_held && mix_rel <= 1e-4;
            logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && xq_same;
            cleared = live && guard_held && inject_held && zeros == n_vocab && mix_zeros == n_embd && apart > 0 &&
                      std::isfinite(apart_rel);
            flipped = live && guard_held && inject_held && flipped_n > 0;
            char buf[180];
            std::snprintf(buf, sizeof buf, "rel %.3e, token %d, inject %d", mix_rel, token, inject_held ? 1 : 0);
            mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows", rel, irel, n_vocab);
            logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d are zero, %d differ, %.2f%% apart", zeros, n_vocab, apart,
                          apart_rel * 100.0);
            zero_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_vocab, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            mix_detail = e.what();
            logits_detail = e.what();
            repeat_detail = e.what();
            zero_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the head mix matches", bytes && mix_ok, mix_detail);
    verdict("the head logits match", bytes && logits_ok, logits_detail);
    verdict("that head repeats", bytes && repeat, repeat_detail);
    verdict("a zero embedding clears the head", bytes && cleared, zero_detail);
    verdict("a changed embedding byte moves the head", bytes && flipped, flip_detail);
}

// Token 7971's layer-0 residual, plus the shared expert, through the same head.
// Expert 0 is weighted exactly 1. The shared vector is added plain. Router logits stay off the block.
// The patterned pass and the earlier head stay unchanged. This is not a token.
void run_shared_on_layer(const PackTable& table, const strata::core::ModelGeometry& g,
                         const std::vector<uint8_t>& blob) {
    std::printf("\ntoken 7971 layer residual plus the shared expert\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int token = 7971;
    const size_t emb_row_n = (size_t) (n_embd / 256) * 110;
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    auto need = [&](const char* name, uint64_t nbytes) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->values->size() != nbytes) return nullptr;
        return t;
    };
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* qkv = need("blk.0.attn_qkv.weight", 13926400);
    const PackTensor* attn_gate = need("blk.0.attn_gate.weight", 8847360);
    const PackTensor* ssm = need("blk.0.ssm_out.weight", 10813440);
    const PackTensor* conv_w = need("blk.0.ssm_conv1d.weight", 163840);
    const PackTensor* alpha_w = need("blk.0.ssm_alpha.weight", 245760);
    const PackTensor* beta_w = need("blk.0.ssm_beta.weight", 245760);
    const PackTensor* ssm_a = need("blk.0.ssm_a", 192);
    const PackTensor* dt_w = need("blk.0.ssm_dt.bias", 192);
    const PackTensor* ssm_norm = need("blk.0.ssm_norm.weight", 512);
    const PackTensor* attn_down = need("blk.0.hc_attn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* attn_up = need("blk.0.hc_attn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* attn_inj = need("blk.0.hc_attn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* attn_norm = need("blk.0.hc_attn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* ffn_down = need("blk.0.hc_ffn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_up = need("blk.0.hc_ffn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* ffn_inj = need("blk.0.hc_ffn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_norm = need("blk.0.hc_ffn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* router = need("blk.0.ffn_gate_inp.weight", (uint64_t) n_embd * (uint64_t) n_expert * 2);
    const PackTensor* sgate = plane("blk.0.ffn_gate_shexp.weight", 870400, n_embd, n_ff);
    const PackTensor* sup = plane("blk.0.ffn_up_shexp.weight", 704000, n_embd, n_ff);
    const PackTensor* sdown = plane("blk.0.ffn_down_shexp.weight", 1126400, n_ff, n_embd);
    const PackTensor* ginp = table.find("blk.0.ffn_gate_inp_shexp.weight");
    const bool ginp_ok = ginp && ginp->values && ginp->kind == strata::core::WeightKind::Bf16InF32 &&
                         ginp->code_bits == 0 && ginp->ne0 == n_embd && ginp->ne1 == 1 &&
                         ginp->values->size() == (uint64_t) n_embd * 2;
    const PackTensor* emb = plane("token_embd.weight", 273152000, n_embd, n_vocab);
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_out->values->size() == (uint64_t) n_vocab * out_row_n && head_down && head_down->values &&
        head_down->kind == strata::core::WeightKind::Bf16InF32 && head_down->code_bits == 0 &&
        head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4;
    const bool bytes =
        qkv && qkv->code_bits == 0 && attn_gate && ssm && conv_w && alpha_w && beta_w && ssm_a && dt_w && ssm_norm &&
        attn_down && attn_up && attn_inj && attn_norm && ffn_down && ffn_up && ffn_inj && ffn_norm && router &&
        sgate && sup && sdown && ginp_ok && emb && head_ok && blob.size() == 1382400 && hc > 0 && n_embd > 0 &&
        (uint64_t) token * emb_row_n + emb_row_n <= emb->values->size();
    bool sh_mix_ok = false;
    bool sh_logits_ok = false;
    bool sh_repeat = false;
    bool sh_moved = false;
    bool sh_flipped = false;
    bool sh_dropped = false;
    std::string sh_mix_detail = "bytes";
    std::string sh_logits_detail = "bytes";
    std::string sh_repeat_detail = "bytes";
    std::string sh_move_detail = "bytes";
    std::string sh_flip_detail = "bytes";
    std::string sh_drop_detail = "bytes";
    if (bytes) {
        try {
            const uint8_t* emb_src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) token * emb_row_n;
            std::vector<float> emb_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(emb_src + (size_t) b * 110, emb_row.data() + (size_t) b * 256);
            double emb_ms = 0;
            for (int i = 0; i < n_embd; ++i) {
                if (!std::isfinite(emb_row[(size_t) i])) throw MetalError("embedding row is not finite");
                emb_ms += (double) emb_row[(size_t) i] * (double) emb_row[(size_t) i];
            }
            if (emb_ms == 0) throw MetalError("embedding row is zero");

            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, attn_gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = ssm_norm->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const size_t gu = iq_row_bytes(kQ2_0, n_embd);
            const int64_t up_off = (int64_t) n_ff * (int64_t) gu;
            const int64_t down_off = up_off * 2;
            Bytes expert_w((uint64_t) blob.size());
            std::memcpy(expert_w.data(), blob.data(), expert_w.size());
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, emb_row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes mid((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4), shared_y((uint64_t) (n_embd + guard) * 4);
            Bytes combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes parts((uint64_t) n_embd * 4), route_w(4);
            static_cast<float*>(route_w.data())[0] = 1.0f;
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kIq4Xs, *sgate->values, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ3_K, *sup->values, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ5_0, *sdown->values, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, *ginp->values, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, bool with_shared) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *attn_norm->values, *attn_down->values, *attn_up->values, attn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, expert_y);
                gr_write(gs, residual, expert_y, inject, mid);
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                iq_quantize_q8_1(mixed, n_embd, exq);
                iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                iq_mmvq(kQ2_0, expert_w, up_off, exq, n_embd, n_ff, 1, eup_y);
                iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                iq_mmvq(kQ2_0, expert_w, down_off, ehq, n_ff, n_embd, 1, expert_y);
                shared_forward(shared_y);
                std::memcpy(parts.data(), expert_y.data(), (size_t) n_embd * 4);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, 1);
                gr_write(gs, mid, combined, inject, out);
            };
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mix.data(), act.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };

            run_pass(R, layer_y, true);
            std::vector<float> got_expert((size_t) n_embd), got_shared((size_t) n_embd), got_combined((size_t) n_embd);
            std::vector<float> got_inject((size_t) hc), got_mixed((size_t) n_embd), got_mid((size_t) hc_dim);
            std::vector<float> got_layer((size_t) hc_dim);
            std::vector<float> got_egate((size_t) n_ff), got_eup((size_t) n_ff);
            std::vector<float> got_sgate((size_t) n_ff), got_sup((size_t) n_ff), got_sdown((size_t) n_embd);
            std::memcpy(got_expert.data(), expert_y.data(), got_expert.size() * 4);
            std::memcpy(got_shared.data(), shared_y.data(), got_shared.size() * 4);
            std::memcpy(got_combined.data(), combined.data(), got_combined.size() * 4);
            std::memcpy(got_inject.data(), inject.data(), got_inject.size() * 4);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_mid.data(), mid.data(), got_mid.size() * 4);
            std::memcpy(got_layer.data(), layer_y.data(), got_layer.size() * 4);
            std::memcpy(got_egate.data(), egate_y.data(), got_egate.size() * 4);
            std::memcpy(got_eup.data(), eup_y.data(), got_eup.size() * 4);
            std::memcpy(got_sgate.data(), sh_gate_y.data(), got_sgate.size() * 4);
            std::memcpy(got_sup.data(), sh_up_y.data(), got_sup.size() * 4);
            std::memcpy(got_sdown.data(), sh_down_y.data(), got_sdown.size() * 4);
            const float got_scale = static_cast<const float*>(sh_scalar.data())[0];
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool shared_guard = static_cast<const float*>(shared_y.data())[n_embd] == 7.f;
            const bool mid_guard = static_cast<const float*>(mid.data())[hc_dim] == 7.f;
            std::vector<uint8_t> exqb((size_t) exq.size()), ehqb((size_t) ehq.size());
            std::vector<uint8_t> sh_xqb((size_t) sh_xq.size()), sh_hqb((size_t) sh_hq.size());
            std::memcpy(exqb.data(), exq.data(), exqb.size());
            std::memcpy(ehqb.data(), ehq.data(), ehqb.size());
            std::memcpy(sh_xqb.data(), sh_xq.data(), sh_xqb.size());
            std::memcpy(sh_hqb.data(), sh_hq.data(), sh_hqb.size());
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            std::vector<float> got_mix((size_t) n_embd);
            std::memcpy(got_mix.data(), mix_y.data(), got_mix.size() * 4);

            std::vector<float> host_egate((size_t) n_ff), host_eup((size_t) n_ff), host_edown((size_t) n_embd);
            q2_ref(blob.data(), exqb.data(), n_embd, n_ff, host_egate.data());
            q2_ref(blob.data() + (size_t) up_off, exqb.data(), n_embd, n_ff, host_eup.data());
            q2_ref(blob.data() + (size_t) down_off, ehqb.data(), n_ff, n_embd, host_edown.data());
            const uint8_t* sgate_b = static_cast<const uint8_t*>(sgate->values->data());
            const uint8_t* sup_b = static_cast<const uint8_t*>(sup->values->data());
            const uint8_t* sdown_b = static_cast<const uint8_t*>(sdown->values->data());
            std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
            xs_ref(sgate_b, sh_xqb.data(), n_embd, n_ff, host_sgate.data());
            q3_ref(sup_b, sh_xqb.data(), n_embd, n_ff, host_sup.data());
            q50_ref(sdown_b, sh_hqb.data(), n_ff, n_embd, host_sdown.data());
            std::vector<float> rounded((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(got_mixed[(size_t) i]);
            const float host_scale =
                bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
            const float sig = gdn_sigmoid(host_scale);
            std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) {
                host_shared[(size_t) i] = host_sdown[(size_t) i] * sig;
                host_combined[(size_t) i] =
                    (float) ((double) got_expert[(size_t) i] + (double) got_shared[(size_t) i]);
            }
            std::vector<float> host_write((size_t) hc_dim);
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(got_inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        got_mid[(size_t) c * n_embd + d] + got_combined[(size_t) d] * scale;
            }
            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(got_layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());

            const AttnHalf host_attn = layer0_attn_host(
                Rp, static_cast<const float*>(attn_norm->values->data()),
                static_cast<const uint16_t*>(attn_down->values->data()),
                static_cast<const uint16_t*>(attn_up->values->data()),
                static_cast<const uint16_t*>(attn_inj->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()), static_cast<const uint8_t*>(attn_gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(ssm_norm->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            const FfnHalf host_ffn = layer0_ffn_host(
                host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()), blob.data(), n_embd, n_ff, n_expert, hc, hc_lr);
            std::vector<float> chain_shexp((size_t) n_embd);
            {
                Bytes act_b((uint64_t) n_embd * 4);
                std::memcpy(act_b.data(), host_ffn.mixed.data(), act_b.size());
                Bytes cxq(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act_b, n_embd, cxq);
                std::vector<uint8_t> cxqb((size_t) cxq.size());
                std::memcpy(cxqb.data(), cxq.data(), cxqb.size());
                std::vector<float> cg((size_t) n_ff), cu((size_t) n_ff), hidden((size_t) n_ff), cdown((size_t) n_embd);
                xs_ref(sgate_b, cxqb.data(), n_embd, n_ff, cg.data());
                q3_ref(sup_b, cxqb.data(), n_embd, n_ff, cu.data());
                for (int i = 0; i < n_ff; ++i) hidden[(size_t) i] = gdn_silu(cg[(size_t) i]) * cu[(size_t) i];
                Bytes hf((uint64_t) n_ff * 4);
                std::memcpy(hf.data(), hidden.data(), hf.size());
                Bytes chq(iq_q8_1_bytes(n_ff));
                iq_quantize_q8_1(hf, n_ff, chq);
                std::vector<uint8_t> chqb((size_t) chq.size());
                std::memcpy(chqb.data(), chq.data(), chqb.size());
                q50_ref(sdown_b, chqb.data(), n_ff, n_embd, cdown.data());
                std::vector<float> cr((size_t) n_embd);
                for (int i = 0; i < n_embd; ++i) cr[(size_t) i] = round_bf16(host_ffn.mixed[(size_t) i]);
                const float cs =
                    bf16_dot(cr.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
                const float csig = gdn_sigmoid(cs);
                for (int i = 0; i < n_embd; ++i) chain_shexp[(size_t) i] = cdown[(size_t) i] * csig;
            }
            std::vector<float> chain_inject((size_t) hc, 0.f), chain_res((size_t) hc_dim, 0.f);
            std::vector<float> chain_block((size_t) n_embd, 0.f);
            bool chain_res_ok = false;
            if (all_finite(host_attn.out.data(), hc_dim) && all_finite(host_ffn.block.data(), n_embd) &&
                all_finite(chain_shexp.data(), n_embd)) {
                const float* src = host_attn.out.data();
                const float* norm_w = static_cast<const float*>(ffn_norm->values->data());
                const uint16_t* iw = static_cast<const uint16_t*>(ffn_inj->values->data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                for (int c = 0; c < hc; ++c)
                    chain_inject[(size_t) c] = bf16_dot(act.data(), iw + (size_t) c * hc_dim, hc_dim);
                for (int d = 0; d < n_embd; ++d)
                    chain_block[(size_t) d] =
                        (float) ((double) host_ffn.block[(size_t) d] + (double) chain_shexp[(size_t) d]);
                for (int c = 0; c < hc; ++c) {
                    const float scale = 2.0f * gdn_sigmoid(chain_inject[(size_t) c] / (float) hc);
                    for (int d = 0; d < n_embd; ++d)
                        chain_res[(size_t) c * n_embd + d] = src[(size_t) c * n_embd + d] + chain_block[(size_t) d] * scale;
                }
                chain_res_ok = true;
            }
            std::vector<float> chain_mix((size_t) n_embd, 0.f);
            if (chain_res_ok)
                chain_mix = output_mix_host(chain_res.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            std::vector<float> chain_logits((size_t) n_vocab);
            bool chain_ran = false;
            if (chain_res_ok && all_finite(chain_mix.data(), n_embd)) {
                Bytes chain_act((uint64_t) n_embd * 4);
                std::memcpy(chain_act.data(), chain_mix.data(), chain_act.size());
                Bytes chain_q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(chain_act, n_embd, chain_q);
                std::vector<uint8_t> chain_qb((size_t) chain_q.size());
                std::memcpy(chain_qb.data(), chain_q.data(), chain_qb.size());
                k_ref(out_bytes, chain_qb.data(), n_embd, n_vocab, true, chain_logits.data());
                chain_ran = true;
            }

            run_pass(R, layer_y2, true);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, true);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);
            run_pass(R, layer_drop, false);

            std::vector<uint8_t> row_copy(emb_row_n);
            auto launch_flip = [&](int local, bool with_logits) {
                std::memcpy(row_copy.data(), emb_src, emb_row_n);
                row_copy[(size_t) local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                auto* Rfp = static_cast<float*>(Rf.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
                run_pass(Rf, layer_yf, true);
                mix_read(layer_yf, mix_yf);
                if (with_logits) project(mix_yf, logits_f);
            };
            int flip_local = 32;
            launch_flip(flip_local, true);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flipped_n = differ_floats(lp, lfp, n_vocab);
            if (flipped_n == 0) {
                for (int i = 0; i < (int) emb_row_n; ++i) {
                    if (i == 32) continue;
                    launch_flip(i, false);
                    if (differ_floats(layer_p, layerf_p, hc_dim) == 0) continue;
                    project(mix_yf, logits_f);
                    flipped_n = differ_floats(lp, lfp, n_vocab);
                    if (flipped_n > 0) {
                        flip_local = i;
                        break;
                    }
                }
            }
            const int flip_at = (int) ((size_t) token * emb_row_n + (size_t) flip_local);
            double chain_mix_rel = std::nan("");
            double chain_logit_rel = std::nan("");
            if (chain_res_ok && all_finite(chain_mix.data(), n_embd))
                chain_mix_rel = rel_l1(chain_mix.data(), mp, n_embd);
            if (chain_ran && all_finite(chain_logits.data(), n_vocab))
                chain_logit_rel = rel_l1(chain_logits.data(), lp, n_vocab);

            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mzp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && combine_guard && shared_guard && mid_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(layer_p, hc_dim) && all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) &&
                all_finite(layerf_p, hc_dim) && all_finite(mp, n_embd) && all_finite(m2p, n_embd) &&
                all_finite(mzp, n_embd) && all_finite(mfp, n_embd) && all_finite(lp, n_vocab) &&
                all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) && all_finite(lfp, n_vocab) &&
                all_finite(got_expert.data(), n_embd) && all_finite(got_shared.data(), n_embd) &&
                all_finite(got_combined.data(), n_embd) && all_finite(got_mixed.data(), n_embd) &&
                all_finite(got_mid.data(), hc_dim) && all_finite(got_layer.data(), hc_dim) &&
                all_finite(got_egate.data(), n_ff) && all_finite(got_eup.data(), n_ff) &&
                all_finite(got_sgate.data(), n_ff) && all_finite(got_sup.data(), n_ff) &&
                all_finite(got_sdown.data(), n_embd) && std::isfinite(got_scale) && std::isfinite(host_scale) &&
                all_finite(host_egate.data(), n_ff) && all_finite(host_eup.data(), n_ff) &&
                all_finite(host_edown.data(), n_embd) && all_finite(host_sgate.data(), n_ff) &&
                all_finite(host_sup.data(), n_ff) && all_finite(host_sdown.data(), n_embd) &&
                all_finite(host_shared.data(), n_embd) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(got_inject.data(), hc) && !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) &&
                !still_seven(lp, n_vocab);
            const double egate_rel = rel_l1(host_egate.data(), got_egate.data(), n_ff);
            const double eup_rel = rel_l1(host_eup.data(), got_eup.data(), n_ff);
            const double expert_rel = rel_l1(host_edown.data(), got_expert.data(), n_embd);
            const double gate_rel = rel_l1(host_sgate.data(), got_sgate.data(), n_ff);
            const double up_rel = rel_l1(host_sup.data(), got_sup.data(), n_ff);
            const double down_rel = rel_l1(host_sdown.data(), got_sdown.data(), n_embd);
            const double scale_rel = rel_l1(&host_scale, &got_scale, 1);
            const double shared_rel = rel_l1(host_shared.data(), got_shared.data(), n_embd);
            const double combine_rel = rel_l1(host_combined.data(), got_combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), got_layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), got_mix.data(), n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            sh_mix_ok = live && guard_held && inject_held && egate_rel <= 1e-5 && eup_rel <= 1e-5 &&
                        expert_rel <= 1e-5 && gate_rel <= 1e-5 && up_rel <= 1e-5 && down_rel <= 1e-5 &&
                        scale_rel <= 1e-4 && shared_rel <= 1e-4 && combine_rel <= 1e-5 && write_rel <= 1e-4 &&
                        mix_rel <= 1e-4;
            sh_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            sh_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            sh_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            sh_flipped = live && guard_held && inject_held && flipped_n > 0;
            sh_dropped = live && guard_held && inject_held && all_finite(drop_p, hc_dim) && drop_n > 0;
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "token %d, expert %.3e, gate %.3e, up %.3e, down %.3e, scale %.3e, shared %.3e, "
                          "combine %.3e, write %.3e, mix %.3e, weight 1, chain mix %.3e",
                          token, expert_rel, gate_rel, up_rel, down_rel, scale_rel, shared_rel, combine_rel,
                          write_rel, mix_rel, chain_mix_rel);
            sh_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain %.3e", rel, irel, n_vocab,
                          chain_logit_rel);
            sh_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            sh_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            sh_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_vocab, flip_at);
            sh_flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            sh_drop_detail = buf;
        } catch (const MetalError& e) {
            sh_mix_detail = e.what();
            sh_logits_detail = e.what();
            sh_repeat_detail = e.what();
            sh_move_detail = e.what();
            sh_flip_detail = e.what();
            sh_drop_detail = e.what();
        }
    }
    verdict("the shared layer mix matches", bytes && sh_mix_ok, sh_mix_detail);
    verdict("the shared layer logits match", bytes && sh_logits_ok, sh_logits_detail);
    verdict("that shared layer head repeats", bytes && sh_repeat, sh_repeat_detail);
    verdict("a zero residual moves that shared layer", bytes && sh_moved, sh_move_detail);
    verdict("a changed embedding byte moves that shared layer", bytes && sh_flipped, sh_flip_detail);
    verdict("dropping the shared expert moves that layer", bytes && sh_dropped, sh_drop_detail);
}

// Softmax, strict top k, lower index on a tie, then divide by the clamped sum.
// The same reference the router parity uses. k is the caller's width.
void route_ref(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = (double) logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    std::vector<double> p((size_t) n_expert);
    double sum = 0;
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = std::exp((double) logits[e] - mx);
        sum += p[(size_t) e];
    }
    for (int e = 0; e < n_expert; ++e) p[(size_t) e] /= sum;
    std::vector<int> idx((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) idx[(size_t) e] = e;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double chosen = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t) i];
        w[i] = (float) p[(size_t) idx[(size_t) i]];
        chosen += p[(size_t) idx[(size_t) i]];
    }
    chosen = std::max(chosen, 6.103515625e-05);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / chosen);
}

// Token 7971's layer-0 router logits, then router_top10. k is qwen4exp.expert_used_count.
// The selected experts are not loaded. This is not a token.
void run_layer_route(const PackTable& table, const strata::core::ModelGeometry& g,
                     const std::vector<uint8_t>& blob) {
    std::printf("\ntoken 7971 layer-0 router selection\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int token = 7971;
    const int k = 10;
    const size_t emb_row_n = (size_t) (n_embd / 256) * 110;
    auto need = [&](const char* name, uint64_t nbytes) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->values->size() != nbytes) return nullptr;
        return t;
    };
    const PackTensor* qkv = need("blk.0.attn_qkv.weight", 13926400);
    const PackTensor* attn_gate = need("blk.0.attn_gate.weight", 8847360);
    const PackTensor* ssm = need("blk.0.ssm_out.weight", 10813440);
    const PackTensor* conv_w = need("blk.0.ssm_conv1d.weight", 163840);
    const PackTensor* alpha_w = need("blk.0.ssm_alpha.weight", 245760);
    const PackTensor* beta_w = need("blk.0.ssm_beta.weight", 245760);
    const PackTensor* ssm_a = need("blk.0.ssm_a", 192);
    const PackTensor* dt_w = need("blk.0.ssm_dt.bias", 192);
    const PackTensor* ssm_norm = need("blk.0.ssm_norm.weight", 512);
    const PackTensor* attn_down = need("blk.0.hc_attn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* attn_up = need("blk.0.hc_attn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* attn_inj = need("blk.0.hc_attn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* attn_norm = need("blk.0.hc_attn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* ffn_down = need("blk.0.hc_ffn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_up = need("blk.0.hc_ffn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* ffn_inj = need("blk.0.hc_ffn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_norm = need("blk.0.hc_ffn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* router = need("blk.0.ffn_gate_inp.weight", (uint64_t) n_embd * (uint64_t) n_expert * 2);
    const PackTensor* emb = table.find("token_embd.weight");
    const bool emb_ok = emb && emb->values && emb->code_bits == 0 && emb->ne0 == n_embd && emb->ne1 == 248320 &&
                        emb->values->size() == (uint64_t) 248320 * emb_row_n &&
                        (uint64_t) token * emb_row_n + emb_row_n <= emb->values->size();
    const bool bytes = qkv && qkv->code_bits == 0 && attn_gate && ssm && conv_w && alpha_w && beta_w && ssm_a &&
                       dt_w && ssm_norm && attn_down && attn_up && attn_inj && attn_norm && ffn_down && ffn_up &&
                       ffn_inj && ffn_norm && router && emb_ok && n_expert == 512 && k <= n_expert && hc > 0;
    bool rt_ok = false;
    bool rt_repeat = false;
    bool rt_moved = false;
    bool rt_flipped = false;
    std::string rt_detail = "bytes";
    std::string rt_repeat_detail = "bytes";
    std::string rt_move_detail = "bytes";
    std::string rt_flip_detail = "bytes";
    if (bytes) {
        try {
            const uint8_t* emb_src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) token * emb_row_n;
            std::vector<float> emb_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(emb_src + (size_t) b * 110, emb_row.data() + (size_t) b * 256);
            double emb_ms = 0;
            for (int i = 0; i < n_embd; ++i) {
                if (!std::isfinite(emb_row[(size_t) i])) throw MetalError("embedding row is not finite");
                emb_ms += (double) emb_row[(size_t) i] * (double) emb_row[(size_t) i];
            }
            if (emb_ms == 0) throw MetalError("embedding row is zero");

            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, attn_gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = ssm_norm->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, emb_row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard = 1;
            Bytes mid((uint64_t) (hc_dim + guard) * 4), block((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) (n_expert + guard) * 4);
            Bytes ids((uint64_t) (k + guard) * 4), weights((uint64_t) (k + guard) * 4);
            Bytes ids2((uint64_t) (k + guard) * 4), weights2((uint64_t) (k + guard) * 4);
            Bytes idsz((uint64_t) (k + guard) * 4), weightsz((uint64_t) (k + guard) * 4);
            Bytes idsf((uint64_t) (k + guard) * 4), weightsf((uint64_t) (k + guard) * 4);
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto run_pass = [&](const Bytes& residual, Bytes& id_out, Bytes& w_out) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) (hc_dim + guard), 7.f);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(route.data()), (size_t) (n_expert + guard), 7.f);
                std::fill_n(static_cast<int*>(id_out.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(w_out.data()), (size_t) (k + guard), 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *attn_norm->values, *attn_down->values, *attn_up->values, attn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, block);
                gr_write(gs, residual, block, inject, mid);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, id_out, w_out);
            };

            run_pass(R, ids, weights);
            std::vector<float> got_mixed((size_t) n_embd), got_route((size_t) n_expert);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_route.data(), route.data(), got_route.size() * 4);
            const bool route_guard = static_cast<const float*>(route.data())[n_expert] == 7.f;
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_w((size_t) k);
            route_ref(got_route.data(), n_expert, k, host_ids.data(), host_w.data());
            std::vector<float> rounded((size_t) n_embd), host_logits((size_t) n_expert);
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(got_mixed[(size_t) i]);
            const uint16_t* rw = static_cast<const uint16_t*>(router->values->data());
            for (int o = 0; o < n_expert; ++o)
                host_logits[(size_t) o] = bf16_dot(rounded.data(), rw + (size_t) o * n_embd, n_embd);

            int chain_id_bad = -1;
            double chain_w_rel = std::nan("");
            bool chain_ran = false;
            if (blob.size() == 1382400 && all_finite(got_mixed.data(), n_embd)) {
                const AttnHalf host_attn = layer0_attn_host(
                    Rp, static_cast<const float*>(attn_norm->values->data()),
                    static_cast<const uint16_t*>(attn_down->values->data()),
                    static_cast<const uint16_t*>(attn_up->values->data()),
                    static_cast<const uint16_t*>(attn_inj->values->data()),
                    static_cast<const uint8_t*>(qkv->values->data()),
                    static_cast<const uint8_t*>(attn_gate->values->data()),
                    static_cast<const uint8_t*>(ssm->values->data()),
                    static_cast<const uint16_t*>(alpha_w->values->data()),
                    static_cast<const uint16_t*>(beta_w->values->data()),
                    static_cast<const float*>(conv_w->values->data()),
                    static_cast<const float*>(ssm_norm->values->data()),
                    static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                    n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
                if (all_finite(host_attn.out.data(), hc_dim)) {
                    const FfnHalf host_ffn = layer0_ffn_host(
                        host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                        static_cast<const uint16_t*>(ffn_down->values->data()),
                        static_cast<const uint16_t*>(ffn_up->values->data()),
                        static_cast<const uint16_t*>(ffn_inj->values->data()), rw, blob.data(), n_embd, n_ff,
                        n_expert, hc, hc_lr);
                    if (all_finite(host_ffn.router.data(), n_expert)) {
                        std::vector<int> chain_ids((size_t) k);
                        std::vector<float> chain_w((size_t) k);
                        route_ref(host_ffn.router.data(), n_expert, k, chain_ids.data(), chain_w.data());
                        chain_id_bad = 0;
                        for (int i = 0; i < k; ++i)
                            if (chain_ids[(size_t) i] != static_cast<const int*>(ids.data())[i]) ++chain_id_bad;
                        chain_w_rel = rel_l1(chain_w.data(), static_cast<const float*>(weights.data()), k);
                        chain_ran = true;
                    }
                }
            }

            run_pass(R, ids2, weights2);
            run_pass(zero_r, idsz, weightsz);
            std::vector<uint8_t> row_copy(emb_row_n);
            auto launch_flip = [&](int local) {
                std::memcpy(row_copy.data(), emb_src, emb_row_n);
                row_copy[(size_t) local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                auto* Rfp = static_cast<float*>(Rf.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
                run_pass(Rf, idsf, weightsf);
            };
            int flip_local = 32;
            launch_flip(flip_local);

            const int* ip = static_cast<const int*>(ids.data());
            const int* i2p = static_cast<const int*>(ids2.data());
            const int* izp = static_cast<const int*>(idsz.data());
            const int* ifp = static_cast<const int*>(idsf.data());
            const float* wp = static_cast<const float*>(weights.data());
            const float* w2p = static_cast<const float*>(weights2.data());
            const float* wzp = static_cast<const float*>(weightsz.data());
            const float* wfp = static_cast<const float*>(weightsf.data());
            auto id_diff = [&](const int* a) {
                int n = 0;
                for (int i = 0; i < k; ++i)
                    if (a[i] != ip[i]) ++n;
                return n;
            };
            int flipped_ids = id_diff(ifp);
            int flipped_w = differ_floats(wp, wfp, k);
            if (flipped_ids == 0 && flipped_w == 0) {
                for (int i = 0; i < (int) emb_row_n; ++i) {
                    if (i == 32) continue;
                    launch_flip(i);
                    flipped_ids = id_diff(ifp);
                    flipped_w = differ_floats(wp, wfp, k);
                    if (flipped_ids > 0 || flipped_w > 0) {
                        flip_local = i;
                        break;
                    }
                }
            }
            const int flip_at = (int) ((size_t) token * emb_row_n + (size_t) flip_local);
            int id_bad = 0;
            bool ids_in_range = true;
            for (int i = 0; i < k; ++i) {
                if (ip[i] != host_ids[(size_t) i]) ++id_bad;
                if (ip[i] < 0 || ip[i] >= n_expert) ids_in_range = false;
            }
            int expert0 = 0;
            for (int i = 0; i < k; ++i)
                if (ip[i] == 0) ++expert0;
            double wsum = 0;
            for (int i = 0; i < k; ++i) wsum += (double) wp[i];
            const double sum_err = std::fabs(wsum - 1.0);
            const bool guard_held = ip[k] == 7 && i2p[k] == 7 && izp[k] == 7 && ifp[k] == 7 && wp[k] == 7.f &&
                                    w2p[k] == 7.f && wzp[k] == 7.f && wfp[k] == 7.f && route_guard;
            const bool live = all_finite(got_mixed.data(), n_embd) && all_finite(got_route.data(), n_expert) &&
                              all_finite(host_logits.data(), n_expert) && all_finite(wp, k) && all_finite(w2p, k) &&
                              all_finite(wzp, k) && all_finite(wfp, k) && all_finite(host_w.data(), k) &&
                              ids_in_range && !still_seven(got_route.data(), n_expert) && !still_seven(wp, k);
            const double logit_rel = rel_l1(host_logits.data(), got_route.data(), n_expert);
            const double weight_rel = rel_l1(host_w.data(), wp, k);
            const int again_ids = id_diff(i2p);
            const int again_w = differ_floats(wp, w2p, k);
            const int apart_ids = id_diff(izp);
            const int apart_w = differ_floats(wp, wzp, k);
            rt_ok = live && guard_held && logit_rel <= 1e-5 && id_bad == 0 && weight_rel <= 1e-5 && sum_err <= 1e-5;
            rt_repeat = live && guard_held && again_ids == 0 && again_w == 0;
            rt_moved = live && guard_held && (apart_ids > 0 || apart_w > 0);
            rt_flipped = live && guard_held && (flipped_ids > 0 || flipped_w > 0);
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(ip[i]);
            }
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "k %d, logits %.3e, ids %d, weights %.3e, sum %.3e, expert0 %d, chain ids %d, "
                          "chain weights %.3e, experts %s",
                          k, logit_rel, id_bad, weight_rel, sum_err, expert0, chain_ran ? chain_id_bad : -1,
                          chain_w_rel, id_list.c_str());
            rt_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again_ids + again_w, guard_held ? 1 : 0);
            rt_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "ids %d, weights %d of %d differ", apart_ids, apart_w, k);
            rt_move_detail = buf;
            std::snprintf(buf, sizeof buf, "ids %d, weights %d of %d differ, byte %d", flipped_ids, flipped_w, k,
                          flip_at);
            rt_flip_detail = buf;
        } catch (const MetalError& e) {
            rt_detail = e.what();
            rt_repeat_detail = e.what();
            rt_move_detail = e.what();
            rt_flip_detail = e.what();
        }
    }
    verdict("the layer-0 router selection matches", bytes && rt_ok, rt_detail);
    verdict("that router selection repeats", bytes && rt_repeat, rt_repeat_detail);
    verdict("a zero residual moves that selection", bytes && rt_moved, rt_move_detail);
    verdict("a changed embedding byte moves that selection", bytes && rt_flipped, rt_flip_detail);
}

// The ten experts router_top10 named for token 7971. Expert 0 is not one of them.
// Their Q2_0 bytes are read from the GGUF. The other 47 layers are not read.
// The shared vector is added plain. This is not a token.
void run_routed_ten(const PackTable& table, const strata::core::ModelGeometry& g,
                    const std::vector<uint8_t>& blob) {
    std::printf("\ntoken 7971 layer residual through the ten routed experts\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int token = 7971;
    const int k = 10;
    const int recorded[10] = {232, 479, 66, 342, 308, 16, 27, 164, 347, 99};
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 746117120;
    const uint64_t k_gate = 983173120;
    const uint64_t k_up = 1222599680;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t emb_row_n = (size_t) (n_embd / 256) * 110;
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    bool layout = false;
    std::string layout_detail = "bytes";
    if (blob.size() == 1382400 && stride_ok && std::filesystem::is_regular_file(shard)) {
        try {
            const auto gate0 = read_at_path(shard, k_data + k_gate, stride);
            const auto up0 = read_at_path(shard, k_data + k_up, stride);
            const auto down0 = read_at_path(shard, k_data + k_down, stride);
            layout = std::memcmp(gate0.data(), blob.data(), stride) == 0 &&
                     std::memcmp(up0.data(), blob.data() + stride, stride) == 0 &&
                     std::memcmp(down0.data(), blob.data() + stride * 2, stride) == 0;
            layout_detail = layout ? "460800 x 3" : "mismatch";
        } catch (const MetalError& e) {
            layout_detail = e.what();
        }
    }
    verdict("expert 0 matches the GGUF planes", layout, layout_detail);

    auto need = [&](const char* name, uint64_t nbytes) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->values->size() != nbytes) return nullptr;
        return t;
    };
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* qkv = need("blk.0.attn_qkv.weight", 13926400);
    const PackTensor* attn_gate = need("blk.0.attn_gate.weight", 8847360);
    const PackTensor* ssm = need("blk.0.ssm_out.weight", 10813440);
    const PackTensor* conv_w = need("blk.0.ssm_conv1d.weight", 163840);
    const PackTensor* alpha_w = need("blk.0.ssm_alpha.weight", 245760);
    const PackTensor* beta_w = need("blk.0.ssm_beta.weight", 245760);
    const PackTensor* ssm_a = need("blk.0.ssm_a", 192);
    const PackTensor* dt_w = need("blk.0.ssm_dt.bias", 192);
    const PackTensor* ssm_norm = need("blk.0.ssm_norm.weight", 512);
    const PackTensor* attn_down = need("blk.0.hc_attn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* attn_up = need("blk.0.hc_attn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* attn_inj = need("blk.0.hc_attn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* attn_norm = need("blk.0.hc_attn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* ffn_down = need("blk.0.hc_ffn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_up = need("blk.0.hc_ffn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* ffn_inj = need("blk.0.hc_ffn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_norm = need("blk.0.hc_ffn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* router = need("blk.0.ffn_gate_inp.weight", (uint64_t) n_embd * (uint64_t) n_expert * 2);
    const PackTensor* sgate = plane("blk.0.ffn_gate_shexp.weight", 870400, n_embd, n_ff);
    const PackTensor* sup = plane("blk.0.ffn_up_shexp.weight", 704000, n_embd, n_ff);
    const PackTensor* sdown = plane("blk.0.ffn_down_shexp.weight", 1126400, n_ff, n_embd);
    const PackTensor* ginp = table.find("blk.0.ffn_gate_inp_shexp.weight");
    const bool ginp_ok = ginp && ginp->values && ginp->kind == strata::core::WeightKind::Bf16InF32 &&
                         ginp->code_bits == 0 && ginp->ne0 == n_embd && ginp->ne1 == 1 &&
                         ginp->values->size() == (uint64_t) n_embd * 2;
    const PackTensor* emb = plane("token_embd.weight", 273152000, n_embd, n_vocab);
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = layout && qkv && qkv->code_bits == 0 && attn_gate && ssm && conv_w && alpha_w && beta_w &&
                       ssm_a && dt_w && ssm_norm && attn_down && attn_up && attn_inj && attn_norm && ffn_down &&
                       ffn_up && ffn_inj && ffn_norm && router && sgate && sup && sdown && ginp_ok && emb && head_ok &&
                       n_expert == 512 && hc > 0;
    bool rt_mix_ok = false;
    bool rt_logits_ok = false;
    bool rt_repeat = false;
    bool rt_moved = false;
    bool rt_flipped = false;
    bool rt_dropped = false;
    std::string rt_mix_detail = "bytes";
    std::string rt_logits_detail = "bytes";
    std::string rt_repeat_detail = "bytes";
    std::string rt_move_detail = "bytes";
    std::string rt_flip_detail = "bytes";
    std::string rt_drop_detail = "bytes";
    if (bytes) {
        try {
            const uint8_t* emb_src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) token * emb_row_n;
            std::vector<float> emb_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(emb_src + (size_t) b * 110, emb_row.data() + (size_t) b * 256);
            double emb_ms = 0;
            for (int i = 0; i < n_embd; ++i) {
                if (!std::isfinite(emb_row[(size_t) i])) throw MetalError("embedding row is not finite");
                emb_ms += (double) emb_row[(size_t) i] * (double) emb_row[(size_t) i];
            }
            if (emb_ms == 0) throw MetalError("embedding row is zero");

            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, attn_gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = ssm_norm->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, emb_row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes mid((uint64_t) guard_n * 4), expert_y((uint64_t) (n_embd + guard) * 4);
            Bytes shared_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, mid, shared, combined, layer;
                std::vector<float> sgate, sup, sdown;
                std::vector<uint8_t> exq, sh_xq, sh_hq;
                float scale = 0;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kIq4Xs, *sgate->values, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ3_K, *sup->values, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ5_0, *sdown->values, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, *ginp->values, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, bool with_shared, Snap* snap) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *attn_norm->values, *attn_down->values, *attn_up->values, attn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, expert_y);
                gr_write(gs, residual, expert_y, inject, mid);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                shared_forward(shared_y);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, k);
                gr_write(gs, mid, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->mid.resize((size_t) hc_dim);
                    snap->shared.resize((size_t) n_embd);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->sgate.resize((size_t) n_ff);
                    snap->sup.resize((size_t) n_ff);
                    snap->sdown.resize((size_t) n_embd);
                    snap->sh_xq.resize(sh_xq.size());
                    snap->sh_hq.resize(sh_hq.size());
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->mid.data(), mid.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->shared.data(), shared_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->sgate.data(), sh_gate_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sup.data(), sh_up_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sdown.data(), sh_down_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->sh_xq.data(), sh_xq.data(), sh_xq.size());
                    std::memcpy(snap->sh_hq.data(), sh_hq.data(), sh_hq.size());
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                    snap->scale = static_cast<const float*>(sh_scalar.data())[0];
                }
            };

            Snap snap;
            run_pass(R, layer_y, true, &snap);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool shared_guard = static_cast<const float*>(shared_y.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            const uint8_t* sgate_b = static_cast<const uint8_t*>(sgate->values->data());
            const uint8_t* sup_b = static_cast<const uint8_t*>(sup->values->data());
            const uint8_t* sdown_b = static_cast<const uint8_t*>(sdown->values->data());
            std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
            xs_ref(sgate_b, snap.sh_xq.data(), n_embd, n_ff, host_sgate.data());
            q3_ref(sup_b, snap.sh_xq.data(), n_embd, n_ff, host_sup.data());
            q50_ref(sdown_b, snap.sh_hq.data(), n_ff, n_embd, host_sdown.data());
            std::vector<float> rounded((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            const float host_scale =
                bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
            const float sig = gdn_sigmoid(host_scale);
            std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                acc += (double) snap.shared[(size_t) d];
                host_combined[(size_t) d] = (float) acc;
                host_shared[(size_t) d] = host_sdown[(size_t) d] * sig;
            }
            std::vector<float> host_write((size_t) hc_dim);
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.mid[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);

            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mix.data(), act.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());

            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            std::vector<float> gpu_route((size_t) n_expert);
            std::memcpy(gpu_route.data(), route.data(), gpu_route.size() * 4);
            route_ref(gpu_route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0, recorded_bad = 0;
            for (int i = 0; i < k; ++i) {
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
                if (snap.ids[(size_t) i] != recorded[i]) ++recorded_bad;
            }
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);

            double chain_mix_rel = std::nan("");
            double chain_logit_rel = std::nan("");
            bool chain_ran = false;
            const AttnHalf host_attn = layer0_attn_host(
                Rp, static_cast<const float*>(attn_norm->values->data()),
                static_cast<const uint16_t*>(attn_down->values->data()),
                static_cast<const uint16_t*>(attn_up->values->data()),
                static_cast<const uint16_t*>(attn_inj->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()),
                static_cast<const uint8_t*>(attn_gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()),
                static_cast<const float*>(ssm_norm->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            if (all_finite(host_attn.out.data(), hc_dim)) {
                const uint16_t* rw = static_cast<const uint16_t*>(router->values->data());
                const FfnHalf host_ffn = layer0_ffn_host(
                    host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                    static_cast<const uint16_t*>(ffn_down->values->data()),
                    static_cast<const uint16_t*>(ffn_up->values->data()),
                    static_cast<const uint16_t*>(ffn_inj->values->data()), rw, blob.data(), n_embd, n_ff, n_expert,
                    hc, hc_lr);
                if (all_finite(host_ffn.mixed.data(), n_embd) && all_finite(host_ffn.router.data(), n_expert)) {
                    std::vector<int> chain_ids((size_t) k);
                    std::vector<float> chain_w((size_t) k);
                    route_ref(host_ffn.router.data(), n_expert, k, chain_ids.data(), chain_w.data());
                    std::vector<std::vector<float>> chain_y((size_t) k, std::vector<float>((size_t) n_embd));
                    bool chain_experts = true;
                    for (int e = 0; e < k; ++e) {
                        std::vector<uint8_t> raw;
                        load_expert(chain_ids[(size_t) e], raw);
                        Bytes act_b((uint64_t) n_embd * 4);
                        std::memcpy(act_b.data(), host_ffn.mixed.data(), act_b.size());
                        Bytes cxq(iq_q8_1_bytes(n_embd));
                        iq_quantize_q8_1(act_b, n_embd, cxq);
                        std::vector<uint8_t> cxqb(cxq.size());
                        std::memcpy(cxqb.data(), cxq.data(), cxqb.size());
                        std::vector<float> cg((size_t) n_ff), cu((size_t) n_ff), hidden((size_t) n_ff);
                        q2_ref(raw.data(), cxqb.data(), n_embd, n_ff, cg.data());
                        q2_ref(raw.data() + stride, cxqb.data(), n_embd, n_ff, cu.data());
                        if (!all_finite(cg.data(), n_ff) || !all_finite(cu.data(), n_ff)) chain_experts = false;
                        for (int i = 0; i < n_ff; ++i) hidden[(size_t) i] = gdn_silu(cg[(size_t) i]) * cu[(size_t) i];
                        Bytes hf((uint64_t) n_ff * 4);
                        std::memcpy(hf.data(), hidden.data(), hf.size());
                        Bytes chq(iq_q8_1_bytes(n_ff));
                        iq_quantize_q8_1(hf, n_ff, chq);
                        std::vector<uint8_t> chqb(chq.size());
                        std::memcpy(chqb.data(), chq.data(), chqb.size());
                        q2_ref(raw.data() + stride * 2, chqb.data(), n_ff, n_embd, chain_y[(size_t) e].data());
                    }
                    Bytes sact((uint64_t) n_embd * 4);
                    std::memcpy(sact.data(), host_ffn.mixed.data(), sact.size());
                    Bytes sxq(iq_q8_1_bytes(n_embd));
                    iq_quantize_q8_1(sact, n_embd, sxq);
                    std::vector<uint8_t> sxqb(sxq.size());
                    std::memcpy(sxqb.data(), sxq.data(), sxqb.size());
                    std::vector<float> sg((size_t) n_ff), su((size_t) n_ff), shid((size_t) n_ff), sdo((size_t) n_embd);
                    xs_ref(sgate_b, sxqb.data(), n_embd, n_ff, sg.data());
                    q3_ref(sup_b, sxqb.data(), n_embd, n_ff, su.data());
                    for (int i = 0; i < n_ff; ++i) shid[(size_t) i] = gdn_silu(sg[(size_t) i]) * su[(size_t) i];
                    Bytes shf((uint64_t) n_ff * 4);
                    std::memcpy(shf.data(), shid.data(), shf.size());
                    Bytes shqq(iq_q8_1_bytes(n_ff));
                    iq_quantize_q8_1(shf, n_ff, shqq);
                    std::vector<uint8_t> shqb(shqq.size());
                    std::memcpy(shqb.data(), shqq.data(), shqb.size());
                    q50_ref(sdown_b, shqb.data(), n_ff, n_embd, sdo.data());
                    std::vector<float> sr((size_t) n_embd);
                    for (int i = 0; i < n_embd; ++i) sr[(size_t) i] = round_bf16(host_ffn.mixed[(size_t) i]);
                    const float cs = bf16_dot(sr.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
                    const float csig = gdn_sigmoid(cs);
                    std::vector<float> chain_shared((size_t) n_embd), chain_block((size_t) n_embd);
                    for (int d = 0; d < n_embd; ++d) {
                        chain_shared[(size_t) d] = sdo[(size_t) d] * csig;
                        double acc = (double) chain_shared[(size_t) d];
                        for (int e = 0; e < k; ++e)
                            acc += (double) chain_w[(size_t) e] * (double) chain_y[(size_t) e][(size_t) d];
                        chain_block[(size_t) d] = (float) acc;
                    }
                    const float* src = host_attn.out.data();
                    const float* norm_w = static_cast<const float*>(ffn_norm->values->data());
                    const uint16_t* iw = static_cast<const uint16_t*>(ffn_inj->values->data());
                    std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim), chain_res((size_t) hc_dim);
                    for (int c = 0; c < hc; ++c) {
                        double ms = 0;
                        for (int d = 0; d < n_embd; ++d) {
                            const double v = src[(size_t) c * n_embd + d];
                            ms += v * v;
                        }
                        ms /= (double) n_embd;
                        const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                        for (int d = 0; d < n_embd; ++d) {
                            const size_t i = (size_t) c * n_embd + d;
                            xn[i] = src[i] * rs * norm_w[i];
                        }
                    }
                    for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                    for (int c = 0; c < hc; ++c) {
                        const float cinj = bf16_dot(act.data(), iw + (size_t) c * hc_dim, hc_dim);
                        const float scale = 2.0f * gdn_sigmoid(cinj / (float) hc);
                        for (int d = 0; d < n_embd; ++d)
                            chain_res[(size_t) c * n_embd + d] =
                                src[(size_t) c * n_embd + d] + chain_block[(size_t) d] * scale;
                    }
                    if (chain_experts && all_finite(chain_res.data(), hc_dim)) {
                        const std::vector<float> chain_mix =
                            output_mix_host(chain_res.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
                        if (all_finite(chain_mix.data(), n_embd)) {
                            chain_mix_rel = rel_l1(chain_mix.data(), static_cast<const float*>(mix_y.data()), n_embd);
                            Bytes chain_act((uint64_t) n_embd * 4);
                            std::memcpy(chain_act.data(), chain_mix.data(), chain_act.size());
                            Bytes chain_q(iq_q8_1_bytes(n_embd));
                            iq_quantize_q8_1(chain_act, n_embd, chain_q);
                            std::vector<uint8_t> chain_qb(chain_q.size());
                            std::memcpy(chain_qb.data(), chain_q.data(), chain_qb.size());
                            std::vector<float> chain_logits((size_t) n_vocab);
                            k_ref(out_bytes, chain_qb.data(), n_embd, n_vocab, true, chain_logits.data());
                            if (all_finite(chain_logits.data(), n_vocab)) {
                                chain_logit_rel =
                                    rel_l1(chain_logits.data(), static_cast<const float*>(logits.data()), n_vocab);
                                chain_ran = true;
                            }
                        }
                    }
                }
            }
            (void) chain_ran;

            run_pass(R, layer_y2, true, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, true, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);
            run_pass(R, layer_drop, false, nullptr);

            std::vector<uint8_t> row_copy(emb_row_n);
            auto launch_flip = [&](int local, bool with_logits) {
                std::memcpy(row_copy.data(), emb_src, emb_row_n);
                row_copy[(size_t) local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                auto* Rfp = static_cast<float*>(Rf.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
                run_pass(Rf, layer_yf, true, nullptr);
                mix_read(layer_yf, mix_yf);
                if (with_logits) project(mix_yf, logits_f);
            };
            int flip_local = 32;
            launch_flip(flip_local, true);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flipped_n = differ_floats(lp, lfp, n_vocab);
            if (flipped_n == 0) {
                for (int i = 0; i < (int) emb_row_n; ++i) {
                    if (i == 32) continue;
                    launch_flip(i, false);
                    if (differ_floats(layer_p, layerf_p, hc_dim) == 0) continue;
                    project(mix_yf, logits_f);
                    flipped_n = differ_floats(lp, lfp, n_vocab);
                    if (flipped_n > 0) {
                        flip_local = i;
                        break;
                    }
                }
            }
            const int flip_at = (int) ((size_t) token * emb_row_n + (size_t) flip_local);
            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mzp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && combine_guard && shared_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.shared.data(), n_embd) &&
                all_finite(snap.combined.data(), n_embd) && all_finite(snap.layer.data(), hc_dim) &&
                all_finite(snap.mid.data(), hc_dim) && all_finite(host_shared.data(), n_embd) &&
                all_finite(host_combined.data(), n_embd) && all_finite(host_write.data(), hc_dim) &&
                all_finite(host_mix.data(), n_embd) && all_finite(host_logits.data(), n_vocab) &&
                all_finite(scalar_logits.data(), n_vocab) && all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) &&
                all_finite(lzp, n_vocab) && all_finite(lfp, n_vocab) && all_finite(mp, n_embd) &&
                all_finite(layer_p, hc_dim) && all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) &&
                all_finite(layerf_p, hc_dim) && std::isfinite(snap.scale) && std::isfinite(host_scale) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double gate_rel = rel_l1(host_sgate.data(), snap.sgate.data(), n_ff);
            const double up_rel = rel_l1(host_sup.data(), snap.sup.data(), n_ff);
            const double down_rel = rel_l1(host_sdown.data(), snap.sdown.data(), n_embd);
            const double scale_rel = rel_l1(&host_scale, &snap.scale, 1);
            const double shared_rel = rel_l1(host_shared.data(), snap.shared.data(), n_embd);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            rt_mix_ok = live && guard_held && inject_held && id_bad == 0 && recorded_bad == 0 &&
                        route_w_rel <= 1e-5 && egate_rel <= 1e-5 && eup_rel <= 1e-5 && expert_rel <= 1e-5 &&
                        gate_rel <= 1e-5 && up_rel <= 1e-5 && down_rel <= 1e-5 && scale_rel <= 1e-4 &&
                        shared_rel <= 1e-4 && combine_rel <= 1e-5 && write_rel <= 1e-4 && mix_rel <= 1e-4;
            rt_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            rt_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            rt_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            rt_flipped = live && guard_held && inject_held && flipped_n > 0;
            rt_dropped = live && guard_held && inject_held && all_finite(drop_p, hc_dim) && drop_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "k %d, expert %.3e, gate %.3e, up %.3e, down %.3e, scale %.3e, shared %.3e, "
                          "combine %.3e, write %.3e, mix %.3e, chain mix %.3e, experts %s",
                          k, expert_rel, gate_rel, up_rel, down_rel, scale_rel, shared_rel, combine_rel, write_rel,
                          mix_rel, chain_mix_rel, id_list.c_str());
            rt_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain %.3e", rel, irel, n_vocab,
                          chain_logit_rel);
            rt_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            rt_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            rt_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_vocab, flip_at);
            rt_flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            rt_drop_detail = buf;
        } catch (const MetalError& e) {
            rt_mix_detail = e.what();
            rt_logits_detail = e.what();
            rt_repeat_detail = e.what();
            rt_move_detail = e.what();
            rt_flip_detail = e.what();
            rt_drop_detail = e.what();
        }
    }
    verdict("the routed layer mix matches", bytes && rt_mix_ok, rt_mix_detail);
    verdict("the routed layer logits match", bytes && rt_logits_ok, rt_logits_detail);
    verdict("that routed layer head repeats", bytes && rt_repeat, rt_repeat_detail);
    verdict("a zero residual moves that routed layer", bytes && rt_moved, rt_move_detail);
    verdict("a changed embedding byte moves that routed layer", bytes && rt_flipped, rt_flip_detail);
    verdict("dropping the shared expert moves that route", bytes && rt_dropped, rt_drop_detail);
}

// Token 1's layer-0 residual, copied out for the layer-1 check. Empty unless that pass finished.
std::vector<float> carried_token1_residual;
// Layer 1's attention residual, copied out for the layer-1 feed-forward. PLE was not applied.
std::vector<float> layer1_attn_residual;
// Layer 1's feed-forward residual, copied out for the next layer. PLE was not applied.
std::vector<float> layer1_ffn_residual;
// Layer 2's attention residual, copied out for the layer-2 feed-forward. Layer 1's PLE was not applied.
std::vector<float> layer2_attn_residual;
// Layer 2's feed-forward residual. The shared expert was not added. Its down plane is Q4_0.
std::vector<float> layer2_ffn_residual;
// Layer 2's feed-forward residual with the shared expert added. Layer 1's PLE was not applied.
std::vector<float> layer2_shared_residual;
// Layer 3's attention residual. Position 0, cache empty. Layer 1's PLE was not applied.
std::vector<float> layer3_attn_residual;
// Layer 3's feed-forward residual. The shared expert was not added.
std::vector<float> layer3_ffn_residual;
// Layer 3's feed-forward residual with the shared expert added. Layer 1's PLE was not applied.
std::vector<float> layer3_shared_residual;

// Token 7971 from a zero state, then token 1 from that convolution and recurrent state.
// Token 1 is a fixed row. It is not an argmax. The other 47 layers are not loaded.
void run_carried_pair(const PackTable& table, const strata::core::ModelGeometry& g,
                      const std::vector<uint8_t>& blob) {
    std::printf("\ntoken 7971 state carried into token 1\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int token_a = 7971;
    const int token_b = 1;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 746117120;
    const uint64_t k_gate = 983173120;
    const uint64_t k_up = 1222599680;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t emb_row_n = (size_t) (n_embd / 256) * 110;
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    auto need = [&](const char* name, uint64_t nbytes) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->values->size() != nbytes) return nullptr;
        return t;
    };
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* qkv = need("blk.0.attn_qkv.weight", 13926400);
    const PackTensor* attn_gate = need("blk.0.attn_gate.weight", 8847360);
    const PackTensor* ssm = need("blk.0.ssm_out.weight", 10813440);
    const PackTensor* conv_w = need("blk.0.ssm_conv1d.weight", 163840);
    const PackTensor* alpha_w = need("blk.0.ssm_alpha.weight", 245760);
    const PackTensor* beta_w = need("blk.0.ssm_beta.weight", 245760);
    const PackTensor* ssm_a = need("blk.0.ssm_a", 192);
    const PackTensor* dt_w = need("blk.0.ssm_dt.bias", 192);
    const PackTensor* ssm_norm = need("blk.0.ssm_norm.weight", 512);
    const PackTensor* attn_down = need("blk.0.hc_attn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* attn_up = need("blk.0.hc_attn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* attn_inj = need("blk.0.hc_attn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* attn_norm = need("blk.0.hc_attn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* ffn_down = need("blk.0.hc_ffn_down.weight", (uint64_t) hc_lr * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_up = need("blk.0.hc_ffn_up.weight", (uint64_t) hc_dim * (uint64_t) hc_lr * 2);
    const PackTensor* ffn_inj = need("blk.0.hc_ffn_inject.weight", (uint64_t) hc * (uint64_t) hc_dim * 2);
    const PackTensor* ffn_norm = need("blk.0.hc_ffn_norm.weight", (uint64_t) hc_dim * 4);
    const PackTensor* router = need("blk.0.ffn_gate_inp.weight", (uint64_t) n_embd * (uint64_t) n_expert * 2);
    const PackTensor* sgate = plane("blk.0.ffn_gate_shexp.weight", 870400, n_embd, n_ff);
    const PackTensor* sup = plane("blk.0.ffn_up_shexp.weight", 704000, n_embd, n_ff);
    const PackTensor* sdown = plane("blk.0.ffn_down_shexp.weight", 1126400, n_ff, n_embd);
    const PackTensor* ginp = table.find("blk.0.ffn_gate_inp_shexp.weight");
    const bool ginp_ok = ginp && ginp->values && ginp->kind == strata::core::WeightKind::Bf16InF32 &&
                         ginp->code_bits == 0 && ginp->ne0 == n_embd && ginp->ne1 == 1 &&
                         ginp->values->size() == (uint64_t) n_embd * 2;
    const PackTensor* emb = plane("token_embd.weight", 273152000, n_embd, n_vocab);
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && std::filesystem::is_regular_file(shard) && blob.size() == 1382400 && qkv &&
                       qkv->code_bits == 0 && attn_gate && ssm && conv_w && alpha_w && beta_w && ssm_a && dt_w &&
                       ssm_norm && attn_down && attn_up && attn_inj && attn_norm && ffn_down && ffn_up && ffn_inj &&
                       ffn_norm && router && sgate && sup && sdown && ginp_ok && emb && head_ok && n_expert == 512 &&
                       hc > 0;
    bool st_ok = false;
    bool cp_mix_ok = false;
    bool cp_logits_ok = false;
    bool cp_repeat = false;
    bool cp_cleared = false;
    bool cp_flipped = false;
    bool cp_dropped = false;
    std::string st_detail = "bytes";
    std::string cp_mix_detail = "bytes";
    std::string cp_logits_detail = "bytes";
    std::string cp_repeat_detail = "bytes";
    std::string cp_clear_detail = "bytes";
    std::string cp_flip_detail = "bytes";
    std::string cp_drop_detail = "bytes";
    if (bytes) {
        try {
            auto row_of = [&](int token, std::vector<float>& dst) {
                const uint8_t* src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) token * emb_row_n;
                dst.resize((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(src + (size_t) b * 110, dst.data() + (size_t) b * 256);
                double ms = 0;
                for (int i = 0; i < n_embd; ++i) {
                    if (!std::isfinite(dst[(size_t) i])) throw MetalError("embedding row is not finite");
                    ms += (double) dst[(size_t) i] * (double) dst[(size_t) i];
                }
                if (ms == 0) throw MetalError("embedding row is zero");
            };
            std::vector<float> row_a, row_b;
            row_of(token_a, row_a);
            row_of(token_b, row_b);
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, attn_gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = ssm_norm->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const int state_n = (int) lg.S * (int) lg.h_v * (int) lg.S;
            const int conv_n = (int) g.ssm_conv_channels * (int) (lg.d_conv - 1);
            Bytes state((uint64_t) state_n * 4), conv((uint64_t) conv_n * 4);
            auto tile = [&](const std::vector<float>& row, Bytes& dst) {
                auto* p = static_cast<float*>(dst.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(p + (size_t) c * n_embd, row.data(), (size_t) n_embd * sizeof(float));
            };
            Bytes Ra((uint64_t) hc_dim * 4), Rb((uint64_t) hc_dim * 4);
            tile(row_a, Ra);
            tile(row_b, Rb);
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes mid((uint64_t) guard_n * 4), gdn_y((uint64_t) n_embd * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yc((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff)), expert_y((uint64_t) (n_embd + guard) * 4);
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes shared_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            auto attn = [&](const Bytes& residual, bool zero, Bytes& out) {
                if (zero) {
                    std::memset(state.data(), 0, (size_t) state.size());
                    std::memset(conv.data(), 0, (size_t) conv.size());
                }
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(gdn_y.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *attn_norm->values, *attn_down->values, *attn_up->values, attn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, gdn_y);
                gr_write(gs, residual, gdn_y, inject, out);
            };
            struct FfnSnap {
                double egate_rel = 0, eup_rel = 0, expert_rel = 0;
                double gate_rel = 0, up_rel = 0, down_rel = 0, scale_rel = 0, shared_rel = 0;
                double combine_rel = 0, write_rel = 0, route_w_rel = 0;
                std::vector<int> ids;
                std::vector<float> route_w;
                float scale = 0;
                int id_bad = 0;
            };
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kIq4Xs, *sgate->values, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ3_K, *sup->values, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ5_0, *sdown->values, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, *ginp->values, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto ffn = [&](const Bytes& residual, bool with_shared, Bytes& out, FfnSnap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                std::vector<uint8_t> exqb;
                std::vector<std::vector<float>> edown;
                if (snap) {
                    exqb.resize(exq.size());
                    std::memcpy(exqb.data(), exq.data(), exqb.size());
                    edown.assign((size_t) k, {});
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                        std::vector<uint8_t> hqb(ehq.size());
                        std::memcpy(hqb.data(), ehq.data(), hqb.size());
                        q2_ref(raw.data(), exqb.data(), n_embd, n_ff, hg.data());
                        q2_ref(raw.data() + stride, exqb.data(), n_embd, n_ff, hu.data());
                        q2_ref(raw.data() + stride * 2, hqb.data(), n_ff, n_embd, hd.data());
                        snap->egate_rel = std::max(snap->egate_rel, rel_l1(hg.data(), static_cast<const float*>(egate_y.data()), n_ff));
                        snap->eup_rel = std::max(snap->eup_rel, rel_l1(hu.data(), static_cast<const float*>(eup_y.data()), n_ff));
                        snap->expert_rel = std::max(snap->expert_rel, rel_l1(hd.data(), static_cast<const float*>(expert_y.data()), n_embd));
                        edown[(size_t) e].resize((size_t) n_embd);
                        std::memcpy(edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                    }
                }
                shared_forward(shared_y);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (!snap) return;
                const uint8_t* sgate_b = static_cast<const uint8_t*>(sgate->values->data());
                const uint8_t* sup_b = static_cast<const uint8_t*>(sup->values->data());
                const uint8_t* sdown_b = static_cast<const uint8_t*>(sdown->values->data());
                std::vector<uint8_t> sx(sh_xq.size()), shq(sh_hq.size());
                std::memcpy(sx.data(), sh_xq.data(), sx.size());
                std::memcpy(shq.data(), sh_hq.data(), shq.size());
                std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
                xs_ref(sgate_b, sx.data(), n_embd, n_ff, host_sgate.data());
                q3_ref(sup_b, sx.data(), n_embd, n_ff, host_sup.data());
                q50_ref(sdown_b, shq.data(), n_ff, n_embd, host_sdown.data());
                std::vector<float> ffn_mixed((size_t) n_embd), rounded((size_t) n_embd);
                std::memcpy(ffn_mixed.data(), mixed.data(), (size_t) n_embd * 4);
                for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(ffn_mixed[(size_t) i]);
                const float host_scale = bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
                const float sig = gdn_sigmoid(host_scale);
                const float gpu_scale = static_cast<const float*>(sh_scalar.data())[0];
                std::vector<float> gpu_shared((size_t) n_embd), gpu_combined((size_t) n_embd), gpu_inj((size_t) hc);
                std::vector<float> gpu_mid((size_t) hc_dim), gpu_layer((size_t) hc_dim);
                std::memcpy(gpu_shared.data(), shared_y.data(), (size_t) n_embd * 4);
                std::memcpy(gpu_combined.data(), combined.data(), (size_t) n_embd * 4);
                std::memcpy(gpu_inj.data(), inject.data(), (size_t) hc * 4);
                std::memcpy(gpu_mid.data(), residual.data(), (size_t) hc_dim * 4);
                std::memcpy(gpu_layer.data(), out.data(), (size_t) hc_dim * 4);
                std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
                std::vector<float> gpu_w((size_t) k);
                std::memcpy(gpu_w.data(), route_w.data(), (size_t) k * 4);
                for (int d = 0; d < n_embd; ++d) {
                    double acc = (double) gpu_shared[(size_t) d];
                    for (int e = 0; e < k; ++e) acc += (double) gpu_w[(size_t) e] * (double) edown[(size_t) e][(size_t) d];
                    host_combined[(size_t) d] = (float) acc;
                    host_shared[(size_t) d] = host_sdown[(size_t) d] * sig;
                }
                std::vector<float> host_write((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    const float scale = 2.0f * gdn_sigmoid(gpu_inj[(size_t) c] / (float) hc);
                    for (int d = 0; d < n_embd; ++d)
                        host_write[(size_t) c * n_embd + d] = gpu_mid[(size_t) c * n_embd + d] + gpu_combined[(size_t) d] * scale;
                }
                std::vector<int> host_ids((size_t) k);
                std::vector<float> host_rw((size_t) k), gpu_route((size_t) n_expert);
                std::memcpy(gpu_route.data(), route.data(), (size_t) n_expert * 4);
                route_ref(gpu_route.data(), n_expert, k, host_ids.data(), host_rw.data());
                snap->id_bad = 0;
                for (int i = 0; i < k; ++i)
                    if (idp[i] != host_ids[(size_t) i]) ++snap->id_bad;
                snap->ids.assign(idp, idp + k);
                snap->route_w = gpu_w;
                snap->route_w_rel = rel_l1(host_rw.data(), gpu_w.data(), k);
                snap->gate_rel = rel_l1(host_sgate.data(), static_cast<const float*>(sh_gate_y.data()), n_ff);
                snap->up_rel = rel_l1(host_sup.data(), static_cast<const float*>(sh_up_y.data()), n_ff);
                snap->down_rel = rel_l1(host_sdown.data(), static_cast<const float*>(sh_down_y.data()), n_embd);
                snap->scale_rel = rel_l1(&host_scale, &gpu_scale, 1);
                snap->shared_rel = rel_l1(host_shared.data(), gpu_shared.data(), n_embd);
                snap->combine_rel = rel_l1(host_combined.data(), gpu_combined.data(), n_embd);
                snap->write_rel = rel_l1(host_write.data(), gpu_layer.data(), hc_dim);
                snap->scale = gpu_scale;
            };
            auto copy_state = [&](std::vector<float>& dst_state, std::vector<float>& dst_conv) {
                dst_state.resize((size_t) state_n);
                dst_conv.resize((size_t) conv_n);
                std::memcpy(dst_state.data(), state.data(), (size_t) state_n * 4);
                std::memcpy(dst_conv.data(), conv.data(), (size_t) conv_n * 4);
            };
            auto restore_state = [&](const std::vector<float>& src_state, const std::vector<float>& src_conv) {
                std::memcpy(state.data(), src_state.data(), (size_t) state_n * 4);
                std::memcpy(conv.data(), src_conv.data(), (size_t) conv_n * 4);
            };
            attn(Ra, true, mid);
            std::vector<float> state_a, conv_a, mixed_a((size_t) n_embd);
            copy_state(state_a, conv_a);
            std::memcpy(mixed_a.data(), mixed.data(), (size_t) n_embd * 4);
            std::vector<float> host_state((size_t) state_n, 0.f), host_conv((size_t) conv_n, 0.f);
            gdn_iq_host(mixed_a.data(), static_cast<const uint8_t*>(qkv->values->data()),
                        static_cast<const uint8_t*>(attn_gate->values->data()),
                        static_cast<const uint8_t*>(ssm->values->data()),
                        static_cast<const uint16_t*>(alpha_w->values->data()),
                        static_cast<const uint16_t*>(beta_w->values->data()),
                        static_cast<const float*>(conv_w->values->data()),
                        static_cast<const float*>(ssm_norm->values->data()),
                        static_cast<const float*>(dt_w->values->data()),
                        static_cast<const float*>(ssm_a->values->data()), n_embd, (int) lg.S, (int) lg.h_k,
                        (int) lg.h_v, (int) lg.d_conv, &host_conv, &host_state);
            const double state_rel = rel_l1(host_state.data(), state_a.data(), state_n);
            const double conv_rel = rel_l1(host_conv.data(), conv_a.data(), conv_n);
            std::vector<float> zeros_s((size_t) state_n, 0.f), zeros_c((size_t) conv_n, 0.f);
            const int state_moved = differ_floats(state_a.data(), zeros_s.data(), state_n);
            const int conv_moved = differ_floats(conv_a.data(), zeros_c.data(), conv_n);

            attn(Rb, false, mid);
            std::vector<float> mixed_b((size_t) n_embd), gdn_b((size_t) n_embd);
            std::memcpy(mixed_b.data(), mixed.data(), (size_t) n_embd * 4);
            std::memcpy(gdn_b.data(), gdn_y.data(), (size_t) n_embd * 4);
            std::vector<float> step_state = state_a, step_conv = conv_a;
            const std::vector<float> host_block = gdn_iq_host(
                mixed_b.data(), static_cast<const uint8_t*>(qkv->values->data()),
                static_cast<const uint8_t*>(attn_gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()),
                static_cast<const float*>(ssm_norm->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv, &step_conv, &step_state);
            const double block_rel = rel_l1(host_block.data(), gdn_b.data(), n_embd);
            FfnSnap snap;
            ffn(mid, true, layer_y, &snap);
            carried_token1_residual.assign(static_cast<const float*>(layer_y.data()),
                                           static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            std::vector<float> layer_live((size_t) hc_dim);
            std::memcpy(layer_live.data(), layer_y.data(), (size_t) hc_dim * 4);
            const std::vector<float> host_mix =
                output_mix_host(layer_live.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yc((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_c((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yc);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_c);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mix.data(), act.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            const double mix_rel = rel_l1(host_mix.data(), static_cast<const float*>(mix_y.data()), n_embd);
            const double rel = rel_l1(scalar_logits.data(), static_cast<const float*>(logits.data()), n_vocab);
            const double irel = rel_l1(host_logits.data(), static_cast<const float*>(logits.data()), n_vocab);

            double chain_mix_rel = std::nan("");
            double chain_logit_rel = std::nan("");
            bool chain_ran = false;
            const AttnHalf host_attn = layer0_attn_host(
                static_cast<const float*>(Ra.data()), static_cast<const float*>(attn_norm->values->data()),
                static_cast<const uint16_t*>(attn_down->values->data()),
                static_cast<const uint16_t*>(attn_up->values->data()),
                static_cast<const uint16_t*>(attn_inj->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()),
                static_cast<const uint8_t*>(attn_gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()),
                static_cast<const float*>(ssm_norm->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            if (all_finite(host_attn.mixed.data(), n_embd)) {
                std::vector<float> chain_state((size_t) state_n, 0.f), chain_conv((size_t) conv_n, 0.f);
                gdn_iq_host(host_attn.mixed.data(), static_cast<const uint8_t*>(qkv->values->data()),
                            static_cast<const uint8_t*>(attn_gate->values->data()),
                            static_cast<const uint8_t*>(ssm->values->data()),
                            static_cast<const uint16_t*>(alpha_w->values->data()),
                            static_cast<const uint16_t*>(beta_w->values->data()),
                            static_cast<const float*>(conv_w->values->data()),
                            static_cast<const float*>(ssm_norm->values->data()),
                            static_cast<const float*>(dt_w->values->data()),
                            static_cast<const float*>(ssm_a->values->data()), n_embd, (int) lg.S, (int) lg.h_k,
                            (int) lg.h_v, (int) lg.d_conv, &chain_conv, &chain_state);
                const AttnHalf host_b = layer0_attn_host(
                    static_cast<const float*>(Rb.data()), static_cast<const float*>(attn_norm->values->data()),
                    static_cast<const uint16_t*>(attn_down->values->data()),
                    static_cast<const uint16_t*>(attn_up->values->data()),
                    static_cast<const uint16_t*>(attn_inj->values->data()),
                    static_cast<const uint8_t*>(qkv->values->data()),
                    static_cast<const uint8_t*>(attn_gate->values->data()),
                    static_cast<const uint8_t*>(ssm->values->data()),
                    static_cast<const uint16_t*>(alpha_w->values->data()),
                    static_cast<const uint16_t*>(beta_w->values->data()),
                    static_cast<const float*>(conv_w->values->data()),
                    static_cast<const float*>(ssm_norm->values->data()),
                    static_cast<const float*>(dt_w->values->data()),
                    static_cast<const float*>(ssm_a->values->data()), n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k,
                    (int) lg.h_v, (int) lg.d_conv);
                if (all_finite(host_b.mixed.data(), n_embd)) {
                    const std::vector<float> chain_block = gdn_iq_host(
                        host_b.mixed.data(), static_cast<const uint8_t*>(qkv->values->data()),
                        static_cast<const uint8_t*>(attn_gate->values->data()),
                        static_cast<const uint8_t*>(ssm->values->data()),
                        static_cast<const uint16_t*>(alpha_w->values->data()),
                        static_cast<const uint16_t*>(beta_w->values->data()),
                        static_cast<const float*>(conv_w->values->data()),
                        static_cast<const float*>(ssm_norm->values->data()),
                        static_cast<const float*>(dt_w->values->data()),
                        static_cast<const float*>(ssm_a->values->data()), n_embd, (int) lg.S, (int) lg.h_k,
                        (int) lg.h_v, (int) lg.d_conv, &chain_conv, &chain_state);
                    const float* src_b = static_cast<const float*>(Rb.data());
                    const float* anorm = static_cast<const float*>(attn_norm->values->data());
                    const uint16_t* ainj = static_cast<const uint16_t*>(attn_inj->values->data());
                    std::vector<float> axn((size_t) hc_dim), aact((size_t) hc_dim), chain_attn((size_t) hc_dim);
                    for (int c = 0; c < hc; ++c) {
                        double ms = 0;
                        for (int d = 0; d < n_embd; ++d) {
                            const double v = src_b[(size_t) c * n_embd + d];
                            ms += v * v;
                        }
                        ms /= (double) n_embd;
                        const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                        for (int d = 0; d < n_embd; ++d) {
                            const size_t i = (size_t) c * n_embd + d;
                            axn[i] = src_b[i] * rs * anorm[i];
                        }
                    }
                    for (int i = 0; i < hc_dim; ++i) aact[(size_t) i] = round_bf16(axn[(size_t) i]);
                    if (all_finite(chain_block.data(), n_embd)) {
                        for (int c = 0; c < hc; ++c) {
                            const float cinj = bf16_dot(aact.data(), ainj + (size_t) c * hc_dim, hc_dim);
                            const float scale = 2.0f * gdn_sigmoid(cinj / (float) hc);
                            for (int d = 0; d < n_embd; ++d)
                                chain_attn[(size_t) c * n_embd + d] =
                                    src_b[(size_t) c * n_embd + d] + chain_block[(size_t) d] * scale;
                        }
                        const uint16_t* rw = static_cast<const uint16_t*>(router->values->data());
                        const FfnHalf host_ffn = layer0_ffn_host(
                            chain_attn.data(), static_cast<const float*>(ffn_norm->values->data()),
                            static_cast<const uint16_t*>(ffn_down->values->data()),
                            static_cast<const uint16_t*>(ffn_up->values->data()),
                            static_cast<const uint16_t*>(ffn_inj->values->data()), rw, blob.data(), n_embd, n_ff,
                            n_expert, hc, hc_lr);
                        if (all_finite(host_ffn.mixed.data(), n_embd) && all_finite(host_ffn.router.data(), n_expert) &&
                            all_finite(chain_attn.data(), hc_dim)) {
                            std::vector<int> chain_ids((size_t) k);
                            std::vector<float> chain_w((size_t) k);
                            route_ref(host_ffn.router.data(), n_expert, k, chain_ids.data(), chain_w.data());
                            std::vector<std::vector<float>> chain_y((size_t) k, std::vector<float>((size_t) n_embd));
                            bool chain_experts = true;
                            for (int e = 0; e < k; ++e) {
                                std::vector<uint8_t> raw;
                                load_expert(chain_ids[(size_t) e], raw);
                                Bytes act_b((uint64_t) n_embd * 4);
                                std::memcpy(act_b.data(), host_ffn.mixed.data(), act_b.size());
                                Bytes cxq(iq_q8_1_bytes(n_embd));
                                iq_quantize_q8_1(act_b, n_embd, cxq);
                                std::vector<uint8_t> cxqb(cxq.size());
                                std::memcpy(cxqb.data(), cxq.data(), cxqb.size());
                                std::vector<float> cg((size_t) n_ff), cu((size_t) n_ff), hidden((size_t) n_ff);
                                q2_ref(raw.data(), cxqb.data(), n_embd, n_ff, cg.data());
                                q2_ref(raw.data() + stride, cxqb.data(), n_embd, n_ff, cu.data());
                                if (!all_finite(cg.data(), n_ff) || !all_finite(cu.data(), n_ff)) chain_experts = false;
                                for (int i = 0; i < n_ff; ++i) hidden[(size_t) i] = gdn_silu(cg[(size_t) i]) * cu[(size_t) i];
                                Bytes hf((uint64_t) n_ff * 4);
                                std::memcpy(hf.data(), hidden.data(), hf.size());
                                Bytes chq(iq_q8_1_bytes(n_ff));
                                iq_quantize_q8_1(hf, n_ff, chq);
                                std::vector<uint8_t> chqb(chq.size());
                                std::memcpy(chqb.data(), chq.data(), chqb.size());
                                q2_ref(raw.data() + stride * 2, chqb.data(), n_ff, n_embd, chain_y[(size_t) e].data());
                            }
                            const uint8_t* sgate_b = static_cast<const uint8_t*>(sgate->values->data());
                            const uint8_t* sup_b = static_cast<const uint8_t*>(sup->values->data());
                            const uint8_t* sdown_b = static_cast<const uint8_t*>(sdown->values->data());
                            Bytes sact((uint64_t) n_embd * 4);
                            std::memcpy(sact.data(), host_ffn.mixed.data(), sact.size());
                            Bytes sxq(iq_q8_1_bytes(n_embd));
                            iq_quantize_q8_1(sact, n_embd, sxq);
                            std::vector<uint8_t> sxqb(sxq.size());
                            std::memcpy(sxqb.data(), sxq.data(), sxqb.size());
                            std::vector<float> sg((size_t) n_ff), su((size_t) n_ff), shid((size_t) n_ff), sdo((size_t) n_embd);
                            xs_ref(sgate_b, sxqb.data(), n_embd, n_ff, sg.data());
                            q3_ref(sup_b, sxqb.data(), n_embd, n_ff, su.data());
                            for (int i = 0; i < n_ff; ++i) shid[(size_t) i] = gdn_silu(sg[(size_t) i]) * su[(size_t) i];
                            Bytes shf((uint64_t) n_ff * 4);
                            std::memcpy(shf.data(), shid.data(), shf.size());
                            Bytes shqq(iq_q8_1_bytes(n_ff));
                            iq_quantize_q8_1(shf, n_ff, shqq);
                            std::vector<uint8_t> shqb(shqq.size());
                            std::memcpy(shqb.data(), shqq.data(), shqb.size());
                            q50_ref(sdown_b, shqb.data(), n_ff, n_embd, sdo.data());
                            std::vector<float> sr((size_t) n_embd);
                            for (int i = 0; i < n_embd; ++i) sr[(size_t) i] = round_bf16(host_ffn.mixed[(size_t) i]);
                            const float cs = bf16_dot(sr.data(), static_cast<const uint16_t*>(ginp->values->data()), n_embd);
                            const float csig = gdn_sigmoid(cs);
                            std::vector<float> chain_block_ffn((size_t) n_embd);
                            for (int d = 0; d < n_embd; ++d) {
                                double acc = (double) sdo[(size_t) d] * (double) csig;
                                for (int e = 0; e < k; ++e)
                                    acc += (double) chain_w[(size_t) e] * (double) chain_y[(size_t) e][(size_t) d];
                                chain_block_ffn[(size_t) d] = (float) acc;
                            }
                            const float* fnorm = static_cast<const float*>(ffn_norm->values->data());
                            const uint16_t* finj = static_cast<const uint16_t*>(ffn_inj->values->data());
                            std::vector<float> fxn((size_t) hc_dim), fact((size_t) hc_dim), chain_res((size_t) hc_dim);
                            for (int c = 0; c < hc; ++c) {
                                double ms = 0;
                                for (int d = 0; d < n_embd; ++d) {
                                    const double v = chain_attn[(size_t) c * n_embd + d];
                                    ms += v * v;
                                }
                                ms /= (double) n_embd;
                                const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                                for (int d = 0; d < n_embd; ++d) {
                                    const size_t i = (size_t) c * n_embd + d;
                                    fxn[i] = chain_attn[i] * rs * fnorm[i];
                                }
                            }
                            for (int i = 0; i < hc_dim; ++i) fact[(size_t) i] = round_bf16(fxn[(size_t) i]);
                            for (int c = 0; c < hc; ++c) {
                                const float cinj = bf16_dot(fact.data(), finj + (size_t) c * hc_dim, hc_dim);
                                const float scale = 2.0f * gdn_sigmoid(cinj / (float) hc);
                                for (int d = 0; d < n_embd; ++d)
                                    chain_res[(size_t) c * n_embd + d] =
                                        chain_attn[(size_t) c * n_embd + d] + chain_block_ffn[(size_t) d] * scale;
                            }
                            if (chain_experts && all_finite(chain_res.data(), hc_dim)) {
                                const std::vector<float> chain_mix = output_mix_host(
                                    chain_res.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
                                if (all_finite(chain_mix.data(), n_embd)) {
                                    chain_mix_rel = rel_l1(chain_mix.data(), static_cast<const float*>(mix_y.data()), n_embd);
                                    Bytes chain_act((uint64_t) n_embd * 4);
                                    std::memcpy(chain_act.data(), chain_mix.data(), chain_act.size());
                                    Bytes chain_q(iq_q8_1_bytes(n_embd));
                                    iq_quantize_q8_1(chain_act, n_embd, chain_q);
                                    std::vector<uint8_t> chain_qb(chain_q.size());
                                    std::memcpy(chain_qb.data(), chain_q.data(), chain_qb.size());
                                    std::vector<float> chain_logits((size_t) n_vocab);
                                    k_ref(out_bytes, chain_qb.data(), n_embd, n_vocab, true, chain_logits.data());
                                    if (all_finite(chain_logits.data(), n_vocab)) {
                                        chain_logit_rel = rel_l1(chain_logits.data(), static_cast<const float*>(logits.data()), n_vocab);
                                        chain_ran = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (void) chain_ran;

            attn(Ra, true, mid);
            attn(Rb, false, mid);
            ffn(mid, true, layer_y2, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);

            attn(Rb, true, mid);
            ffn(mid, true, layer_yc, nullptr);
            mix_read(layer_yc, mix_yc);
            project(mix_yc, logits_c);

            const uint8_t* emb_src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) token_b * emb_row_n;
            std::vector<uint8_t> row_copy(emb_row_n);
            auto launch_flip = [&](int local, bool with_logits) {
                restore_state(state_a, conv_a);
                std::memcpy(row_copy.data(), emb_src, emb_row_n);
                row_copy[(size_t) local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                tile(flipped_row, Rf);
                attn(Rf, false, mid);
                ffn(mid, true, layer_yf, nullptr);
                mix_read(layer_yf, mix_yf);
                if (with_logits) project(mix_yf, logits_f);
            };
            int flip_local = 32;
            launch_flip(flip_local, true);
            restore_state(state_a, conv_a);
            attn(Rb, false, mid);
            ffn(mid, false, layer_drop, nullptr);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerc_p = static_cast<const float*>(layer_yc.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mcp = static_cast<const float*>(mix_yc.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lcp = static_cast<const float*>(logits_c.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flipped_n = differ_floats(lp, lfp, n_vocab);
            if (flipped_n == 0) {
                for (int i = 0; i < (int) emb_row_n; ++i) {
                    if (i == 32) continue;
                    launch_flip(i, false);
                    if (differ_floats(layer_p, layerf_p, hc_dim) == 0) continue;
                    project(mix_yf, logits_f);
                    flipped_n = differ_floats(lp, lfp, n_vocab);
                    if (flipped_n > 0) {
                        flip_local = i;
                        break;
                    }
                }
            }
            const int flip_at = (int) ((size_t) token_b * emb_row_n + (size_t) flip_local);
            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerc_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mcp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lcp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live = all_finite(state_a.data(), state_n) && all_finite(conv_a.data(), conv_n) &&
                              all_finite(host_state.data(), state_n) && all_finite(host_conv.data(), conv_n) &&
                              all_finite(gdn_b.data(), n_embd) && all_finite(host_block.data(), n_embd) &&
                              all_finite(layer_p, hc_dim) && all_finite(mp, n_embd) && all_finite(lp, n_vocab) &&
                              all_finite(l2p, n_vocab) && all_finite(lcp, n_vocab) && all_finite(lfp, n_vocab) &&
                              all_finite(host_mix.data(), n_embd) && all_finite(host_logits.data(), n_vocab) &&
                              all_finite(scalar_logits.data(), n_vocab) && std::isfinite(snap.scale) &&
                              !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lcp, n_vocab);
            const double apart_rel = rel_l1(lp, lcp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            st_ok = live && state_rel <= 1e-5 && conv_rel <= 1e-5 && state_moved > 0 && conv_moved > 0;
            cp_mix_ok = live && guard_held && inject_held && snap.id_bad == 0 && snap.route_w_rel <= 1e-5 &&
                        block_rel <= 1e-5 && snap.egate_rel <= 1e-5 && snap.eup_rel <= 1e-5 &&
                        snap.expert_rel <= 1e-5 && snap.gate_rel <= 1e-5 && snap.up_rel <= 1e-5 &&
                        snap.down_rel <= 1e-5 && snap.scale_rel <= 1e-4 && snap.shared_rel <= 1e-4 &&
                        snap.combine_rel <= 1e-5 && snap.write_rel <= 1e-4 && mix_rel <= 1e-4;
            cp_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            cp_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            cp_cleared = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            cp_flipped = live && guard_held && inject_held && flipped_n > 0;
            cp_dropped = live && guard_held && inject_held && all_finite(drop_p, hc_dim) && drop_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[640];
            std::snprintf(buf, sizeof buf, "state %.3e, conv %.3e, state moved %d, conv moved %d", state_rel, conv_rel,
                          state_moved, conv_moved);
            st_detail = buf;
            std::snprintf(buf, sizeof buf,
                          "block %.3e, expert %.3e, gate %.3e, up %.3e, down %.3e, scale %.3e, shared %.3e, "
                          "combine %.3e, write %.3e, mix %.3e, chain mix %.3e, experts %s",
                          block_rel, snap.expert_rel, snap.gate_rel, snap.up_rel, snap.down_rel, snap.scale_rel,
                          snap.shared_rel, snap.combine_rel, snap.write_rel, mix_rel, chain_mix_rel, id_list.c_str());
            cp_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain %.3e", rel, irel, n_vocab,
                          chain_logit_rel);
            cp_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            cp_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            cp_clear_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_vocab, flip_at);
            cp_flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            cp_drop_detail = buf;
        } catch (const MetalError& e) {
            st_detail = e.what();
            cp_mix_detail = e.what();
            cp_logits_detail = e.what();
            cp_repeat_detail = e.what();
            cp_clear_detail = e.what();
            cp_flip_detail = e.what();
            cp_drop_detail = e.what();
        }
    }
    verdict("the carried GDN state matches", bytes && st_ok, st_detail);
    verdict("the second token mix matches", bytes && cp_mix_ok, cp_mix_detail);
    verdict("the second token logits match", bytes && cp_logits_ok, cp_logits_detail);
    verdict("that second token repeats", bytes && cp_repeat, cp_repeat_detail);
    verdict("a cleared state moves that second token", bytes && cp_cleared, cp_clear_detail);
    verdict("a changed embedding byte moves that second token", bytes && cp_flipped, cp_flip_detail);
    verdict("dropping the shared expert moves that second token", bytes && cp_dropped, cp_drop_detail);
}

// Layer 1's GDN step on token 1's layer-0 residual. The six blk.1.ple_* tensors are not added.
// Layer 1 is the only layer that carries them, and that update sits in front of this read.
void run_layer1_gdn(const strata::core::ModelGeometry& g) {
    layer1_attn_residual.clear();
    std::printf("\nlayer 1 GDN step on token 1, PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const uint64_t k_data = 11024672;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const bool bytes = (int) carried_token1_residual.size() == hc_dim && g.ssm_state_size == 128 &&
                       g.ssm_k_heads == 16 && g.ssm_v_heads == 48 && g.ssm_d_conv == 4 &&
                       g.ssm_conv_channels == 10240 && std::filesystem::is_regular_file(shard);
    bool step_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false;
    std::string step_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes qkv = load(1505518976, 13926400);
            Bytes gate = load(1497163136, 8355840);
            Bytes ssm = load(2279044352, 8847360);
            Bytes conv_w = load(2278879808, 163840);
            Bytes alpha_w = load(2278388288, 245760);
            Bytes beta_w = load(2278634048, 245760);
            Bytes ssm_a = load(2278388096, 192);
            Bytes dt_w = load(2279043648, 192);
            Bytes ssm_norm = load(2279043840, 512);
            Bytes attn_down = load(2231243136, 6553600);
            Bytes attn_up = load(2237919616, 6553600);
            Bytes attn_inj = load(2237796736, 81920);
            Bytes attn_norm = load(2237878656, 40960);
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, &qkv};
            w.attn_gate = GdnIqPlane{kIq4Xs, &gate};
            w.ssm_out = GdnIqPlane{kQ4_K, &ssm};
            w.alpha = &alpha_w;
            w.beta = &beta_w;
            w.conv = &conv_w;
            w.ssm_norm = &ssm_norm;
            w.dt = &dt_w;
            w.ssm_a = &ssm_a;
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const int state_n = (int) lg.S * (int) lg.h_v * (int) lg.S;
            const int conv_n = (int) g.ssm_conv_channels * (int) (lg.d_conv - 1);
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            Bytes state((uint64_t) state_n * 4), conv((uint64_t) conv_n * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4), gdn_y((uint64_t) n_embd * 4);
            Bytes out((uint64_t) guard_n * 4), out2((uint64_t) guard_n * 4), outz((uint64_t) guard_n * 4);
            Bytes outf((uint64_t) guard_n * 4);
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), carried_token1_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto step = [&](const Bytes& residual, Bytes& y) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(gdn_y.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, attn_norm, attn_down, attn_up, &attn_inj, 1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, gdn_y);
                gr_write(gs, residual, gdn_y, inject, y);
            };
            step(R, out);
            layer1_attn_residual.assign(static_cast<const float*>(out.data()),
                                        static_cast<const float*>(out.data()) + hc_dim);
            std::vector<float> gpu_state((size_t) state_n), gpu_conv((size_t) conv_n), gpu_mixed((size_t) n_embd);
            std::vector<float> gpu_block((size_t) n_embd), gpu_inj((size_t) hc);
            std::memcpy(gpu_state.data(), state.data(), (size_t) state_n * 4);
            std::memcpy(gpu_conv.data(), conv.data(), (size_t) conv_n * 4);
            std::memcpy(gpu_mixed.data(), mixed.data(), (size_t) n_embd * 4);
            std::memcpy(gpu_block.data(), gdn_y.data(), (size_t) n_embd * 4);
            std::memcpy(gpu_inj.data(), inject.data(), (size_t) hc * 4);
            std::vector<float> host_state((size_t) state_n, 0.f), host_conv((size_t) conv_n, 0.f);
            const std::vector<float> host_block = gdn_iq_host(
                gpu_mixed.data(), static_cast<const uint8_t*>(qkv.data()), static_cast<const uint8_t*>(gate.data()),
                static_cast<const uint8_t*>(ssm.data()), static_cast<const uint16_t*>(alpha_w.data()),
                static_cast<const uint16_t*>(beta_w.data()), static_cast<const float*>(conv_w.data()),
                static_cast<const float*>(ssm_norm.data()), static_cast<const float*>(dt_w.data()),
                static_cast<const float*>(ssm_a.data()), n_embd, (int) lg.S, (int) lg.h_k, (int) lg.h_v,
                (int) lg.d_conv, &host_conv, &host_state, kIq4Xs, kQ4_K);
            std::vector<float> host_write((size_t) hc_dim);
            const float* Rp = static_cast<const float*>(R.data());
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(gpu_inj[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] = Rp[(size_t) c * n_embd + d] + gpu_block[(size_t) d] * scale;
            }
            const double state_rel = rel_l1(host_state.data(), gpu_state.data(), state_n);
            const double conv_rel = rel_l1(host_conv.data(), gpu_conv.data(), conv_n);
            const double block_rel = rel_l1(host_block.data(), gpu_block.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), static_cast<const float*>(out.data()), hc_dim);
            std::vector<float> zeros_s((size_t) state_n, 0.f), zeros_c((size_t) conv_n, 0.f);
            const int state_moved = differ_floats(gpu_state.data(), zeros_s.data(), state_n);
            const int conv_moved = differ_floats(gpu_conv.data(), zeros_c.data(), conv_n);
            step(R, out2);
            step(zero_r, outz);
            const int candidates[] = {32, 2, 8, 136, 1360, 588 * 1360};
            int flip_at = -1;
            int flip_n = 0;
            Bytes flipped((uint64_t) qkv.size());
            for (int local : candidates) {
                if (local < 0 || (uint64_t) local >= qkv.size()) continue;
                std::memcpy(flipped.data(), qkv.data(), qkv.size());
                static_cast<uint8_t*>(flipped.data())[local] ^= 1u;
                w.qkv = GdnIqPlane{kIq4Xs, &flipped};
                step(R, outf);
                flip_n = differ_floats(static_cast<const float*>(out.data()), static_cast<const float*>(outf.data()), hc_dim);
                if (flip_n > 0) {
                    flip_at = local;
                    break;
                }
            }
            w.qkv = GdnIqPlane{kIq4Xs, &qkv};
            const float* op = static_cast<const float*>(out.data());
            const float* o2 = static_cast<const float*>(out2.data());
            const float* oz = static_cast<const float*>(outz.data());
            const float* of = static_cast<const float*>(outf.data());
            const bool guard_held = op[hc_dim] == 7.f && o2[hc_dim] == 7.f && oz[hc_dim] == 7.f &&
                                    (flip_at < 0 || of[hc_dim] == 7.f);
            const bool live = all_finite(op, hc_dim) && all_finite(o2, hc_dim) && all_finite(oz, hc_dim) &&
                              all_finite(host_block.data(), n_embd) && all_finite(host_write.data(), hc_dim) &&
                              all_finite(gpu_state.data(), state_n) && all_finite(host_state.data(), state_n) &&
                              !still_seven(op, hc_dim);
            const int again = differ_floats(op, o2, hc_dim);
            const int apart = differ_floats(op, oz, hc_dim);
            const double apart_rel = rel_l1(op, oz, hc_dim);
            step_ok = live && guard_held && block_rel <= 1e-5 && state_rel <= 1e-5 && conv_rel <= 1e-5 &&
                      write_rel <= 1e-4 && state_moved > 0 && conv_moved > 0;
            repeat_ok = live && guard_held && again == 0;
            moved_ok = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && flip_n > 0;
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "block %.3e, state %.3e, conv %.3e, write %.3e, state moved %d, conv moved %d, PLE not applied",
                          block_rel, state_rel, conv_rel, write_rel, state_moved, conv_moved);
            step_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, hc_dim, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            step_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the layer-1 GDN step matches", bytes && step_ok, step_detail);
    verdict("that layer-1 step repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-1 step", bytes && moved_ok, move_detail);
    verdict("a changed query byte moves that layer-1 step", bytes && flipped_ok, flip_detail);
}

// Layer 1's routed experts and its Q2_0 shared expert, on the attention residual above.
// The six blk.1.ple_* tensors are not added. This is not the model residual.
void run_layer1_ffn(const PackTable& table, const strata::core::ModelGeometry& g) {
    layer1_ffn_residual.clear();
    std::printf("\nlayer 1 routed experts on the attention residual, PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 1519445376;
    const uint64_t k_down_s = 1755374976;
    const uint64_t k_gate = 1755835776;
    const uint64_t k_router = 1991765376;
    const uint64_t k_ginp = 1994386816;
    const uint64_t k_gate_s = 1994391936;
    const uint64_t k_up = 1994852736;
    const uint64_t k_up_s = 2230782336;
    const uint64_t k_ffn_down = 2244473216;
    const uint64_t k_ffn_inj = 2251026816;
    const uint64_t k_ffn_norm = 2251108736;
    const uint64_t k_ffn_up = 2251149696;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && head_ok && n_expert == 512 && hc > 0 && n_ff == 640 &&
                       (int) layer1_attn_residual.size() == hc_dim && std::filesystem::is_regular_file(shard);
    bool l1_mix_ok = false;
    bool l1_logits_ok = false;
    bool l1_repeat = false;
    bool l1_moved = false;
    bool l1_flipped = false;
    bool l1_dropped = false;
    std::string l1_mix_detail = "bytes";
    std::string l1_logits_detail = "bytes";
    std::string l1_repeat_detail = "bytes";
    std::string l1_move_detail = "bytes";
    std::string l1_flip_detail = "bytes";
    std::string l1_drop_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes ffn_down = load(k_ffn_down, 6553600);
            Bytes ffn_up = load(k_ffn_up, 6553600);
            Bytes ffn_inj = load(k_ffn_inj, 81920);
            Bytes ffn_norm = load(k_ffn_norm, 40960);
            Bytes router = load(k_router, 2621440);
            Bytes ginp = load(k_ginp, 5120);
            Bytes sgate = load(k_gate_s, 460800);
            Bytes sup = load(k_up_s, 460800);
            Bytes sdown = load(k_down_s, 460800);
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer1_attn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4);
            Bytes shared_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, residual, shared, combined, layer;
                std::vector<float> sgate, sup, sdown, route;
                std::vector<uint8_t> exq, sh_xq, sh_hq;
                float scale = 0;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kQ2_0, sgate, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ2_0, sup, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ2_0, sdown, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, ginp, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, bool with_shared, Snap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, ffn_norm, ffn_down, ffn_up, &ffn_inj, 1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, router, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                shared_forward(shared_y);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->residual.resize((size_t) hc_dim);
                    snap->shared.resize((size_t) n_embd);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->sgate.resize((size_t) n_ff);
                    snap->sup.resize((size_t) n_ff);
                    snap->sdown.resize((size_t) n_embd);
                    snap->route.resize((size_t) n_expert);
                    snap->sh_xq.resize(sh_xq.size());
                    snap->sh_hq.resize(sh_hq.size());
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->residual.data(), residual.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->shared.data(), shared_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->sgate.data(), sh_gate_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sup.data(), sh_up_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sdown.data(), sh_down_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->route.data(), route.data(), (size_t) n_expert * 4);
                    std::memcpy(snap->sh_xq.data(), sh_xq.data(), sh_xq.size());
                    std::memcpy(snap->sh_hq.data(), sh_hq.data(), sh_hq.size());
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                    snap->scale = static_cast<const float*>(sh_scalar.data())[0];
                }
            };

            Snap snap;
            run_pass(R, layer_y, true, &snap);
            layer1_ffn_residual.assign(static_cast<const float*>(layer_y.data()),
                                       static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool shared_guard = static_cast<const float*>(shared_y.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = snap.residual.data();
                const float* norm_w = static_cast<const float*>(ffn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(ffn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(ffn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(ffn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> rounded((size_t) n_embd), host_route((size_t) n_expert);
            const uint16_t* router_w = static_cast<const uint16_t*>(router.data());
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            for (int o = 0; o < n_expert; ++o)
                host_route[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);

            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
            q2_ref(static_cast<const uint8_t*>(sgate.data()), snap.sh_xq.data(), n_embd, n_ff, host_sgate.data());
            q2_ref(static_cast<const uint8_t*>(sup.data()), snap.sh_xq.data(), n_embd, n_ff, host_sup.data());
            q2_ref(static_cast<const uint8_t*>(sdown.data()), snap.sh_hq.data(), n_ff, n_embd, host_sdown.data());
            const float host_scale = bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp.data()), n_embd);
            const float sig = gdn_sigmoid(host_scale);
            std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                acc += (double) snap.shared[(size_t) d];
                host_combined[(size_t) d] = (float) acc;
                host_shared[(size_t) d] = host_sdown[(size_t) d] * sig;
            }
            std::vector<float> host_write((size_t) hc_dim);
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.residual[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            route_ref(snap.route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0;
            for (int i = 0; i < k; ++i)
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);
            const double read_rel = rel_l1(host_mixed.data(), snap.mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), snap.inject.data(), hc);
            const double router_rel = rel_l1(host_route.data(), snap.route.data(), n_expert);

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);

            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mix.data(), act.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());

            run_pass(R, layer_y2, true, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, true, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);
            run_pass(R, layer_drop, false, nullptr);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());

            int flip_at = -1;
            int flip_n = 0;
            std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
            const int flip_candidates[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256};
            Bytes Rf((uint64_t) hc_dim * 4);
            for (int local : flip_candidates) {
                if (local < 0 || local >= hc_dim * 4) continue;
                std::memcpy(Rf.data(), R.data(), Rf.size());
                static_cast<uint8_t*>(Rf.data())[local] ^= 1u;
                if (!all_finite(static_cast<const float*>(Rf.data()), hc_dim)) continue;
                run_pass(Rf, layer_yf, true, nullptr);
                if (!all_finite(static_cast<const float*>(layer_yf.data()), hc_dim)) {
                    std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                    continue;
                }
                mix_read(layer_yf, mix_yf);
                project(mix_yf, logits_f);
                if (!all_finite(lfp, n_vocab) || !all_finite(mfp, n_embd)) {
                    fill7(logits_f);
                    fill7(mix_yf);
                    std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                    continue;
                }
                flip_n = differ_floats(lp, lfp, n_vocab);
                if (flip_n > 0) {
                    flip_at = local;
                    break;
                }
            }

            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mzp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && combine_guard && shared_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.shared.data(), n_embd) &&
                all_finite(snap.combined.data(), n_embd) && all_finite(snap.layer.data(), hc_dim) &&
                all_finite(snap.residual.data(), hc_dim) && all_finite(host_mixed.data(), n_embd) &&
                all_finite(host_inject.data(), hc) && all_finite(host_route.data(), n_expert) &&
                all_finite(host_shared.data(), n_embd) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                all_finite(lfp, n_vocab) && all_finite(mp, n_embd) && all_finite(layer_p, hc_dim) &&
                all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) && all_finite(layerf_p, hc_dim) &&
                all_finite(drop_p, hc_dim) && std::isfinite(snap.scale) && std::isfinite(host_scale) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double gate_rel = rel_l1(host_sgate.data(), snap.sgate.data(), n_ff);
            const double up_rel = rel_l1(host_sup.data(), snap.sup.data(), n_ff);
            const double down_rel = rel_l1(host_sdown.data(), snap.sdown.data(), n_embd);
            const double scale_rel = rel_l1(&host_scale, &snap.scale, 1);
            const double shared_rel = rel_l1(host_shared.data(), snap.shared.data(), n_embd);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            l1_mix_ok = live && guard_held && inject_held && id_bad == 0 && route_w_rel <= 1e-5 &&
                        read_rel <= 1e-4 && inject_rel <= 1e-4 && router_rel <= 1e-5 && egate_rel <= 1e-5 &&
                        eup_rel <= 1e-5 && expert_rel <= 1e-5 && gate_rel <= 1e-5 && up_rel <= 1e-5 &&
                        down_rel <= 1e-5 && scale_rel <= 1e-4 && shared_rel <= 1e-4 && combine_rel <= 1e-5 &&
                        write_rel <= 1e-4 && mix_rel <= 1e-4;
            l1_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            l1_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            l1_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            l1_flipped = live && guard_held && inject_held && flip_n > 0;
            l1_dropped = live && guard_held && inject_held && drop_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[768];
            std::snprintf(buf, sizeof buf,
                          "k %d, read %.3e, inject %.3e, router %.3e, expert %.3e, gate %.3e, up %.3e, down %.3e, "
                          "scale %.3e, shared %.3e, combine %.3e, write %.3e, mix %.3e, chain not run, experts %s, "
                          "PLE not applied",
                          k, read_rel, inject_rel, router_rel, expert_rel, gate_rel, up_rel, down_rel, scale_rel,
                          shared_rel, combine_rel, write_rel, mix_rel, id_list.c_str());
            l1_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain not run", rel, irel, n_vocab);
            l1_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            l1_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            l1_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_vocab, flip_at);
            l1_flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            l1_drop_detail = buf;
        } catch (const MetalError& e) {
            l1_mix_detail = e.what();
            l1_logits_detail = e.what();
            l1_repeat_detail = e.what();
            l1_move_detail = e.what();
            l1_flip_detail = e.what();
            l1_drop_detail = e.what();
        }
    }
    verdict("the layer-1 routed mix matches", bytes && l1_mix_ok, l1_mix_detail);
    verdict("the layer-1 routed logits match", bytes && l1_logits_ok, l1_logits_detail);
    verdict("that layer-1 route repeats", bytes && l1_repeat, l1_repeat_detail);
    verdict("a zero residual moves that layer-1 route", bytes && l1_moved, l1_move_detail);
    verdict("a changed residual byte moves that layer-1 route", bytes && l1_flipped, l1_flip_detail);
    verdict("dropping the shared expert moves that layer-1 route", bytes && l1_dropped, l1_drop_detail);
}

// Layer 2's GDN step on layer 1's feed-forward residual. Layer 2 has no ple tensors.
// The residual it reads skipped the six blk.1.ple_* tensors. Query, gate, and output are IQ4_XS.
void run_layer2_gdn(const strata::core::ModelGeometry& g) {
    layer2_attn_residual.clear();
    std::printf("\nlayer 2 GDN step on layer 1's feed-forward residual, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const uint64_t k_data = 11024672;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const bool bytes = (int) layer1_ffn_residual.size() == hc_dim && g.ssm_state_size == 128 &&
                       g.ssm_k_heads == 16 && g.ssm_v_heads == 48 && g.ssm_d_conv == 4 &&
                       g.ssm_conv_channels == 10240 && std::filesystem::is_regular_file(shard);
    bool step_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false;
    std::string step_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes qkv = load(2296247552, 13926400);
            Bytes gate = load(2287891712, 8355840);
            Bytes ssm = load(3049958528, 8355840);
            Bytes conv_w = load(3049793984, 163840);
            Bytes alpha_w = load(3049302464, 245760);
            Bytes beta_w = load(3049548224, 245760);
            Bytes ssm_a = load(3049302272, 192);
            Bytes dt_w = load(3049957824, 192);
            Bytes ssm_norm = load(3049958016, 512);
            Bytes attn_down = load(3022842112, 6553600);
            Bytes attn_up = load(3029518592, 6553600);
            Bytes attn_inj = load(3029395712, 81920);
            Bytes attn_norm = load(3029477632, 40960);
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, &qkv};
            w.attn_gate = GdnIqPlane{kIq4Xs, &gate};
            w.ssm_out = GdnIqPlane{kIq4Xs, &ssm};
            w.alpha = &alpha_w;
            w.beta = &beta_w;
            w.conv = &conv_w;
            w.ssm_norm = &ssm_norm;
            w.dt = &dt_w;
            w.ssm_a = &ssm_a;
            const GrShapes gs{n_embd, hc, hc_lr};
            const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
            const int state_n = (int) lg.S * (int) lg.h_v * (int) lg.S;
            const int conv_n = (int) g.ssm_conv_channels * (int) (lg.d_conv - 1);
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            Bytes state((uint64_t) state_n * 4), conv((uint64_t) conv_n * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4), gdn_y((uint64_t) n_embd * 4);
            Bytes out((uint64_t) guard_n * 4), out2((uint64_t) guard_n * 4), outz((uint64_t) guard_n * 4);
            Bytes outf((uint64_t) guard_n * 4);
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer1_ffn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto step = [&](const Bytes& residual, Bytes& y) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(gdn_y.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, attn_norm, attn_down, attn_up, &attn_inj, 1e-6f, false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, gdn_y);
                gr_write(gs, residual, gdn_y, inject, y);
            };
            step(R, out);
            layer2_attn_residual.assign(static_cast<const float*>(out.data()),
                                        static_cast<const float*>(out.data()) + hc_dim);
            std::vector<float> gpu_state((size_t) state_n), gpu_conv((size_t) conv_n), gpu_mixed((size_t) n_embd);
            std::vector<float> gpu_block((size_t) n_embd), gpu_inj((size_t) hc);
            std::memcpy(gpu_state.data(), state.data(), (size_t) state_n * 4);
            std::memcpy(gpu_conv.data(), conv.data(), (size_t) conv_n * 4);
            std::memcpy(gpu_mixed.data(), mixed.data(), (size_t) n_embd * 4);
            std::memcpy(gpu_block.data(), gdn_y.data(), (size_t) n_embd * 4);
            std::memcpy(gpu_inj.data(), inject.data(), (size_t) hc * 4);
            std::vector<float> host_state((size_t) state_n, 0.f), host_conv((size_t) conv_n, 0.f);
            const std::vector<float> host_block = gdn_iq_host(
                gpu_mixed.data(), static_cast<const uint8_t*>(qkv.data()), static_cast<const uint8_t*>(gate.data()),
                static_cast<const uint8_t*>(ssm.data()), static_cast<const uint16_t*>(alpha_w.data()),
                static_cast<const uint16_t*>(beta_w.data()), static_cast<const float*>(conv_w.data()),
                static_cast<const float*>(ssm_norm.data()), static_cast<const float*>(dt_w.data()),
                static_cast<const float*>(ssm_a.data()), n_embd, (int) lg.S, (int) lg.h_k, (int) lg.h_v,
                (int) lg.d_conv, &host_conv, &host_state, kIq4Xs, kIq4Xs);
            std::vector<float> host_write((size_t) hc_dim);
            const float* Rp = static_cast<const float*>(R.data());
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(gpu_inj[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] = Rp[(size_t) c * n_embd + d] + gpu_block[(size_t) d] * scale;
            }
            const double state_rel = rel_l1(host_state.data(), gpu_state.data(), state_n);
            const double conv_rel = rel_l1(host_conv.data(), gpu_conv.data(), conv_n);
            const double block_rel = rel_l1(host_block.data(), gpu_block.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), static_cast<const float*>(out.data()), hc_dim);
            std::vector<float> zeros_s((size_t) state_n, 0.f), zeros_c((size_t) conv_n, 0.f);
            const int state_moved = differ_floats(gpu_state.data(), zeros_s.data(), state_n);
            const int conv_moved = differ_floats(gpu_conv.data(), zeros_c.data(), conv_n);
            step(R, out2);
            step(zero_r, outz);
            const int candidates[] = {32, 2, 8, 136, 1360, 588 * 1360};
            int flip_at = -1;
            int flip_n = 0;
            Bytes flipped((uint64_t) qkv.size());
            for (int local : candidates) {
                if (local < 0 || (uint64_t) local >= qkv.size()) continue;
                std::memcpy(flipped.data(), qkv.data(), qkv.size());
                static_cast<uint8_t*>(flipped.data())[local] ^= 1u;
                w.qkv = GdnIqPlane{kIq4Xs, &flipped};
                step(R, outf);
                flip_n = differ_floats(static_cast<const float*>(out.data()), static_cast<const float*>(outf.data()), hc_dim);
                if (flip_n > 0) {
                    flip_at = local;
                    break;
                }
            }
            w.qkv = GdnIqPlane{kIq4Xs, &qkv};
            const float* op = static_cast<const float*>(out.data());
            const float* o2 = static_cast<const float*>(out2.data());
            const float* oz = static_cast<const float*>(outz.data());
            const float* of = static_cast<const float*>(outf.data());
            const bool guard_held = op[hc_dim] == 7.f && o2[hc_dim] == 7.f && oz[hc_dim] == 7.f &&
                                    (flip_at < 0 || of[hc_dim] == 7.f);
            const bool live = all_finite(op, hc_dim) && all_finite(o2, hc_dim) && all_finite(oz, hc_dim) &&
                              all_finite(host_block.data(), n_embd) && all_finite(host_write.data(), hc_dim) &&
                              all_finite(gpu_state.data(), state_n) && all_finite(host_state.data(), state_n) &&
                              !still_seven(op, hc_dim);
            const int again = differ_floats(op, o2, hc_dim);
            const int apart = differ_floats(op, oz, hc_dim);
            const double apart_rel = rel_l1(op, oz, hc_dim);
            step_ok = live && guard_held && block_rel <= 1e-5 && state_rel <= 1e-5 && conv_rel <= 1e-5 &&
                      write_rel <= 1e-4 && state_moved > 0 && conv_moved > 0;
            repeat_ok = live && guard_held && again == 0;
            moved_ok = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && flip_n > 0;
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "block %.3e, state %.3e, conv %.3e, write %.3e, state moved %d, conv moved %d, "
                          "layer 1 PLE not applied",
                          block_rel, state_rel, conv_rel, write_rel, state_moved, conv_moved);
            step_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, hc_dim, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            step_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the layer-2 GDN step matches", bytes && step_ok, step_detail);
    verdict("that layer-2 step repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-2 step", bytes && moved_ok, move_detail);
    verdict("a changed query byte moves that layer-2 step", bytes && flipped_ok, flip_detail);
}

// Layer 2's ten routed experts on the attention residual above. The shared expert is not added.
// Its down plane is Q4_0. iq_mmvq dispatches that product. This pass still leaves the shared
// expert out. Layer 1's PLE was not applied.
void run_layer2_routed(const PackTable& table, const strata::core::ModelGeometry& g) {
    layer2_ffn_residual.clear();
    std::printf("\nlayer 2 routed experts, shared expert not added, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 2310173952;
    const uint64_t k_gate = 2547025152;
    const uint64_t k_router = 2782954752;
    const uint64_t k_up = 2786451712;
    const uint64_t k_ffn_down = 3036072192;
    const uint64_t k_ffn_inj = 3042625792;
    const uint64_t k_ffn_norm = 3042707712;
    const uint64_t k_ffn_up = 3042748672;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && head_ok && n_expert == 512 && hc > 0 && n_ff == 640 &&
                       (int) layer2_attn_residual.size() == hc_dim && std::filesystem::is_regular_file(shard);
    bool l2_mix_ok = false, l2_logits_ok = false, l2_repeat = false, l2_moved = false, l2_flipped = false;
    std::string l2_mix_detail = "bytes";
    std::string l2_logits_detail = "bytes";
    std::string l2_repeat_detail = "bytes";
    std::string l2_move_detail = "bytes";
    std::string l2_flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes ffn_down = load(k_ffn_down, 6553600);
            Bytes ffn_up = load(k_ffn_up, 6553600);
            Bytes ffn_inj = load(k_ffn_inj, 81920);
            Bytes ffn_norm = load(k_ffn_norm, 40960);
            Bytes router = load(k_router, 2621440);
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer2_attn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            GrScratch gr = make_gr_scratch(gs);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, residual, combined, layer, route;
                std::vector<uint8_t> exq;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, Snap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, ffn_norm, ffn_down, ffn_up, &ffn_inj, 1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, router, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                moe_combine(parts, route_w, nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->residual.resize((size_t) hc_dim);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->route.resize((size_t) n_expert);
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->residual.data(), residual.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->route.data(), route.data(), (size_t) n_expert * 4);
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                }
            };

            Snap snap;
            run_pass(R, layer_y, &snap);
            layer2_ffn_residual.assign(static_cast<const float*>(layer_y.data()),
                                       static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = snap.residual.data();
                const float* norm_w = static_cast<const float*>(ffn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(ffn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(ffn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(ffn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> rounded((size_t) n_embd), host_route((size_t) n_expert);
            const uint16_t* router_w = static_cast<const uint16_t*>(router.data());
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            for (int o = 0; o < n_expert; ++o)
                host_route[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);
            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            std::vector<float> host_combined((size_t) n_embd), host_write((size_t) hc_dim);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                host_combined[(size_t) d] = (float) acc;
            }
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.residual[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            route_ref(snap.route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0;
            for (int i = 0; i < k; ++i)
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);
            const double read_rel = rel_l1(host_mixed.data(), snap.mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), snap.inject.data(), hc);
            const double router_rel = rel_l1(host_route.data(), snap.route.data(), n_expert);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes actb((uint64_t) n_embd * 4);
                std::memcpy(actb.data(), mix.data(), actb.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(actb, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            run_pass(R, layer_y2, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flip_at = -1;
            int flip_n = 0;
            std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
            const float* rp = static_cast<const float*>(R.data());
            int tries = 0;
            Bytes Rf((uint64_t) hc_dim * 4);
            for (int i = 0; i < hc_dim && tries < 8 && flip_n == 0; ++i) {
                if (!std::isfinite(rp[i]) || std::fabs(rp[i]) < 1e-3f) continue;
                for (int b = 2; b <= 3 && tries < 8 && flip_n == 0; ++b) {
                    const int local = i * 4 + b;
                    ++tries;
                    std::memcpy(Rf.data(), R.data(), Rf.size());
                    static_cast<uint8_t*>(Rf.data())[local] ^= 1u;
                    if (!all_finite(static_cast<const float*>(Rf.data()), hc_dim)) continue;
                    run_pass(Rf, layer_yf, nullptr);
                    if (!all_finite(static_cast<const float*>(layer_yf.data()), hc_dim)) {
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    mix_read(layer_yf, mix_yf);
                    project(mix_yf, logits_f);
                    if (!all_finite(lfp, n_vocab) || !all_finite(mfp, n_embd)) {
                        fill7(logits_f);
                        fill7(mix_yf);
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    flip_n = differ_floats(lp, lfp, n_vocab);
                    if (flip_n > 0) flip_at = local;
                }
            }
            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && mp[n_embd] == 7.f && m2p[n_embd] == 7.f &&
                                    mzp[n_embd] == 7.f && mfp[n_embd] == 7.f && lp[n_vocab] == 7.f &&
                                    l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f && lfp[n_vocab] == 7.f &&
                                    combine_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.combined.data(), n_embd) &&
                all_finite(snap.layer.data(), hc_dim) && all_finite(snap.residual.data(), hc_dim) &&
                all_finite(host_mixed.data(), n_embd) && all_finite(host_inject.data(), hc) &&
                all_finite(host_route.data(), n_expert) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                all_finite(lfp, n_vocab) && all_finite(mp, n_embd) && all_finite(layer_p, hc_dim) &&
                all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) && all_finite(layerf_p, hc_dim) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            l2_mix_ok = live && guard_held && inject_held && id_bad == 0 && route_w_rel <= 1e-5 &&
                        read_rel <= 1e-4 && inject_rel <= 1e-4 && router_rel <= 1e-5 && egate_rel <= 1e-5 &&
                        eup_rel <= 1e-5 && expert_rel <= 1e-5 && combine_rel <= 1e-5 && write_rel <= 1e-4 &&
                        mix_rel <= 1e-4;
            l2_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            l2_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            l2_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            l2_flipped = live && guard_held && inject_held && flip_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "k %d, read %.3e, inject %.3e, router %.3e, expert %.3e, combine %.3e, write %.3e, "
                          "mix %.3e, chain not run, shared not added, down is Q4_0, experts %s, "
                          "layer 1 PLE not applied",
                          k, read_rel, inject_rel, router_rel, expert_rel, combine_rel, write_rel, mix_rel,
                          id_list.c_str());
            l2_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain not run", rel, irel, n_vocab);
            l2_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            l2_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            l2_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_vocab, flip_at);
            l2_flip_detail = buf;
        } catch (const MetalError& e) {
            l2_mix_detail = e.what();
            l2_logits_detail = e.what();
            l2_repeat_detail = e.what();
            l2_move_detail = e.what();
            l2_flip_detail = e.what();
        }
    }
    verdict("the layer-2 routed mix matches", bytes && l2_mix_ok, l2_mix_detail);
    verdict("the layer-2 routed logits match", bytes && l2_logits_ok, l2_logits_detail);
    verdict("that layer-2 route repeats", bytes && l2_repeat, l2_repeat_detail);
    verdict("a zero residual moves that layer-2 route", bytes && l2_moved, l2_move_detail);
    verdict("a changed residual byte moves that layer-2 route", bytes && l2_flipped, l2_flip_detail);
}

// Layer 2's shared expert on the same attention residual as the routed check.
// Gate is IQ4_XS, up is Q2_0, down is Q4_0. The shared vector is added plain.
// Layer 1's PLE was not applied. This does not load layers 3 through 47.
void run_layer2_shared(const PackTable& table, const strata::core::ModelGeometry& g) {
    layer2_shared_residual.clear();
    std::printf("\nlayer 2 shared expert on the routed mix, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 2310173952;
    const uint64_t k_down_s = 2546103552;
    const uint64_t k_gate = 2547025152;
    const uint64_t k_router = 2782954752;
    const uint64_t k_ginp = 2785576192;
    const uint64_t k_gate_s = 2785581312;
    const uint64_t k_up = 2786451712;
    const uint64_t k_up_s = 3022381312;
    const uint64_t k_ffn_down = 3036072192;
    const uint64_t k_ffn_inj = 3042625792;
    const uint64_t k_ffn_norm = 3042707712;
    const uint64_t k_ffn_up = 3042748672;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    size_t gate_n = 0, up_n = 0, down_n = 0;
    if (n_embd > 0 && n_ff > 0 && n_embd % 256 == 0 && n_ff % 32 == 0) {
        gate_n = (size_t) n_ff * iq_row_bytes(kIq4Xs, n_embd);
        up_n = (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd);
        down_n = (size_t) n_embd * iq_row_bytes(kQ4_0, n_ff);
    }
    const bool shexp_ok = gate_n == 870400 && up_n == 460800 && down_n == 921600;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && shexp_ok && head_ok && n_expert == 512 && hc > 0 && n_ff == 640 &&
                       (int) layer2_attn_residual.size() == hc_dim && std::filesystem::is_regular_file(shard);
    bool mix_ok = false, logits_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false, dropped_ok = false;
    std::string mix_detail = "bytes";
    std::string logits_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    std::string drop_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes ffn_down = load(k_ffn_down, 6553600);
            Bytes ffn_up = load(k_ffn_up, 6553600);
            Bytes ffn_inj = load(k_ffn_inj, 81920);
            Bytes ffn_norm = load(k_ffn_norm, 40960);
            Bytes router = load(k_router, 2621440);
            Bytes ginp = load(k_ginp, 5120);
            Bytes sgate = load(k_gate_s, gate_n);
            Bytes sup = load(k_up_s, up_n);
            Bytes sdown = load(k_down_s, down_n);
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer2_attn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4);
            Bytes shared_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, residual, shared, combined, layer;
                std::vector<float> sgate, sup, sdown, route;
                std::vector<uint8_t> exq, sh_xq, sh_hq;
                float scale = 0;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kIq4Xs, sgate, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ2_0, sup, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ4_0, sdown, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, ginp, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, bool with_shared, Snap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, ffn_norm, ffn_down, ffn_up, &ffn_inj, 1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, router, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                shared_forward(shared_y);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->residual.resize((size_t) hc_dim);
                    snap->shared.resize((size_t) n_embd);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->sgate.resize((size_t) n_ff);
                    snap->sup.resize((size_t) n_ff);
                    snap->sdown.resize((size_t) n_embd);
                    snap->route.resize((size_t) n_expert);
                    snap->sh_xq.resize(sh_xq.size());
                    snap->sh_hq.resize(sh_hq.size());
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->residual.data(), residual.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->shared.data(), shared_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->sgate.data(), sh_gate_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sup.data(), sh_up_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sdown.data(), sh_down_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->route.data(), route.data(), (size_t) n_expert * 4);
                    std::memcpy(snap->sh_xq.data(), sh_xq.data(), sh_xq.size());
                    std::memcpy(snap->sh_hq.data(), sh_hq.data(), sh_hq.size());
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                    snap->scale = static_cast<const float*>(sh_scalar.data())[0];
                }
            };

            Snap snap;
            run_pass(R, layer_y, true, &snap);
            layer2_shared_residual.assign(static_cast<const float*>(layer_y.data()),
                                          static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool shared_guard = static_cast<const float*>(shared_y.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = snap.residual.data();
                const float* norm_w = static_cast<const float*>(ffn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(ffn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(ffn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(ffn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> rounded((size_t) n_embd), host_route((size_t) n_expert);
            const uint16_t* router_w = static_cast<const uint16_t*>(router.data());
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            for (int o = 0; o < n_expert; ++o)
                host_route[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);
            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
            xs_ref(static_cast<const uint8_t*>(sgate.data()), snap.sh_xq.data(), n_embd, n_ff, host_sgate.data());
            q2_ref(static_cast<const uint8_t*>(sup.data()), snap.sh_xq.data(), n_embd, n_ff, host_sup.data());
            q40_ref(static_cast<const uint8_t*>(sdown.data()), snap.sh_hq.data(), n_ff, n_embd, host_sdown.data());
            const float host_scale = bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp.data()), n_embd);
            const float sig = gdn_sigmoid(host_scale);
            std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                acc += (double) snap.shared[(size_t) d];
                host_combined[(size_t) d] = (float) acc;
                host_shared[(size_t) d] = host_sdown[(size_t) d] * sig;
            }
            std::vector<float> host_write((size_t) hc_dim);
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.residual[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            route_ref(snap.route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0;
            for (int i = 0; i < k; ++i)
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);
            const double read_rel = rel_l1(host_mixed.data(), snap.mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), snap.inject.data(), hc);
            const double router_rel = rel_l1(host_route.data(), snap.route.data(), n_expert);

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes actb((uint64_t) n_embd * 4);
                std::memcpy(actb.data(), mix.data(), actb.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(actb, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            run_pass(R, layer_y2, true, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, true, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);
            run_pass(R, layer_drop, false, nullptr);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flip_at = -1;
            int flip_n = 0;
            std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
            const float* rp = static_cast<const float*>(R.data());
            int tries = 0;
            Bytes Rf((uint64_t) hc_dim * 4);
            for (int i = 0; i < hc_dim && tries < 8 && flip_n == 0; ++i) {
                if (!std::isfinite(rp[i]) || std::fabs(rp[i]) < 1e-3f) continue;
                for (int b = 2; b <= 3 && tries < 8 && flip_n == 0; ++b) {
                    const int local = i * 4 + b;
                    ++tries;
                    std::memcpy(Rf.data(), R.data(), Rf.size());
                    static_cast<uint8_t*>(Rf.data())[local] ^= 1u;
                    if (!all_finite(static_cast<const float*>(Rf.data()), hc_dim)) continue;
                    run_pass(Rf, layer_yf, true, nullptr);
                    if (!all_finite(static_cast<const float*>(layer_yf.data()), hc_dim)) {
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    mix_read(layer_yf, mix_yf);
                    project(mix_yf, logits_f);
                    if (!all_finite(lfp, n_vocab) || !all_finite(mfp, n_embd)) {
                        fill7(logits_f);
                        fill7(mix_yf);
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    flip_n = differ_floats(lp, lfp, n_vocab);
                    if (flip_n > 0) flip_at = local;
                }
            }
            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mzp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && combine_guard && shared_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.shared.data(), n_embd) &&
                all_finite(snap.combined.data(), n_embd) && all_finite(snap.layer.data(), hc_dim) &&
                all_finite(snap.residual.data(), hc_dim) && all_finite(host_mixed.data(), n_embd) &&
                all_finite(host_inject.data(), hc) && all_finite(host_route.data(), n_expert) &&
                all_finite(host_shared.data(), n_embd) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                all_finite(lfp, n_vocab) && all_finite(mp, n_embd) && all_finite(layer_p, hc_dim) &&
                all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) && all_finite(layerf_p, hc_dim) &&
                all_finite(drop_p, hc_dim) && std::isfinite(snap.scale) && std::isfinite(host_scale) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double gate_rel = rel_l1(host_sgate.data(), snap.sgate.data(), n_ff);
            const double up_rel = rel_l1(host_sup.data(), snap.sup.data(), n_ff);
            const double down_rel = rel_l1(host_sdown.data(), snap.sdown.data(), n_embd);
            const double scale_rel = rel_l1(&host_scale, &snap.scale, 1);
            const double shared_rel = rel_l1(host_shared.data(), snap.shared.data(), n_embd);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            mix_ok = live && guard_held && inject_held && id_bad == 0 && route_w_rel <= 1e-5 && read_rel <= 1e-4 &&
                     inject_rel <= 1e-4 && router_rel <= 1e-5 && egate_rel <= 1e-5 && eup_rel <= 1e-5 &&
                     expert_rel <= 1e-5 && gate_rel <= 1e-5 && up_rel <= 1e-5 && down_rel <= 1e-5 &&
                     scale_rel <= 1e-4 && shared_rel <= 1e-4 && combine_rel <= 1e-5 && write_rel <= 1e-4 &&
                     mix_rel <= 1e-4;
            logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            repeat_ok = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            moved_ok = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && inject_held && flip_n > 0;
            dropped_ok = live && guard_held && inject_held && drop_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[768];
            std::snprintf(buf, sizeof buf,
                          "k %d, read %.3e, inject %.3e, router %.3e, expert %.3e, gate %.3e, up %.3e, down %.3e, "
                          "scale %.3e, shared %.3e, combine %.3e, write %.3e, mix %.3e, chain not run, experts %s, "
                          "layer 1 PLE not applied",
                          k, read_rel, inject_rel, router_rel, expert_rel, gate_rel, up_rel, down_rel, scale_rel,
                          shared_rel, combine_rel, write_rel, mix_rel, id_list.c_str());
            mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain not run", rel, irel, n_vocab);
            logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_vocab, flip_at);
            flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            drop_detail = buf;
        } catch (const MetalError& e) {
            mix_detail = e.what();
            logits_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
            drop_detail = e.what();
        }
    }
    verdict("the layer-2 shared mix matches", bytes && mix_ok, mix_detail);
    verdict("the layer-2 shared logits match", bytes && logits_ok, logits_detail);
    verdict("that layer-2 shared pass repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-2 shared pass", bytes && moved_ok, move_detail);
    verdict("a changed residual byte moves that layer-2 shared pass", bytes && flipped_ok, flip_detail);
    verdict("dropping the shared expert moves that layer-2 pass", bytes && dropped_ok, drop_detail);
}


// Layer 3's four attention products. The QSA block is not called. Layer 1's PLE was not applied.
void run_layer3_attn() {
    std::printf("\nlayer 3 attention products, QSA block not run, layer 1 PLE not applied\n");
    const int n_embd = 2560;
    const int q_out = 12288;
    const int kv_out = 512;
    const int attn_out_in = 6144;
    const uint64_t k_data = 11024672;
    const uint64_t k_q = 3067725952;
    const uint64_t k_k = 3058314368;
    const uint64_t k_v = 3076574336;
    const uint64_t k_o = 3058878592;
    const uint64_t n_q = 8847360;
    const uint64_t n_k = 563200;
    const uint64_t n_v = 901120;
    const uint64_t n_o = 8847360;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const bool layout = iq_row_bytes(kQ2_0, n_embd) * (size_t) q_out == n_q &&
                        iq_row_bytes(kQ3_K, n_embd) * (size_t) kv_out == n_k &&
                        iq_row_bytes(kQ5_K, n_embd) * (size_t) kv_out == n_v &&
                        iq_row_bytes(kQ4_K, attn_out_in) * (size_t) n_embd == n_o &&
                        std::filesystem::is_regular_file(shard);
    std::vector<uint8_t> q, k, v, o;
    if (layout) {
        auto load = [&](uint64_t off, uint64_t n, std::vector<uint8_t>& dst) {
            dst = read_at_path(shard, k_data + off, (size_t) n);
        };
        load(k_q, n_q, q);
        load(k_k, n_k, k);
        load(k_v, n_v, v);
        load(k_o, n_o, o);
    }
    const uint8_t* qp = layout ? q.data() : nullptr;
    const uint8_t* kp = layout ? k.data() : nullptr;
    const uint8_t* vp = layout ? v.data() : nullptr;
    const uint8_t* op = layout ? o.data() : nullptr;
    check_rows("the layer-3 query rows match", "a changed layer-3 query code moves that product", qp, layout ? n_q : 0,
               n_embd, q_out, kQ2_0, 2, q2_ref, nullptr);
    check_rows("the layer-3 key rows match", "a changed layer-3 key code moves that product", kp, layout ? n_k : 0,
               n_embd, kv_out, kQ3_K, 32, q3_ref, q3_scalar);
    check_k("the layer-3 value rows match", "a changed layer-3 value code moves that product", vp, layout ? n_v : 0,
            n_embd, kv_out, true, 48);
    check_k("the layer-3 output rows match", "a changed layer-3 output code moves that product", op, layout ? n_o : 0,
            attn_out_in, n_embd, false, 16);
}

void qsa_rms_rows(std::vector<float>& x, const float* w, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float part[32];
        for (int i = 0; i < 32; ++i) part[i] = 0;
        float* row = x.data() + (size_t) r * (size_t) cols;
        for (int c = 0; c < cols; ++c) part[c & 31] += row[c] * row[c];
        for (int off = 16; off > 0; off >>= 1) {
            float next[32];
            for (int i = 0; i < 32; ++i) {
                const float other = (i + off < 32) ? part[i + off] : 0.0f;
                next[i] = part[i] + other;
            }
            for (int i = 0; i < 32; ++i) part[i] = next[i];
        }
        const float inv = 1.0f / std::sqrt(part[0] / (float) cols + 1e-6f);
        for (int c = 0; c < cols; ++c) row[c] = row[c] * w[c] * inv;
    }
}

void qsa_rope_rows(std::vector<float>& x, int rows, int head_dim, int n_rot, int pos, const float* cos_tab,
                   const float* sin_tab) {
    const int pairs = n_rot / 2;
    for (int r = 0; r < rows; ++r) {
        float* row = x.data() + (size_t) r * (size_t) head_dim;
        for (int i = 0; i < pairs; ++i) {
            const float a = row[i];
            const float b = row[pairs + i];
            const float c = cos_tab[(size_t) pos * (size_t) pairs + (size_t) i];
            const float s = sin_tab[(size_t) pos * (size_t) pairs + (size_t) i];
            const float ac = a * c;
            const float bs = b * s;
            const float as_ = a * s;
            const float bc = b * c;
            row[i] = ac - bs;
            row[pairs + i] = as_ + bc;
        }
    }
}

std::vector<float> qsa_attend(const std::vector<float>& q, const std::vector<uint16_t>& k, const std::vector<uint16_t>& v,
                              int n_kv, int n_head, int n_head_kv, int head_dim) {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    const int group = n_head / n_head_kv;
    std::vector<float> out((size_t) n_head * (size_t) head_dim, 0.f);
    for (int h = 0; h < n_head; ++h) {
        const int kvh = h / group;
        std::vector<double> sc((size_t) n_kv);
        for (int j = 0; j < n_kv; ++j) {
            double dot = 0;
            for (int d = 0; d < head_dim; ++d) {
                const size_t at =
                    ((size_t) j * (size_t) n_head_kv + (size_t) kvh) * (size_t) head_dim + (size_t) d;
                dot += (double) strata::kernels::f32_from_f16(k[at]) * (double) q[(size_t) h * (size_t) head_dim + d];
            }
            sc[(size_t) j] = (double) scale * dot;
        }
        double mx = sc[0];
        for (double s : sc) mx = std::max(mx, s);
        double sum = 0;
        for (double& s : sc) {
            s = std::exp(s - mx);
            sum += s;
        }
        for (double& s : sc) s /= sum;
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0;
            for (int j = 0; j < n_kv; ++j) {
                const size_t at =
                    ((size_t) j * (size_t) n_head_kv + (size_t) kvh) * (size_t) head_dim + (size_t) d;
                acc += sc[(size_t) j] * (double) strata::kernels::f32_from_f16(v[at]);
            }
            out[(size_t) h * (size_t) head_dim + (size_t) d] = (float) acc;
        }
    }
    return out;
}

std::vector<float> qsa_gate(const std::vector<float>& attn, const std::vector<float>& q_full, int n_head, int head_dim) {
    std::vector<float> out((size_t) n_head * (size_t) head_dim);
    for (int h = 0; h < n_head; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            const float gate = q_full[(size_t) h * (size_t) head_dim * 2u + (size_t) head_dim + (size_t) d];
            const float s = 1.0f / (1.0f + std::exp(-gate));
            out[(size_t) h * (size_t) head_dim + (size_t) d] =
                attn[(size_t) h * (size_t) head_dim + (size_t) d] * s;
        }
    }
    return out;
}

struct QsaHost {
    std::vector<float> y0;
    std::vector<float> y1;
    std::vector<float> k1;
};

// Two cells of one activation. Position 0, then position 1. The page table is not in this walk:
// the cells are stored in order. K and V are the fp16 values the pool stores.
QsaHost qsa_iq_host(const float* x, int n_embd, const uint8_t* qw, const uint8_t* kw, const uint8_t* vw,
                    const uint8_t* ow, const float* qn, const float* kn, const float* cos_tab, const float* sin_tab) {
    const int n_head = 24;
    const int n_head_kv = 2;
    const int head_dim = 256;
    const int n_rot = 64;
    const int kv = n_head_kv * head_dim;
    const int q_full_n = n_head * head_dim * 2;
    const int q_n = n_head * head_dim;
    std::vector<uint16_t> hk((size_t) 2 * (size_t) kv), hv(hk.size());
    QsaHost host;
    for (int t = 0; t < 2; ++t) {
        Bytes xin((uint64_t) n_embd * 4u);
        std::memcpy(xin.data(), x, (size_t) n_embd * 4u);
        Bytes xq(iq_q8_1_bytes(n_embd));
        iq_quantize_q8_1(xin, n_embd, xq);
        const uint8_t* qb = static_cast<const uint8_t*>(xq.data());
        std::vector<float> kval((size_t) kv), vval((size_t) kv), qfull((size_t) q_full_n);
        q3_ref(kw, qb, n_embd, kv, kval.data());
        k_ref(vw, qb, n_embd, kv, true, vval.data());
        q2_ref(qw, qb, n_embd, q_full_n, qfull.data());
        qsa_rms_rows(kval, kn, n_head_kv, head_dim);
        qsa_rope_rows(kval, n_head_kv, head_dim, n_rot, t, cos_tab, sin_tab);
        if (t == 1) host.k1 = kval;
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t src = (size_t) h * (size_t) head_dim + (size_t) d;
                const size_t at = ((size_t) t * (size_t) n_head_kv + (size_t) h) * (size_t) head_dim + (size_t) d;
                hk[at] = strata::kernels::f16_from_f32(kval[src]);
                hv[at] = strata::kernels::f16_from_f32(vval[src]);
            }
        }
        std::vector<float> q((size_t) q_n);
        for (int h = 0; h < n_head; ++h)
            std::memcpy(q.data() + (size_t) h * (size_t) head_dim,
                        qfull.data() + (size_t) h * (size_t) head_dim * 2u, (size_t) head_dim * 4u);
        qsa_rms_rows(q, qn, n_head, head_dim);
        qsa_rope_rows(q, n_head, head_dim, n_rot, t, cos_tab, sin_tab);
        const std::vector<float> attn = qsa_attend(q, hk, hv, t + 1, n_head, n_head_kv, head_dim);
        const std::vector<float> gated = qsa_gate(attn, qfull, n_head, head_dim);
        Bytes gin((uint64_t) q_n * 4u);
        std::memcpy(gin.data(), gated.data(), (size_t) q_n * 4u);
        Bytes gq(iq_q8_1_bytes(q_n));
        iq_quantize_q8_1(gin, q_n, gq);
        std::vector<float> y((size_t) n_embd);
        k_ref(ow, static_cast<const uint8_t*>(gq.data()), q_n, n_embd, false, y.data());
        if (t == 0) host.y0 = std::move(y);
        else host.y1 = std::move(y);
    }
    return host;
}

// Layer 3's QSA block on layer 2's shared residual. Position 0 starts from an empty cache.
// The same activation is repeated at position 1 so the rotation is in that output.
// The residual write is not run. Layer 1's PLE was not applied.
void run_layer3_qsa(const strata::core::ModelGeometry& g) {
    const auto sh = strata::kernels::qsa_real_shapes();
    if (sh.n_head != 24 || sh.n_head_kv != 2 || sh.head_dim != 256 || sh.n_rot != 64 || sh.idx_n_head != 4 ||
        sh.idx_dim != 128 || sh.idx_block != 4 || sh.idx_top_k != 2048 || sh.page_size != 4 ||
        strata::kernels::qsa_freq_base() != 1.0e7)
        throw MetalError("qsa geometry moved");
    std::printf("\nlayer 3 QSA block on layer 2's shared residual, position 0, cache empty, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int q_out = 12288;
    const int kv_out = 512;
    const int attn_in = 6144;
    const uint64_t k_data = 11024672;
    const uint64_t n_q = 8847360;
    const uint64_t n_k = 563200;
    const uint64_t n_v = 901120;
    const uint64_t n_o = 8847360;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const bool bytes = (int) layer2_shared_residual.size() == hc_dim && g.n_embd == 2560 &&
                       iq_row_bytes(kQ2_0, n_embd) * (size_t) q_out == n_q &&
                       iq_row_bytes(kQ3_K, n_embd) * (size_t) kv_out == n_k &&
                       iq_row_bytes(kQ5_K, n_embd) * (size_t) kv_out == n_v &&
                       iq_row_bytes(kQ4_K, attn_in) * (size_t) n_embd == n_o &&
                       std::filesystem::is_regular_file(shard);
    bool step_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false;
    std::string step_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes q = load(3067725952, n_q);
            Bytes k = load(3058314368, n_k);
            Bytes v = load(3076574336, n_v);
            Bytes o = load(3058878592, n_o);
            Bytes q_norm = load(3076573312, 1024);
            Bytes k_norm = load(3058877568, 1024);
            Bytes idx_q_norm = load(3816850048, 512);
            Bytes idx_k_norm = load(3816194176, 512);
            Bytes idx_q = load(3816850560, 2621440);
            Bytes idx_k = load(3816194688, 655360);
            Bytes attn_down = load(3789734016, 6553600);
            Bytes attn_up = load(3796410496, 6553600);
            Bytes attn_inj = load(3796287616, 81920);
            Bytes attn_norm = load(3796369536, 40960);
            const int pairs = (int) sh.n_rot / 2;
            const int table_rows = 2;
            std::vector<float> cos_h((size_t) table_rows * (size_t) pairs), sin_h(cos_h.size());
            build_rope_table((int) sh.n_rot, strata::kernels::qsa_freq_base(), table_rows, cos_h.data(), sin_h.data());
            Bytes b_cos((uint64_t) cos_h.size() * 4u), b_sin((uint64_t) sin_h.size() * 4u);
            std::memcpy(b_cos.data(), cos_h.data(), b_cos.size());
            std::memcpy(b_sin.data(), sin_h.data(), b_sin.size());
            int page = 0;
            Bytes b_table(4);
            std::memcpy(b_table.data(), &page, 4);
            const size_t pool_n = (size_t) sh.n_head_kv * (size_t) sh.page_size * (size_t) sh.head_dim;
            Bytes k_pool(pool_n * 2u), v_pool(pool_n * 2u);
            Bytes idx_tail((size_t) (sh.idx_block - 1) * (size_t) sh.idx_dim * 4u);
            Bytes idx_dead((size_t) sh.idx_dim * 4u);
            Bytes idx_pooled((size_t) sh.idx_dim * 4u);
            Bytes idx_bpos(4);
            QsaLayerGeom lg;
            lg.n_embd = n_embd;
            lg.max_cells = 2;
            lg.qsa = sh;
            QsaIqWeights w;
            w.k = QsaIqPlane{kQ3_K, &k};
            w.v = QsaIqPlane{kQ5_K, &v};
            w.q = QsaIqPlane{kQ2_0, &q};
            w.out = QsaIqPlane{kQ4_K, &o};
            w.idx_k = &idx_k;
            w.idx_q = &idx_q;
            w.k_norm = &k_norm;
            w.q_norm = &q_norm;
            w.idx_k_norm = &idx_k_norm;
            w.idx_q_norm = &idx_q_norm;
            w.cos_tab = &b_cos;
            w.sin_tab = &b_sin;
            const GrShapes gs{n_embd, hc, hc_lr};
            GrScratch gr = make_gr_scratch(gs);
            QsaLayerScratch sc = make_qsa_layer_scratch(lg);
            const int guard = 1;
            Bytes mixed((uint64_t) n_embd * 4u), inject((uint64_t) hc * 4u);
            Bytes y0((uint64_t) (n_embd + guard) * 4u), y1((uint64_t) (n_embd + guard) * 4u);
            Bytes y0b((uint64_t) (n_embd + guard) * 4u), y1b((uint64_t) (n_embd + guard) * 4u);
            Bytes y0z((uint64_t) (n_embd + guard) * 4u), y1z((uint64_t) (n_embd + guard) * 4u);
            Bytes y0f((uint64_t) (n_embd + guard) * 4u), y1f((uint64_t) (n_embd + guard) * 4u);
            Bytes R((uint64_t) hc_dim * 4u), zero_r((uint64_t) hc_dim * 4u);
            std::memcpy(R.data(), layer2_shared_residual.data(), (size_t) hc_dim * 4u);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            auto run_two = [&](const Bytes& residual, Bytes& out0, Bytes& out1) {
                std::memset(k_pool.data(), 0, (size_t) k_pool.size());
                std::memset(v_pool.data(), 0, (size_t) v_pool.size());
                std::memset(idx_tail.data(), 0, (size_t) idx_tail.size());
                std::memset(idx_dead.data(), 0, (size_t) idx_dead.size());
                std::memset(idx_pooled.data(), 0, (size_t) idx_pooled.size());
                std::memset(idx_bpos.data(), 0, (size_t) idx_bpos.size());
                std::fill_n(static_cast<float*>(mixed.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(out0.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out1.data()), (size_t) (n_embd + guard), 7.f);
                QsaLayerState st;
                st.k_pool = &k_pool;
                st.v_pool = &v_pool;
                st.page_table = &b_table;
                st.idx_tail = &idx_tail;
                st.idx_dead = &idx_dead;
                st.idx_pooled = &idx_pooled;
                st.idx_block_pos = &idx_bpos;
                gr_read(gs, residual, attn_norm, attn_down, attn_up, &attn_inj, 1e-6f, false, gr, mixed, inject);
                qsa_layer_iq(lg, w, st, sc, 0, 0, mixed, out0);
                qsa_layer_iq(lg, w, st, sc, 1, 0, mixed, out1);
            };
            run_two(R, y0, y1);
            std::vector<float> gpu_mixed((size_t) n_embd), gpu_inj((size_t) hc), gpu_k((size_t) kv_out);
            std::memcpy(gpu_mixed.data(), mixed.data(), (size_t) n_embd * 4u);
            std::memcpy(gpu_inj.data(), inject.data(), (size_t) hc * 4u);
            std::memcpy(gpu_k.data(), sc.kcur.data(), (size_t) kv_out * 4u);
            const int* idp = static_cast<const int*>(sc.ids.data());
            const bool ids_ok = idp[0] == 0 && idp[1] == 1;
            const QsaHost host = qsa_iq_host(gpu_mixed.data(), n_embd, static_cast<const uint8_t*>(q.data()),
                                             static_cast<const uint8_t*>(k.data()), static_cast<const uint8_t*>(v.data()),
                                             static_cast<const uint8_t*>(o.data()),
                                             static_cast<const float*>(q_norm.data()),
                                             static_cast<const float*>(k_norm.data()), cos_h.data(), sin_h.data());
            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = static_cast<const float*>(R.data());
                const float* norm_w = static_cast<const float*>(attn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(attn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(attn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(attn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double val = src[(size_t) c * n_embd + d];
                        ms += val * val;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            const double read_rel = rel_l1(host_mixed.data(), gpu_mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), gpu_inj.data(), hc);
            const double block_rel = rel_l1(host.y0.data(), static_cast<const float*>(y0.data()), n_embd);
            const double pos_rel = rel_l1(host.y1.data(), static_cast<const float*>(y1.data()), n_embd);
            const double key_rel = rel_l1(host.k1.data(), gpu_k.data(), kv_out);
            run_two(R, y0b, y1b);
            run_two(zero_r, y0z, y1z);
            // A one-bit quant change can round away in the fp16 value. Keep bytes that change that fp16.
            std::vector<int> tries;
            {
                Bytes xin((uint64_t) n_embd * 4u);
                std::memcpy(xin.data(), gpu_mixed.data(), (size_t) n_embd * 4u);
                Bytes xq(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(xin, n_embd, xq);
                const uint8_t* qb = static_cast<const uint8_t*>(xq.data());
                const int row_bytes = 1760;
                const int offs[] = {16, 48, 80, 120, 160, 192, 400, 880};
                for (int row = 0; row < 64 && (int) tries.size() < 8; ++row) {
                    const uint8_t* rowp = static_cast<const uint8_t*>(v.data()) + (size_t) row * (size_t) row_bytes;
                    float base = 0.f;
                    k_ref(rowp, qb, n_embd, 1, true, &base);
                    const uint16_t base16 = strata::kernels::f16_from_f32(base);
                    for (int off : offs) {
                        if ((off % 176) < 16 || off >= row_bytes) continue;
                        std::vector<uint8_t> tmp(rowp, rowp + row_bytes);
                        tmp[(size_t) off] ^= 1u;
                        float alt = 0.f;
                        k_ref(tmp.data(), qb, n_embd, 1, true, &alt);
                        if (strata::kernels::f16_from_f32(alt) != base16) {
                            tries.push_back(row * row_bytes + off);
                            break;
                        }
                    }
                }
            }
            int flip_at = -1;
            int flip_n = 0;
            Bytes flipped(v.size());
            for (int local : tries) {
                std::memcpy(flipped.data(), v.data(), (size_t) v.size());
                static_cast<uint8_t*>(flipped.data())[local] ^= 1u;
                w.v = QsaIqPlane{kQ5_K, &flipped};
                run_two(R, y0f, y1f);
                const int n0 = differ_floats(static_cast<const float*>(y0.data()), static_cast<const float*>(y0f.data()),
                                             n_embd);
                const int n1 = differ_floats(static_cast<const float*>(y1.data()), static_cast<const float*>(y1f.data()),
                                             n_embd);
                flip_n = n0 > 0 ? n0 : n1;
                if (flip_n > 0) {
                    flip_at = local;
                    break;
                }
            }
            w.v = QsaIqPlane{kQ5_K, &v};
            const float* a0 = static_cast<const float*>(y0.data());
            const float* a1 = static_cast<const float*>(y1.data());
            const float* b0 = static_cast<const float*>(y0b.data());
            const float* b1 = static_cast<const float*>(y1b.data());
            const float* z0 = static_cast<const float*>(y0z.data());
            const bool guard_held = a0[n_embd] == 7.f && a1[n_embd] == 7.f && b0[n_embd] == 7.f && b1[n_embd] == 7.f &&
                                    z0[n_embd] == 7.f && (flip_at < 0 || static_cast<const float*>(y0f.data())[n_embd] == 7.f);
            const bool live = all_finite(a0, n_embd) && all_finite(a1, n_embd) && all_finite(host.y0.data(), n_embd) &&
                              all_finite(host.y1.data(), n_embd) && all_finite(host.k1.data(), kv_out) &&
                              all_finite(gpu_mixed.data(), n_embd) && !still_seven(a0, n_embd) && !still_seven(a1, n_embd);
            const int again = differ_floats(a0, b0, n_embd) + differ_floats(a1, b1, n_embd);
            const int apart = differ_floats(a0, z0, n_embd);
            const double apart_rel = rel_l1(a0, z0, n_embd);
            step_ok = live && guard_held && ids_ok && read_rel <= 1e-4 && inject_rel <= 1e-4 && block_rel <= 1e-5 &&
                      pos_rel <= 1e-5 && key_rel <= 1e-5;
            repeat_ok = live && guard_held && again == 0;
            moved_ok = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && flip_n > 0;
            char buf[480];
            std::snprintf(buf, sizeof buf,
                          "read %.3e, inject %.3e, block %.3e, position 1 %.3e, key %.3e, ids %d, "
                          "rope 64 at 1e7, sections not applied, selection is the identity, "
                          "residual write not run, layer 1 PLE not applied",
                          read_rel, inject_rel, block_rel, pos_rel, key_rel, ids_ok ? 0 : 1);
            step_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_embd, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_embd, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            step_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the layer-3 QSA block matches", bytes && step_ok, step_detail);
    verdict("that layer-3 block repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-3 block", bytes && moved_ok, move_detail);
    verdict("a changed value byte moves that layer-3 block", bytes && flipped_ok, flip_detail);
}

// The position-0 block written back onto layer 2's shared residual. The cache starts empty.
// Layer 1's PLE was not applied.
void run_layer3_write(const strata::core::ModelGeometry& g) {
    layer3_attn_residual.clear();
    const auto sh = strata::kernels::qsa_real_shapes();
    if (sh.n_head != 24 || sh.n_head_kv != 2 || sh.head_dim != 256 || sh.n_rot != 64 || sh.idx_n_head != 4 ||
        sh.idx_dim != 128 || sh.idx_block != 4 || sh.idx_top_k != 2048 || sh.page_size != 4 ||
        strata::kernels::qsa_freq_base() != 1.0e7)
        throw MetalError("qsa geometry moved");
    std::printf("\nlayer 3 attention write, position 0, cache empty, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int q_out = 12288;
    const int kv_out = 512;
    const int attn_in = 6144;
    const uint64_t k_data = 11024672;
    const uint64_t n_q = 8847360;
    const uint64_t n_k = 563200;
    const uint64_t n_v = 901120;
    const uint64_t n_o = 8847360;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const bool bytes = (int) layer2_shared_residual.size() == hc_dim && g.n_embd == 2560 &&
                       iq_row_bytes(kQ2_0, n_embd) * (size_t) q_out == n_q &&
                       iq_row_bytes(kQ3_K, n_embd) * (size_t) kv_out == n_k &&
                       iq_row_bytes(kQ5_K, n_embd) * (size_t) kv_out == n_v &&
                       iq_row_bytes(kQ4_K, attn_in) * (size_t) n_embd == n_o &&
                       std::filesystem::is_regular_file(shard);
    bool step_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false;
    std::string step_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes q = load(3067725952, n_q);
            Bytes k = load(3058314368, n_k);
            Bytes v = load(3076574336, n_v);
            Bytes o = load(3058878592, n_o);
            Bytes q_norm = load(3076573312, 1024);
            Bytes k_norm = load(3058877568, 1024);
            Bytes idx_q_norm = load(3816850048, 512);
            Bytes idx_k_norm = load(3816194176, 512);
            Bytes idx_q = load(3816850560, 2621440);
            Bytes idx_k = load(3816194688, 655360);
            Bytes attn_down = load(3789734016, 6553600);
            Bytes attn_up = load(3796410496, 6553600);
            Bytes attn_inj = load(3796287616, 81920);
            Bytes attn_norm = load(3796369536, 40960);
            const int pairs = (int) sh.n_rot / 2;
            std::vector<float> cos_h((size_t) pairs), sin_h(cos_h.size());
            build_rope_table((int) sh.n_rot, strata::kernels::qsa_freq_base(), 1, cos_h.data(), sin_h.data());
            Bytes b_cos((uint64_t) cos_h.size() * 4u), b_sin((uint64_t) sin_h.size() * 4u);
            std::memcpy(b_cos.data(), cos_h.data(), b_cos.size());
            std::memcpy(b_sin.data(), sin_h.data(), b_sin.size());
            int page = 0;
            Bytes b_table(4);
            std::memcpy(b_table.data(), &page, 4);
            const size_t pool_n = (size_t) sh.n_head_kv * (size_t) sh.page_size * (size_t) sh.head_dim;
            Bytes k_pool(pool_n * 2u), v_pool(pool_n * 2u);
            Bytes idx_tail((size_t) (sh.idx_block - 1) * (size_t) sh.idx_dim * 4u);
            Bytes idx_dead((size_t) sh.idx_dim * 4u);
            Bytes idx_pooled((size_t) sh.idx_dim * 4u);
            Bytes idx_bpos(4);
            QsaLayerGeom lg;
            lg.n_embd = n_embd;
            lg.max_cells = 1;
            lg.qsa = sh;
            QsaIqWeights w;
            w.k = QsaIqPlane{kQ3_K, &k};
            w.v = QsaIqPlane{kQ5_K, &v};
            w.q = QsaIqPlane{kQ2_0, &q};
            w.out = QsaIqPlane{kQ4_K, &o};
            w.idx_k = &idx_k;
            w.idx_q = &idx_q;
            w.k_norm = &k_norm;
            w.q_norm = &q_norm;
            w.idx_k_norm = &idx_k_norm;
            w.idx_q_norm = &idx_q_norm;
            w.cos_tab = &b_cos;
            w.sin_tab = &b_sin;
            const GrShapes gs{n_embd, hc, hc_lr};
            GrScratch gr = make_gr_scratch(gs);
            QsaLayerScratch sc = make_qsa_layer_scratch(lg);
            const int guard = 1;
            Bytes mixed((uint64_t) n_embd * 4u), inject((uint64_t) hc * 4u), block((uint64_t) n_embd * 4u);
            Bytes out((uint64_t) (hc_dim + guard) * 4u), out2((uint64_t) (hc_dim + guard) * 4u);
            Bytes outz((uint64_t) (hc_dim + guard) * 4u), outf((uint64_t) (hc_dim + guard) * 4u);
            Bytes R((uint64_t) hc_dim * 4u), zero_r((uint64_t) hc_dim * 4u);
            std::memcpy(R.data(), layer2_shared_residual.data(), (size_t) hc_dim * 4u);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            auto step = [&](const Bytes& residual, Bytes& y) {
                std::memset(k_pool.data(), 0, (size_t) k_pool.size());
                std::memset(v_pool.data(), 0, (size_t) v_pool.size());
                std::memset(idx_tail.data(), 0, (size_t) idx_tail.size());
                std::memset(idx_dead.data(), 0, (size_t) idx_dead.size());
                std::memset(idx_pooled.data(), 0, (size_t) idx_pooled.size());
                std::memset(idx_bpos.data(), 0, (size_t) idx_bpos.size());
                std::fill_n(static_cast<float*>(mixed.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(block.data()), (size_t) n_embd, 7.f);
                std::fill_n(static_cast<float*>(y.data()), (size_t) (hc_dim + guard), 7.f);
                QsaLayerState st;
                st.k_pool = &k_pool;
                st.v_pool = &v_pool;
                st.page_table = &b_table;
                st.idx_tail = &idx_tail;
                st.idx_dead = &idx_dead;
                st.idx_pooled = &idx_pooled;
                st.idx_block_pos = &idx_bpos;
                gr_read(gs, residual, attn_norm, attn_down, attn_up, &attn_inj, 1e-6f, false, gr, mixed, inject);
                qsa_layer_iq(lg, w, st, sc, 0, 0, mixed, block);
                gr_write(gs, residual, block, inject, y);
            };
            step(R, out);
            layer3_attn_residual.assign(static_cast<const float*>(out.data()),
                                        static_cast<const float*>(out.data()) + hc_dim);
            std::vector<float> gpu_mixed((size_t) n_embd), gpu_inj((size_t) hc), gpu_block((size_t) n_embd);
            std::memcpy(gpu_mixed.data(), mixed.data(), (size_t) n_embd * 4u);
            std::memcpy(gpu_inj.data(), inject.data(), (size_t) hc * 4u);
            std::memcpy(gpu_block.data(), block.data(), (size_t) n_embd * 4u);
            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = static_cast<const float*>(R.data());
                const float* norm_w = static_cast<const float*>(attn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(attn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(attn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(attn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double val = src[(size_t) c * n_embd + d];
                        ms += val * val;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> host_write((size_t) hc_dim);
            const float* Rp = static_cast<const float*>(R.data());
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(gpu_inj[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] = Rp[(size_t) c * n_embd + d] + gpu_block[(size_t) d] * scale;
            }
            const double read_rel = rel_l1(host_mixed.data(), gpu_mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), gpu_inj.data(), hc);
            const double write_rel = rel_l1(host_write.data(), static_cast<const float*>(out.data()), hc_dim);
            step(R, out2);
            step(zero_r, outz);
            const int candidates[] = {3600, 48, 1808, 3616, 1760 + 48};
            int flip_at = -1;
            int flip_n = 0;
            Bytes flipped(v.size());
            for (int local : candidates) {
                if (local < 0 || (uint64_t) local >= v.size()) continue;
                std::memcpy(flipped.data(), v.data(), (size_t) v.size());
                static_cast<uint8_t*>(flipped.data())[local] ^= 1u;
                w.v = QsaIqPlane{kQ5_K, &flipped};
                step(R, outf);
                flip_n = differ_floats(static_cast<const float*>(out.data()), static_cast<const float*>(outf.data()),
                                       hc_dim);
                if (flip_n > 0) {
                    flip_at = local;
                    break;
                }
            }
            w.v = QsaIqPlane{kQ5_K, &v};
            const float* op = static_cast<const float*>(out.data());
            const float* o2 = static_cast<const float*>(out2.data());
            const float* oz = static_cast<const float*>(outz.data());
            const bool guard_held = op[hc_dim] == 7.f && o2[hc_dim] == 7.f && oz[hc_dim] == 7.f &&
                                    (flip_at < 0 || static_cast<const float*>(outf.data())[hc_dim] == 7.f);
            const bool live = all_finite(op, hc_dim) && all_finite(o2, hc_dim) && all_finite(gpu_block.data(), n_embd) &&
                              all_finite(host_write.data(), hc_dim) && all_finite(gpu_mixed.data(), n_embd) &&
                              !still_seven(op, hc_dim);
            const int again = differ_floats(op, o2, hc_dim);
            const int apart = differ_floats(op, oz, hc_dim);
            const double apart_rel = rel_l1(op, oz, hc_dim);
            step_ok = live && guard_held && read_rel <= 1e-4 && inject_rel <= 1e-4 && write_rel <= 1e-4;
            repeat_ok = live && guard_held && again == 0;
            moved_ok = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && flip_n > 0;
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "read %.3e, inject %.3e, write %.3e, position 0, cache empty, layer 1 PLE not applied",
                          read_rel, inject_rel, write_rel);
            step_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, hc_dim, flip_at);
            flip_detail = buf;
        } catch (const MetalError& e) {
            step_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("the layer-3 attention write matches", bytes && step_ok, step_detail);
    verdict("that layer-3 write repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-3 write", bytes && moved_ok, move_detail);
    verdict("a changed value byte moves that layer-3 write", bytes && flipped_ok, flip_detail);
}

// Layer 3's ten Q2_0 routed experts on the attention residual.
// The shared expert is not added. The route comes from this residual.
// Layer 1's PLE was not applied. This does not load layers 4 through 47.
void run_layer3_routed(const PackTable& table, const strata::core::ModelGeometry& g) {
    layer3_ffn_residual.clear();
    std::printf("\nlayer 3 routed experts, shared expert not added, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 3077475456;
    const uint64_t k_gate = 3314326656;
    const uint64_t k_router = 3550256256;
    const uint64_t k_up = 3553343616;
    const uint64_t k_ffn_down = 3802964096;
    const uint64_t k_ffn_inj = 3809517696;
    const uint64_t k_ffn_norm = 3809599616;
    const uint64_t k_ffn_up = 3809640576;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && head_ok && n_expert == 512 && hc > 0 && n_ff == 640 &&
                       (int) layer3_attn_residual.size() == hc_dim && std::filesystem::is_regular_file(shard);
    bool l3_mix_ok = false, l3_logits_ok = false, l3_repeat = false, l3_moved = false, l3_flipped = false;
    std::string l3_mix_detail = "bytes";
    std::string l3_logits_detail = "bytes";
    std::string l3_repeat_detail = "bytes";
    std::string l3_move_detail = "bytes";
    std::string l3_flip_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes ffn_down = load(k_ffn_down, 6553600);
            Bytes ffn_up = load(k_ffn_up, 6553600);
            Bytes ffn_inj = load(k_ffn_inj, 81920);
            Bytes ffn_norm = load(k_ffn_norm, 40960);
            Bytes router = load(k_router, 2621440);
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer3_attn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            GrScratch gr = make_gr_scratch(gs);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, residual, combined, layer, route;
                std::vector<uint8_t> exq;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, Snap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, ffn_norm, ffn_down, ffn_up, &ffn_inj, 1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, router, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                moe_combine(parts, route_w, nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->residual.resize((size_t) hc_dim);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->route.resize((size_t) n_expert);
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->residual.data(), residual.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->route.data(), route.data(), (size_t) n_expert * 4);
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                }
            };

            Snap snap;
            run_pass(R, layer_y, &snap);
            layer3_ffn_residual.assign(static_cast<const float*>(layer_y.data()),
                                       static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = snap.residual.data();
                const float* norm_w = static_cast<const float*>(ffn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(ffn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(ffn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(ffn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> rounded((size_t) n_embd), host_route((size_t) n_expert);
            const uint16_t* router_w = static_cast<const uint16_t*>(router.data());
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            for (int o = 0; o < n_expert; ++o)
                host_route[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);
            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            std::vector<float> host_combined((size_t) n_embd), host_write((size_t) hc_dim);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                host_combined[(size_t) d] = (float) acc;
            }
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.residual[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            route_ref(snap.route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0;
            for (int i = 0; i < k; ++i)
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);
            const double read_rel = rel_l1(host_mixed.data(), snap.mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), snap.inject.data(), hc);
            const double router_rel = rel_l1(host_route.data(), snap.route.data(), n_expert);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes actb((uint64_t) n_embd * 4);
                std::memcpy(actb.data(), mix.data(), actb.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(actb, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            run_pass(R, layer_y2, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* repeat_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flip_at = -1;
            int flip_n = 0;
            std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
            const float* rp = static_cast<const float*>(R.data());
            int tries = 0;
            Bytes Rf((uint64_t) hc_dim * 4);
            for (int i = 0; i < hc_dim && tries < 8 && flip_n == 0; ++i) {
                if (!std::isfinite(rp[i]) || std::fabs(rp[i]) < 1e-3f) continue;
                for (int b = 2; b <= 3 && tries < 8 && flip_n == 0; ++b) {
                    const int local = i * 4 + b;
                    ++tries;
                    std::memcpy(Rf.data(), R.data(), Rf.size());
                    static_cast<uint8_t*>(Rf.data())[local] ^= 1u;
                    if (!all_finite(static_cast<const float*>(Rf.data()), hc_dim)) continue;
                    run_pass(Rf, layer_yf, nullptr);
                    if (!all_finite(static_cast<const float*>(layer_yf.data()), hc_dim)) {
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    mix_read(layer_yf, mix_yf);
                    project(mix_yf, logits_f);
                    if (!all_finite(lfp, n_vocab) || !all_finite(mfp, n_embd)) {
                        fill7(logits_f);
                        fill7(mix_yf);
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    flip_n = differ_floats(lp, lfp, n_vocab);
                    if (flip_n > 0) flip_at = local;
                }
            }
            const bool guard_held = layer_p[hc_dim] == 7.f && repeat_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && mp[n_embd] == 7.f && m2p[n_embd] == 7.f &&
                                    mzp[n_embd] == 7.f && mfp[n_embd] == 7.f && lp[n_vocab] == 7.f &&
                                    l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f && lfp[n_vocab] == 7.f &&
                                    combine_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.combined.data(), n_embd) &&
                all_finite(snap.layer.data(), hc_dim) && all_finite(snap.residual.data(), hc_dim) &&
                all_finite(host_mixed.data(), n_embd) && all_finite(host_inject.data(), hc) &&
                all_finite(host_route.data(), n_expert) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                all_finite(lfp, n_vocab) && all_finite(mp, n_embd) && all_finite(layer_p, hc_dim) &&
                all_finite(repeat_p, hc_dim) && all_finite(layerz_p, hc_dim) && all_finite(layerf_p, hc_dim) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, repeat_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            l3_mix_ok = live && guard_held && inject_held && id_bad == 0 && route_w_rel <= 1e-5 &&
                        read_rel <= 1e-4 && inject_rel <= 1e-4 && router_rel <= 1e-5 && egate_rel <= 1e-5 &&
                        eup_rel <= 1e-5 && expert_rel <= 1e-5 && combine_rel <= 1e-5 && write_rel <= 1e-4 &&
                        mix_rel <= 1e-4;
            l3_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            l3_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            l3_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            l3_flipped = live && guard_held && inject_held && flip_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "k %d, read %.3e, inject %.3e, router %.3e, expert %.3e, combine %.3e, write %.3e, "
                          "mix %.3e, chain not run, shared not added, experts %s, "
                          "layer 1 PLE not applied",
                          k, read_rel, inject_rel, router_rel, expert_rel, combine_rel, write_rel, mix_rel,
                          id_list.c_str());
            l3_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain not run", rel, irel, n_vocab);
            l3_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            l3_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            l3_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_vocab, flip_at);
            l3_flip_detail = buf;
        } catch (const MetalError& e) {
            l3_mix_detail = e.what();
            l3_logits_detail = e.what();
            l3_repeat_detail = e.what();
            l3_move_detail = e.what();
            l3_flip_detail = e.what();
        }
    }
    verdict("the layer-3 routed mix matches", bytes && l3_mix_ok, l3_mix_detail);
    verdict("the layer-3 routed logits match", bytes && l3_logits_ok, l3_logits_detail);
    verdict("that layer-3 route repeats", bytes && l3_repeat, l3_repeat_detail);
    verdict("a zero residual moves that layer-3 route", bytes && l3_moved, l3_move_detail);
    verdict("a changed residual byte moves that layer-3 route", bytes && l3_flipped, l3_flip_detail);
}

// Layer 3's shared expert on the same attention residual as the routed check.
// Gate is Q2_0, up is Q2_0, down is Q4_0. The shared vector is added plain.
// Layer 1's PLE was not applied. This does not load layers 4 through 47.
void run_layer3_shared(const PackTable& table, const strata::core::ModelGeometry& g) {
    layer3_shared_residual.clear();
    std::printf("\nlayer 3 shared expert on the routed mix, gate Q2_0, up Q2_0, down Q4_0, layer 1 PLE not applied\n");
    const int n_embd = (int) g.n_embd;
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const int hc = (int) g.hc;
    const int hc_lr = (int) g.hc_lr;
    const int hc_dim = hc * n_embd;
    const int n_vocab = 248320;
    const int k = 10;
    const uint64_t k_data = 11024672;
    const uint64_t k_down = 3077475456;
    const uint64_t k_down_s = 3313405056;
    const uint64_t k_gate = 3314326656;
    const uint64_t k_router = 3550256256;
    const uint64_t k_ginp = 3552877696;
    const uint64_t k_gate_s = 3552882816;
    const uint64_t k_up = 3553343616;
    const uint64_t k_up_s = 3789273216;
    const uint64_t k_ffn_down = 3802964096;
    const uint64_t k_ffn_inj = 3809517696;
    const uint64_t k_ffn_norm = 3809599616;
    const uint64_t k_ffn_up = 3809640576;
    const std::filesystem::path shard =
        "/Users/kamp/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const size_t stride = (n_embd > 0 && n_ff > 0) ? (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd) : 0;
    const bool stride_ok = stride == 460800 && (size_t) n_embd * iq_row_bytes(kQ2_0, n_ff) == stride;
    size_t gate_n = 0, up_n = 0, down_n = 0;
    if (n_embd > 0 && n_ff > 0 && n_embd % 256 == 0 && n_ff % 32 == 0) {
        gate_n = (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd);
        up_n = (size_t) n_ff * iq_row_bytes(kQ2_0, n_embd);
        down_n = (size_t) n_embd * iq_row_bytes(kQ4_0, n_ff);
    }
    const bool shexp_ok = gate_n == 460800 && up_n == 460800 && down_n == 921600;
    auto plane = [&](const char* name, uint64_t nbytes, int64_t ne0, int64_t ne1) -> const PackTensor* {
        const PackTensor* t = table.find(name);
        if (!t || !t->values || t->code_bits != 0 || t->ne0 != ne0 || t->ne1 != ne1 || t->values->size() != nbytes)
            return nullptr;
        return t;
    };
    const PackTensor* head_out = plane("output.weight", 437043200, n_embd, n_vocab);
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const bool head_ok =
        head_out && head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 && head_up && head_up->values &&
        head_up->kind == strata::core::WeightKind::Bf16InF32 && head_up->code_bits == 0 && head_up->ne0 == hc_lr &&
        head_up->ne1 == hc_dim && head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 && head_norm &&
        head_norm->values && head_norm->kind == strata::core::WeightKind::F32 && head_norm->code_bits == 0 &&
        head_norm->ne0 == hc_dim && head_norm->ne1 == 1 && head_norm->values->size() == (uint64_t) hc_dim * 4 &&
        head_out->values->size() == (uint64_t) n_vocab * out_row_n;
    const bool bytes = stride_ok && shexp_ok && head_ok && n_expert == 512 && hc > 0 && n_ff == 640 &&
                       (int) layer3_attn_residual.size() == hc_dim && std::filesystem::is_regular_file(shard);
    bool mix_ok = false, logits_ok = false, repeat_ok = false, moved_ok = false, flipped_ok = false, dropped_ok = false;
    std::string mix_detail = "bytes";
    std::string logits_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    std::string drop_detail = "bytes";
    if (bytes) {
        try {
            auto load = [&](uint64_t off, uint64_t n) {
                const auto raw = read_at_path(shard, k_data + off, (size_t) n);
                Bytes b(n);
                std::memcpy(b.data(), raw.data(), (size_t) n);
                return b;
            };
            Bytes ffn_down = load(k_ffn_down, 6553600);
            Bytes ffn_up = load(k_ffn_up, 6553600);
            Bytes ffn_inj = load(k_ffn_inj, 81920);
            Bytes ffn_norm = load(k_ffn_norm, 40960);
            Bytes router = load(k_router, 2621440);
            Bytes ginp = load(k_ginp, 5120);
            Bytes sgate = load(k_gate_s, gate_n);
            Bytes sup = load(k_up_s, up_n);
            Bytes sdown = load(k_down_s, down_n);
            const GrShapes gs{n_embd, hc, hc_lr};
            const int guard = 1;
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes R((uint64_t) hc_dim * 4), zero_r((uint64_t) hc_dim * 4);
            std::memcpy(R.data(), layer3_attn_residual.data(), (size_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes layer_drop((uint64_t) guard_n * 4);
            Bytes expert_y((uint64_t) (n_embd + guard) * 4);
            Bytes shared_y((uint64_t) (n_embd + guard) * 4), combined((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), egate_y((uint64_t) n_ff * 4), eup_y((uint64_t) n_ff * 4);
            Bytes ehq(iq_q8_1_bytes(n_ff));
            Bytes sh_xq(iq_q8_1_bytes(n_embd)), sh_gate_y((uint64_t) n_ff * 4), sh_up_y((uint64_t) n_ff * 4);
            Bytes sh_hq(iq_q8_1_bytes(n_ff)), sh_down_y((uint64_t) (n_embd + guard) * 4), sh_scalar(4);
            Bytes parts((uint64_t) k * (uint64_t) n_embd * 4);
            Bytes ids((uint64_t) (k + guard) * 4), route_w((uint64_t) (k + guard) * 4);
            struct SeN {
                int n;
                int pad0;
                int pad1;
                int pad2;
            };
            static_assert(sizeof(SeN) == 16, "SeN must match the shader");
            GrScratch gr = make_gr_scratch(gs);
            auto load_expert = [&](int id, std::vector<uint8_t>& dst) {
                if (id < 0 || id >= n_expert) throw MetalError("routed expert id is outside the pool");
                dst.resize(stride * 3);
                const uint64_t base = (uint64_t) id * stride;
                const auto gate_b = read_at_path(shard, k_data + k_gate + base, stride);
                const auto up_b = read_at_path(shard, k_data + k_up + base, stride);
                const auto down_b = read_at_path(shard, k_data + k_down + base, stride);
                std::memcpy(dst.data(), gate_b.data(), stride);
                std::memcpy(dst.data() + stride, up_b.data(), stride);
                std::memcpy(dst.data() + stride * 2, down_b.data(), stride);
            };
            struct Snap {
                std::vector<float> mixed, inject, residual, shared, combined, layer;
                std::vector<float> sgate, sup, sdown, route;
                std::vector<uint8_t> exq, sh_xq, sh_hq;
                float scale = 0;
                std::vector<int> ids;
                std::vector<float> route_w;
                std::vector<std::vector<float>> egate, eup, edown;
                std::vector<std::vector<uint8_t>> ehq, blob;
            };
            auto shared_forward = [&](Bytes& out) {
                iq_quantize_q8_1(mixed, n_embd, sh_xq);
                iq_mmvq(kQ2_0, sgate, 0, sh_xq, n_embd, n_ff, 1, sh_gate_y);
                iq_mmvq(kQ2_0, sup, 0, sh_xq, n_embd, n_ff, 1, sh_up_y);
                iq_swiglu_q8(sh_gate_y, sh_up_y, n_ff, sh_hq);
                iq_mmvq(kQ4_0, sdown, 0, sh_hq, n_ff, n_embd, 1, sh_down_y);
                std::fill_n(static_cast<float*>(out.data()), (size_t) (n_embd + guard), 7.f);
                std::memcpy(out.data(), sh_down_y.data(), (size_t) n_embd * 4);
                Bytes x_bf16_s((uint64_t) n_embd * 2);
                f32_to_bf16_bulk(mixed, x_bf16_s, n_embd);
                bf16_gemv(x_bf16_s, ginp, sh_scalar, n_embd, 1);
                const SeN dim{n_embd, 0, 0, 0};
                const uint32_t groups = (uint32_t) ((n_embd + 255) / 256);
                const Bytes* scb[] = {&out, &sh_scalar};
                launch("se_sigmoid_scale", scb, 2, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
            };
            auto run_pass = [&](const Bytes& residual, Bytes& out, bool with_shared, Snap* snap) {
                std::fill_n(static_cast<float*>(expert_y.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(combined.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<int*>(ids.data()), (size_t) (k + guard), 7);
                std::fill_n(static_cast<float*>(route_w.data()), (size_t) (k + guard), 7.f);
                gr_read(gs, residual, ffn_norm, ffn_down, ffn_up, &ffn_inj, 1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, router, route, n_embd, n_expert);
                router_top10(route, 1, n_expert, k, ids, route_w);
                iq_quantize_q8_1(mixed, n_embd, exq);
                const int* idp = static_cast<const int*>(ids.data());
                if (snap) {
                    snap->egate.assign((size_t) k, {});
                    snap->eup.assign((size_t) k, {});
                    snap->edown.assign((size_t) k, {});
                    snap->ehq.assign((size_t) k, {});
                    snap->blob.assign((size_t) k, {});
                    snap->exq.assign((size_t) exq.size(), 0);
                    std::memcpy(snap->exq.data(), exq.data(), snap->exq.size());
                }
                for (int e = 0; e < k; ++e) {
                    std::vector<uint8_t> raw;
                    load_expert(idp[e], raw);
                    Bytes expert_w((uint64_t) raw.size());
                    std::memcpy(expert_w.data(), raw.data(), raw.size());
                    const int64_t up_b = (int64_t) stride;
                    const int64_t down_b = up_b * 2;
                    iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, egate_y);
                    iq_mmvq(kQ2_0, expert_w, up_b, exq, n_embd, n_ff, 1, eup_y);
                    iq_swiglu_q8(egate_y, eup_y, n_ff, ehq);
                    iq_mmvq(kQ2_0, expert_w, down_b, ehq, n_ff, n_embd, 1, expert_y);
                    std::memcpy(static_cast<float*>(parts.data()) + (size_t) e * n_embd, expert_y.data(),
                                (size_t) n_embd * 4);
                    if (snap) {
                        snap->egate[(size_t) e].resize((size_t) n_ff);
                        snap->eup[(size_t) e].resize((size_t) n_ff);
                        snap->edown[(size_t) e].resize((size_t) n_embd);
                        snap->ehq[(size_t) e].resize(ehq.size());
                        snap->blob[(size_t) e] = raw;
                        std::memcpy(snap->egate[(size_t) e].data(), egate_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->eup[(size_t) e].data(), eup_y.data(), (size_t) n_ff * 4);
                        std::memcpy(snap->edown[(size_t) e].data(), expert_y.data(), (size_t) n_embd * 4);
                        std::memcpy(snap->ehq[(size_t) e].data(), ehq.data(), ehq.size());
                    }
                }
                shared_forward(shared_y);
                moe_combine(parts, route_w, with_shared ? &shared_y : nullptr, combined, n_embd, k);
                gr_write(gs, residual, combined, inject, out);
                if (snap) {
                    snap->mixed.resize((size_t) n_embd);
                    snap->inject.resize((size_t) hc);
                    snap->residual.resize((size_t) hc_dim);
                    snap->shared.resize((size_t) n_embd);
                    snap->combined.resize((size_t) n_embd);
                    snap->layer.resize((size_t) hc_dim);
                    snap->sgate.resize((size_t) n_ff);
                    snap->sup.resize((size_t) n_ff);
                    snap->sdown.resize((size_t) n_embd);
                    snap->route.resize((size_t) n_expert);
                    snap->sh_xq.resize(sh_xq.size());
                    snap->sh_hq.resize(sh_hq.size());
                    snap->ids.assign(idp, idp + k);
                    snap->route_w.resize((size_t) k);
                    std::memcpy(snap->mixed.data(), mixed.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->inject.data(), inject.data(), (size_t) hc * 4);
                    std::memcpy(snap->residual.data(), residual.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->shared.data(), shared_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->combined.data(), combined.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->layer.data(), out.data(), (size_t) hc_dim * 4);
                    std::memcpy(snap->sgate.data(), sh_gate_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sup.data(), sh_up_y.data(), (size_t) n_ff * 4);
                    std::memcpy(snap->sdown.data(), sh_down_y.data(), (size_t) n_embd * 4);
                    std::memcpy(snap->route.data(), route.data(), (size_t) n_expert * 4);
                    std::memcpy(snap->sh_xq.data(), sh_xq.data(), sh_xq.size());
                    std::memcpy(snap->sh_hq.data(), sh_hq.data(), sh_hq.size());
                    std::memcpy(snap->route_w.data(), route_w.data(), (size_t) k * 4);
                    snap->scale = static_cast<const float*>(sh_scalar.data())[0];
                }
            };

            Snap snap;
            run_pass(R, layer_y, true, &snap);
            layer3_shared_residual.assign(static_cast<const float*>(layer_y.data()),
                                          static_cast<const float*>(layer_y.data()) + hc_dim);
            const bool combine_guard = static_cast<const float*>(combined.data())[n_embd] == 7.f;
            const bool shared_guard = static_cast<const float*>(shared_y.data())[n_embd] == 7.f;
            const bool id_guard = static_cast<const int*>(ids.data())[k] == 7;
            const bool weight_guard = static_cast<const float*>(route_w.data())[k] == 7.f;

            std::vector<float> host_mixed((size_t) n_embd, 0.f), host_inject((size_t) hc);
            {
                const float* src = snap.residual.data();
                const float* norm_w = static_cast<const float*>(ffn_norm.data());
                const uint16_t* down_w = static_cast<const uint16_t*>(ffn_down.data());
                const uint16_t* up_w = static_cast<const uint16_t*>(ffn_up.data());
                const uint16_t* inj_w = static_cast<const uint16_t*>(ffn_inj.data());
                std::vector<float> xn((size_t) hc_dim), act((size_t) hc_dim);
                for (int c = 0; c < hc; ++c) {
                    double ms = 0;
                    for (int d = 0; d < n_embd; ++d) {
                        const double v = src[(size_t) c * n_embd + d];
                        ms += v * v;
                    }
                    ms /= (double) n_embd;
                    const float rs = (float) (1.0 / std::sqrt(ms + 1e-6));
                    for (int d = 0; d < n_embd; ++d) {
                        const size_t i = (size_t) c * n_embd + d;
                        xn[i] = src[i] * rs * norm_w[i];
                    }
                }
                for (int i = 0; i < hc_dim; ++i) act[(size_t) i] = round_bf16(xn[(size_t) i]);
                std::vector<float> lo((size_t) hc_lr), lq((size_t) hc_lr);
                for (int r = 0; r < hc_lr; ++r) {
                    const float p = bf16_dot(act.data(), down_w + (size_t) r * hc_dim, hc_dim);
                    lo[(size_t) r] = gdn_silu(p / (float) hc);
                    lq[(size_t) r] = round_bf16(lo[(size_t) r]);
                }
                for (int d = 0; d < n_embd; ++d) {
                    float m = 0.f;
                    for (int c = 0; c < hc; ++c) {
                        const int i = c * n_embd + d;
                        const float gv = bf16_dot(lq.data(), up_w + (size_t) i * hc_lr, hc_lr);
                        m += xn[(size_t) i] * gdn_sigmoid(gv);
                    }
                    host_mixed[(size_t) d] = m / (float) hc;
                }
                for (int c = 0; c < hc; ++c)
                    host_inject[(size_t) c] = bf16_dot(act.data(), inj_w + (size_t) c * hc_dim, hc_dim);
            }
            std::vector<float> rounded((size_t) n_embd), host_route((size_t) n_expert);
            const uint16_t* router_w = static_cast<const uint16_t*>(router.data());
            for (int i = 0; i < n_embd; ++i) rounded[(size_t) i] = round_bf16(snap.mixed[(size_t) i]);
            for (int o = 0; o < n_expert; ++o)
                host_route[(size_t) o] = bf16_dot(rounded.data(), router_w + (size_t) o * n_embd, n_embd);
            double expert_rel = 0, egate_rel = 0, eup_rel = 0;
            for (int e = 0; e < k; ++e) {
                std::vector<float> hg((size_t) n_ff), hu((size_t) n_ff), hd((size_t) n_embd);
                const uint8_t* raw = snap.blob[(size_t) e].data();
                q2_ref(raw, snap.exq.data(), n_embd, n_ff, hg.data());
                q2_ref(raw + stride, snap.exq.data(), n_embd, n_ff, hu.data());
                q2_ref(raw + stride * 2, snap.ehq[(size_t) e].data(), n_ff, n_embd, hd.data());
                egate_rel = std::max(egate_rel, rel_l1(hg.data(), snap.egate[(size_t) e].data(), n_ff));
                eup_rel = std::max(eup_rel, rel_l1(hu.data(), snap.eup[(size_t) e].data(), n_ff));
                expert_rel = std::max(expert_rel, rel_l1(hd.data(), snap.edown[(size_t) e].data(), n_embd));
            }
            std::vector<float> host_sgate((size_t) n_ff), host_sup((size_t) n_ff), host_sdown((size_t) n_embd);
            q2_ref(static_cast<const uint8_t*>(sgate.data()), snap.sh_xq.data(), n_embd, n_ff, host_sgate.data());
            q2_ref(static_cast<const uint8_t*>(sup.data()), snap.sh_xq.data(), n_embd, n_ff, host_sup.data());
            q40_ref(static_cast<const uint8_t*>(sdown.data()), snap.sh_hq.data(), n_ff, n_embd, host_sdown.data());
            const float host_scale = bf16_dot(rounded.data(), static_cast<const uint16_t*>(ginp.data()), n_embd);
            const float sig = gdn_sigmoid(host_scale);
            std::vector<float> host_shared((size_t) n_embd), host_combined((size_t) n_embd);
            for (int d = 0; d < n_embd; ++d) {
                double acc = 0;
                for (int e = 0; e < k; ++e)
                    acc += (double) snap.route_w[(size_t) e] * (double) snap.edown[(size_t) e][(size_t) d];
                acc += (double) snap.shared[(size_t) d];
                host_combined[(size_t) d] = (float) acc;
                host_shared[(size_t) d] = host_sdown[(size_t) d] * sig;
            }
            std::vector<float> host_write((size_t) hc_dim);
            for (int c = 0; c < hc; ++c) {
                const float scale = 2.0f * gdn_sigmoid(snap.inject[(size_t) c] / (float) hc);
                for (int d = 0; d < n_embd; ++d)
                    host_write[(size_t) c * n_embd + d] =
                        snap.residual[(size_t) c * n_embd + d] + snap.combined[(size_t) d] * scale;
            }
            std::vector<int> host_ids((size_t) k);
            std::vector<float> host_rw((size_t) k);
            route_ref(snap.route.data(), n_expert, k, host_ids.data(), host_rw.data());
            int id_bad = 0;
            for (int i = 0; i < k; ++i)
                if (snap.ids[(size_t) i] != host_ids[(size_t) i]) ++id_bad;
            const double route_w_rel = rel_l1(host_rw.data(), snap.route_w.data(), k);
            const double read_rel = rel_l1(host_mixed.data(), snap.mixed.data(), n_embd);
            const double inject_rel = rel_l1(host_inject.data(), snap.inject.data(), hc);
            const double router_rel = rel_l1(host_route.data(), snap.route.data(), n_expert);

            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(snap.layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes actb((uint64_t) n_embd * 4);
                std::memcpy(actb.data(), mix.data(), actb.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(actb, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            run_pass(R, layer_y2, true, nullptr);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz, true, nullptr);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);
            run_pass(R, layer_drop, false, nullptr);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* repeat_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* drop_p = static_cast<const float*>(layer_drop.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flip_at = -1;
            int flip_n = 0;
            std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
            const float* rp = static_cast<const float*>(R.data());
            int tries = 0;
            Bytes Rf((uint64_t) hc_dim * 4);
            for (int i = 0; i < hc_dim && tries < 8 && flip_n == 0; ++i) {
                if (!std::isfinite(rp[i]) || std::fabs(rp[i]) < 1e-3f) continue;
                for (int b = 2; b <= 3 && tries < 8 && flip_n == 0; ++b) {
                    const int local = i * 4 + b;
                    ++tries;
                    std::memcpy(Rf.data(), R.data(), Rf.size());
                    static_cast<uint8_t*>(Rf.data())[local] ^= 1u;
                    if (!all_finite(static_cast<const float*>(Rf.data()), hc_dim)) continue;
                    run_pass(Rf, layer_yf, true, nullptr);
                    if (!all_finite(static_cast<const float*>(layer_yf.data()), hc_dim)) {
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    mix_read(layer_yf, mix_yf);
                    project(mix_yf, logits_f);
                    if (!all_finite(lfp, n_vocab) || !all_finite(mfp, n_embd)) {
                        fill7(logits_f);
                        fill7(mix_yf);
                        std::fill_n(static_cast<float*>(layer_yf.data()), (size_t) guard_n, 7.f);
                        continue;
                    }
                    flip_n = differ_floats(lp, lfp, n_vocab);
                    if (flip_n > 0) flip_at = local;
                }
            }
            const bool guard_held = layer_p[hc_dim] == 7.f && repeat_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && drop_p[hc_dim] == 7.f && mp[n_embd] == 7.f &&
                                    m2p[n_embd] == 7.f && mzp[n_embd] == 7.f && mfp[n_embd] == 7.f &&
                                    lp[n_vocab] == 7.f && l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f &&
                                    lfp[n_vocab] == 7.f && combine_guard && shared_guard && id_guard && weight_guard;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool live =
                all_finite(snap.mixed.data(), n_embd) && all_finite(snap.shared.data(), n_embd) &&
                all_finite(snap.combined.data(), n_embd) && all_finite(snap.layer.data(), hc_dim) &&
                all_finite(snap.residual.data(), hc_dim) && all_finite(host_mixed.data(), n_embd) &&
                all_finite(host_inject.data(), hc) && all_finite(host_route.data(), n_expert) &&
                all_finite(host_shared.data(), n_embd) && all_finite(host_combined.data(), n_embd) &&
                all_finite(host_write.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                all_finite(lp, n_vocab) && all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) &&
                all_finite(lfp, n_vocab) && all_finite(mp, n_embd) && all_finite(layer_p, hc_dim) &&
                all_finite(repeat_p, hc_dim) && all_finite(layerz_p, hc_dim) && all_finite(layerf_p, hc_dim) &&
                all_finite(drop_p, hc_dim) && std::isfinite(snap.scale) && std::isfinite(host_scale) &&
                !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double gate_rel = rel_l1(host_sgate.data(), snap.sgate.data(), n_ff);
            const double up_rel = rel_l1(host_sup.data(), snap.sup.data(), n_ff);
            const double down_rel = rel_l1(host_sdown.data(), snap.sdown.data(), n_embd);
            const double scale_rel = rel_l1(&host_scale, &snap.scale, 1);
            const double shared_rel = rel_l1(host_shared.data(), snap.shared.data(), n_embd);
            const double combine_rel = rel_l1(host_combined.data(), snap.combined.data(), n_embd);
            const double write_rel = rel_l1(host_write.data(), snap.layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, repeat_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            const int drop_n = differ_floats(layer_p, drop_p, hc_dim);
            mix_ok = live && guard_held && inject_held && id_bad == 0 && route_w_rel <= 1e-5 && read_rel <= 1e-4 &&
                     inject_rel <= 1e-4 && router_rel <= 1e-5 && egate_rel <= 1e-5 && eup_rel <= 1e-5 &&
                     expert_rel <= 1e-5 && gate_rel <= 1e-5 && up_rel <= 1e-5 && down_rel <= 1e-5 &&
                     scale_rel <= 1e-4 && shared_rel <= 1e-4 && combine_rel <= 1e-5 && write_rel <= 1e-4 &&
                     mix_rel <= 1e-4;
            logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            repeat_ok = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            moved_ok = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            flipped_ok = live && guard_held && inject_held && flip_n > 0;
            dropped_ok = live && guard_held && inject_held && drop_n > 0;
            std::string id_list;
            for (int i = 0; i < k; ++i) {
                if (i) id_list += ' ';
                id_list += std::to_string(snap.ids[(size_t) i]);
            }
            char buf[768];
            std::snprintf(buf, sizeof buf,
                          "k %d, read %.3e, inject %.3e, router %.3e, expert %.3e, gate %.3e, up %.3e, down %.3e, "
                          "scale %.3e, shared %.3e, combine %.3e, write %.3e, mix %.3e, chain not run, gate Q2_0, up Q2_0, down Q4_0, experts %s, "
                          "layer 1 PLE not applied",
                          k, read_rel, inject_rel, router_rel, expert_rel, gate_rel, up_rel, down_rel, scale_rel,
                          shared_rel, combine_rel, write_rel, mix_rel, id_list.c_str());
            mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain not run", rel, irel, n_vocab);
            logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flip_n, n_vocab, flip_at);
            flip_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ", drop_n, hc_dim);
            drop_detail = buf;
        } catch (const MetalError& e) {
            mix_detail = e.what();
            logits_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
            drop_detail = e.what();
        }
    }
    verdict("the layer-3 shared mix matches", bytes && mix_ok, mix_detail);
    verdict("the layer-3 shared logits match", bytes && logits_ok, logits_detail);
    verdict("that layer-3 shared pass repeats", bytes && repeat_ok, repeat_detail);
    verdict("a zero residual moves that layer-3 shared pass", bytes && moved_ok, move_detail);
    verdict("a changed residual byte moves that layer-3 shared pass", bytes && flipped_ok, flip_detail);
    verdict("dropping the shared expert moves that layer-3 pass", bytes && dropped_ok, drop_detail);
}

void run_q2_layer() {
    const char* env = std::getenv("STRATA_Q2_LAYER");
    const bool asked = env != nullptr && env[0] != '\0';
    const std::filesystem::path dir =
        asked ? std::filesystem::path(env) : std::filesystem::path("/Users/kamp/Developer/Strata-data/extract/q2_0-layer0");
    if (!std::filesystem::is_regular_file(dir / "index.txt")) {
        if (asked) throw MetalError("STRATA_Q2_LAYER has no index.txt");
        return;
    }
    strata::core::ModelGeometry g;
    if (g.n_embd != 2560 || g.n_expert != 512 || g.n_ff != 640 || g.hc != 4 || g.hc_lr != 320 ||
        g.qsa_interval != 4 || g.n_layers != 48 || g.ssm_state_size != 128 || g.ssm_k_heads != 16 ||
        g.ssm_v_heads != 48 || g.ssm_d_conv != 4 || g.ssm_conv_channels != 10240 || g.ssm_value_dim != 6144) {
        throw MetalError("model geometry defaults moved");
    }
    g.n_layers = 1;

    std::printf("\none Q2_0 layer from the extracted pack\n");
    const PackTable table = PackTable::load(dir.string());
    std::string shape_err;
    const bool named = check_pack_layer(table, g, 0, shape_err);
    verdict("one Q2_0 layer passes the name check", named, named ? "n_embd 2560, layer 0" : shape_err);
    if (!named) return;

    const PackTensor* qkv = table.find("blk.0.attn_qkv.weight");
    const bool qkv_bytes = qkv && qkv->values && qkv->values->size() == 13926400 && qkv->code_bits == 0;
    std::string bind_got;
    const bool refused = threw("Q8_K", [&] { bind_layer(table, g, 0); }, bind_got);
    const bool qkv_msg = bind_got.find("attn_qkv.weight") != std::string::npos &&
                         bind_got.find("code_bits 0") != std::string::npos;
    verdict("the Q2_0 query projection is not Q8_K", qkv_bytes && refused && qkv_msg,
            qkv_bytes && refused ? "IQ4_XS, 13926400 B, " + bind_got : bind_got);

    const int q_in = 2560;
    const int q_out = 10240;
    const size_t q_row = (size_t) (q_in / 256) * 136;
    const bool q_layout = qkv_bytes && qkv->values->size() == (uint64_t) q_out * q_row;
    std::vector<float> qx((size_t) q_in);
    for (int i = 0; i < q_in; ++i) qx[(size_t) i] = ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes qxf((uint64_t) q_in * 4);
    std::memcpy(qxf.data(), qx.data(), qxf.size());
    Bytes qxq(iq_q8_1_bytes(q_in));
    iq_quantize_q8_1(qxf, q_in, qxq);
    std::vector<uint8_t> qxqb((size_t) qxq.size());
    std::memcpy(qxqb.data(), qxq.data(), qxqb.size());
    std::vector<float> q_got((size_t) q_out), q_host((size_t) q_out);
    double q_rel = 1;
    bool q_ok = false;
    if (q_layout) {
        Bytes qy((uint64_t) q_out * 4);
        iq_mmvq(kIq4Xs, *qkv->values, 0, qxq, q_in, q_out, 1, qy);
        std::memcpy(q_got.data(), qy.data(), q_got.size() * 4);
        xs_ref(static_cast<const uint8_t*>(qkv->values->data()), qxqb.data(), q_in, q_out, q_host.data());
        q_rel = rel_l1(q_host.data(), q_got.data(), q_out);
        q_ok = all_finite(q_got.data(), q_out) && q_rel <= 1e-5;
    }
    char q_detail[80];
    std::snprintf(q_detail, sizeof q_detail, "rel %.3e, %d rows", q_rel, q_out);
    verdict("the IQ4_XS query rows match", q_layout && q_ok, q_layout ? q_detail : "query bytes");

    int q_moved = 0;
    if (q_ok) {
        Bytes flipped(qkv->values->size());
        std::memcpy(flipped.data(), qkv->values->data(), flipped.size());
        static_cast<uint8_t*>(flipped.data())[8] ^= 1u;
        Bytes y2((uint64_t) q_out * 4);
        iq_mmvq(kIq4Xs, flipped, 0, qxq, q_in, q_out, 1, y2);
        const float* y2p = static_cast<const float*>(y2.data());
        for (int i = 0; i < q_out; ++i)
            if (y2p[i] != q_got[(size_t) i]) ++q_moved;
    }
    char q_move_detail[80];
    std::snprintf(q_move_detail, sizeof q_move_detail, "%d of %d differ", q_moved, q_out);
    verdict("a changed IQ4_XS code moves the query", q_ok && q_moved > 0, q_move_detail);

    const PackTensor* gate = table.find("blk.0.attn_gate.weight");
    const PackTensor* ssm = table.find("blk.0.ssm_out.weight");
    check_k("the Q4_K gate rows match", "a changed Q4_K code moves the gate",
            gate && gate->values ? static_cast<const uint8_t*>(gate->values->data()) : nullptr,
            gate && gate->values ? gate->values->size() : 0, 2560, 6144, false, 16);
    check_k("the Q5_K output rows match", "a changed Q5_K code moves the output",
            ssm && ssm->values ? static_cast<const uint8_t*>(ssm->values->data()) : nullptr,
            ssm && ssm->values ? ssm->values->size() : 0, 6144, 2560, true, 48);

    const PackTensor* router = table.find("blk.0.ffn_gate_inp.weight");
    const int n_in = 2560;
    const int n_out = 512;
    const bool router_bytes =
        router && router->values && router->kind == strata::core::WeightKind::Bf16InF32 && router->code_bits == 0 &&
        router->values->size() == (uint64_t) n_in * n_out * 2;
    std::vector<uint16_t> xh((size_t) n_in);
    std::vector<float> want((size_t) n_out), got((size_t) n_out), zero((size_t) n_out);
    bool router_ok = false;
    int moved = 0;
    bool zero_clear = false;
    if (router_bytes) {
        for (int i = 0; i < n_in; ++i) {
            const float v = ((i * 17) % 29) * 0.01f - 0.1f;
            xh[(size_t) i] = strata::kernels::bf16_from_f32(v);
        }
        const uint16_t* w = static_cast<const uint16_t*>(router->values->data());
        for (int o = 0; o < n_out; ++o) {
            double acc = 0;
            const uint16_t* row = w + (size_t) o * n_in;
            for (int i = 0; i < n_in; ++i)
                acc += (double) strata::kernels::f32_from_bf16(xh[(size_t) i]) *
                       (double) strata::kernels::f32_from_bf16(row[i]);
            want[(size_t) o] = (float) acc;
        }
        Bytes dx((uint64_t) n_in * 2);
        Bytes dy((uint64_t) n_out * 4);
        std::memcpy(dx.data(), xh.data(), dx.size());
        bf16_gemv(dx, *router->values, dy, n_in, n_out);
        std::memcpy(got.data(), dy.data(), got.size() * 4);
        Bytes zx((uint64_t) n_in * 2);
        std::memset(zx.data(), 0, zx.size());
        bf16_gemv(zx, *router->values, dy, n_in, n_out);
        std::memcpy(zero.data(), dy.data(), zero.size() * 4);
        zero_clear = true;
        for (int i = 0; i < n_out; ++i) {
            if (zero[(size_t) i] != 0.f) zero_clear = false;
            if (got[(size_t) i] != zero[(size_t) i]) ++moved;
        }
        router_ok = all_finite(got.data(), n_out) && rel_l1(want.data(), got.data(), n_out) <= 1e-5;
    }
    char router_detail[120];
    std::snprintf(router_detail, sizeof router_detail, "rel %.3e, %d of %d differ from zero",
                  router_bytes ? rel_l1(want.data(), got.data(), n_out) : 1.0, moved, n_out);
    verdict("the router matches the file bytes", router_ok, router_bytes ? router_detail : "router bytes");
    verdict("a zero activation moves the router", router_ok && zero_clear && moved == n_out, router_detail);

    const auto experts = dir / "experts.bin";
    const auto blob = read_at_path(experts, 0, 1382400);
    const bool blob_n = blob.size() == 1382400;
    std::vector<float> x((size_t) n_in);
    for (int i = 0; i < n_in; ++i) x[(size_t) i] = ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes dxf((uint64_t) n_in * 4);
    std::memcpy(dxf.data(), x.data(), dxf.size());
    Bytes xq(iq_q8_1_bytes(n_in));
    iq_quantize_q8_1(dxf, n_in, xq);
    std::vector<uint8_t> xqb((size_t) xq.size());
    std::memcpy(xqb.data(), xq.data(), xqb.size());

    auto role = [&](uint64_t off, int role_in, int role_out, const Bytes& xin, const uint8_t* xbytes,
                    std::vector<float>& y) {
        const size_t nbytes = (size_t) role_out * (size_t) (role_in / 64) * 18;
        Bytes dw(nbytes);
        std::memcpy(dw.data(), blob.data() + off, nbytes);
        Bytes dy((uint64_t) role_out * 4);
        iq_mmvq(kQ2_0, dw, 0, xin, role_in, role_out, 1, dy);
        y.resize((size_t) role_out);
        std::memcpy(y.data(), dy.data(), y.size() * 4);
        std::vector<float> host((size_t) role_out);
        q2_ref(blob.data() + off, xbytes, role_in, role_out, host.data());
        return rel_l1(host.data(), y.data(), role_out);
    };
    std::vector<float> xdown((size_t) g.n_ff);
    for (int i = 0; i < (int) g.n_ff; ++i) xdown[(size_t) i] = ((i * 17) % 29) * 0.02f - 0.2f;
    Bytes dxdown((uint64_t) g.n_ff * 4);
    std::memcpy(dxdown.data(), xdown.data(), dxdown.size());
    Bytes xqdown(iq_q8_1_bytes(g.n_ff));
    iq_quantize_q8_1(dxdown, g.n_ff, xqdown);
    std::vector<uint8_t> xqdownb((size_t) xqdown.size());
    std::memcpy(xqdownb.data(), xqdown.data(), xqdownb.size());
    std::vector<float> gate_y, up_y, down_y;
    const double gate_rel = role(0, n_in, (int) g.n_ff, xq, xqb.data(), gate_y);
    const double up_rel = role(460800, n_in, (int) g.n_ff, xq, xqb.data(), up_y);
    const double down_rel = role(921600, (int) g.n_ff, n_in, xqdown, xqdownb.data(), down_y);
    const bool gate_ok = all_finite(gate_y.data(), (int) gate_y.size()) && gate_rel <= 1e-5;
    const bool up_ok = all_finite(up_y.data(), (int) up_y.size()) && up_rel <= 1e-5;
    const bool down_ok = all_finite(down_y.data(), (int) down_y.size()) && down_rel <= 1e-5;
    char gemv_detail[160];
    std::snprintf(gemv_detail, sizeof gemv_detail, "gate rel %.3e, up rel %.3e, down rel %.3e", gate_rel, up_rel,
                  down_rel);
    verdict("expert 0 Q2_0 rows match", blob_n && gate_ok && up_ok && down_ok, gemv_detail);

    std::vector<uint8_t> flipped(blob.begin(), blob.begin() + 460800);
    flipped[20] ^= 0x1;
    Bytes df(flipped.size());
    std::memcpy(df.data(), flipped.data(), df.size());
    Bytes y2((uint64_t) g.n_ff * 4);
    iq_mmvq(kQ2_0, df, 0, xq, n_in, g.n_ff, 1, y2);
    int code_moved = 0;
    const float* y2p = static_cast<const float*>(y2.data());
    for (int i = 0; i < (int) g.n_ff; ++i)
        if (y2p[i] != gate_y[(size_t) i]) ++code_moved;
    char code_detail[80];
    std::snprintf(code_detail, sizeof code_detail, "%d of %d differ", code_moved, (int) g.n_ff);
    verdict("a changed Q2_0 code moves the gate", gate_ok && code_moved > 0, code_detail);

    Bytes refused_y((uint64_t) g.n_embd * 4);
    std::string expert_got;
    const bool expert_refused =
        threw("down quant", [&] { run_routed_expert(table, dir.string(), g, 0, 0, xq, refused_y); }, expert_got);
    char expert_detail[200];
    std::snprintf(expert_detail, sizeof expert_detail, "blob %zu, %s", blob.size(), expert_got.c_str());
    verdict("a Q2_0 gate is refused", blob_n && expert_refused, expert_detail);

    const PackTensor* conv_w = table.find("blk.0.ssm_conv1d.weight");
    const PackTensor* alpha_w = table.find("blk.0.ssm_alpha.weight");
    const PackTensor* beta_w = table.find("blk.0.ssm_beta.weight");
    const PackTensor* ssm_a = table.find("blk.0.ssm_a");
    const PackTensor* dt_w = table.find("blk.0.ssm_dt.bias");
    const PackTensor* norm_w = table.find("blk.0.ssm_norm.weight");
    const bool step_bytes = qkv_bytes && gate && gate->values && gate->values->size() == 8847360 && ssm &&
                            ssm->values && ssm->values->size() == 10813440 && conv_w && conv_w->values &&
                            conv_w->values->size() == 163840 && alpha_w && alpha_w->values &&
                            alpha_w->values->size() == 245760 && beta_w && beta_w->values &&
                            beta_w->values->size() == 245760 && ssm_a && ssm_a->values && ssm_a->values->size() == 192 &&
                            dt_w && dt_w->values && dt_w->values->size() == 192 && norm_w && norm_w->values &&
                            norm_w->values->size() == 512;
    const int n_embd = 2560;
    const int guard = 1;
    const GdnLayerGeom lg{g.n_embd, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv};
    bool step_ok = false;
    bool step_repeat = false;
    bool step_moved = false;
    bool step_flipped = false;
    std::string step_detail = "bytes";
    std::string repeat_detail = "bytes";
    std::string move_detail = "bytes";
    std::string flip_detail = "bytes";
    if (step_bytes) {
        try {
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = norm_w->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();

            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4), state_b(state_n * 4), conv_b(conv_n * 4);
            Bytes state_z(state_n * 4), conv_z(conv_n * 4), state_f(state_n * 4), conv_f(conv_n * 4);
            std::memset(state.data(), 0, (size_t) state.size());
            std::memset(conv.data(), 0, (size_t) conv.size());
            std::memset(state_b.data(), 0, (size_t) state_b.size());
            std::memset(conv_b.data(), 0, (size_t) conv_b.size());
            std::memset(state_z.data(), 0, (size_t) state_z.size());
            std::memset(conv_z.data(), 0, (size_t) conv_z.size());
            std::memset(state_f.data(), 0, (size_t) state_f.size());
            std::memset(conv_f.data(), 0, (size_t) conv_f.size());
            Bytes x((uint64_t) n_embd * 4);
            auto* xp = static_cast<float*>(x.data());
            for (int i = 0; i < n_embd; ++i) xp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
            Bytes y((uint64_t) (n_embd + guard) * 4);
            Bytes y2((uint64_t) (n_embd + guard) * 4);
            Bytes yz((uint64_t) (n_embd + guard) * 4);
            Bytes yf((uint64_t) (n_embd + guard) * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) (n_embd + guard), 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) (n_embd + guard), 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) (n_embd + guard), 7.f);
            std::fill_n(static_cast<float*>(yf.data()), (size_t) (n_embd + guard), 7.f);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            GdnLayerState st{&state, &conv};
            GdnLayerState st_b{&state_b, &conv_b};
            GdnLayerState st_z{&state_z, &conv_z};
            GdnLayerState st_f{&state_f, &conv_f};
            gdn_layer_iq(lg, w, st, sc, x, y);
            gdn_layer_iq(lg, w, st_b, sc, x, y2);
            const std::vector<float> host = gdn_iq_host(
                xp, static_cast<const uint8_t*>(qkv->values->data()),
                static_cast<const uint8_t*>(gate->values->data()), static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(norm_w->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            Bytes zero((uint64_t) n_embd * 4);
            std::memset(zero.data(), 0, (size_t) zero.size());
            gdn_layer_iq(lg, w, st_z, sc, zero, yz);
            // Row 0's first code changes the query product and does not reach this output.
            // The query channel with the largest last tap carries one code bit through the step.
            const float* cw = static_cast<const float*>(conv_w->values->data());
            int channel = 0;
            float best_tap = 0.f;
            const int q_channels = (int) (lg.S * lg.h_k);
            for (int c = 0; c < q_channels; ++c) {
                const float tap = std::fabs(cw[(size_t) c * lg.d_conv + (lg.d_conv - 1)]);
                if (tap > best_tap) {
                    best_tap = tap;
                    channel = c;
                }
            }
            const int flip_at = channel * (int) iq_row_bytes(kIq4Xs, n_embd) + 8;
            Bytes flipped(qkv->values->size());
            std::memcpy(flipped.data(), qkv->values->data(), flipped.size());
            static_cast<uint8_t*>(flipped.data())[flip_at] ^= 1u;
            GdnIqWeights wf = w;
            wf.qkv.weights = &flipped;
            gdn_layer_iq(lg, wf, st_f, sc, x, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = yp[n_embd] == 7.f && y2p[n_embd] == 7.f && yzp[n_embd] == 7.f && yfp[n_embd] == 7.f;
            const bool live = all_finite(yp, n_embd) && all_finite(y2p, n_embd) && all_finite(yzp, n_embd) &&
                              all_finite(yfp, n_embd) && all_finite(host.data(), n_embd) && !still_seven(yp, n_embd);
            const double rel = rel_l1(host.data(), yp, n_embd);
            const int again = differ_floats(yp, y2p, n_embd);
            const int apart = differ_floats(yp, yzp, n_embd);
            const int flipped_n = differ_floats(yp, yfp, n_embd);
            const double apart_rel = rel_l1(yp, yzp, n_embd);
            step_ok = live && guard_held && rel <= 1e-5;
            step_repeat = live && guard_held && again == 0;
            step_moved = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            step_flipped = live && guard_held && flipped_n > 0;
            char buf[160];
            std::snprintf(buf, sizeof buf, "rel %.3e, %d rows", rel, n_embd);
            step_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_embd, apart_rel * 100.0);
            move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, channel %d", flipped_n, n_embd, channel);
            flip_detail = buf;
        } catch (const MetalError& e) {
            step_detail = e.what();
            repeat_detail = e.what();
            move_detail = e.what();
            flip_detail = e.what();
        }
    }
    verdict("one Q2_0 GDN step matches", step_bytes && step_ok, step_detail);
    verdict("that GDN step repeats", step_bytes && step_repeat, repeat_detail);
    verdict("a zero activation moves that GDN step", step_bytes && step_moved, move_detail);
    verdict("a changed query code moves that GDN step", step_bytes && step_flipped, flip_detail);

    // The S-form layer is not this path. These projections are ggml blocks. The attention
    // half around the step is the bf16 residual, which already has a kernel.
    const PackTensor* down_w = table.find("blk.0.hc_attn_down.weight");
    const PackTensor* up_w = table.find("blk.0.hc_attn_up.weight");
    const PackTensor* inj_w = table.find("blk.0.hc_attn_inject.weight");
    const PackTensor* norm_h = table.find("blk.0.hc_attn_norm.weight");
    const int hc = 4;
    const int hc_lr = 320;
    const int hc_dim = hc * n_embd;
    const bool half_bytes =
        step_bytes && down_w && down_w->values && down_w->values->size() == (uint64_t) hc_lr * hc_dim * 2 && up_w &&
        up_w->values && up_w->values->size() == (uint64_t) hc_dim * hc_lr * 2 && inj_w && inj_w->values &&
        inj_w->values->size() == (uint64_t) hc * hc_dim * 2 && norm_h && norm_h->values &&
        norm_h->values->size() == (uint64_t) hc_dim * 4;
    bool half_ok = false;
    bool half_repeat = false;
    bool half_moved = false;
    bool half_flipped = false;
    std::string half_detail = "bytes";
    std::string half_repeat_detail = "bytes";
    std::string half_move_detail = "bytes";
    std::string half_flip_detail = "bytes";
    if (half_bytes) {
        try {
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = norm_w->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const GrShapes gs{n_embd, hc, hc_lr};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            auto fresh = [&](Bytes& state, Bytes& conv) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
            };
            Bytes state(state_n * 4), conv(conv_n * 4), state_b(state_n * 4), conv_b(conv_n * 4);
            Bytes state_z(state_n * 4), conv_z(conv_n * 4), state_f(state_n * 4), conv_f(conv_n * 4);
            fresh(state, conv);
            fresh(state_b, conv_b);
            fresh(state_z, conv_z);
            fresh(state_f, conv_f);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int i = 0; i < hc_dim; ++i) Rp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard_n = hc_dim + guard;
            Bytes y((uint64_t) guard_n * 4), y2((uint64_t) guard_n * 4), yz((uint64_t) guard_n * 4);
            Bytes yf((uint64_t) guard_n * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yf.data()), (size_t) guard_n, 7.f);
            Bytes mixed((uint64_t) n_embd * 4), block((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto run_half = [&](Bytes& st_b, Bytes& cv_b, const Bytes& residual, const GdnIqWeights& ww, Bytes& out) {
                GdnLayerState st{&st_b, &cv_b};
                gr_read(gs, residual, *norm_h->values, *down_w->values, *up_w->values, inj_w->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                gdn_layer_iq(lg, ww, st, sc, mixed, block);
                gr_write(gs, residual, block, inject, out);
            };
            run_half(state, conv, R, w, y);
            const AttnHalf host = layer0_attn_host(
                Rp, static_cast<const float*>(norm_h->values->data()),
                static_cast<const uint16_t*>(down_w->values->data()),
                static_cast<const uint16_t*>(up_w->values->data()),
                static_cast<const uint16_t*>(inj_w->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()), static_cast<const uint8_t*>(gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(norm_w->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            std::vector<float> got_mixed((size_t) n_embd), got_block((size_t) n_embd);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_block.data(), block.data(), got_block.size() * 4);
            run_half(state_b, conv_b, R, w, y2);
            run_half(state_z, conv_z, zero_r, w, yz);

            const float* cw = static_cast<const float*>(conv_w->values->data());
            int channel = 0;
            float best_tap = 0.f;
            for (int c = 0; c < (int) (lg.S * lg.h_k); ++c) {
                const float tap = std::fabs(cw[(size_t) c * lg.d_conv + (lg.d_conv - 1)]);
                if (tap > best_tap) {
                    best_tap = tap;
                    channel = c;
                }
            }
            const int flip_at = channel * (int) iq_row_bytes(kIq4Xs, n_embd) + 8;
            Bytes flipped(qkv->values->size());
            std::memcpy(flipped.data(), qkv->values->data(), flipped.size());
            static_cast<uint8_t*>(flipped.data())[flip_at] ^= 1u;
            GdnIqWeights wf = w;
            wf.qkv.weights = &flipped;
            run_half(state_f, conv_f, R, wf, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = yp[hc_dim] == 7.f && y2p[hc_dim] == 7.f && yzp[hc_dim] == 7.f && yfp[hc_dim] == 7.f;
            const bool live = all_finite(yp, hc_dim) && all_finite(y2p, hc_dim) && all_finite(yzp, hc_dim) &&
                              all_finite(yfp, hc_dim) && all_finite(host.mixed.data(), n_embd) &&
                              all_finite(host.block.data(), n_embd) && all_finite(host.out.data(), hc_dim) &&
                              all_finite(got_mixed.data(), n_embd) && all_finite(got_block.data(), n_embd) &&
                              !still_seven(yp, hc_dim);
            const double read_rel = rel_l1(host.mixed.data(), got_mixed.data(), n_embd);
            const double step_rel = rel_l1(host.block.data(), got_block.data(), n_embd);
            const double write_rel = rel_l1(host.out.data(), yp, hc_dim);
            const int again = differ_floats(yp, y2p, hc_dim);
            const int apart = differ_floats(yp, yzp, hc_dim);
            const int flipped_n = differ_floats(yp, yfp, hc_dim);
            const double apart_rel = rel_l1(yp, yzp, hc_dim);
            half_ok = live && guard_held && read_rel <= 1e-4 && step_rel <= 1e-4 && write_rel <= 1e-4;
            half_repeat = live && guard_held && again == 0;
            half_moved = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            half_flipped = live && guard_held && flipped_n > 0;
            char buf[200];
            std::snprintf(buf, sizeof buf, "read %.3e, step %.3e, write %.3e", read_rel, step_rel, write_rel);
            half_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            half_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            half_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, channel %d", flipped_n, hc_dim, channel);
            half_flip_detail = buf;
        } catch (const MetalError& e) {
            half_detail = e.what();
            half_repeat_detail = e.what();
            half_move_detail = e.what();
            half_flip_detail = e.what();
        }
    }
    verdict("the layer-0 attention half matches", half_bytes && half_ok, half_detail);
    verdict("that attention half repeats", half_bytes && half_repeat, half_repeat_detail);
    verdict("a zero residual moves that half", half_bytes && half_moved, half_move_detail);
    verdict("a changed query code moves that half", half_bytes && half_flipped, half_flip_detail);

    // Expert 0 is three raw Q2_0 planes. The wrapper still refuses that gate.
    // The block written back is that expert. The router is checked and does not scale it.
    const PackTensor* ffn_down = table.find("blk.0.hc_ffn_down.weight");
    const PackTensor* ffn_up = table.find("blk.0.hc_ffn_up.weight");
    const PackTensor* ffn_inj = table.find("blk.0.hc_ffn_inject.weight");
    const PackTensor* ffn_norm = table.find("blk.0.hc_ffn_norm.weight");
    const int n_ff = (int) g.n_ff;
    const int n_expert = (int) g.n_expert;
    const bool ffn_bytes =
        blob_n && router && router->values && router->values->size() == (uint64_t) n_embd * (uint64_t) n_expert * 2 &&
        ffn_down && ffn_down->values && ffn_down->values->size() == (uint64_t) hc_lr * hc_dim * 2 && ffn_up &&
        ffn_up->values && ffn_up->values->size() == (uint64_t) hc_dim * hc_lr * 2 && ffn_inj && ffn_inj->values &&
        ffn_inj->values->size() == (uint64_t) hc * hc_dim * 2 && ffn_norm && ffn_norm->values &&
        ffn_norm->values->size() == (uint64_t) hc_dim * 4;
    bool ffn_ok = false;
    bool ffn_repeat = false;
    bool ffn_moved = false;
    bool ffn_flipped = false;
    std::string ffn_detail = "bytes";
    std::string ffn_repeat_detail = "bytes";
    std::string ffn_move_detail = "bytes";
    std::string ffn_flip_detail = "bytes";
    if (ffn_bytes) {
        try {
            const size_t gu = iq_row_bytes(kQ2_0, n_embd);
            const int64_t up_off = (int64_t) n_ff * (int64_t) gu;
            const int64_t down_off = up_off * 2;
            Bytes expert_w((uint64_t) blob.size());
            std::memcpy(expert_w.data(), blob.data(), expert_w.size());
            const GrShapes gs{n_embd, hc, hc_lr};
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int i = 0; i < hc_dim; ++i) Rp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard_n = hc_dim + guard;
            Bytes y((uint64_t) guard_n * 4), y2((uint64_t) guard_n * 4), yz((uint64_t) guard_n * 4);
            Bytes yf((uint64_t) guard_n * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yf.data()), (size_t) guard_n, 7.f);
            Bytes mixed((uint64_t) n_embd * 4), block((uint64_t) (n_embd + guard) * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes xq(iq_q8_1_bytes(n_embd)), gate_y((uint64_t) n_ff * 4), up_y((uint64_t) n_ff * 4);
            Bytes hq(iq_q8_1_bytes(n_ff));
            GrScratch gr = make_gr_scratch(gs);
            auto run_ffn = [&](const Bytes& residual, const Bytes& weights, Bytes& out) {
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                gr_read(gs, residual, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(),
                        1e-6f, false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                iq_quantize_q8_1(mixed, n_embd, xq);
                iq_mmvq(kQ2_0, weights, 0, xq, n_embd, n_ff, 1, gate_y);
                iq_mmvq(kQ2_0, weights, up_off, xq, n_embd, n_ff, 1, up_y);
                iq_swiglu_q8(gate_y, up_y, n_ff, hq);
                iq_mmvq(kQ2_0, weights, down_off, hq, n_ff, n_embd, 1, block);
                gr_write(gs, residual, block, inject, out);
            };
            run_ffn(R, expert_w, y);
            const FfnHalf host = layer0_ffn_host(
                Rp, static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            std::vector<float> got_mixed((size_t) n_embd), got_route((size_t) n_expert), got_block((size_t) n_embd);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_route.data(), route.data(), got_route.size() * 4);
            std::memcpy(got_block.data(), block.data(), got_block.size() * 4);
            // Gate byte 20 changes the direct activation and does not reach this residual.
            // A down-row code that this hidden vector dots does.
            const size_t drow = iq_row_bytes(kQ2_0, n_ff);
            std::vector<uint8_t> hqb((size_t) hq.size());
            std::memcpy(hqb.data(), hq.data(), hqb.size());
            int flip_at = -1;
            std::vector<uint8_t> one_row(drow);
            const uint8_t* down = static_cast<const uint8_t*>(expert_w.data()) + down_off;
            for (int r = 0; r < n_embd && flip_at < 0; ++r) {
                const uint8_t* src = down + (size_t) r * drow;
                const float base = q2_row(src, hqb.data(), n_ff);
                for (int b = 0; b < (int) drow; ++b) {
                    if (b % 18 < 2) continue;
                    std::memcpy(one_row.data(), src, drow);
                    one_row[(size_t) b] ^= 1u;
                    if (q2_row(one_row.data(), hqb.data(), n_ff) != base) {
                        flip_at = (int) down_off + r * (int) drow + b;
                        break;
                    }
                }
            }
            run_ffn(R, expert_w, y2);
            run_ffn(zero_r, expert_w, yz);
            Bytes flipped(expert_w.size());
            std::memcpy(flipped.data(), expert_w.data(), flipped.size());
            if (flip_at >= 0) static_cast<uint8_t*>(flipped.data())[flip_at] ^= 1u;
            run_ffn(R, flipped, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const float* bp = static_cast<const float*>(block.data());
            const bool guard_held = yp[hc_dim] == 7.f && y2p[hc_dim] == 7.f && yzp[hc_dim] == 7.f && yfp[hc_dim] == 7.f &&
                                    bp[n_embd] == 7.f;
            const bool live = all_finite(yp, hc_dim) && all_finite(y2p, hc_dim) && all_finite(yzp, hc_dim) &&
                              all_finite(yfp, hc_dim) && all_finite(host.mixed.data(), n_embd) &&
                              all_finite(host.router.data(), n_expert) && all_finite(host.block.data(), n_embd) &&
                              all_finite(host.out.data(), hc_dim) && all_finite(got_mixed.data(), n_embd) &&
                              all_finite(got_route.data(), n_expert) && all_finite(got_block.data(), n_embd) &&
                              !still_seven(yp, hc_dim) && !still_seven(got_block.data(), n_embd);
            const double read_rel = rel_l1(host.mixed.data(), got_mixed.data(), n_embd);
            const double route_rel = rel_l1(host.router.data(), got_route.data(), n_expert);
            const double expert_rel = rel_l1(host.block.data(), got_block.data(), n_embd);
            const double write_rel = rel_l1(host.out.data(), yp, hc_dim);
            const int again = differ_floats(yp, y2p, hc_dim);
            const int apart = differ_floats(yp, yzp, hc_dim);
            const int flipped_n = differ_floats(yp, yfp, hc_dim);
            const double apart_rel = rel_l1(yp, yzp, hc_dim);
            ffn_ok = live && guard_held && read_rel <= 1e-4 && route_rel <= 1e-4 && expert_rel <= 1e-4 &&
                     write_rel <= 1e-4;
            ffn_repeat = live && guard_held && again == 0;
            ffn_moved = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            ffn_flipped = live && guard_held && flipped_n > 0;
            char buf[220];
            std::snprintf(buf, sizeof buf, "read %.3e, router %.3e, expert %.3e, write %.3e", read_rel, route_rel,
                          expert_rel, write_rel);
            ffn_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            ffn_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            ffn_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, hc_dim, flip_at);
            ffn_flip_detail = buf;
        } catch (const MetalError& e) {
            ffn_detail = e.what();
            ffn_repeat_detail = e.what();
            ffn_move_detail = e.what();
            ffn_flip_detail = e.what();
        }
    }
    verdict("the layer-0 feed-forward half matches", ffn_bytes && ffn_ok, ffn_detail);
    verdict("that feed-forward half repeats", ffn_bytes && ffn_repeat, ffn_repeat_detail);
    verdict("a zero residual moves the feed-forward half", ffn_bytes && ffn_moved, ffn_move_detail);
    verdict("a changed expert code moves that half", ffn_bytes && ffn_flipped, ffn_flip_detail);

    // One layer-0 pass. The attention write is the residual the feed-forward half reads.
    // The router logits stay off the expert block. The shared-expert rows are not read.
    const bool pass_bytes = half_bytes && ffn_bytes;
    bool pass_ok = false;
    bool pass_repeat = false;
    bool pass_moved = false;
    bool pass_flipped = false;
    std::string pass_detail = "bytes";
    std::string pass_repeat_detail = "bytes";
    std::string pass_move_detail = "bytes";
    std::string pass_flip_detail = "bytes";
    if (pass_bytes) {
        try {
            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = norm_w->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const size_t gu = iq_row_bytes(kQ2_0, n_embd);
            const int64_t up_off = (int64_t) n_ff * (int64_t) gu;
            const int64_t down_off = up_off * 2;
            Bytes expert_w((uint64_t) blob.size());
            std::memcpy(expert_w.data(), blob.data(), expert_w.size());
            const GrShapes gs{n_embd, hc, hc_lr};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int i = 0; i < hc_dim; ++i) Rp[i] = ((i * 17) % 29) * 0.02f - 0.2f;
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard_n = hc_dim + guard;
            Bytes y((uint64_t) guard_n * 4), y2((uint64_t) guard_n * 4), yz((uint64_t) guard_n * 4);
            Bytes yf((uint64_t) guard_n * 4), mid((uint64_t) guard_n * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yf.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
            Bytes mixed((uint64_t) n_embd * 4), block((uint64_t) (n_embd + guard) * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes xq(iq_q8_1_bytes(n_embd)), gate_y((uint64_t) n_ff * 4), up_y((uint64_t) n_ff * 4);
            Bytes hq(iq_q8_1_bytes(n_ff));
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto run_pass = [&](const Bytes& residual, const GdnIqWeights& ww, const Bytes& expert, Bytes& out) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *norm_h->values, *down_w->values, *up_w->values, inj_w->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                gdn_layer_iq(lg, ww, st, sc, mixed, block);
                gr_write(gs, residual, block, inject, mid);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                iq_quantize_q8_1(mixed, n_embd, xq);
                iq_mmvq(kQ2_0, expert, 0, xq, n_embd, n_ff, 1, gate_y);
                iq_mmvq(kQ2_0, expert, up_off, xq, n_embd, n_ff, 1, up_y);
                iq_swiglu_q8(gate_y, up_y, n_ff, hq);
                iq_mmvq(kQ2_0, expert, down_off, hq, n_ff, n_embd, 1, block);
                gr_write(gs, mid, block, inject, out);
            };
            run_pass(R, w, expert_w, y);
            std::vector<float> got_attn((size_t) hc_dim), got_mixed((size_t) n_embd), got_route((size_t) n_expert);
            std::vector<float> got_block((size_t) n_embd);
            std::memcpy(got_attn.data(), mid.data(), got_attn.size() * 4);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_route.data(), route.data(), got_route.size() * 4);
            std::memcpy(got_block.data(), block.data(), got_block.size() * 4);
            const AttnHalf host_attn = layer0_attn_host(
                Rp, static_cast<const float*>(norm_h->values->data()),
                static_cast<const uint16_t*>(down_w->values->data()),
                static_cast<const uint16_t*>(up_w->values->data()),
                static_cast<const uint16_t*>(inj_w->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()), static_cast<const uint8_t*>(gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(norm_w->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            // The feed-forward host reads the residual this pass wrote. A second host, from the
            // attention host's residual, records how that small gap moves the expert.
            const FfnHalf host_ffn = layer0_ffn_host(
                got_attn.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const FfnHalf host_chain = layer0_ffn_host(
                host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const bool block_guard = static_cast<const float*>(block.data())[n_embd] == 7.f;
            const bool mid_guard = static_cast<const float*>(mid.data())[hc_dim] == 7.f;
            run_pass(R, w, expert_w, y2);
            run_pass(zero_r, w, expert_w, yz);
            const float* cw = static_cast<const float*>(conv_w->values->data());
            int channel = 0;
            float best_tap = 0.f;
            for (int c = 0; c < (int) (lg.S * lg.h_k); ++c) {
                const float tap = std::fabs(cw[(size_t) c * lg.d_conv + (lg.d_conv - 1)]);
                if (tap > best_tap) {
                    best_tap = tap;
                    channel = c;
                }
            }
            const int flip_at = channel * (int) iq_row_bytes(kIq4Xs, n_embd) + 8;
            Bytes flipped(qkv->values->size());
            std::memcpy(flipped.data(), qkv->values->data(), flipped.size());
            static_cast<uint8_t*>(flipped.data())[flip_at] ^= 1u;
            GdnIqWeights wf = w;
            wf.qkv.weights = &flipped;
            run_pass(R, wf, expert_w, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = yp[hc_dim] == 7.f && y2p[hc_dim] == 7.f && yzp[hc_dim] == 7.f && yfp[hc_dim] == 7.f &&
                                    mid_guard && block_guard;
            const bool live = all_finite(yp, hc_dim) && all_finite(y2p, hc_dim) && all_finite(yzp, hc_dim) &&
                              all_finite(yfp, hc_dim) && all_finite(host_attn.out.data(), hc_dim) &&
                              all_finite(host_ffn.mixed.data(), n_embd) && all_finite(host_ffn.router.data(), n_expert) &&
                              all_finite(host_ffn.block.data(), n_embd) && all_finite(host_ffn.out.data(), hc_dim) &&
                              all_finite(host_chain.block.data(), n_embd) && all_finite(host_chain.out.data(), hc_dim) &&
                              all_finite(got_attn.data(), hc_dim) && all_finite(got_mixed.data(), n_embd) &&
                              all_finite(got_route.data(), n_expert) && all_finite(got_block.data(), n_embd) &&
                              !still_seven(yp, hc_dim) && !still_seven(got_attn.data(), hc_dim);
            const double attn_rel = rel_l1(host_attn.out.data(), got_attn.data(), hc_dim);
            const double read_rel = rel_l1(host_ffn.mixed.data(), got_mixed.data(), n_embd);
            const double route_rel = rel_l1(host_ffn.router.data(), got_route.data(), n_expert);
            const double expert_rel = rel_l1(host_ffn.block.data(), got_block.data(), n_embd);
            const double write_rel = rel_l1(host_ffn.out.data(), yp, hc_dim);
            const double chain_expert = rel_l1(host_chain.block.data(), got_block.data(), n_embd);
            const double chain_write = rel_l1(host_chain.out.data(), yp, hc_dim);
            const int again = differ_floats(yp, y2p, hc_dim);
            const int apart = differ_floats(yp, yzp, hc_dim);
            const int flipped_n = differ_floats(yp, yfp, hc_dim);
            const double apart_rel = rel_l1(yp, yzp, hc_dim);
            pass_ok = live && guard_held && attn_rel <= 1e-4 && read_rel <= 1e-4 && route_rel <= 1e-4 &&
                      expert_rel <= 1e-4 && write_rel <= 1e-4;
            pass_repeat = live && guard_held && again == 0;
            pass_moved = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            pass_flipped = live && guard_held && flipped_n > 0;
            char buf[240];
            std::snprintf(buf, sizeof buf,
                          "attn %.3e, read %.3e, router %.3e, expert %.3e, write %.3e, chain expert %.3e, chain write %.3e",
                          attn_rel, read_rel, route_rel, expert_rel, write_rel, chain_expert, chain_write);
            pass_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            pass_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            pass_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, channel %d", flipped_n, hc_dim, channel);
            pass_flip_detail = buf;
        } catch (const MetalError& e) {
            pass_detail = e.what();
            pass_repeat_detail = e.what();
            pass_move_detail = e.what();
            pass_flip_detail = e.what();
        }
    }
    verdict("the layer-0 pass matches", pass_bytes && pass_ok, pass_detail);
    verdict("that layer-0 pass repeats", pass_bytes && pass_repeat, pass_repeat_detail);
    verdict("a zero residual moves the layer-0 pass", pass_bytes && pass_moved, pass_move_detail);
    verdict("a changed query code moves the layer-0 pass", pass_bytes && pass_flipped, pass_flip_detail);

    // Token 7971 is the embedding row with the largest sum of squares. It is copied into each
    // residual stream and run through the same layer-0 pass. The patterned pass above is unchanged.
    // The head is not called. No token is drawn.
    std::printf("\ntoken 7971 through the layer-0 pass\n");
    const int emb_token = 7971;
    const size_t emb_row_n = (size_t) (n_embd / 256) * 110;
    const PackTensor* emb = table.find("token_embd.weight");
    const bool emb_bytes =
        pass_bytes && emb && emb->values && emb->code_bits == 0 && emb->ne0 == n_embd && emb->ne1 == 248320 &&
        emb->values->size() == (uint64_t) 248320 * emb_row_n &&
        (uint64_t) emb_token * emb_row_n + emb_row_n <= emb->values->size();
    bool emb_ok = false;
    bool emb_repeat = false;
    bool emb_moved = false;
    bool emb_flipped = false;
    std::string emb_detail = "bytes";
    std::string emb_repeat_detail = "bytes";
    std::string emb_move_detail = "bytes";
    std::string emb_flip_detail = "bytes";
    if (emb_bytes) {
        try {
            const uint8_t* emb_src = static_cast<const uint8_t*>(emb->values->data()) + (size_t) emb_token * emb_row_n;
            std::vector<float> emb_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(emb_src + (size_t) b * 110, emb_row.data() + (size_t) b * 256);
            double emb_ms = 0;
            for (int i = 0; i < n_embd; ++i) {
                if (!std::isfinite(emb_row[(size_t) i])) throw MetalError("embedding row is not finite");
                emb_ms += (double) emb_row[(size_t) i] * (double) emb_row[(size_t) i];
            }
            if (emb_ms == 0) throw MetalError("embedding row is zero");

            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = norm_w->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const size_t gu = iq_row_bytes(kQ2_0, n_embd);
            const int64_t up_off = (int64_t) n_ff * (int64_t) gu;
            const int64_t down_off = up_off * 2;
            Bytes expert_w((uint64_t) blob.size());
            std::memcpy(expert_w.data(), blob.data(), expert_w.size());
            const GrShapes gs{n_embd, hc, hc_lr};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, emb_row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard_n = hc_dim + guard;
            Bytes y((uint64_t) guard_n * 4), y2((uint64_t) guard_n * 4), yz((uint64_t) guard_n * 4);
            Bytes yf((uint64_t) guard_n * 4), mid((uint64_t) guard_n * 4);
            std::fill_n(static_cast<float*>(y.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(y2.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yz.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(yf.data()), (size_t) guard_n, 7.f);
            std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
            Bytes mixed((uint64_t) n_embd * 4), block((uint64_t) (n_embd + guard) * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), gate_y((uint64_t) n_ff * 4), up_y((uint64_t) n_ff * 4);
            Bytes hq(iq_q8_1_bytes(n_ff));
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto run_pass = [&](const Bytes& residual, Bytes& out) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *norm_h->values, *down_w->values, *up_w->values, inj_w->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, block);
                gr_write(gs, residual, block, inject, mid);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                iq_quantize_q8_1(mixed, n_embd, exq);
                iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, gate_y);
                iq_mmvq(kQ2_0, expert_w, up_off, exq, n_embd, n_ff, 1, up_y);
                iq_swiglu_q8(gate_y, up_y, n_ff, hq);
                iq_mmvq(kQ2_0, expert_w, down_off, hq, n_ff, n_embd, 1, block);
                gr_write(gs, mid, block, inject, out);
            };
            run_pass(R, y);
            std::vector<float> got_attn((size_t) hc_dim), got_mixed((size_t) n_embd), got_route((size_t) n_expert);
            std::vector<float> got_block((size_t) n_embd);
            std::memcpy(got_attn.data(), mid.data(), got_attn.size() * 4);
            std::memcpy(got_mixed.data(), mixed.data(), got_mixed.size() * 4);
            std::memcpy(got_route.data(), route.data(), got_route.size() * 4);
            std::memcpy(got_block.data(), block.data(), got_block.size() * 4);
            const AttnHalf host_attn = layer0_attn_host(
                Rp, static_cast<const float*>(norm_h->values->data()),
                static_cast<const uint16_t*>(down_w->values->data()),
                static_cast<const uint16_t*>(up_w->values->data()),
                static_cast<const uint16_t*>(inj_w->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()), static_cast<const uint8_t*>(gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(norm_w->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            const FfnHalf host_ffn = layer0_ffn_host(
                got_attn.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const FfnHalf host_chain = layer0_ffn_host(
                host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const bool block_guard = static_cast<const float*>(block.data())[n_embd] == 7.f;
            const bool mid_guard = static_cast<const float*>(mid.data())[hc_dim] == 7.f;
            run_pass(R, y2);
            run_pass(zero_r, yz);
            const int flip_local = 32;
            const int flip_at = (int) ((size_t) emb_token * emb_row_n + (size_t) flip_local);
            std::vector<uint8_t> row_copy(emb_row_n);
            std::memcpy(row_copy.data(), emb_src, emb_row_n);
            row_copy[(size_t) flip_local] ^= 1u;
            std::vector<float> flipped_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
            Bytes Rf((uint64_t) hc_dim * 4);
            auto* Rfp = static_cast<float*>(Rf.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
            run_pass(Rf, yf);

            const float* yp = static_cast<const float*>(y.data());
            const float* y2p = static_cast<const float*>(y2.data());
            const float* yzp = static_cast<const float*>(yz.data());
            const float* yfp = static_cast<const float*>(yf.data());
            const bool guard_held = yp[hc_dim] == 7.f && y2p[hc_dim] == 7.f && yzp[hc_dim] == 7.f && yfp[hc_dim] == 7.f &&
                                    mid_guard && block_guard;
            const bool live = all_finite(yp, hc_dim) && all_finite(y2p, hc_dim) && all_finite(yzp, hc_dim) &&
                              all_finite(yfp, hc_dim) && all_finite(host_attn.out.data(), hc_dim) &&
                              all_finite(host_ffn.mixed.data(), n_embd) && all_finite(host_ffn.router.data(), n_expert) &&
                              all_finite(host_ffn.block.data(), n_embd) && all_finite(host_ffn.out.data(), hc_dim) &&
                              all_finite(host_chain.block.data(), n_embd) && all_finite(host_chain.out.data(), hc_dim) &&
                              all_finite(got_attn.data(), hc_dim) && all_finite(got_mixed.data(), n_embd) &&
                              all_finite(got_route.data(), n_expert) && all_finite(got_block.data(), n_embd) &&
                              !still_seven(yp, hc_dim) && !still_seven(got_attn.data(), hc_dim);
            const double attn_rel = rel_l1(host_attn.out.data(), got_attn.data(), hc_dim);
            const double read_rel = rel_l1(host_ffn.mixed.data(), got_mixed.data(), n_embd);
            const double route_rel = rel_l1(host_ffn.router.data(), got_route.data(), n_expert);
            const double expert_rel = rel_l1(host_ffn.block.data(), got_block.data(), n_embd);
            const double write_rel = rel_l1(host_ffn.out.data(), yp, hc_dim);
            const double chain_expert = rel_l1(host_chain.block.data(), got_block.data(), n_embd);
            const double chain_write = rel_l1(host_chain.out.data(), yp, hc_dim);
            const int again = differ_floats(yp, y2p, hc_dim);
            const int apart = differ_floats(yp, yzp, hc_dim);
            const int flipped_n = differ_floats(yp, yfp, hc_dim);
            const double apart_rel = rel_l1(yp, yzp, hc_dim);
            emb_ok = live && guard_held && attn_rel <= 1e-4 && read_rel <= 1e-4 && route_rel <= 1e-4 &&
                     expert_rel <= 1e-4 && write_rel <= 1e-4;
            emb_repeat = live && guard_held && again == 0;
            emb_moved = live && guard_held && apart > 0 && std::isfinite(apart_rel);
            emb_flipped = live && guard_held && flipped_n > 0;
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "token %d, attn %.3e, read %.3e, router %.3e, expert %.3e, write %.3e, chain expert %.3e, "
                          "chain write %.3e",
                          emb_token, attn_rel, read_rel, route_rel, expert_rel, write_rel, chain_expert, chain_write);
            emb_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            emb_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, hc_dim, apart_rel * 100.0);
            emb_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, hc_dim, flip_at);
            emb_flip_detail = buf;
        } catch (const MetalError& e) {
            emb_detail = e.what();
            emb_repeat_detail = e.what();
            emb_move_detail = e.what();
            emb_flip_detail = e.what();
        }
    }
    verdict("token 7971 through the layer-0 pass matches", emb_bytes && emb_ok, emb_detail);
    verdict("that embedding pass repeats", emb_bytes && emb_repeat, emb_repeat_detail);
    verdict("a zero residual moves that embedding pass", emb_bytes && emb_moved, emb_move_detail);
    verdict("a changed embedding byte moves that pass", emb_bytes && emb_flipped, emb_flip_detail);

    // The layer-0 write of token 7971 goes through the output mixer and the Q5_K output weight.
    // The mixer is checked against that GPU residual. The logits are checked against that GPU mix.
    // The host-of-host chain is reported. The checks above are unchanged. No token is drawn.
    std::printf("\ntoken 7971 layer residual through the head\n");
    const PackTensor* head_out = table.find("output.weight");
    const PackTensor* head_down = table.find("output_hc_down.weight");
    const PackTensor* head_up = table.find("output_hc_up.weight");
    const PackTensor* head_norm = table.find("output_hc_norm.weight");
    const int n_vocab = 248320;
    const size_t out_row_n = (size_t) (n_embd / 256) * 176;
    const bool head_bytes =
        emb_bytes && head_out && head_out->values && head_out->code_bits == 0 && head_out->ne0 == n_embd &&
        head_out->ne1 == n_vocab && head_out->values->size() == (uint64_t) n_vocab * out_row_n &&
        head_down && head_down->values && head_down->kind == strata::core::WeightKind::Bf16InF32 &&
        head_down->code_bits == 0 && head_down->ne0 == hc_dim && head_down->ne1 == hc_lr &&
        head_down->values->size() == (uint64_t) hc_lr * (uint64_t) hc_dim * 2 &&
        head_up && head_up->values && head_up->kind == strata::core::WeightKind::Bf16InF32 &&
        head_up->code_bits == 0 && head_up->ne0 == hc_lr && head_up->ne1 == hc_dim &&
        head_up->values->size() == (uint64_t) hc_dim * (uint64_t) hc_lr * 2 &&
        head_norm && head_norm->values && head_norm->kind == strata::core::WeightKind::F32 &&
        head_norm->code_bits == 0 && head_norm->ne0 == hc_dim && head_norm->ne1 == 1 &&
        head_norm->values->size() == (uint64_t) hc_dim * 4;
    bool lh_mix_ok = false;
    bool lh_logits_ok = false;
    bool lh_repeat = false;
    bool lh_moved = false;
    bool lh_flipped = false;
    std::string lh_mix_detail = "bytes";
    std::string lh_logits_detail = "bytes";
    std::string lh_repeat_detail = "bytes";
    std::string lh_move_detail = "bytes";
    std::string lh_flip_detail = "bytes";
    if (head_bytes) {
        try {
            const int head_token = 7971;
            const size_t head_row_n = (size_t) (n_embd / 256) * 110;
            const uint8_t* head_src =
                static_cast<const uint8_t*>(emb->values->data()) + (size_t) head_token * head_row_n;
            std::vector<float> head_row((size_t) n_embd);
            for (int b = 0; b < n_embd / 256; ++b)
                dequant_q3(head_src + (size_t) b * 110, head_row.data() + (size_t) b * 256);
            double head_ms = 0;
            for (int i = 0; i < n_embd; ++i) {
                if (!std::isfinite(head_row[(size_t) i])) throw MetalError("embedding row is not finite");
                head_ms += (double) head_row[(size_t) i] * (double) head_row[(size_t) i];
            }
            if (head_ms == 0) throw MetalError("embedding row is zero");

            GdnIqWeights w;
            w.qkv = GdnIqPlane{kIq4Xs, qkv->values.get()};
            w.attn_gate = GdnIqPlane{kQ4_K, gate->values.get()};
            w.ssm_out = GdnIqPlane{kQ5_K, ssm->values.get()};
            w.alpha = alpha_w->values.get();
            w.beta = beta_w->values.get();
            w.conv = conv_w->values.get();
            w.ssm_norm = norm_w->values.get();
            w.dt = dt_w->values.get();
            w.ssm_a = ssm_a->values.get();
            const size_t gu = iq_row_bytes(kQ2_0, n_embd);
            const int64_t up_off = (int64_t) n_ff * (int64_t) gu;
            const int64_t down_off = up_off * 2;
            Bytes expert_w((uint64_t) blob.size());
            std::memcpy(expert_w.data(), blob.data(), expert_w.size());
            const GrShapes gs{n_embd, hc, hc_lr};
            const uint64_t state_n = (uint64_t) lg.S * (uint64_t) lg.h_v * (uint64_t) lg.S;
            const uint64_t conv_n = (uint64_t) g.ssm_conv_channels * (uint64_t) (lg.d_conv - 1);
            Bytes state(state_n * 4), conv(conv_n * 4);
            Bytes R((uint64_t) hc_dim * 4);
            auto* Rp = static_cast<float*>(R.data());
            for (int c = 0; c < hc; ++c)
                std::memcpy(Rp + (size_t) c * n_embd, head_row.data(), (size_t) n_embd * sizeof(float));
            Bytes zero_r((uint64_t) hc_dim * 4);
            std::memset(zero_r.data(), 0, (size_t) zero_r.size());
            const int guard_n = hc_dim + guard;
            const int mix_n = n_embd + guard;
            const int logit_n = n_vocab + guard;
            Bytes layer_y((uint64_t) guard_n * 4), layer_y2((uint64_t) guard_n * 4);
            Bytes layer_yz((uint64_t) guard_n * 4), layer_yf((uint64_t) guard_n * 4);
            Bytes mid((uint64_t) guard_n * 4), block((uint64_t) (n_embd + guard) * 4);
            Bytes mixed((uint64_t) n_embd * 4), inject((uint64_t) hc * 4);
            Bytes x_bf16((uint64_t) n_embd * 2), route((uint64_t) n_expert * 4);
            Bytes exq(iq_q8_1_bytes(n_embd)), gate_y((uint64_t) n_ff * 4), up_y((uint64_t) n_ff * 4);
            Bytes hq(iq_q8_1_bytes(n_ff));
            GrScratch gr = make_gr_scratch(gs);
            GdnLayerScratch sc = make_gdn_layer_scratch(lg);
            auto run_pass = [&](const Bytes& residual, Bytes& out) {
                std::memset(state.data(), 0, (size_t) state.size());
                std::memset(conv.data(), 0, (size_t) conv.size());
                std::fill_n(static_cast<float*>(mid.data()), (size_t) guard_n, 7.f);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) guard_n, 7.f);
                GdnLayerState st{&state, &conv};
                gr_read(gs, residual, *norm_h->values, *down_w->values, *up_w->values, inj_w->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                gdn_layer_iq(lg, w, st, sc, mixed, block);
                gr_write(gs, residual, block, inject, mid);
                std::fill_n(static_cast<float*>(block.data()), (size_t) (n_embd + guard), 7.f);
                gr_read(gs, mid, *ffn_norm->values, *ffn_down->values, *ffn_up->values, ffn_inj->values.get(), 1e-6f,
                        false, gr, mixed, inject);
                f32_to_bf16_bulk(mixed, x_bf16, n_embd);
                bf16_gemv(x_bf16, *router->values, route, n_embd, n_expert);
                iq_quantize_q8_1(mixed, n_embd, exq);
                iq_mmvq(kQ2_0, expert_w, 0, exq, n_embd, n_ff, 1, gate_y);
                iq_mmvq(kQ2_0, expert_w, up_off, exq, n_embd, n_ff, 1, up_y);
                iq_swiglu_q8(gate_y, up_y, n_ff, hq);
                iq_mmvq(kQ2_0, expert_w, down_off, hq, n_ff, n_embd, 1, block);
                gr_write(gs, mid, block, inject, out);
            };
            Bytes mix_y((uint64_t) mix_n * 4), mix_y2((uint64_t) mix_n * 4);
            Bytes mix_yz((uint64_t) mix_n * 4), mix_yf((uint64_t) mix_n * 4);
            Bytes logits((uint64_t) logit_n * 4), logits2((uint64_t) logit_n * 4);
            Bytes logits_z((uint64_t) logit_n * 4), logits_f((uint64_t) logit_n * 4);
            Bytes head_inject((uint64_t) (hc + guard) * 4);
            auto fill7 = [](Bytes& b) { std::fill_n(static_cast<float*>(b.data()), (size_t) (b.size() / 4), 7.f); };
            fill7(mix_y);
            fill7(mix_y2);
            fill7(mix_yz);
            fill7(mix_yf);
            fill7(logits);
            fill7(logits2);
            fill7(logits_z);
            fill7(logits_f);
            fill7(head_inject);
            bool inject_held = true;
            auto mix_read = [&](const Bytes& residual, Bytes& out) {
                std::fill_n(static_cast<float*>(head_inject.data()), (size_t) (hc + guard), 7.f);
                std::fill_n(static_cast<float*>(out.data()), (size_t) mix_n, 7.f);
                gr_read(gs, residual, *head_norm->values, *head_down->values, *head_up->values, nullptr, 1e-6f, false,
                        gr, out, head_inject);
                if (!still_seven(static_cast<const float*>(head_inject.data()), hc + guard)) inject_held = false;
            };
            auto project = [&](const Bytes& mix, Bytes& out) {
                Bytes act((uint64_t) n_embd * 4);
                std::memcpy(act.data(), mix.data(), act.size());
                Bytes q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(act, n_embd, q);
                fill7(out);
                iq_mmvq(kQ5_K, *head_out->values, 0, q, n_embd, n_vocab, 1, out);
                return q;
            };

            run_pass(R, layer_y);
            std::vector<float> got_attn((size_t) hc_dim), got_layer((size_t) hc_dim);
            std::memcpy(got_attn.data(), mid.data(), got_attn.size() * 4);
            std::memcpy(got_layer.data(), layer_y.data(), got_layer.size() * 4);
            mix_read(layer_y, mix_y);
            const Bytes head_q = project(mix_y, logits);
            std::vector<uint8_t> xqb((size_t) head_q.size());
            std::memcpy(xqb.data(), head_q.data(), xqb.size());
            const AttnHalf host_attn = layer0_attn_host(
                Rp, static_cast<const float*>(norm_h->values->data()),
                static_cast<const uint16_t*>(down_w->values->data()),
                static_cast<const uint16_t*>(up_w->values->data()),
                static_cast<const uint16_t*>(inj_w->values->data()),
                static_cast<const uint8_t*>(qkv->values->data()), static_cast<const uint8_t*>(gate->values->data()),
                static_cast<const uint8_t*>(ssm->values->data()),
                static_cast<const uint16_t*>(alpha_w->values->data()),
                static_cast<const uint16_t*>(beta_w->values->data()),
                static_cast<const float*>(conv_w->values->data()), static_cast<const float*>(norm_w->values->data()),
                static_cast<const float*>(dt_w->values->data()), static_cast<const float*>(ssm_a->values->data()),
                n_embd, hc, hc_lr, (int) lg.S, (int) lg.h_k, (int) lg.h_v, (int) lg.d_conv);
            const FfnHalf host_ffn = layer0_ffn_host(
                got_attn.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const FfnHalf host_chain = layer0_ffn_host(
                host_attn.out.data(), static_cast<const float*>(ffn_norm->values->data()),
                static_cast<const uint16_t*>(ffn_down->values->data()),
                static_cast<const uint16_t*>(ffn_up->values->data()),
                static_cast<const uint16_t*>(ffn_inj->values->data()),
                static_cast<const uint16_t*>(router->values->data()),
                static_cast<const uint8_t*>(expert_w.data()), n_embd, n_ff, n_expert, hc, hc_lr);
            const float* norm_mix = static_cast<const float*>(head_norm->values->data());
            const uint16_t* down_mix = static_cast<const uint16_t*>(head_down->values->data());
            const uint16_t* up_mix = static_cast<const uint16_t*>(head_up->values->data());
            const std::vector<float> host_mix =
                output_mix_host(got_layer.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            const std::vector<float> chain_mix =
                output_mix_host(host_chain.out.data(), norm_mix, down_mix, up_mix, n_embd, hc, hc_lr, nullptr);
            const uint8_t* out_bytes = static_cast<const uint8_t*>(head_out->values->data());
            std::vector<float> host_logits((size_t) n_vocab), scalar_logits((size_t) n_vocab);
            k_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, host_logits.data());
            scalar_ref(out_bytes, xqb.data(), n_embd, n_vocab, true, scalar_logits.data());
            std::vector<float> chain_logits((size_t) n_vocab);
            double chain_logit_rel = std::nan("");
            bool chain_ran = false;
            if (all_finite(chain_mix.data(), n_embd)) {
                Bytes chain_act((uint64_t) n_embd * 4);
                std::memcpy(chain_act.data(), chain_mix.data(), chain_act.size());
                Bytes chain_q(iq_q8_1_bytes(n_embd));
                iq_quantize_q8_1(chain_act, n_embd, chain_q);
                std::vector<uint8_t> chain_qb((size_t) chain_q.size());
                std::memcpy(chain_qb.data(), chain_q.data(), chain_qb.size());
                k_ref(out_bytes, chain_qb.data(), n_embd, n_vocab, true, chain_logits.data());
                chain_ran = true;
            }

            run_pass(R, layer_y2);
            mix_read(layer_y2, mix_y2);
            const Bytes head_q2 = project(mix_y2, logits2);
            run_pass(zero_r, layer_yz);
            mix_read(layer_yz, mix_yz);
            project(mix_yz, logits_z);

            std::vector<uint8_t> row_copy(head_row_n);
            auto launch_flip = [&](int local, bool with_logits) {
                std::memcpy(row_copy.data(), head_src, head_row_n);
                row_copy[(size_t) local] ^= 1u;
                std::vector<float> flipped_row((size_t) n_embd);
                for (int b = 0; b < n_embd / 256; ++b)
                    dequant_q3(row_copy.data() + (size_t) b * 110, flipped_row.data() + (size_t) b * 256);
                Bytes Rf((uint64_t) hc_dim * 4);
                auto* Rfp = static_cast<float*>(Rf.data());
                for (int c = 0; c < hc; ++c)
                    std::memcpy(Rfp + (size_t) c * n_embd, flipped_row.data(), (size_t) n_embd * sizeof(float));
                run_pass(Rf, layer_yf);
                mix_read(layer_yf, mix_yf);
                if (with_logits) project(mix_yf, logits_f);
            };
            int flip_local = 32;
            launch_flip(flip_local, true);

            const float* layer_p = static_cast<const float*>(layer_y.data());
            const float* layer2_p = static_cast<const float*>(layer_y2.data());
            const float* layerz_p = static_cast<const float*>(layer_yz.data());
            const float* layerf_p = static_cast<const float*>(layer_yf.data());
            const float* mp = static_cast<const float*>(mix_y.data());
            const float* m2p = static_cast<const float*>(mix_y2.data());
            const float* mzp = static_cast<const float*>(mix_yz.data());
            const float* mfp = static_cast<const float*>(mix_yf.data());
            const float* lp = static_cast<const float*>(logits.data());
            const float* l2p = static_cast<const float*>(logits2.data());
            const float* lzp = static_cast<const float*>(logits_z.data());
            const float* lfp = static_cast<const float*>(logits_f.data());
            int flipped_n = differ_floats(lp, lfp, n_vocab);
            if (flipped_n == 0) {
                for (int i = 0; i < (int) head_row_n; ++i) {
                    if (i == 32) continue;
                    launch_flip(i, false);
                    if (differ_floats(mp, mfp, n_embd) == 0) continue;
                    project(mix_yf, logits_f);
                    flipped_n = differ_floats(lp, lfp, n_vocab);
                    if (flipped_n > 0) {
                        flip_local = i;
                        break;
                    }
                }
            }
            const int flip_at = (int) ((size_t) head_token * head_row_n + (size_t) flip_local);
            if (chain_ran && all_finite(chain_logits.data(), n_vocab))
                chain_logit_rel = rel_l1(chain_logits.data(), lp, n_vocab);

            const bool guard_held = layer_p[hc_dim] == 7.f && layer2_p[hc_dim] == 7.f && layerz_p[hc_dim] == 7.f &&
                                    layerf_p[hc_dim] == 7.f && mp[n_embd] == 7.f && m2p[n_embd] == 7.f &&
                                    mzp[n_embd] == 7.f && mfp[n_embd] == 7.f && lp[n_vocab] == 7.f &&
                                    l2p[n_vocab] == 7.f && lzp[n_vocab] == 7.f && lfp[n_vocab] == 7.f;
            const bool q_same = head_q.size() == head_q2.size() &&
                                std::memcmp(head_q.data(), head_q2.data(), head_q.size()) == 0;
            const bool chain_mix_ok = all_finite(chain_mix.data(), n_embd);
            const double chain_mix_rel = chain_mix_ok ? rel_l1(chain_mix.data(), mp, n_embd) : std::nan("");
            const bool live = all_finite(layer_p, hc_dim) && all_finite(layer2_p, hc_dim) && all_finite(layerz_p, hc_dim) &&
                              all_finite(layerf_p, hc_dim) && all_finite(mp, n_embd) && all_finite(m2p, n_embd) &&
                              all_finite(mzp, n_embd) && all_finite(mfp, n_embd) && all_finite(lp, n_vocab) &&
                              all_finite(l2p, n_vocab) && all_finite(lzp, n_vocab) && all_finite(lfp, n_vocab) &&
                              all_finite(host_ffn.out.data(), hc_dim) && all_finite(host_mix.data(), n_embd) &&
                              all_finite(host_logits.data(), n_vocab) && all_finite(scalar_logits.data(), n_vocab) &&
                              !still_seven(layer_p, hc_dim) && !still_seven(mp, n_embd) && !still_seven(lp, n_vocab);
            const double write_rel = rel_l1(host_ffn.out.data(), got_layer.data(), hc_dim);
            const double mix_rel = rel_l1(host_mix.data(), mp, n_embd);
            const double rel = rel_l1(scalar_logits.data(), lp, n_vocab);
            const double irel = rel_l1(host_logits.data(), lp, n_vocab);
            const int again = differ_floats(lp, l2p, n_vocab);
            const int mix_again = differ_floats(mp, m2p, n_embd);
            const int layer_again = differ_floats(layer_p, layer2_p, hc_dim);
            const int apart = differ_floats(lp, lzp, n_vocab);
            const double apart_rel = rel_l1(lp, lzp, n_vocab);
            lh_mix_ok = live && guard_held && inject_held && write_rel <= 1e-4 && mix_rel <= 1e-4;
            lh_logits_ok = live && guard_held && inject_held && rel <= 1e-5 && irel <= 1e-5;
            lh_repeat = live && guard_held && inject_held && again == 0 && mix_again == 0 && layer_again == 0 && q_same;
            lh_moved = live && guard_held && inject_held && apart > 0 && std::isfinite(apart_rel);
            lh_flipped = live && guard_held && inject_held && flipped_n > 0;
            char buf[320];
            std::snprintf(buf, sizeof buf, "token %d, write %.3e, mix %.3e, inject %d, chain mix %.3e", head_token,
                          write_rel, mix_rel, inject_held ? 1 : 0, chain_mix_rel);
            lh_mix_detail = buf;
            std::snprintf(buf, sizeof buf, "rel %.3e, integer %.3e, %d rows, chain %.3e", rel, irel, n_vocab,
                          chain_logit_rel);
            lh_logits_detail = buf;
            std::snprintf(buf, sizeof buf, "%d differ, guard %d", again, guard_held ? 1 : 0);
            lh_repeat_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, %.2f%% apart", apart, n_vocab, apart_rel * 100.0);
            lh_move_detail = buf;
            std::snprintf(buf, sizeof buf, "%d of %d differ, byte %d", flipped_n, n_vocab, flip_at);
            lh_flip_detail = buf;
        } catch (const MetalError& e) {
            lh_mix_detail = e.what();
            lh_logits_detail = e.what();
            lh_repeat_detail = e.what();
            lh_move_detail = e.what();
            lh_flip_detail = e.what();
        }
    }
    verdict("the layer-head mix matches", head_bytes && lh_mix_ok, lh_mix_detail);
    verdict("the layer-head logits match", head_bytes && lh_logits_ok, lh_logits_detail);
    verdict("that layer head repeats", head_bytes && lh_repeat, lh_repeat_detail);
    verdict("a zero residual moves that layer head", head_bytes && lh_moved, lh_move_detail);
    verdict("a changed embedding byte moves that layer head", head_bytes && lh_flipped, lh_flip_detail);

    run_shared_on_layer(table, g, blob);
    run_layer_route(table, g, blob);
    run_routed_ten(table, g, blob);
    run_carried_pair(table, g, blob);
    run_layer1_gdn(g);
    run_layer1_ffn(table, g);
    run_layer2_gdn(g);
    run_layer2_routed(table, g);
    run_layer2_shared(table, g);
    run_layer3_attn();
    run_layer3_qsa(g);
    run_layer3_write(g);
    run_layer3_routed(table, g);
    run_layer3_shared(table, g);
    run_shared_rows(table, g);
    run_vocab_rows(table, g);
    run_output_mix(table, g);
    run_head(table, g);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-pack [--selftest]\n");
        return 2;
    }
    std::printf(
        "pack: index conversions, one product on loaded bytes, name checks, one embedding row, one shared expert, "
        "one routed expert, one 2-bit query, one stack token, one model layer, one IQ4_XS query, "
        "one Q4_K gate, one Q5_K output, one Q4_0 product, one GDN step, one attention half, one feed-forward half, "
        "one layer-0 pass, one embedding through that pass, one layer residual through the head, "
        "one shared expert on that residual, one router selection on that residual, "
        "one routed ten on that residual, one carried state into a second token, "
        "one layer-1 GDN step on that residual, one layer-1 routed expert pass on that residual, "
        "one layer-2 GDN step on that residual, one layer-2 routed expert pass on that residual, "
        "one layer-2 shared expert on that residual, "
        "one layer-3 query, one layer-3 key, one layer-3 value, one layer-3 output, one layer-3 QSA block, one layer-3 attention write, one layer-3 routed expert pass, one layer-3 shared expert, "
        "one shared expert, one Q3_K embedding, "
        "one output weight, one output mixer, one head\n\n");
    try {
        run_conversions();
        run_refusals();
        run_shapes();
        run_bind();
        run_embed();
        run_shexp();
        run_experts();
        run_attn();
        run_stack();
        run_model_layer();
        run_kquant();
        run_q2_layer();
    } catch (const strata::metal::MetalError& e) {
        std::fprintf(stderr, "pack: %s\n", e.what());
        return 2;
    }
    std::printf("\npack: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");
    if (g_fail) return 1;
    std::printf("pack_parity OK\n");
    return 0;
}
