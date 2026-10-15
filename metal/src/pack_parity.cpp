// The pack index, loaded into the kernels that already pass. One stack token is drawn from a pack.
#include "strata/metal/pack.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"
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
                               int h_v, int d_conv) {
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

    std::vector<float> z((size_t) V);
    scalar_ref(gate_w, xqb.data(), n_embd, V, false, z.data());
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
    scalar_ref(out_w, yqb.data(), V, n_embd, true, out.data());
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

// Layer 0 of the Q2_0 extract. The other 47 layers are not in the directory.
// The loader does not open the GGUF. A missing directory skips this section.
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
        "one Q4_K gate, one Q5_K output, one GDN step, one attention half\n\n");
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
