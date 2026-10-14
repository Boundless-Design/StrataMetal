// The pack index, loaded into the kernels that already pass. No token is drawn.
#include "strata/metal/pack.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/s2_gemv.hpp"
#include "strata/metal/sgemv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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
               uint8_t code0) {
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
    for (uint64_t i = 0; i < groups; ++i) {
        const uint16_t h = 0x3c00;
        std::memcpy(src.data() + codes_bytes + i * 2, &h, 2);
    }
    for (uint64_t i = 0; i < groups && has_offset; ++i) {
        const float o = 0.25f;
        std::memcpy(src.data() + codes_bytes + scale_src + i * 4, &o, 4);
    }
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
    for (uint64_t i = 0; i < groups; ++i) {
        const uint16_t h = 0x3c00;
        std::memcpy(src.data() + codes_bytes + i * 2, &h, 2);
    }
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

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-pack [--selftest]\n");
        return 2;
    }
    std::printf(
        "pack: index conversions, one product on loaded bytes, name checks, one embedding row, one shared expert, "
        "one routed expert, one 2-bit query\n\n");
    try {
        run_conversions();
        run_refusals();
        run_shapes();
        run_bind();
        run_embed();
        run_shexp();
        run_experts();
        run_attn();
    } catch (const strata::metal::MetalError& e) {
        std::fprintf(stderr, "pack: %s\n", e.what());
        return 2;
    }
    std::printf("\npack: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");
    if (g_fail) return 1;
    std::printf("pack_parity OK\n");
    return 0;
}
