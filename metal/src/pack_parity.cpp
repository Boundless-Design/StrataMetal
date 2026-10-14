// The pack index, loaded into the kernels that already pass. No token is drawn.
#include "strata/metal/pack.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/sgemv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
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

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-pack [--selftest]\n");
        return 2;
    }
    std::printf("pack: index conversions, one product on loaded bytes, name checks\n\n");
    try {
        run_conversions();
        run_refusals();
        run_shapes();
        run_bind();
    } catch (const strata::metal::MetalError& e) {
        std::fprintf(stderr, "pack: %s\n", e.what());
        return 2;
    }
    std::printf("\npack: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");
    if (g_fail) return 1;
    std::printf("pack_parity OK\n");
    return 0;
}
