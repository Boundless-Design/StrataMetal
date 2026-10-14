// metal/src/pack.cpp - read index.txt into one buffer per plane.
#include "strata/metal/pack.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr uint64_t kChunk = 8ull << 20;

enum class Conv : int { Copy = 0, Bf16High = 1, ToF16 = 2, WidenF16 = 3 };

struct IndexRow {
    std::string name;
    int file = 0;
    strata::core::WeightKind kind = strata::core::WeightKind::Verbatim;
    uint64_t src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0;
    int act_kind = 0;
    bool raw16 = false;
};

struct Seg {
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    Conv conv = Conv::Copy;
    Bytes* dst = nullptr;
};

const char* file_name(int id) {
    switch (id) {
        case 0: return "dense.bin";
        case 1: return "embd.bin";
        case 2: return "experts.bin";
        case 3: return "extra.bin";
        default: return nullptr;
    }
}

struct FileClose {
    void operator()(std::FILE* f) const {
        if (f) std::fclose(f);
    }
};
using File = std::unique_ptr<std::FILE, FileClose>;

void read_at(std::FILE* f, uint64_t off, void* dst, size_t n, const std::string& name) {
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) throw MetalError(name + ": seek failed");
    if (std::fread(dst, 1, n, f) != n) throw MetalError(name + ": short read");
}

std::unique_ptr<Bytes> own(uint64_t n, const std::string& name) {
    if (n == 0) throw MetalError(name + ": empty plane");
    return std::make_unique<Bytes>(n);
}

void fill_plane(std::FILE* f, uint64_t file_off, const Seg& s, const std::string& name) {
    const uint64_t dst_bytes = s.dst->size();
    std::vector<uint8_t> stage((size_t) std::min(kChunk, s.src_bytes ? s.src_bytes : 1));
    uint64_t done = 0;
    while (done < s.src_bytes) {
        uint64_t n = std::min(kChunk, s.src_bytes - done);
        const bool widen = s.conv == Conv::WidenF16;
        if (widen && (n & 1ull)) {
            if (n == 1) throw MetalError(name + ": an fp16 plane ending on a half element");
            n -= 1;
        }
        if ((s.conv == Conv::Bf16High || s.conv == Conv::ToF16) && (n % 4ull))
            throw MetalError(name + ": a promoted plane ending on a partial element");
        read_at(f, file_off + s.src_off + done, stage.data(), (size_t) n, name);
        uint8_t* out = static_cast<uint8_t*>(s.dst->data());
        const uint8_t* src = stage.data();
        switch (s.conv) {
            case Conv::Copy:
                if (done + n > dst_bytes) throw MetalError(name + ": a copied plane overruns the engine bytes");
                std::memcpy(out + done, src, (size_t) n);
                break;
            case Conv::Bf16High: {
                // The high 16 bits are the bf16 encoding. The pack promoted the value
                // without rounding it, so the low half is discarded, not rounded.
                const uint64_t elems = n / 4;
                const uint64_t at = done / 2;
                if (at + elems * 2 > dst_bytes) throw MetalError(name + ": a bf16 plane overruns the engine bytes");
                for (uint64_t i = 0; i < elems; ++i) {
                    uint32_t v;
                    std::memcpy(&v, src + i * 4, 4);
                    const uint16_t h = (uint16_t) (v >> 16);
                    std::memcpy(out + at + i * 2, &h, 2);
                }
                break;
            }
            case Conv::ToF16: {
                const uint64_t elems = n / 4;
                const uint64_t at = done / 2;
                if (at + elems * 2 > dst_bytes) throw MetalError(name + ": an f16 plane overruns the engine bytes");
                for (uint64_t i = 0; i < elems; ++i) {
                    float v;
                    std::memcpy(&v, src + i * 4, 4);
                    const uint16_t h = strata::kernels::f16_from_f32(v);
                    std::memcpy(out + at + i * 2, &h, 2);
                }
                break;
            }
            case Conv::WidenF16: {
                const uint64_t elems = n / 2;
                const uint64_t at = done * 2;
                if (at + elems * 4 > dst_bytes) throw MetalError(name + ": a widened scale overruns the engine bytes");
                float* o = reinterpret_cast<float*>(out + at);
                for (uint64_t i = 0; i < elems; ++i) {
                    uint16_t h;
                    std::memcpy(&h, src + i * 2, 2);
                    o[i] = strata::kernels::f32_from_f16(h);
                }
                break;
            }
        }
        done += n;
    }
    uint64_t expect = s.src_bytes;
    if (s.conv == Conv::Bf16High || s.conv == Conv::ToF16) expect = s.src_bytes / 2;
    if (s.conv == Conv::WidenF16) expect = s.src_bytes * 2;
    if (expect != dst_bytes) {
        throw MetalError(name + ": converted " + std::to_string(expect) + " B into a plane of " +
                         std::to_string(dst_bytes) + " B");
    }
}

std::string layer_name(int64_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

bool fail_ne(std::string& err, int64_t layer, const std::string& name, const char* what, int64_t got,
             int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld", (long long) layer,
                  name.c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

struct Want2 {
    const char* suffix;
    int64_t ne0;
    int64_t ne1;
    bool qsa_only;
    bool gdn_only;
};

struct Want1 {
    const char* suffix;
    int64_t elements;
    strata::core::WeightKind kind;
    bool qsa_only;
    bool gdn_only;
};

const PackTensor* need_row(const PackTable& table, int64_t layer, const char* suffix, std::string& err) {
    const std::string name = layer_name(layer, suffix);
    const PackTensor* r = table.find(name);
    if (!r) {
        err = "layer " + std::to_string(layer) + ": missing " + name;
        return nullptr;
    }
    return r;
}

void require_schedule(const strata::core::ModelGeometry& g, int64_t layer) {
    if (g.qsa_interval < 1) throw MetalError("qsa_interval must be positive");
    if (layer < 0 || layer >= g.n_layers) {
        throw MetalError("layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1));
    }
}

bool group_ok(int group) {
    if (group <= 0 || (group % 16) != 0) return false;
    return (group & (group - 1)) == 0;
}

SForm form_of(const PackTensor& t) {
    SForm form;
    form.code_bits = t.code_bits;
    form.code_bias = t.code_bias;
    form.group_elems = t.group_elems;
    form.codebook = t.codebook_iq4nl ? Codebook::Iq4Nl : Codebook::Affine;
    form.has_offset = t.has_offset;
    return form;
}

// n_in is ne0, the contiguous axis. Q8_K projections are 4-bit or 8-bit only.
GdnPlane bind_q8k_plane(const PackTensor& t) {
    if (t.code_bits != 4 && t.code_bits != 8)
        throw MetalError(t.name + ": code_bits " + std::to_string(t.code_bits) + " has no Q8_K contract");
    if (t.act_kind != 1)
        throw MetalError(t.name + ": activation kind " + std::to_string(t.act_kind) + " is not Q8_K");
    if (!group_ok(t.group_elems))
        throw MetalError(t.name + ": group_elems " + std::to_string(t.group_elems) +
                         " is not a power of two and a multiple of 16");
    if (t.ne0 <= 0 || t.ne1 <= 0) throw MetalError(t.name + ": empty shape");
    if (t.ne0 % t.group_elems != 0) throw MetalError(t.name + ": n_in is not a multiple of group_elems");
    if (t.ne0 % 256 != 0) throw MetalError(t.name + ": n_in is not a multiple of 256");
    if (!t.codes || !t.scales) throw MetalError(t.name + ": codes or scales are missing");
    SForm form = form_of(t);
    if (t.codes->size() < sgemv_code_bytes(form, t.ne0, t.ne1)) throw MetalError(t.name + ": codes are short");
    if (t.scales->size() < sgemv_group_bytes(form, t.ne0, t.ne1)) throw MetalError(t.name + ": scales are short");
    if (t.has_offset) {
        if (!t.offset) throw MetalError(t.name + ": offset is required");
        if (t.offset->size() < sgemv_group_bytes(form, t.ne0, t.ne1)) throw MetalError(t.name + ": offset is short");
    }
    GdnPlane p;
    p.form = form;
    p.codes = t.codes.get();
    p.scales = t.scales.get();
    p.offset = t.has_offset ? t.offset.get() : nullptr;
    return p;
}

const Bytes* need_values(const PackTensor& t, strata::core::WeightKind kind, int width, const char* what) {
    if (t.kind != kind || t.code_bits != 0) throw MetalError(t.name + ": " + what);
    if (!t.values) throw MetalError(t.name + ": missing bytes");
    const uint64_t want = (uint64_t) t.elements * (uint64_t) width;
    if (t.elements <= 0 || t.values->size() < want) throw MetalError(t.name + ": value plane is short");
    return t.values.get();
}

const PackTensor& row_of(const PackTable& table, int64_t layer, const char* suffix) {
    const std::string name = layer_name(layer, suffix);
    const PackTensor* r = table.find(name);
    if (!r) throw MetalError("layer " + std::to_string(layer) + ": missing " + name);
    return *r;
}

TokenGr bind_gr_half(const PackTable& table, int64_t layer, const char* stem) {
    const std::string down = std::string(stem) + "_down.weight";
    const std::string up = std::string(stem) + "_up.weight";
    const std::string inject = std::string(stem) + "_inject.weight";
    const std::string norm = std::string(stem) + "_norm.weight";
    TokenGr g;
    g.down = need_values(row_of(table, layer, down.c_str()), strata::core::WeightKind::Bf16InF32, 2,
                         "the gated residual reads bf16");
    g.up = need_values(row_of(table, layer, up.c_str()), strata::core::WeightKind::Bf16InF32, 2,
                       "the gated residual reads bf16");
    g.inject = need_values(row_of(table, layer, inject.c_str()), strata::core::WeightKind::Bf16InF32, 2,
                           "the gated residual reads bf16");
    g.norm = need_values(row_of(table, layer, norm.c_str()), strata::core::WeightKind::F32, 4,
                         "the gated residual reads f32");
    return g;
}

}  // namespace

PackTable PackTable::load(const std::string& pack_dir) {
    const std::string path = pack_dir + "/index.txt";
    File idx(std::fopen(path.c_str(), "rb"));
    if (!idx) throw MetalError("cannot open " + path + " (run tools/pack_index.py)");

    uint64_t pool = 0;
    std::vector<IndexRow> rows;
    char line[1024];
    while (std::fgets(line, sizeof line, idx.get())) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, t = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) pool = p;
            continue;
        }
        IndexRow r;
        char name[256] = {0};
        int kind = 0;
        const int n = std::sscanf(line, "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d",
                                  name, &r.file, &kind, &r.src_off, &r.src_bytes, &r.dst_off, &r.dst_bytes, &r.ne0,
                                  &r.ne1, &r.code_bits, &r.code_bias, &r.group_elems, &r.codebook, &r.has_offset,
                                  &r.codes_bytes, &r.scales_bytes, &r.offset_bytes, &r.scales_fp16, &r.act_kind);
        if (n != 19) {
            throw MetalError("index.txt: could not parse a row (" + std::to_string(n) + " of 19 fields)");
        }
        r.name = name;
        if (kind == 4 || kind == 5) {
            r.raw16 = true;
            kind = kind == 4 ? (int) strata::core::WeightKind::Bf16InF32 : (int) strata::core::WeightKind::F16InF32;
        }
        r.kind = (strata::core::WeightKind) kind;
        rows.push_back(std::move(r));
    }
    if (pool == 0 || rows.empty()) throw MetalError("index.txt has no header or no rows");

    PackTable table;
    File cur;
    int cur_file = -1;
    for (const IndexRow& r : rows) {
        if (r.code_bits != 0 && r.dst_bytes == 0) {
            throw MetalError(r.name + ": this pack holds the tensor only in its GGUF form (run with --native SHARD1)");
        }

        PackTensor tensor;
        tensor.name = r.name;
        tensor.kind = r.kind;
        tensor.ne0 = r.ne0;
        tensor.ne1 = r.ne1;
        tensor.elements = r.ne0 * (r.ne1 > 0 ? r.ne1 : 1);
        tensor.code_bits = r.code_bits;
        tensor.code_bias = r.code_bias;
        tensor.group_elems = r.group_elems;
        tensor.codebook_iq4nl = r.codebook != 0;
        tensor.has_offset = r.has_offset != 0;
        tensor.scales_fp16 = r.scales_fp16 != 0;
        tensor.raw16 = r.raw16;
        tensor.act_kind = r.act_kind;
        tensor.file_id = r.file;
        tensor.src_off = r.src_off;
        tensor.src_bytes = r.src_bytes;
        tensor.codes_bytes = r.codes_bytes;
        tensor.scales_bytes = r.scales_bytes;
        tensor.offset_bytes = r.offset_bytes;

        std::vector<Seg> segs;
        uint64_t sum_src = 0, sum_dst = 0;
        if (r.codes_bytes != 0) {
            tensor.codes = own(r.codes_bytes, r.name);
            segs.push_back(Seg{0, r.codes_bytes, Conv::Copy, tensor.codes.get()});
            sum_src += r.codes_bytes;
            sum_dst += r.codes_bytes;
            if (r.scales_bytes) {
                const uint64_t src_scale = r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes;
                if (r.scales_fp16 && (r.scales_bytes % 2 || src_scale * 2 != r.scales_bytes))
                    throw MetalError(r.name + ": an fp16 scale plane of an odd byte count");
                tensor.scales = own(r.scales_bytes, r.name);
                segs.push_back(Seg{r.codes_bytes, src_scale, r.scales_fp16 ? Conv::WidenF16 : Conv::Copy,
                                   tensor.scales.get()});
                sum_src += src_scale;
                sum_dst += r.scales_bytes;
            }
            if (r.offset_bytes) {
                const uint64_t after = r.codes_bytes + (r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes);
                tensor.offset = own(r.offset_bytes, r.name);
                segs.push_back(Seg{after, r.offset_bytes, Conv::Copy, tensor.offset.get()});
                sum_src += r.offset_bytes;
                sum_dst += r.offset_bytes;
            }
        } else {
            Conv c = Conv::Copy;
            if (r.raw16) c = Conv::Copy;
            else if (r.kind == strata::core::WeightKind::Bf16InF32) c = Conv::Bf16High;
            else if (r.kind == strata::core::WeightKind::F16InF32) c = Conv::ToF16;
            const uint64_t elems = (uint64_t) r.ne0 * (uint64_t) (r.ne1 > 0 ? r.ne1 : 1);
            if (c != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2)) {
                char buf[384];
                std::snprintf(buf, sizeof buf,
                              "%s: a promoted tensor of %llu elements is %llu B in and %llu B out, not %llu/%llu",
                              r.name.c_str(), (unsigned long long) elems, (unsigned long long) r.src_bytes,
                              (unsigned long long) r.dst_bytes, (unsigned long long) (elems * 4),
                              (unsigned long long) (elems * 2));
                throw MetalError(buf);
            }
            if (r.dst_bytes == 0) throw MetalError(r.name + ": empty tensor");
            tensor.values = own(r.dst_bytes, r.name);
            segs.push_back(Seg{0, r.src_bytes, c, tensor.values.get()});
            sum_src += r.src_bytes;
            sum_dst += r.dst_bytes;
        }
        if (sum_src != r.src_bytes || sum_dst != r.dst_bytes) {
            char buf[420];
            std::snprintf(buf, sizeof buf,
                          "%s: the segments cover %llu/%llu B but the index says %llu/%llu - the plane layout "
                          "this loader built is not the one the index describes",
                          r.name.c_str(), (unsigned long long) sum_src, (unsigned long long) sum_dst,
                          (unsigned long long) r.src_bytes, (unsigned long long) r.dst_bytes);
            throw MetalError(buf);
        }

        if (r.file != cur_file) {
            const char* fn = file_name(r.file);
            const std::string p = pack_dir + "/" + (fn ? fn : "?");
            cur.reset(std::fopen(p.c_str(), "rb"));
            if (!cur) throw MetalError("cannot open " + p);
            cur_file = r.file;
        }
        for (const Seg& s : segs) {
            if (s.conv == Conv::WidenF16 && (s.src_bytes & 1ull))
                throw MetalError(r.name + ": an fp16 plane with an odd source byte count");
            fill_plane(cur.get(), r.src_off, s, r.name);
        }
        table.rows_[r.name] = std::move(tensor);
    }
    return table;
}

const PackTensor* PackTable::find(const std::string& name) const {
    const auto it = rows_.find(name);
    return it == rows_.end() ? nullptr : &it->second;
}

bool check_pack_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer,
                      std::string& err) {
    if (g.qsa_interval < 1) {
        err = "qsa_interval must be positive";
        return false;
    }
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    const bool qsa = strata::core::is_qsa_layer(g, layer);
    const int64_t hc_dim = g.hc_dim();
    const Want2 want2[] = {
        {"hc_attn_down.weight", hc_dim, g.hc_lr, false, false},
        {"hc_attn_up.weight", g.hc_lr, hc_dim, false, false},
        {"hc_attn_inject.weight", hc_dim, g.hc, false, false},
        {"hc_ffn_down.weight", hc_dim, g.hc_lr, false, false},
        {"hc_ffn_up.weight", g.hc_lr, hc_dim, false, false},
        {"hc_ffn_inject.weight", hc_dim, g.hc, false, false},
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, false, false},
        {"attn_qkv.weight", g.n_embd, g.ssm_conv_channels, false, true},
        {"attn_gate.weight", g.n_embd, g.ssm_value_dim, false, true},
        {"ssm_out.weight", g.ssm_value_dim, g.n_embd, false, true},
        {"ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, false, true},
        {"ssm_alpha.weight", g.n_embd, g.ssm_v_heads, false, true},
        {"ssm_beta.weight", g.n_embd, g.ssm_v_heads, false, true},
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, true, false},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, true, false},
        {"indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, true, false},
        {"indexer.k_proj.weight", g.n_embd, g.idx_key_dim, true, false},
    };
    for (const Want2& w : want2) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const PackTensor* r = need_row(table, layer, w.suffix, err);
        if (!r) return false;
        if (r->ne0 != w.ne0) return fail_ne(err, layer, r->name, "ne0", r->ne0, w.ne0);
        if (r->ne1 != w.ne1) return fail_ne(err, layer, r->name, "ne1", r->ne1, w.ne1);
    }
    const Want1 want1[] = {
        {"hc_attn_norm.weight", hc_dim, strata::core::WeightKind::F32, false, false},
        {"hc_ffn_norm.weight", hc_dim, strata::core::WeightKind::F32, false, false},
        {"ffn_gate_inp_shexp.weight", g.n_embd, strata::core::WeightKind::Bf16InF32, false, false},
        {"ssm_a", g.ssm_v_heads, strata::core::WeightKind::F32, false, true},
        {"ssm_dt.bias", g.ssm_v_heads, strata::core::WeightKind::F32, false, true},
        {"ssm_norm.weight", g.ssm_state_size, strata::core::WeightKind::F32, false, true},
        {"attn_q_norm.weight", g.head_dim, strata::core::WeightKind::F32, true, false},
        {"attn_k_norm.weight", g.head_dim, strata::core::WeightKind::F32, true, false},
        {"indexer.q_norm.weight", g.idx_key_dim, strata::core::WeightKind::F32, true, false},
        {"indexer.k_norm.weight", g.idx_key_dim, strata::core::WeightKind::F32, true, false},
    };
    for (const Want1& w : want1) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const PackTensor* r = need_row(table, layer, w.suffix, err);
        if (!r) return false;
        if (r->elements != w.elements) return fail_ne(err, layer, r->name, "elements", r->elements, w.elements);
        if (r->kind != w.kind) {
            const uint64_t want_bytes =
                (uint64_t) w.elements * (w.kind == strata::core::WeightKind::Bf16InF32 ? 2u : 4u);
            const uint64_t got_bytes = r->values ? r->values->size() : 0;
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, r->name.c_str(), (int) r->kind, (unsigned long long) got_bytes,
                          (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

GdnLayerWeights bind_gdn_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer) {
    require_schedule(g, layer);
    if (strata::core::is_qsa_layer(g, layer))
        throw MetalError("layer " + std::to_string(layer) + " is full attention");
    std::string err;
    if (!check_pack_layer(table, g, layer, err)) throw MetalError(err);
    GdnLayerWeights w;
    w.qkv = bind_q8k_plane(row_of(table, layer, "attn_qkv.weight"));
    w.attn_gate = bind_q8k_plane(row_of(table, layer, "attn_gate.weight"));
    w.ssm_out = bind_q8k_plane(row_of(table, layer, "ssm_out.weight"));
    w.alpha = need_values(row_of(table, layer, "ssm_alpha.weight"), strata::core::WeightKind::Bf16InF32, 2,
                          "the layer reads bf16");
    w.beta = need_values(row_of(table, layer, "ssm_beta.weight"), strata::core::WeightKind::Bf16InF32, 2,
                         "the layer reads bf16");
    w.conv = need_values(row_of(table, layer, "ssm_conv1d.weight"), strata::core::WeightKind::F32, 4,
                         "the convolution reads f32");
    w.ssm_norm = need_values(row_of(table, layer, "ssm_norm.weight"), strata::core::WeightKind::F32, 4,
                             "the layer reads f32");
    w.dt = need_values(row_of(table, layer, "ssm_dt.bias"), strata::core::WeightKind::F32, 4, "the layer reads f32");
    w.ssm_a = need_values(row_of(table, layer, "ssm_a"), strata::core::WeightKind::F32, 4, "the layer reads f32");
    return w;
}

QsaLayerWeights bind_qsa_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer) {
    require_schedule(g, layer);
    if (!strata::core::is_qsa_layer(g, layer))
        throw MetalError("layer " + std::to_string(layer) + " is not full attention");
    std::string err;
    if (!check_pack_layer(table, g, layer, err)) throw MetalError(err);
    auto plane = [](const GdnPlane& p) {
        QsaPlane q;
        q.form = p.form;
        q.codes = p.codes;
        q.scales = p.scales;
        q.offset = p.offset;
        return q;
    };
    QsaLayerWeights w;
    w.q = plane(bind_q8k_plane(row_of(table, layer, "attn_q.weight")));
    w.k = plane(bind_q8k_plane(row_of(table, layer, "attn_k.weight")));
    w.v = plane(bind_q8k_plane(row_of(table, layer, "attn_v.weight")));
    w.out = plane(bind_q8k_plane(row_of(table, layer, "attn_output.weight")));
    w.idx_q = need_values(row_of(table, layer, "indexer.q_proj.weight"), strata::core::WeightKind::Bf16InF32, 2,
                          "quantized; the attention block reads it as bf16");
    w.idx_k = need_values(row_of(table, layer, "indexer.k_proj.weight"), strata::core::WeightKind::Bf16InF32, 2,
                          "quantized; the attention block reads it as bf16");
    w.q_norm = need_values(row_of(table, layer, "attn_q_norm.weight"), strata::core::WeightKind::F32, 4,
                           "the layer reads f32");
    w.k_norm = need_values(row_of(table, layer, "attn_k_norm.weight"), strata::core::WeightKind::F32, 4,
                           "the layer reads f32");
    w.idx_q_norm = need_values(row_of(table, layer, "indexer.q_norm.weight"), strata::core::WeightKind::F32, 4,
                               "the layer reads f32");
    w.idx_k_norm = need_values(row_of(table, layer, "indexer.k_norm.weight"), strata::core::WeightKind::F32, 4,
                               "the layer reads f32");
    w.cos_tab = nullptr;
    w.sin_tab = nullptr;
    return w;
}

PackLayer bind_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer) {
    require_schedule(g, layer);
    std::string err;
    if (!check_pack_layer(table, g, layer, err)) throw MetalError(err);
    PackLayer out;
    out.attn = bind_gr_half(table, layer, "hc_attn");
    out.ffn = bind_gr_half(table, layer, "hc_ffn");
    out.router = need_values(row_of(table, layer, "ffn_gate_inp.weight"), strata::core::WeightKind::Bf16InF32, 2,
                             "the router reads bf16");
    if (strata::core::is_qsa_layer(g, layer)) {
        out.qsa_layer = true;
        out.qsa = bind_qsa_layer(table, g, layer);
    } else {
        out.gdn = bind_gdn_layer(table, g, layer);
    }
    return out;
}

}  // namespace strata::metal
