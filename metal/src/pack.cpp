// metal/src/pack.cpp - read index.txt into one buffer per plane.
#include "strata/metal/pack.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/bf16_gemv.hpp"
#include "strata/metal/device.hpp"
#include "strata/metal/elementwise.hpp"
#include "strata/metal/launch.hpp"
#include "strata/metal/s2_gemv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

struct SeN {
    int n;
    int pad0;
    int pad1;
    int pad2;
};
static_assert(sizeof(SeN) == 16, "SeN must match the shader");

void launch_se(const char* fn, const Bytes* const* bufs, int nbuf, int n) {
    const SeN dim{n, 0, 0, 0};
    const uint32_t groups = (uint32_t) ((n + 255) / 256);
    launch(fn, bufs, nbuf, &dim, sizeof dim, Grid{groups, 1, 1, 256, 1, 1});
}

enum class ShexpAct { TwoBit, Q8K, Q80 };

const PackTensor& shexp_row(const PackTable& table, int64_t layer, const char* suffix) {
    const std::string name = layer_name(layer, suffix);
    const PackTensor* r = table.find(name);
    if (!r) throw MetalError(name + " is missing");
    return *r;
}

void need_ne(const PackTensor& t, int64_t ne0, int64_t ne1) {
    if (t.ne0 != ne0)
        throw MetalError(t.name + " ne0 is " + std::to_string(t.ne0) + ", the shared expert requires " +
                         std::to_string(ne0));
    if (t.ne1 != ne1)
        throw MetalError(t.name + " ne1 is " + std::to_string(t.ne1) + ", the shared expert requires " +
                         std::to_string(ne1));
}

void need_planes(const PackTensor& t, int64_t n_in, int64_t n_out) {
    const SForm form = form_of(t);
    const uint64_t codes = sgemv_code_bytes(form, n_in, n_out);
    const uint64_t groups = sgemv_group_bytes(form, n_in, n_out);
    if (!t.codes || t.codes->size() < codes) throw MetalError(t.name + ": codes are short");
    if (!t.scales || t.scales->size() < groups) throw MetalError(t.name + ": scales are short");
    if (t.has_offset != (t.offset_bytes != 0))
        throw MetalError(t.name + ": has_offset does not match the offset plane");
    if (t.has_offset && (!t.offset || t.offset->size() < groups)) throw MetalError(t.name + ": offset is short");
}

ShexpAct classify_shexp(const PackTensor& t, int64_t n_in, const char* role) {
    if (t.code_bits == 2) {
        if (t.group_elems != 64 || t.code_bias != -1 || t.has_offset || t.codebook_iq4nl)
            throw MetalError(t.name + ": the 2-bit product reads group 64, bias -1, no offset, and an affine codebook");
        if (n_in % 64 != 0) throw MetalError(t.name + ": n_in is not a multiple of 64");
        need_planes(t, n_in, t.ne1);
        return ShexpAct::TwoBit;
    }
    if (t.code_bits != 4 && t.code_bits != 8)
        throw MetalError(t.name + ": code_bits " + std::to_string(t.code_bits) + " is not an S2/S4/S8 form");
    if (!group_ok(t.group_elems))
        throw MetalError(t.name + ": group_elems " + std::to_string(t.group_elems) +
                         " is not a power of two and a multiple of 16");
    if (t.ne0 % t.group_elems != 0) throw MetalError(t.name + ": n_in is not a multiple of group_elems");
    if (t.act_kind == 1) {
        if (n_in % 256 != 0)
            throw MetalError(t.name + ": the " + role + " weight wants Q8_K but n_in " + std::to_string(n_in) +
                             " is not a multiple of 256");
        need_planes(t, n_in, t.ne1);
        return ShexpAct::Q8K;
    }
    if (t.act_kind != 0)
        throw MetalError(t.name + ": activation kind " + std::to_string(t.act_kind) + " is not Q8_0 or Q8_K");
    if (n_in % 32 != 0) throw MetalError(t.name + ": n_in is not a multiple of 32");
    need_planes(t, n_in, t.ne1);
    return ShexpAct::Q80;
}

void project_shexp(ShexpAct act, const PackTensor& t, const Bytes& x80, const Bytes* xq8k, Bytes& y, int64_t n_in,
                   int64_t n_out) {
    const SForm form = form_of(t);
    const Bytes* off = t.has_offset ? t.offset.get() : nullptr;
    if (act == ShexpAct::TwoBit) {
        s2_gemv_q8(x80, *t.codes, *t.scales, y, n_in, n_out, 32);
    } else if (act == ShexpAct::Q8K) {
        s_gemv_q8k_split(*xq8k, *t.codes, *t.scales, off, y, n_in, n_out, form);
    } else {
        s_gemv_q8_0_split(x80, *t.codes, *t.scales, off, y, n_in, n_out, form);
    }
}

struct NativeLine {
    int64_t layer = 0;
    int gu = 0;
    int d = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct NativePick {
    int gu_type = 0;
    int d_type = 0;
    uint64_t file_off = 0;
    uint64_t blob_bytes = 0;
    int64_t n_expert = 0;
};

// The GGUF offsets and any shard name sit past the five fields this loader reads.
// false means native_experts.txt is not in the pack. A bad line throws.
bool pick_native_expert(const std::string& pack_dir, int64_t layer, int64_t expert, NativePick& out) {
    const std::string path = pack_dir + "/native_experts.txt";
    File in(std::fopen(path.c_str(), "rb"));
    if (!in) return false;
    int version = 0;
    int64_t n_expert = 0;
    std::vector<NativeLine> lines;
    char buf[1024];
    while (std::fgets(buf, sizeof buf, in.get())) {
        size_t len = std::strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';
        if (len == 0) continue;
        if (buf[0] == '#') {
            static const char tag[] = "# strata native experts v";
            if (std::strncmp(buf, tag, sizeof tag - 1) == 0) version = std::atoi(buf + (sizeof tag - 1));
            const char* at = std::strstr(buf, "(n_expert ");
            if (at) n_expert = std::atoll(at + 10);
            continue;
        }
        long long found = -1;
        int gu = 0, d = 0;
        unsigned long long off = 0, blob = 0;
        if (std::sscanf(buf, "%lld %d %d %llu %llu", &found, &gu, &d, &off, &blob) != 5 || found < 0)
            throw MetalError(std::string("native_experts.txt: a malformed line: ") + buf);
        for (const NativeLine& prev : lines) {
            if (prev.layer == (int64_t) found)
                throw MetalError("native_experts.txt: layer " + std::to_string(found) + " is repeated");
        }
        lines.push_back(NativeLine{(int64_t) found, gu, d, (uint64_t) off, (uint64_t) blob});
    }
    if (version > 4)
        throw MetalError("native_experts.txt is v" + std::to_string(version) + "; this loader reads up to v4");
    if (n_expert <= 0) throw MetalError("native_experts.txt does not say n_expert");
    if (expert < 0 || expert >= n_expert)
        throw MetalError("expert " + std::to_string(expert) + " is outside 0.." + std::to_string(n_expert - 1));
    const NativeLine* row = nullptr;
    for (const NativeLine& line : lines)
        if (line.layer == layer) row = &line;
    if (!row)
        throw MetalError("native_experts.txt: layer " + std::to_string(layer) + " is missing or not contiguous");
    uint64_t at = 0;
    for (int64_t l = 0; l <= layer; ++l) {
        const NativeLine* step = nullptr;
        for (const NativeLine& line : lines)
            if (line.layer == l) step = &line;
        if (!step || step->offset != at)
            throw MetalError("native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous");
        at += step->bytes * (uint64_t) n_expert;
    }
    out.gu_type = row->gu;
    out.d_type = row->d;
    out.file_off = row->offset + (uint64_t) expert * row->bytes;
    out.blob_bytes = row->bytes;
    out.n_expert = n_expert;
    return true;
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

void gather_token_embd(const PackTable& table, const strata::core::ModelGeometry& g, int64_t token, Bytes& out) {
    const PackTensor* w = table.find("token_embd.weight");
    if (!w) throw MetalError("token_embd.weight is missing");
    if (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)
        throw MetalError("embed_row: token_embd.weight is not an S2/S4/S8 tensor");
    if (w->codebook_iq4nl) throw MetalError("embed_row: an IQ4NL embedding is not supported");
    const int per_byte = 8 / w->code_bits;
    if (!w->codes || !w->scales || w->ne0 <= 0 || w->ne1 <= 0 || w->ne0 != g.n_embd || w->group_elems <= 0 ||
        w->ne0 % w->group_elems != 0 || w->ne0 % per_byte != 0) {
        throw MetalError("embed_row: invalid embedding pointers, dimensions or group size");
    }
    if (token < 0 || token >= w->ne1) {
        throw MetalError("embed_row: token " + std::to_string(token) + " is outside 0.." +
                         std::to_string(w->ne1 - 1));
    }
    const uint64_t row_codes = (uint64_t) (w->ne0 / per_byte);
    const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
    const uint64_t rows = (uint64_t) w->ne1;
    const bool planes = w->codes_bytes != 0 && w->scales_bytes != 0 && w->codes->size() == w->codes_bytes &&
                        w->scales->size() == w->scales_bytes && w->codes_bytes % row_codes == 0 &&
                        w->codes_bytes / row_codes == rows && w->codes_bytes % alignof(float) == 0 &&
                        w->scales_bytes % sizeof(float) == 0 &&
                        (w->scales_bytes / sizeof(float)) % row_groups == 0 &&
                        (w->scales_bytes / sizeof(float)) / row_groups == rows &&
                        w->has_offset == (w->offset_bytes != 0) &&
                        (!w->has_offset || (w->offset && w->offset->size() == w->offset_bytes &&
                                            w->offset_bytes == w->scales_bytes));
    if (!planes) throw MetalError("embed_row: embedding planes do not match the loaded tensor shape");
    embedding_gather(*w->codes, *w->scales, w->has_offset ? w->offset.get() : nullptr, w->ne0, w->code_bits,
                     w->code_bias, w->group_elems, token, out, 0);
}

void run_shared_expert(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer, const Bytes& x,
                       Bytes& out) {
    if (layer < 0 || layer >= g.n_layers)
        throw MetalError("layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1));
    if (g.n_embd <= 0 || g.n_ff <= 0) throw MetalError("shared expert: empty shape");
    if (x.size() < (uint64_t) g.n_embd * 4) throw MetalError("shared expert: the activation is short");
    if (out.size() < (uint64_t) g.n_embd * 4) throw MetalError("shared expert: the output is short");

    const PackTensor& gate = shexp_row(table, layer, "ffn_gate_shexp.weight");
    const PackTensor& up = shexp_row(table, layer, "ffn_up_shexp.weight");
    const PackTensor& down = shexp_row(table, layer, "ffn_down_shexp.weight");
    const PackTensor& ginp = shexp_row(table, layer, "ffn_gate_inp_shexp.weight");
    need_ne(gate, g.n_embd, g.n_ff);
    need_ne(up, g.n_embd, g.n_ff);
    need_ne(down, g.n_ff, g.n_embd);
    if (ginp.elements != g.n_embd)
        throw MetalError(ginp.name + " elements is " + std::to_string(ginp.elements) +
                         ", the shared expert requires " + std::to_string(g.n_embd));
    if (ginp.kind != strata::core::WeightKind::Bf16InF32 || ginp.code_bits != 0 || !ginp.values ||
        ginp.values->size() < (uint64_t) g.n_embd * 2) {
        const uint64_t got = ginp.values ? ginp.values->size() : 0;
        throw MetalError(ginp.name + " is engine form " + std::to_string((int) ginp.kind) + " (" +
                         std::to_string(got) + " B); the scalar gate reads it as bf16, so it must be form 1 (" +
                         std::to_string((uint64_t) g.n_embd * 2) + " B)");
    }

    const ShexpAct gate_act = classify_shexp(gate, g.n_embd, "gate");
    const ShexpAct up_act = classify_shexp(up, g.n_embd, "up");
    const ShexpAct down_act = classify_shexp(down, g.n_ff, "down");
    const bool x80 = gate_act != ShexpAct::Q8K || up_act != ShexpAct::Q8K;
    const bool xk = gate_act == ShexpAct::Q8K || up_act == ShexpAct::Q8K;
    const bool h80 = down_act != ShexpAct::Q8K;

    Bytes x_bf16((uint64_t) g.n_embd * 2);
    std::unique_ptr<Bytes> x_q8_0;
    std::unique_ptr<Bytes> x_q8k;
    if (x80) {
        x_q8_0 = std::make_unique<Bytes>(q8_0_bytes(g.n_embd));
        quantize_q8_0(x, g.n_embd, *x_q8_0);
    }
    if (xk) {
        x_q8k = std::make_unique<Bytes>(q8k_bytes(g.n_embd));
        quantize_q8_k(x, g.n_embd, *x_q8k);
    }
    f32_to_bf16_bulk(x, x_bf16, g.n_embd);

    Bytes gate_y((uint64_t) g.n_ff * 4);
    Bytes up_y((uint64_t) g.n_ff * 4);
    const Bytes& act80 = x80 ? *x_q8_0 : *x_q8k;
    project_shexp(gate_act, gate, act80, x_q8k.get(), gate_y, g.n_embd, g.n_ff);
    project_shexp(up_act, up, act80, x_q8k.get(), up_y, g.n_embd, g.n_ff);
    const Bytes* sw[] = {&gate_y, &up_y, &gate_y};
    launch_se("se_swiglu", sw, 3, (int) g.n_ff);

    std::unique_ptr<Bytes> h_q8;
    if (h80) {
        h_q8 = std::make_unique<Bytes>(q8_0_bytes(g.n_ff));
        quantize_q8_0(gate_y, g.n_ff, *h_q8);
    } else {
        h_q8 = std::make_unique<Bytes>(q8k_bytes(g.n_ff));
        quantize_q8_k(gate_y, g.n_ff, *h_q8);
    }
    project_shexp(down_act, down, *h_q8, h_q8.get(), out, g.n_ff, g.n_embd);

    Bytes scalar(4);
    bf16_gemv(x_bf16, *ginp.values, scalar, g.n_embd, 1);
    const Bytes* sc[] = {&out, &scalar};
    launch_se("se_sigmoid_scale", sc, 2, (int) g.n_embd);
}

void run_routed_expert(const PackTable& table, const std::string& pack_dir, const strata::core::ModelGeometry& g,
                       int64_t layer, int64_t expert, const Bytes& x_q8, Bytes& y) {
    if (layer < 0 || layer >= g.n_layers)
        throw MetalError("layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1));
    if (g.n_embd <= 0 || g.n_ff <= 0) throw MetalError("routed expert: empty shape");
    if (x_q8.size() < iq_q8_1_bytes(g.n_embd)) throw MetalError("routed expert: the activation is short");
    if (y.size() < (uint64_t) g.n_embd * 4) throw MetalError("routed expert: the output is short");

    NativePick pick;
    if (!pick_native_expert(pack_dir, layer, expert, pick)) {
        // The index name is the 3-D role span. It is not the concatenated blob, even when
        // the row is loaded. Its bytes are not passed to the expert kernel.
        const std::string role = layer_name(layer, "ffn_gate_exps.weight");
        const PackTensor* row = table.find(role);
        const uint64_t n = (row && row->values) ? row->values->size() : 0;
        if (n != 0) {
            throw MetalError(role + " is " + std::to_string(n) +
                             " bytes and is a 3-D role span, not the concatenated expert blob");
        }
        throw MetalError(role + " is a 3-D role span, not the concatenated expert blob");
    }
    IqExpert shape{pick.gu_type, pick.d_type, g.n_embd, g.n_ff};
    const size_t want = iq_expert_bytes(shape);
    if (pick.blob_bytes != want) {
        throw MetalError("native_experts.txt: layer " + std::to_string(layer) + " blob is " +
                         std::to_string(pick.blob_bytes) + " B but its formats make " + std::to_string(want));
    }
    File bin(std::fopen((pack_dir + "/experts.bin").c_str(), "rb"));
    if (!bin) throw MetalError("experts.bin is missing; a GGUF shard is not this loader");
    Bytes blob(pick.blob_bytes);
    read_at(bin.get(), pick.file_off, blob.data(), (size_t) pick.blob_bytes, "experts.bin");
    iq_expert(blob, x_q8, shape, 1, y);
}

void run_attn_q(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer, const Bytes& x,
                Bytes& y) {
    require_schedule(g, layer);
    if (!strata::core::is_qsa_layer(g, layer))
        throw MetalError("layer " + std::to_string(layer) + " is not full attention");
    const std::string name = layer_name(layer, "attn_q.weight");
    const PackTensor* w = table.find(name);
    if (!w) throw MetalError(name + " is missing");
    if (g.n_embd <= 0 || g.n_head <= 0 || g.head_dim <= 0) throw MetalError("attn_q: empty shape");
    const int64_t n_out = 2 * g.n_head * g.head_dim;
    if (n_out <= 0) throw MetalError("attn_q: empty shape");
    if (w->ne0 != g.n_embd)
        throw MetalError(w->name + " ne0 is " + std::to_string(w->ne0) + ", the query requires " +
                         std::to_string(g.n_embd));
    if (w->ne1 != n_out)
        throw MetalError(w->name + " ne1 is " + std::to_string(w->ne1) + ", the query requires " +
                         std::to_string(n_out));
    if (w->code_bits != 2)
        throw MetalError(w->name + ": code_bits " + std::to_string(w->code_bits) + " is not an S2 form");
    if (w->has_offset || w->offset) throw MetalError(w->name + ": an S2 form must have no offset plane");
    if (w->group_elems != 64 || w->code_bias != -1) {
        throw MetalError(w->name + ": an S2 form must be group 64 with bias -1, this one is group " +
                         std::to_string(w->group_elems) + " with bias " + std::to_string(w->code_bias));
    }
    if (w->codebook_iq4nl) throw MetalError(w->name + ": an S2 form is affine");
    if (!w->codes || !w->scales) throw MetalError(w->name + ": codes or scales are missing");
    if (x.size() < (uint64_t) g.n_embd * 4) throw MetalError("attn_q: the activation is short");
    if (y.size() < (uint64_t) n_out * 4) throw MetalError("attn_q: the output is short");
    // The kernel refuses an n_in that is not a multiple of 64 before it reads the planes.
    Bytes act(q8_0_bytes(g.n_embd));
    quantize_q8_0(x, g.n_embd, act);
    s2_gemv_q8(act, *w->codes, *w->scales, y, g.n_embd, n_out, 32);
}

bool stack_geometry_matches(const strata::core::ModelGeometry& g, const StackGeom& stack) {
    if (g.n_layers < 1 || stack.n_layers < 1 || g.qsa_interval < 1 || stack.qsa_interval < 1) return false;
    if (g.n_embd < 1 || g.n_expert < 1 || g.n_ff < 1 || g.hc < 1 || g.hc_lr < 1) return false;
    if (stack.n_layers != g.n_layers || stack.qsa_interval != g.qsa_interval) return false;
    if (stack.token.n_embd != g.n_embd || stack.token.hc != g.hc || stack.token.hc_lr != g.hc_lr) return false;
    if (stack.token.n_expert != g.n_expert || stack.token.n_ff_shared != g.n_ff) return false;
    if (stack.token.expert.n_embd != g.n_embd || stack.token.expert.n_ff != g.n_ff) return false;
    if (stack.token.qsa.n_embd != g.n_embd) return false;
    const strata::kernels::QsaShapes& q = stack.token.qsa.qsa;
    if (q.n_head != g.n_head || q.n_head_kv != g.n_head_kv || q.head_dim != g.head_dim) return false;
    if (q.idx_n_head != g.idx_q_heads || q.idx_dim != g.idx_key_dim) return false;
    if (stack.gdn.n_embd != g.n_embd || stack.gdn.S != g.ssm_state_size || stack.gdn.h_k != g.ssm_k_heads ||
        stack.gdn.h_v != g.ssm_v_heads || stack.gdn.d_conv != g.ssm_d_conv) {
        return false;
    }
    if (stack.gdn.S <= 0 || stack.gdn.h_v <= 0 || stack.gdn.h_k <= 0) return false;
    const int64_t mixed = 2 * stack.gdn.h_k + stack.gdn.h_v;
    if (mixed <= 0) return false;
    if (g.ssm_value_dim % stack.gdn.h_v != 0 || g.ssm_value_dim / stack.gdn.h_v != stack.gdn.S) return false;
    if (g.ssm_conv_channels % stack.gdn.S != 0 || g.ssm_conv_channels / stack.gdn.S != mixed) return false;
    return true;
}

void bind_stack_embedding(const PackTable& table, const StackGeom& stack, StackWeights& w) {
    const PackTensor* emb = table.find("token_embd.weight");
    if (!emb) throw MetalError("token_embd.weight is missing");
    if (emb->codebook_iq4nl) throw MetalError("token_embd.weight: an IQ4NL embedding is not the stack embedding");
    const TokenGeom& tg = stack.token;
    if (emb->code_bits != tg.emb_code_bits || emb->code_bias != tg.emb_code_bias || emb->group_elems != tg.emb_group ||
        emb->ne0 != tg.n_embd || emb->ne1 != tg.n_vocab) {
        throw MetalError("token_embd.weight does not match the stack embedding");
    }
    if (emb->code_bits != 2 && emb->code_bits != 4 && emb->code_bits != 8)
        throw MetalError("token_embd.weight is not an S2/S4/S8 tensor");
    if (!emb->codes || !emb->scales) throw MetalError("token_embd.weight: codes or scales are missing");
    if (emb->has_offset && !emb->offset) throw MetalError("token_embd.weight: offset is required");
    w.emb_codes = emb->codes.get();
    w.emb_scales = emb->scales.get();
    w.emb_offsets = emb->has_offset ? emb->offset.get() : nullptr;
}

void bind_stack_head(const PackTable& table, const strata::core::ModelGeometry& g, const StackGeom& stack,
                     StackWeights& w) {
    const PackTensor* norm = table.find("output_hc_norm.weight");
    const PackTensor* down = table.find("output_hc_down.weight");
    const PackTensor* up = table.find("output_hc_up.weight");
    if (!norm || !down || !up) throw MetalError("lm_head: an output_hc_* weight is missing");
    if (norm->kind != strata::core::WeightKind::F32 || down->kind != strata::core::WeightKind::Bf16InF32 ||
        up->kind != strata::core::WeightKind::Bf16InF32 || norm->code_bits != 0 || down->code_bits != 0 ||
        up->code_bits != 0) {
        throw MetalError("lm_head: the output_hc_* weights have the wrong engine forms");
    }
    const int64_t hc_dim = g.hc * g.n_embd;
    if (norm->elements != hc_dim || down->ne0 != hc_dim || down->ne1 != g.hc_lr || up->ne0 != g.hc_lr ||
        up->ne1 != hc_dim) {
        throw MetalError("lm_head: the output_hc_* weights have the wrong shape");
    }
    w.head.norm = need_values(*norm, strata::core::WeightKind::F32, 4, "the gated residual reads f32");
    w.head.down = need_values(*down, strata::core::WeightKind::Bf16InF32, 2, "the gated residual reads bf16");
    w.head.up = need_values(*up, strata::core::WeightKind::Bf16InF32, 2, "the gated residual reads bf16");
    w.head.inject = nullptr;

    const PackTensor* out = table.find("output.weight");
    if (!out) throw MetalError("output.weight is missing");
    if (out->ne0 != g.n_embd || out->ne1 != stack.token.n_vocab)
        throw MetalError("output.weight does not match the stack");
    const GdnPlane plane = bind_q8k_plane(*out);
    w.out.form = plane.form;
    w.out.codes = plane.codes;
    w.out.scales = plane.scales;
    w.out.offset = plane.offset;
}

void bind_stack_shexp(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer,
                      StackLayerWeights& w) {
    auto one = [&](const char* suffix, int64_t n_in, int64_t n_out, const Bytes*& codes, const Bytes*& scales) {
        const PackTensor& t = shexp_row(table, layer, suffix);
        if (t.ne0 != n_in || t.ne1 != n_out || t.code_bits != 2 || t.group_elems != 64 || t.code_bias != -1 ||
            t.has_offset || t.codebook_iq4nl || n_in % 64 != 0) {
            throw MetalError(t.name + ": the stack's shared expert is 2-bit, group 64, bias -1, no offset, and affine");
        }
        need_planes(t, n_in, t.ne1);
        codes = t.codes.get();
        scales = t.scales.get();
    };
    one("ffn_gate_shexp.weight", g.n_embd, g.n_ff, w.sh_gate_codes, w.sh_gate_scales);
    one("ffn_up_shexp.weight", g.n_embd, g.n_ff, w.sh_up_codes, w.sh_up_scales);
    one("ffn_down_shexp.weight", g.n_ff, g.n_embd, w.sh_down_codes, w.sh_down_scales);
    const PackTensor& ginp = shexp_row(table, layer, "ffn_gate_inp_shexp.weight");
    if (ginp.elements != g.n_embd)
        throw MetalError(ginp.name + " elements is " + std::to_string(ginp.elements) +
                         ", the shared expert requires " + std::to_string(g.n_embd));
    w.sh_gate_inp = need_values(ginp, strata::core::WeightKind::Bf16InF32, 2, "the scalar gate reads it as bf16");
}

void bind_stack_experts(const std::string& pack_dir, const strata::core::ModelGeometry& g, const StackGeom& stack,
                        int64_t layer, std::vector<Bytes>& blobs, File& bin) {
    blobs.reserve((size_t) g.n_expert);
    const size_t want = iq_expert_bytes(stack.token.expert);
    for (int64_t expert = 0; expert < g.n_expert; ++expert) {
        NativePick pick;
        if (!pick_native_expert(pack_dir, layer, expert, pick))
            throw MetalError("native_experts.txt is missing; a GGUF shard is not this loader");
        if (pick.n_expert != g.n_expert || pick.n_expert != stack.token.n_expert)
            throw MetalError("native_experts.txt n_expert does not match the stack");
        if (pick.gu_type != stack.token.expert.gu_type || pick.d_type != stack.token.expert.d_type)
            throw MetalError("native_experts.txt: layer " + std::to_string(layer) +
                             " types do not match the stack expert");
        if (pick.blob_bytes != want) {
            throw MetalError("native_experts.txt: layer " + std::to_string(layer) + " blob is " +
                             std::to_string(pick.blob_bytes) + " B but its formats make " + std::to_string(want));
        }
        if (!bin) {
            bin.reset(std::fopen((pack_dir + "/experts.bin").c_str(), "rb"));
            if (!bin) throw MetalError("experts.bin is missing; a GGUF shard is not this loader");
        }
        Bytes blob(pick.blob_bytes);
        read_at(bin.get(), pick.file_off, blob.data(), (size_t) pick.blob_bytes, "experts.bin");
        blobs.push_back(std::move(blob));
    }
}

void bind_pack_stack(const PackTable& table, const std::string& pack_dir, const strata::core::ModelGeometry& g,
                     const StackGeom& stack, const Bytes* cos_tab, const Bytes* sin_tab, PackStack& out) {
    out.experts.clear();
    out.layers.clear();
    out.weights = StackWeights{};
    if (!stack_geometry_matches(g, stack)) throw MetalError("the stack geometry does not match the pack");

    bool any_qsa = false;
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        if (strata::core::is_qsa_layer(g, layer)) any_qsa = true;
    }
    if (any_qsa && (cos_tab == nullptr || sin_tab == nullptr))
        throw MetalError("the rope table is not in the pack");

    bind_stack_embedding(table, stack, out.weights);
    bind_stack_head(table, g, stack, out.weights);

    out.layers.resize((size_t) g.n_layers);
    out.experts.resize((size_t) g.n_layers);
    File bin;
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        const PackLayer loaded = bind_layer(table, g, layer);
        StackLayerWeights& w = out.layers[(size_t) layer];
        w.attn = loaded.attn;
        w.ffn = loaded.ffn;
        w.router = loaded.router;
        w.gdn = loaded.gdn;
        w.qsa = loaded.qsa;
        if (loaded.qsa_layer) {
            w.qsa.cos_tab = cos_tab;
            w.qsa.sin_tab = sin_tab;
        }
        bind_stack_shexp(table, g, layer, w);
        bind_stack_experts(pack_dir, g, stack, layer, out.experts[(size_t) layer], bin);
        w.experts = out.experts[(size_t) layer].data();
    }
    out.weights.layers = out.layers.data();
}

}  // namespace strata::metal
