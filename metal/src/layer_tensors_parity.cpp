// metal/src/layer_tensors_parity.cpp - strata-metal-layer-tensors: the by-name tensor lookup.
//
// --selftest writes a small two-shard GGUF and checks the lookup against it: names, offsets, byte
// counts, types, the layer kind, a tensor in the second shard, a missing second shard, and a
// missing name. Host code only: no Metal, so this builds and runs anywhere.

#include "strata/metal/layer_tensors.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-58s %s%s%s\n", name, ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}

// ---- a minimal GGUF v3 writer, enough for the lookup -------------------------------------------------

struct OutTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type;
    uint64_t bytes;
};

struct OutKv {
    std::string key;
    uint32_t type;  // 4 = u32, 8 = string
    uint32_t u = 0;
    std::string s;
};

void put_u32(std::vector<uint8_t>& b, uint32_t v) { b.insert(b.end(), (uint8_t*) &v, (uint8_t*) &v + 4); }
void put_u64(std::vector<uint8_t>& b, uint64_t v) { b.insert(b.end(), (uint8_t*) &v, (uint8_t*) &v + 8); }
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// Writes the file; returns each tensor's offset inside the data section.
std::vector<uint64_t> write_gguf(const std::filesystem::path& p, const std::vector<OutKv>& kvs,
                                 const std::vector<OutTensor>& ts) {
    const uint64_t align = 32;
    std::vector<uint8_t> h;
    put_u32(h, 0x46554747u);
    put_u32(h, 3);
    put_u64(h, ts.size());
    put_u64(h, kvs.size());
    for (const auto& kv : kvs) {
        put_str(h, kv.key);
        put_u32(h, kv.type);
        if (kv.type == 8) put_str(h, kv.s);
        else put_u32(h, kv.u);
    }
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const auto& t : ts) {
        put_str(h, t.name);
        put_u32(h, (uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put_u64(h, d);
        put_u32(h, t.type);
        put_u64(h, off);
        offs.push_back(off);
        off = (off + t.bytes + align - 1) / align * align;
    }
    while (h.size() % align) h.push_back(0);
    std::vector<uint8_t> data(off, 0);
    for (size_t i = 0; i < ts.size(); ++i)
        for (uint64_t j = 0; j < ts[i].bytes; ++j) data[offs[i] + j] = (uint8_t) (i * 31 + j);
    std::ofstream f(p, std::ios::binary);
    f.write((const char*) h.data(), (std::streamsize) h.size());
    f.write((const char*) data.data(), (std::streamsize) data.size());
    return offs;
}

std::string fmt(const char* f, unsigned long long a, unsigned long long b) {
    char buf[160];
    std::snprintf(buf, sizeof buf, f, a, b);
    return buf;
}

void run_synthetic() {
    std::printf("\nlookup on a written two-shard GGUF\n");
    const auto dir = std::filesystem::temp_directory_path() / "strata_layer_tensors_check";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto s1 = dir / "tiny-00001-of-00002.gguf";
    const auto s2 = dir / "tiny-00002-of-00002.gguf";
    // Layer 0 is GDN, layer 1 is QSA, and layer 1's output projection sits in shard 2.
    const std::vector<OutTensor> t1 = {
        {"blk.0.attn_gate.weight", {256, 4}, 23, 4 * 136},      // IQ4_XS: 256 per 136 B
        {"blk.0.attn_qkv.weight", {256, 6}, 11, 6 * 110},       // Q3_K
        {"blk.0.ssm_a", {48}, 0, 48 * 4},                       // F32
        {"blk.1.attn_k.weight", {256, 2}, 13, 2 * 176},         // Q5_K
        {"blk.1.attn_q.weight", {256, 8}, 23, 8 * 136},
        {"blk.1.ffn_gate_exps.weight", {64, 4, 8}, 42, 8 * 4 * 18},  // Q2_0: 64 per 18 B, 8 experts
    };
    const std::vector<OutTensor> t2 = {{"blk.1.attn_output.weight", {512, 256}, 13, 256 * 2 * 176}};
    const std::vector<OutKv> kv1 = {{"general.architecture", 8, 0, "qwen4exp"},
                                    {"qwen4exp.block_count", 4, 2, ""},
                                    {"split.count", 4, 2, ""},
                                    {"split.no", 4, 0, ""}};
    const std::vector<OutKv> kv2 = {{"split.count", 4, 2, ""}, {"split.no", 4, 1, ""}};
    const auto o1 = write_gguf(s1, kv1, t1);
    const auto o2 = write_gguf(s2, kv2, t2);

    try {
        strata::metal::ModelTensors m(s2.string());  // any shard names the model
        verdict("both shards open from the second shard's name", m.shard_count() == 2 && m.shard_present(1),
                m.path());
        verdict("block_count comes from the metadata", m.n_layers() == 2, std::to_string(m.n_layers()));

        const auto& qkv = m.at(0, "attn_qkv.weight");
        verdict("a GDN tensor resolves to its directory offset",
                qkv.rel_offset == o1[1] && qkv.bytes == 6 * 110 && qkv.type == 11 && qkv.shard == 0,
                fmt("offset %llu, %llu B", qkv.rel_offset, qkv.bytes) + ", " + qkv.type_name());
        const auto raw = m.read(qkv);
        verdict("its bytes are the bytes written there", raw.size() == 660 && raw[0] == 31 && raw[5] == 36, "");
        const auto part = m.read(m.at(1, "ffn_gate_exps.weight"), 3 * 72, 72);
        verdict("one expert's rows read at an offset inside a tensor",
                part.size() == 72 && part[0] == (uint8_t) (5 * 31 + 3 * 72), "");

        const auto& out = m.at(1, "attn_output.weight");
        verdict("a tensor in shard 2 resolves there",
                out.shard == 1 && out.rel_offset == o2[0] && out.bytes == 256 * 2 * 176 && out.type == 13,
                out.path);
        verdict("file offset is data start + directory offset", out.file_offset > out.rel_offset, "");

        verdict("attn_qkv makes layer 0 GDN", m.kind(0) == strata::metal::LayerKind::Gdn, "");
        verdict("attn_q makes layer 1 QSA", m.kind(1) == strata::metal::LayerKind::Qsa, "");

        std::string got;
        try {
            m.at(1, "ffn_up_exps.weight");
        } catch (const std::exception& e) {
            got = e.what();
        }
        verdict("a missing name is an error that names it", got.find("blk.1.ffn_up_exps.weight") != std::string::npos,
                got);
        got.clear();
        try {
            m.kind(5);
        } catch (const std::exception& e) {
            got = e.what();
        }
        verdict("a layer with no attention is refused", got.find("neither") != std::string::npos, got);
    } catch (const std::exception& e) {
        verdict("the two-shard model opens", false, e.what());
    }

    std::filesystem::remove(s2);
    try {
        strata::metal::ModelTensors m(s1.string());
        verdict("shard 1 alone still opens", !m.shard_present(1), "");
        std::string got;
        try {
            m.at(1, "attn_output.weight");
        } catch (const std::exception& e) {
            got = e.what();
        }
        verdict("a tensor in the absent shard names that shard", got.find("00002-of-00002") != std::string::npos, got);
        verdict("shard-1 tensors still resolve", m.at(1, "attn_q.weight").rel_offset == o1[4], "");
    } catch (const std::exception& e) {
        verdict("shard 1 alone still opens", false, e.what());
    }
    std::filesystem::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-layer-tensors [--selftest]\n");
        return 2;
    }
    std::printf("layer tensors: by-name lookup, two shards, a missing shard\n");
    run_synthetic();
    std::printf("\nlayer tensors: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");
    if (g_fail) return 1;
    std::printf("layer_tensors_parity OK\n");
    return 0;
}
