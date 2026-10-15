// metal/include/strata/metal/layer_tensors.hpp - a layer's tensors, found by name in the GGUF.
//
// The per-layer parity checks used to carry each tensor's byte offset as a literal, one block of
// constants per layer, copied and edited by hand. This header replaces those literals with a lookup:
// `blk.<L>.<suffix>` is resolved in the model's own tensor directory, which gives its shard, its
// offset, its byte count, and its quant type. A wrong name is an error that names the tensor. A
// wrong offset can no longer happen.
//
// Host code only. Nothing here touches Metal, so it builds and runs on any platform.
//
// The model is found from STRATA_MODEL_GGUF, which may name any shard of a split model. Without it,
// the default is $HOME/Developer/Strata-data/models/Q2_0/<the Q2_0 shard 1>, which is where the
// literals came from. A split model's later shards may be absent (a partial download): the shards
// that are present are opened, and a tensor in a missing shard is an error that names that shard.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {

struct LayerTensor {
    std::string name;
    std::string path;          // the shard that holds it
    size_t shard = 0;          // 0 = the metadata shard
    uint64_t rel_offset = 0;   // offset inside that shard's data section, as the directory lists it
    uint64_t file_offset = 0;  // data start + rel_offset: the byte to read from `path`
    uint64_t bytes = 0;        // from the shape and the type's block geometry
    int type = -1;             // ggml type id; the same numbers as kQ2_0, kIq4Xs, ... in iq.hpp
    std::vector<uint64_t> shape;
    const char* type_name() const { return strata::ggml_type_name((uint32_t) type); }
};

enum class LayerKind { Gdn, Qsa };

inline const char* layer_kind_name(LayerKind k) { return k == LayerKind::Gdn ? "GDN" : "QSA"; }

inline std::string layer_tensor_name(int layer, const std::string& suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

// The shard paths of a split model, in split order, WITHOUT requiring that they exist.
// A path without the -NNNNN-of-MMMMM.gguf suffix is a model of one shard.
inline std::vector<std::string> split_paths_unchecked(const std::string& any) {
    static const char tag[] = "-of-";
    const size_t at = any.rfind(tag);
    if (at == std::string::npos || at < 6 || any[at - 6] != '-' || any.size() != at + 4 + 5 + 5 ||
        any.compare(at + 9, 5, ".gguf") != 0)
        return {any};
    int no = 0, n = 0;
    for (size_t i = 0; i < 5; ++i) {
        const char a = any[at - 5 + i], b = any[at + 4 + i];
        if (a < '0' || a > '9' || b < '0' || b > '9') return {any};
        no = no * 10 + (a - '0');
        n = n * 10 + (b - '0');
    }
    if (n < 1 || no < 1 || no > n) return {any};
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char num[16];
        std::snprintf(num, sizeof num, "%05d", i);
        std::string p = any;
        p.replace(at - 5, 5, num);
        out.push_back(std::move(p));
    }
    return out;
}

// STRATA_MODEL_GGUF when set, otherwise the Q2_0 shard 1 under $HOME. Empty when neither can be formed.
inline std::string default_model_shard() {
    if (const char* env = std::getenv("STRATA_MODEL_GGUF"); env && env[0]) return env;
    const char* home = std::getenv("HOME");
    if (!home || !home[0]) return {};
    return std::string(home) +
           "/Developer/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
}

class ModelTensors {
public:
    // Opens every shard of `any_shard`'s model that exists on disk. Shard 1 must exist: it carries the metadata.
    explicit ModelTensors(const std::string& any_shard) {
        paths_ = split_paths_unchecked(any_shard);
        files_.resize(paths_.size());
        for (size_t i = 0; i < paths_.size(); ++i) {
            if (!std::filesystem::is_regular_file(paths_[i])) {
                if (i == 0) throw std::runtime_error("model shard 1 is missing: " + paths_[i]);
                continue;
            }
            files_[i] = std::make_unique<strata::GgufFile>(paths_[i]);
            for (const auto& t : files_[i]->tensors()) {
                if (!index_.emplace(t.name, std::make_pair(i, &t)).second)
                    throw std::runtime_error("GGUF: tensor " + t.name + " is in two shards");
            }
        }
        if (const strata::MetaValue* bc = files_[0]->get("qwen4exp.block_count")) n_layers_ = (int) bc->u;
    }

    // The default model (default_model_shard()), or nullptr with `why` set when it is absent or unreadable.
    static std::unique_ptr<ModelTensors> open_default(std::string* why = nullptr) {
        const std::string p = default_model_shard();
        if (p.empty() || !std::filesystem::is_regular_file(p)) {
            if (why) *why = p.empty() ? "no STRATA_MODEL_GGUF and no HOME" : "no model at " + p;
            return nullptr;
        }
        try {
            return std::make_unique<ModelTensors>(p);
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return nullptr;
        }
    }

    const std::string& path() const { return paths_[0]; }
    size_t shard_count() const { return paths_.size(); }
    bool shard_present(size_t i) const { return i < files_.size() && files_[i] != nullptr; }
    // From qwen4exp.block_count; 0 when the metadata does not say.
    int n_layers() const { return n_layers_; }

    // A metadata value from shard 1, or nullptr. Arrays keep every entry up to the reader's sample cap.
    const strata::MetaValue* meta(const std::string& key) const { return files_[0]->get(key); }

    // A tensor by its full name, or nullptr when no present shard lists it.
    const LayerTensor* find_global(const std::string& name) const {
        auto cached = cache_.find(name);
        if (cached != cache_.end()) return &cached->second;
        auto it = index_.find(name);
        if (it == index_.end()) return nullptr;
        const size_t s = it->second.first;
        const strata::TensorInfo& ti = *it->second.second;
        LayerTensor t;
        t.name = name;
        t.path = paths_[s];
        t.shard = s;
        t.rel_offset = ti.offset;
        t.file_offset = files_[s]->data_start() + ti.offset;
        t.bytes = strata::tensor_payload_bytes(ti);
        t.type = (int) ti.type;
        t.shape = ti.shape;
        const uint64_t payload = files_[s]->file_size() - files_[s]->data_start();
        if (t.bytes == 0 || t.rel_offset > payload || t.bytes > payload - t.rel_offset)
            throw std::runtime_error("GGUF: tensor " + name + " (" + t.type_name() + ") does not fit inside " + t.path);
        return &cache_.emplace(name, std::move(t)).first->second;
    }

    const LayerTensor& global(const std::string& name) const {
        if (const LayerTensor* t = find_global(name)) return *t;
        throw std::runtime_error(missing_message(name));
    }

    const LayerTensor* find(int layer, const std::string& suffix) const {
        return find_global(layer_tensor_name(layer, suffix));
    }

    // Throws naming the tensor, and the missing shard when some shard is not on disk.
    const LayerTensor& at(int layer, const std::string& suffix) const {
        return global(layer_tensor_name(layer, suffix));
    }

    // A QSA layer has attn_q; a GDN layer has attn_qkv. Anything else is refused.
    LayerKind kind(int layer) const {
        const bool q = find(layer, "attn_q.weight") != nullptr;
        const bool qkv = find(layer, "attn_qkv.weight") != nullptr;
        if (q && !qkv) return LayerKind::Qsa;
        if (qkv && !q) return LayerKind::Gdn;
        throw std::runtime_error("layer " + std::to_string(layer) +
                                 (q ? " has both attn_q and attn_qkv" : " has neither attn_q nor attn_qkv") +
                                 (missing_shards().empty() ? "" : "; missing shards: " + missing_shards()));
    }

    // The tensor's bytes, copied out of the mapped shard.
    std::vector<uint8_t> read(const LayerTensor& t) const {
        const uint8_t* p = files_[t.shard]->tensor_data(*index_.at(t.name).second);
        return std::vector<uint8_t>(p, p + t.bytes);
    }
    // `n` bytes at `off` inside the tensor (one expert's rows, say).
    std::vector<uint8_t> read(const LayerTensor& t, uint64_t off, uint64_t n) const {
        if (off > t.bytes || n > t.bytes - off)
            throw std::runtime_error("read past the end of " + t.name);
        const uint8_t* p = files_[t.shard]->tensor_data(*index_.at(t.name).second) + off;
        return std::vector<uint8_t>(p, p + n);
    }

    std::string missing_shards() const {
        std::string s;
        for (size_t i = 0; i < paths_.size(); ++i)
            if (!files_[i]) s += (s.empty() ? "" : ", ") + paths_[i];
        return s;
    }

private:
    std::string missing_message(const std::string& name) const {
        std::string m = "GGUF: no tensor " + name + " in " + paths_[0];
        const std::string gone = missing_shards();
        if (!gone.empty()) m += " (it may be in a shard that is not on disk: " + gone + ")";
        return m;
    }

    std::vector<std::string> paths_;
    std::vector<std::unique_ptr<strata::GgufFile>> files_;
    std::map<std::string, std::pair<size_t, const strata::TensorInfo*>> index_;
    mutable std::map<std::string, LayerTensor> cache_;
    int n_layers_ = 0;
};

}  // namespace strata::metal
