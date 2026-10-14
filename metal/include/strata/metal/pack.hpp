// metal/include/strata/metal/pack.hpp - the pack index, in the form the kernels read.
//
// The engine does not read a GGUF block. The index is nineteen fields, written by
// tools/pack_index.py, plus the bins beside it. This loader applies the same conversions
// and puts each plane in its own shared buffer. A scale plane can widen from fp16 to f32,
// and that widen is exact. A bf16 value promoted to f32 gives back its high 16 bits,
// which is a copy, not a rounding. The kernels take Bytes, so a pointer into the middle
// of one allocation is not a plane.
//
// Names and shapes are the layer contract. A GDN layer has attn_qkv and no attn_q.
// A full-attention layer is the other way around. The interval is the pack rule,
// layer % interval == interval - 1. Rope tables, the page table, and the recurrence
// are built at runtime. They are not in the pack.
//
// The embedding is gathered only as an S2, S4, or S8 row. The shared expert's three
// projections each go to the kernel that accepts that form. The 2-bit shared_expert
// function is not called. A 3-D routed-expert span is not passed to the IQ expert.
// One concatenated blob from experts.bin is. A 2-bit plane is not bound onto a Q8_K
// projection. This does not run the model. There is no token rate.
#pragma once

#include "strata/metal/gdn_layer.hpp"
#include "strata/metal/token.hpp"

#include "strata/core/layout.hpp"

#include <map>
#include <memory>
#include <string>

namespace strata::metal {

// One tensor after conversion. Unquantized rows fill `values`. Quantized rows fill
// `codes`, `scales`, and `offset` when the form has one. The table owns the buffers.
// A pointer taken from here is valid while the table is.
struct PackTensor {
    std::string name;
    strata::core::WeightKind kind = strata::core::WeightKind::Verbatim;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t elements = 0;
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;
    bool scales_fp16 = false;  // the pack's width, before the widen
    bool raw16 = false;        // kinds 4 and 5: already 16 bits in the file
    int act_kind = 0;          // 0 is Q8_0, 1 is Q8_K
    int file_id = 0;
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t offset_bytes = 0;
    std::unique_ptr<Bytes> values;
    std::unique_ptr<Bytes> codes;
    std::unique_ptr<Bytes> scales;
    std::unique_ptr<Bytes> offset;
};

// What one layer's kernels can read. The mixer that does not run is left empty.
// cos_tab and sin_tab stay null: the rope table is not in the pack.
// The routed experts are not here. The embedding and the shared expert are run on
// their own. The embedding is only an S2, S4, or S8 row.
struct PackLayer {
    bool qsa_layer = false;
    TokenGr attn;
    TokenGr ffn;
    const Bytes* router = nullptr;
    GdnLayerWeights gdn;
    QsaLayerWeights qsa;
};

class PackTable {
public:
    PackTable() = default;
    PackTable(const PackTable&) = delete;
    PackTable& operator=(const PackTable&) = delete;
    PackTable(PackTable&&) = default;
    PackTable& operator=(PackTable&&) = default;

    // Throws MetalError. The messages for a bad index match the engine loader.
    // WeightTable::load is not called. It uploads with CUDA.
    static PackTable load(const std::string& pack_dir);

    const PackTensor* find(const std::string& name) const;
    size_t size() const { return rows_.size(); }

private:
    std::map<std::string, PackTensor> rows_;
};

// Names, shapes, and the 1-D engine form. The same rules as check_layer.
// A non-positive interval is refused before any modulo. False fills `err`.
bool check_pack_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer,
                      std::string& err);

// The planes gdn_layer reads. Throws when the layer is full attention, a tensor is
// missing or the wrong shape, or a projection is not a 4-bit or 8-bit Q8_K plane.
GdnLayerWeights bind_gdn_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer);

// The planes qsa_layer reads, except the rope table. Throws when the layer is not full
// attention, or a projection is not a 4-bit or 8-bit Q8_K plane, or an indexer matrix
// is not bf16.
QsaLayerWeights bind_qsa_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer);

// Gated residual, router, and the mixer for this layer. The table must outlive the pointers.
// The embedding and the shared expert are not among those planes.
PackLayer bind_layer(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer);

// One row of token_embd.weight. The row length is g.n_embd and the token selects the row.
// S2, S4, and S8 only. An IQ4NL codebook is refused. A bf16 table is refused. The residual
// is not written, and output.weight is not read.
void gather_token_embd(const PackTable& table, const strata::core::ModelGeometry& g, int64_t token, Bytes& out);

// h = silu(x @ gate) * (x @ up), then h @ down, then times sigmoid(x · gate_inp).
// A 2-bit plane must be group 64, bias -1, and affine, with no offset. A 4-bit or 8-bit
// plane with activation kind 1 uses the Q8_K product, and that product needs n_in to be
// a multiple of 256. Anything else uses the Q8_0 product, whose n_in is a multiple of 32.
// The down weight's n_in is n_ff. The residual is not written, and the routed experts are
// not read. x holds n_embd floats. out holds at least n_embd floats.
void run_shared_expert(const PackTable& table, const strata::core::ModelGeometry& g, int64_t layer, const Bytes& x,
                       Bytes& out);

// One routed expert. experts.bin holds [gate rows | up rows | down rows] for each expert.
// native_experts.txt says the type, the file offset, and the blob size. The count in that
// file bounds the expert index. blk.L.ffn_*_exps.weight is a 3-D role span and is not read.
// A GGUF shard is not opened. x_q8 is one q8_1 row of n_embd. y holds at least n_embd floats.
// The residual is not written. This does not draw a token.
void run_routed_expert(const PackTable& table, const std::string& pack_dir, const strata::core::ModelGeometry& g,
                       int64_t layer, int64_t expert, const Bytes& x_q8, Bytes& y);

}  // namespace strata::metal
