// metal/src/qsa.mm - launch the paged split-K attention. The scratch layout is the CUDA one:
// per query, [acc: chunks * n_head * 256][max: chunks * n_head][sum: chunks * n_head] and 64 floats of slack.
#include "strata/metal/qsa.hpp"

#include "strata/metal/device.hpp"
#include "strata/metal/launch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::metal {
namespace {

constexpr int kHd = 256;
constexpr int kG = 12;
constexpr int kChunk = 64;
constexpr int kThreads = 256;

struct QsaChunkDim {
    int n_kv_heads;
    int page_size;
    int n_chunks;
    int cap;
    int stride;
    float scale;
};
struct QsaMergeDim {
    int n_chunks;
    int n_head;
    int stride;
};

static_assert(sizeof(QsaChunkDim) == 24, "QsaChunkDim must match the shader");
static_assert(sizeof(QsaMergeDim) == 12, "QsaMergeDim must match the shader");
static_assert(offsetof(QsaChunkDim, scale) == 20, "scale is the last field");

void need(const char* what, const Bytes& buf, uint64_t bytes) {
    if (buf.size() < bytes)
        throw MetalError(std::string(what) + " buffer is " + std::to_string(buf.size()) + " bytes, needs " +
                         std::to_string(bytes));
}

void present(const char* what, const Bytes* buf) {
    if (!buf || buf->size() == 0)
        throw MetalError(std::string("qsa_decode_attn: incomplete KV pools (") + what + ")");
}

// k_q4 wins, then hybrid, then int8, then fp16. Same tests as qsa_decode_attn.cu.
int kv_mode(const QsaPools& p) {
    if (p.k_q4) return 2;
    if (p.k_q8 && p.v_q4) return 3;
    if (p.k_q8) return 1;
    return 0;
}

}  // namespace

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const strata::kernels::QsaShapes& s) {
    const int64_t chunks = cap > 0 ? (cap + kChunk - 1) / kChunk : 0;
    return (uint64_t) chunks * (uint64_t) s.n_head * (kHd + 2) + 64;
}

void qsa_decode_attn(const Bytes& q, const QsaPools& pools, const Bytes& ids, const Bytes& step, int64_t cap,
                     const strata::kernels::QsaShapes& s, Bytes& scratch, Bytes& attn, int64_t n_q) {
    if (n_q == 0) return;
    if (s.head_dim != kHd || s.n_head != (int64_t) kG * s.n_head_kv || s.n_head_kv <= 0 || s.page_size <= 0 ||
        cap <= 0 || n_q < 0 || n_q > 65535 || !pools.page_table) {
        throw MetalError("qsa_decode_attn: unsupported geometry or missing buffers");
    }
    const int mode = kv_mode(pools);
    if (mode == 3) {
        present("k_q8", pools.k_q8);
        present("k_scale", pools.k_scale);
        present("v_q4", pools.v_q4);
    } else if (mode == 2) {
        present("k_q4", pools.k_q4);
        present("v_q4", pools.v_q4);
    } else if (mode == 1) {
        present("k_q8", pools.k_q8);
        present("v_q8", pools.v_q8);
        present("k_scale", pools.k_scale);
        present("v_scale", pools.v_scale);
    } else {
        present("k_f16", pools.k_f16);
        present("v_f16", pools.v_f16);
    }
    present("page_table", pools.page_table);

    const uint64_t stride = qsa_decode_attn_scratch_floats(cap, s);
    if (stride > 0x7fffffffull) throw MetalError("qsa_decode_attn: scratch stride does not fit");
    const int n_chunks = (int) ((cap + kChunk - 1) / kChunk);
    const uint64_t nq = (uint64_t) n_q;
    const uint64_t heads = (uint64_t) s.n_head;
    need("qsa q", q, nq * heads * kHd * 4u);
    need("qsa ids", ids, nq * (uint64_t) cap * 4u);
    need("qsa step", step, nq * (uint64_t) strata::kernels::kStepCount * 4u);
    need("qsa scratch", scratch, nq * stride * 4u);
    need("qsa attn", attn, nq * heads * kHd * 4u);

    QsaChunkDim dim;
    dim.n_kv_heads = (int) s.n_head_kv;
    dim.page_size = (int) s.page_size;
    dim.n_chunks = n_chunks;
    dim.cap = (int) cap;
    dim.stride = (int) stride;
    dim.scale = 1.0f / std::sqrt((float) kHd);

    const Bytes* bufs[9];
    int n = 0;
    const char* kernel = "qsa_chunk_f16";
    bufs[n++] = &q;
    bufs[n++] = &ids;
    bufs[n++] = &step;
    bufs[n++] = pools.page_table;
    bufs[n++] = &scratch;
    if (mode == 0) {
        bufs[n++] = pools.k_f16;
        bufs[n++] = pools.v_f16;
        kernel = "qsa_chunk_f16";
    } else if (mode == 1) {
        bufs[n++] = pools.k_q8;
        bufs[n++] = pools.v_q8;
        bufs[n++] = pools.k_scale;
        bufs[n++] = pools.v_scale;
        kernel = "qsa_chunk_q8";
    } else if (mode == 2) {
        bufs[n++] = pools.k_q4;
        bufs[n++] = pools.v_q4;
        kernel = "qsa_chunk_q4";
    } else {
        bufs[n++] = pools.k_q8;
        bufs[n++] = pools.k_scale;
        bufs[n++] = pools.v_q4;
        kernel = "qsa_chunk_k8v4";
    }
    const Grid chunk{(uint32_t) n_chunks, (uint32_t) s.n_head_kv, (uint32_t) n_q, (uint32_t) kThreads, 1u, 1u};
    launch(kernel, bufs, n, &dim, sizeof(dim), chunk);

    QsaMergeDim merge;
    merge.n_chunks = n_chunks;
    merge.n_head = (int) s.n_head;
    merge.stride = (int) stride;
    const Bytes* mb[] = {&scratch, &attn};
    launch("qsa_attn_merge", mb, 2, &merge, sizeof(merge),
           Grid{(uint32_t) s.n_head, (uint32_t) n_q, 1u, (uint32_t) kHd, 1u, 1u});
}

}  // namespace strata::metal
