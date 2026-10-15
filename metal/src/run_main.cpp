// strata-metal-run: token ids in, a logit row per position out.
//
// --tokens are the prompt, with no BOS and no chat template. --max-new 0 scores the prompt except its
// last token. A positive --max-new also scores that last prompt token and then max-new - 1 further
// positions. --teacher-force replaces the engine's own picks after the prompt; a short list falls
// back to the argmax, and the lowest id wins a tie. --dump-logits writes upstream's header (int32
// vocab, int32 rows) and then the f32 rows, one per position from 0 through the last token fed.
#include "strata/metal/forward.hpp"

#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> parse_i32_list(const std::string& text, const char* flag) {
    std::vector<int32_t> out;
    std::string cur;
    auto flush = [&] {
        if (cur.empty()) return;
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(cur.c_str(), &end, 10);
        if (end == cur.c_str() || *end != '\0' || errno == ERANGE || v < std::numeric_limits<int32_t>::min() ||
            v > std::numeric_limits<int32_t>::max())
            throw std::runtime_error(std::string(flag) + " has a bad id: " + cur);
        out.push_back((int32_t) v);
        cur.clear();
    };
    for (char c : text) {
        if (c == ',')
            flush();
        else if (c != ' ' && c != '\t')
            cur.push_back(c);
    }
    flush();
    if (out.empty()) throw std::runtime_error(std::string(flag) + " is empty");
    return out;
}

int32_t argmax_row(const float* row, int n) {
    int32_t best = 0;
    float top = row[0];
    for (int i = 1; i < n; ++i) {
        if (row[i] > top) {
            top = row[i];
            best = i;
        }
    }
    return best;
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-metal-run --tokens a,b,c [--max-new N] [--teacher-force b,c,...] "
                 "[--dump-logits PATH]\n"
                 "  STRATA_MODEL_GGUF selects the GGUF. The default is the Q2_0 shard under $HOME.\n"
                 "  Rows are positions 0 .. (prompt length - 1 + max-new - 1). max-new defaults to 0.\n");
}

}  // namespace

int main(int argc, char** argv) {
    static_assert(std::endian::native == std::endian::little, "the logit dump is little-endian");
    std::string tokens_s, force_s, dump_path;
    int max_new = 0;
    bool saw_tokens = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto need = [&](const char* flag) -> const char* {
                if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
                return argv[++i];
            };
            if (a == "--tokens") {
                tokens_s = need("--tokens");
                saw_tokens = true;
            } else if (a == "--max-new") {
                const std::string v = need("--max-new");
                char* end = nullptr;
                errno = 0;
                const long n = std::strtol(v.c_str(), &end, 10);
                if (end == v.c_str() || *end || errno == ERANGE || n < 0 || n > std::numeric_limits<int>::max())
                    throw std::runtime_error("--max-new is not a non-negative integer");
                max_new = (int) n;
            } else if (a == "--teacher-force") {
                force_s = need("--teacher-force");
            } else if (a == "--dump-logits") {
                dump_path = need("--dump-logits");
            } else if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument " + a);
            }
        }
        if (!saw_tokens) throw std::runtime_error("--tokens is required");
        const std::vector<int32_t> prompt = parse_i32_list(tokens_s, "--tokens");
        const std::vector<int32_t> force = force_s.empty() ? std::vector<int32_t>{} : parse_i32_list(force_s, "--teacher-force");
        const int64_t n_rows_64 = (int64_t) prompt.size() - 1 + max_new;
        if (n_rows_64 < 1 || n_rows_64 > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("the position count is not a positive int32");
        const int n_rows = (int) n_rows_64;
        const std::string path = strata::metal::default_model_shard();
        if (path.empty()) throw std::runtime_error("no STRATA_MODEL_GGUF and no HOME");

        const auto t_load = std::chrono::steady_clock::now();
        strata::metal::Model model(path);
        const int max_pos = n_rows > 512 ? n_rows : 512;
        strata::metal::Session session(model, max_pos);
        std::fprintf(stderr, "loaded in %.3f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
        const int n_vocab = session.n_vocab();
        std::vector<float> row((size_t) n_vocab);
        std::ofstream dump;
        bool header = false;
        if (!dump_path.empty()) {
            dump.open(dump_path, std::ios::binary | std::ios::trunc);
            if (!dump) throw std::runtime_error("cannot write " + dump_path);
        }

        const auto t0 = std::chrono::steady_clock::now();
        int32_t tok = prompt[0];
        size_t force_i = 0;
        for (int pos = 0; pos < n_rows; ++pos) {
            const auto t_pos = std::chrono::steady_clock::now();
            session.forward(tok, pos, row.data());
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pos).count();
            const int32_t pick = argmax_row(row.data(), n_vocab);
            std::fprintf(stderr, "position %d token %d argmax %d max %.6f  %.3f s\n", pos, tok, pick, row[(size_t) pick],
                         sec);
            if (dump) {
                if (!header) {
                    const int32_t hdr[2] = {n_vocab, n_rows};
                    dump.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
                    header = true;
                }
                dump.write(reinterpret_cast<const char*>(row.data()), (std::streamsize) row.size() * 4);
                if (!dump) throw std::runtime_error("the logit dump failed mid-write");
            }
            if (pos + 1 >= n_rows) break;
            if (pos + 1 < (int) prompt.size())
                tok = prompt[(size_t) (pos + 1)];
            else if (force_i < force.size())
                tok = force[force_i++];
            else
                tok = pick;
        }
        if (dump) dump.flush();
        const double all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "%d positions in %.3f s, %.3f s/token\n", n_rows, all, n_rows ? all / n_rows : 0.0);
        if (!dump_path.empty()) std::fprintf(stderr, "logits %s (%d x %d)\n", dump_path.c_str(), n_vocab, n_rows);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-metal-run: %s\n", e.what());
        if (argc <= 1) usage();
        return 1;
    }
    return 0;
}
