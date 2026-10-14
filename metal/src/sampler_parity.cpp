// The sampler. Picks are exact. The order is penalties, top-k, top-p, min-p, temperature, one draw.
// A wrong order still returns a token, so the fixtures are built so the order changes the cut.
// The cluster argmax, the split top-k, and CUDA graphs are not in this port.
#include "strata/metal/sampler.hpp"
#include "strata/metal/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

int g_fail = 0;

void verdict(const char* name, bool ok, const std::string& detail) {
    std::printf("  %-34s %s", name, ok ? "yes" : "*** NO ***");
    if (!detail.empty()) std::printf("  %s", detail.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

void line_match(const char* name, int bad, int n, const std::string& extra) {
    std::printf("  %-34s %s (%d of %d differ)", name, bad ? "*** WRONG ***" : "matches", bad, n);
    if (!extra.empty()) std::printf("   %s", extra.c_str());
    std::printf("\n");
    g_fail += bad;
}

Bytes upload(const void* p, size_t n) {
    Bytes b(n ? n : 16);
    if (n) std::memcpy(b.data(), p, n);
    return b;
}
template <class T>
Bytes upload(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
}

template <class F>
bool throws_metal(F&& fn) {
    try {
        fn();
        return false;
    } catch (const MetalError&) {
        return true;
    }
}

float host_philox(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3;
        const uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

float penalise(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

int count_of(const std::vector<int>& hist, int v) {
    int c = 0;
    for (int h : hist)
        if (h == v) ++c;
    return c;
}

int sampled_k(int top_k, int nv) { return std::min(nv, (top_k > 0 && top_k < 64) ? top_k : 64); }

int reference_pick(const std::vector<float>& l, const SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<int> ids;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) {
                best = v;
                bv = val(v);
            }
        }
        taken[(size_t) best] = 1;
        ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) {
                cut = (int) i + 1;
                break;
            }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        ids.resize((size_t) cut);
    }
    return ids[0];
}

int reference_cut(const std::vector<float>& l, const SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<int> ids;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) {
                best = v;
                bv = val(v);
            }
        }
        taken[(size_t) best] = 1;
        ids.push_back(best);
    }
    if (p.top_p >= 1.0f) return (int) ids.size();
    float mx = val(ids[0]);
    for (int v : ids) mx = std::fmax(mx, val(v));
    double sum = 0;
    for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
    double cum = 0;
    int cut = (int) ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
        if (cum >= (double) p.top_p) {
            cut = (int) i + 1;
            break;
        }
    }
    if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
    return cut;
}

int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist, const SamplerParams& p, int row) {
    const int nv = (int) l.size();
    const int k = sampled_k(p.top_k, nv);
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penalise(l[(size_t) v], count_of(hist, v), p);
            if (best < 0 || s > bv) {
                best = v;
                bv = s;
            }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best);
        sel_logit.push_back(bv);
    }
    int n_keep = k;
    if (p.top_p < 1.0f) {
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[(size_t) i] < thresh) {
                n_keep = i;
                break;
            }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sel_logit[(size_t) i] * inv_t);
    double sum = 0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (sel_logit[(size_t) i] * inv_t) - (double) smx);
    const float u = host_philox(p.seed, p.counter + (uint64_t) row);
    double cum = 0;
    int pick = sel_ids[(size_t) (n_keep > 0 ? n_keep - 1 : 0)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (sel_logit[(size_t) i] * inv_t) - (double) smx) / sum;
        if ((double) u < cum) {
            pick = sel_ids[(size_t) i];
            break;
        }
    }
    return pick;
}

int sampled_cut(const std::vector<float>& l, const std::vector<int>& hist, const SamplerParams& p) {
    const int nv = (int) l.size();
    const int k = sampled_k(p.top_k, nv);
    std::vector<float> sel;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penalise(l[(size_t) v], count_of(hist, v), p);
            if (best < 0 || s > bv) {
                best = v;
                bv = s;
            }
        }
        taken[(size_t) best] = 1;
        sel.push_back(bv);
    }
    int n_minp = (int) sel.size();
    if (p.min_p > 0.0f) {
        const float thresh = sel[0] + std::log(p.min_p);
        for (int i = 0; i < n_minp; ++i)
            if (sel[(size_t) i] < thresh) {
                n_minp = i;
                break;
            }
    }
    if (p.top_p >= 1.0f) return n_minp;
    double sum = 0;
    for (int i = 0; i < n_minp; ++i) sum += std::exp((double) sel[(size_t) i] - (double) sel[0]);
    double cum = 0;
    int cut = n_minp;
    for (int i = 0; i < n_minp; ++i) {
        cum += std::exp((double) sel[(size_t) i] - (double) sel[0]) / sum;
        if (cum >= (double) p.top_p) {
            cut = i + 1;
            break;
        }
    }
    if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
    return cut;
}

struct SelList {
    std::vector<int> ids;
    std::vector<float> logit;
};

SelList mirror_select(const float* l, int nv, const std::vector<int>& window, const SamplerParams& p, int k) {
    std::vector<float> s((size_t) nv);
    for (int v = 0; v < nv; ++v) s[(size_t) v] = penalise(l[v], count_of(window, v), p);
    SelList out;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = nv;
        float bv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < nv; ++v)
            if (!taken[(size_t) v] && s[(size_t) v] > bv) {
                bv = s[(size_t) v];
                best = v;
            }
        const int id = best < nv ? best : 0;
        taken[(size_t) id] = 1;
        out.ids.push_back(id);
        out.logit.push_back(bv);
    }
    return out;
}

int mirror_pick(const SelList& sel, int k, const SamplerParams& p, int row) {
    int n_keep = k;
    float mx = sel.logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel.logit[(size_t) i]);
    if (p.top_p < 1.0f) {
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel.logit[(size_t) i] - (double) mx);
        double cum = 0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel.logit[(size_t) i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel.logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel.logit[(size_t) i] < thresh) {
                n_keep = i;
                break;
            }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    float smx = sel.logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sel.logit[(size_t) i] * inv_t);
    double sum = 0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (sel.logit[(size_t) i] * inv_t) - (double) smx);
    const float u = host_philox(p.seed, p.counter + (uint64_t) row);
    double cum = 0;
    int pick = sel.ids[(size_t) (n_keep > 0 ? n_keep - 1 : 0)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (sel.logit[(size_t) i] * inv_t) - (double) smx) / sum;
        if ((double) u < cum) {
            pick = sel.ids[(size_t) i];
            break;
        }
    }
    return pick;
}

std::vector<int> window_of(const std::vector<int>& hist, int hist_len, int t, int last_n) {
    if (hist_len <= 0 || last_n <= 0) return {};
    const int h = std::min(last_n, hist_len);
    const int* row = hist.data() + (size_t) t * hist_len;
    return std::vector<int>(row + (hist_len - h), row + hist_len);
}

int run(const char* name, const std::vector<float>& logits, int n_tokens, const SamplerParams& p,
        const std::vector<int>& want, const std::vector<int>& hist = {}, int hist_len = 0) {
    const int nv = (int) (logits.size() / (size_t) n_tokens);
    Bytes d_l = upload(logits);
    Bytes d_o((uint64_t) n_tokens * sizeof(int));
    std::fill((int*) d_o.data(), (int*) d_o.data() + n_tokens, -1);
    Bytes d_h(16);
    const Bytes* hp = nullptr;
    if (hist_len > 0) {
        d_h = upload(hist);
        hp = &d_h;
    }
    sample_tokens(d_l, n_tokens, nv, hp, hist_len, p, d_o);
    int bad = 0;
    int first_want = 0, first_got = 0;
    const int* got = (const int*) d_o.data();
    for (int t = 0; t < n_tokens; ++t) {
        if (got[t] != want[(size_t) t]) {
            if (!bad) {
                first_want = want[(size_t) t];
                first_got = got[t];
            }
            ++bad;
        }
    }
    char extra[64] = "";
    if (bad) std::snprintf(extra, sizeof extra, "first: want %d got %d", first_want, first_got);
    line_match(name, bad, n_tokens, extra);
    return bad;
}

std::vector<float> row_of(const std::vector<float>& l, int nv, int t) {
    return std::vector<float>(l.begin() + (size_t) t * nv, l.begin() + (size_t) (t + 1) * nv);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: strata-metal-sampler [--selftest]\n");
        return 2;
    }
    std::printf("sampler: penalties, top-k, top-p, min-p, temperature, one Philox draw\n\n");
    try {
        const int NV = 512, NT = 4;

        {
            SamplerParams p;
            p.top_k = 0;
            p.top_p = 1.0f;
            p.temperature = 1.0f;
            p.greedy = true;
            std::mt19937 rng(3);
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) NV * NT);
            for (auto& v : l) v = g(rng);
            std::vector<int> want((size_t) NT);
            for (int t = 0; t < NT; ++t) want[(size_t) t] = reference_pick(row_of(l, NV, t), p, false);
            run("greedy argmax", l, NT, p, want);
        }

        {
            SamplerParams p;
            p.top_k = 0;
            p.top_p = 0.5f;
            p.temperature = 0.5f;
            p.min_keep = 1;
            p.greedy = true;
            std::vector<float> l((size_t) NV * NT, -1000.0f);
            for (int t = 0; t < NT; ++t) {
                l[(size_t) t * NV + 0] = 3.0f;
                for (int v = 1; v < 8; ++v) l[(size_t) t * NV + v] = 2.6f - 0.05f * (float) v;
            }
            std::vector<int> want((size_t) NT);
            int cut_spec = 0, cut_alt = 0;
            for (int t = 0; t < NT; ++t) {
                const std::vector<float> row = row_of(l, NV, t);
                want[(size_t) t] = reference_pick(row, p, false);
                cut_spec += reference_cut(row, p, false);
                cut_alt += reference_cut(row, p, true);
            }
            char detail[64];
            std::snprintf(detail, sizeof detail, "(survivors: spec %d, temp-first %d)", cut_spec, cut_alt);
            verdict("order is observable on this fixture", cut_spec != cut_alt, detail);
            run("greedy over this fixture", l, NT, p, want);
        }

        {
            SamplerParams a;
            a.top_k = 20;
            a.top_p = 0.95f;
            a.temperature = 1.0f;
            a.greedy = true;
            a.seed = 1;
            SamplerParams b = a;
            b.seed = 999999;
            std::mt19937 rng(5);
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) NV * NT);
            for (auto& v : l) v = g(rng);
            std::vector<int> wa((size_t) NT);
            for (int t = 0; t < NT; ++t) wa[(size_t) t] = reference_pick(row_of(l, NV, t), a, false);
            Bytes d_l = upload(l);
            Bytes d_a((uint64_t) NT * 4), d_b((uint64_t) NT * 4);
            sample_tokens(d_l, NT, NV, nullptr, 0, a, d_a);
            sample_tokens(d_l, NT, NV, nullptr, 0, b, d_b);
            int mismatch = 0, wrong = 0;
            const int* ga = (const int*) d_a.data();
            const int* gb = (const int*) d_b.data();
            for (int t = 0; t < NT; ++t) {
                if (ga[t] != gb[t]) ++mismatch;
                if (ga[t] != wa[(size_t) t]) ++wrong;
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "(seed-independent: %d differ; vs reference: %d wrong)", mismatch,
                          wrong);
            verdict("greedy ignores the seed", mismatch == 0 && wrong == 0, detail);
        }

        {
            auto pick = [](const std::vector<float>& l, const std::vector<int>& hist, const SamplerParams& p,
                           bool divide_always) {
                int best = 0;
                float bv = 0;
                bool first = true;
                for (int v = 0; v < (int) l.size(); ++v) {
                    const int c = count_of(hist, v);
                    float s;
                    if (divide_always && c > 0)
                        s = l[(size_t) v] / p.penalty_repeat - (float) c * p.penalty_freq - p.penalty_present;
                    else
                        s = penalise(l[(size_t) v], c, p);
                    if (first || s > bv) {
                        bv = s;
                        best = v;
                        first = false;
                    }
                }
                return best;
            };
            const int NV2 = 8, NT2 = 2;
            SamplerParams p;
            p.top_k = 0;
            p.top_p = 1.0f;
            p.temperature = 1.0f;
            p.greedy = true;
            p.penalty_last_n = 4;
            p.penalty_repeat = 2.0f;
            std::vector<float> la((size_t) NV2 * NT2, -8.0f);
            for (int t = 0; t < NT2; ++t) {
                la[(size_t) t * NV2 + 0] = -1.0f;
                la[(size_t) t * NV2 + 1] = -1.2f;
            }
            std::vector<int> hist_a((size_t) NT2 * 4, -1);
            for (int t = 0; t < NT2; ++t) hist_a[(size_t) t * 4] = 0;
            std::vector<int> want_a((size_t) NT2), alt_a((size_t) NT2);
            for (int t = 0; t < NT2; ++t) {
                const std::vector<float> row = row_of(la, NV2, t);
                const std::vector<int> h(hist_a.begin() + (size_t) t * 4, hist_a.begin() + (size_t) (t + 1) * 4);
                want_a[(size_t) t] = pick(row, h, p, false);
                alt_a[(size_t) t] = pick(row, h, p, true);
            }
            char detail[64];
            std::snprintf(detail, sizeof detail, "(multiply-rule %d, divide-always %d)", want_a[0], alt_a[0]);
            verdict("multiply-or-divide is observable", want_a[0] != alt_a[0], detail);
            if (want_a[0] != alt_a[0]) run("penalties: repeat on negatives", la, NT2, p, want_a, hist_a, 4);

            std::vector<float> lb((size_t) NV2 * NT2, -8.0f);
            for (int t = 0; t < NT2; ++t) {
                lb[(size_t) t * NV2 + 0] = 5.0f;
                lb[(size_t) t * NV2 + 1] = 3.4f;
            }
            std::vector<int> hist_b((size_t) NT2 * 4, -1);
            for (int t = 0; t < NT2; ++t) {
                hist_b[(size_t) t * 4] = 0;
                hist_b[(size_t) t * 4 + 1] = 0;
            }
            SamplerParams q = p;
            q.penalty_present = 1.5f;
            q.penalty_freq = 0.0f;
            std::vector<int> want_b((size_t) NT2);
            for (int t = 0; t < NT2; ++t) {
                const std::vector<float> row = row_of(lb, NV2, t);
                const std::vector<int> h(hist_b.begin() + (size_t) t * 4, hist_b.begin() + (size_t) (t + 1) * 4);
                want_b[(size_t) t] = pick(row, h, q, false);
            }
            std::snprintf(detail, sizeof detail, "(want token %d)", want_b[0]);
            verdict("presence penalty is a boolean", want_b[0] == 1, detail);
            run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);
        }

        {
            const int NV3 = 512, NT3 = 4;
            std::vector<float> l((size_t) NV3 * NT3, -5.0f);
            std::vector<int> want((size_t) NT3);
            for (int t = 0; t < NT3; ++t) {
                const int best = 100 + t;
                l[(size_t) t * NV3 + best] = 3.0f;
                l[(size_t) t * NV3] = -9.0f;
                want[(size_t) t] = best;
            }
            SamplerParams p0;
            p0.top_k = 0;
            p0.top_p = 1.0f;
            p0.temperature = 0.0f;
            p0.greedy = false;
            run("T=0 greedy=false is the argmax", l, NT3, p0, want);
            SamplerParams p1 = p0;
            p1.greedy = true;
            run("T=0 greedy=true  is the argmax", l, NT3, p1, want);
            SamplerParams p2 = p0;
            p2.greedy = true;
            p2.temperature = 1.0f;
            run("T=1 greedy=true  is the argmax", l, NT3, p2, want);
        }

        {
            const int NV2 = 8, NT2 = 2;
            SamplerParams p;
            p.top_k = 5;
            p.top_p = 0.9f;
            p.temperature = 0.8f;
            p.seed = 9;
            p.penalty_last_n = 4;
            p.penalty_repeat = 3.0f;
            p.penalty_freq = 0.2f;
            p.penalty_present = 0.6f;
            std::vector<float> l((size_t) NV2 * NT2, -8.0f);
            std::vector<int> hist((size_t) NT2 * 4, -1);
            for (int t = 0; t < NT2; ++t) {
                float* row = l.data() + (size_t) t * NV2;
                row[0] = 9.0f;
                row[1] = 4.5f;
                row[2] = 4.4f;
                row[3] = 4.3f;
                hist[(size_t) t * 4] = 0;
                hist[(size_t) t * 4 + 1] = t == 1 ? 0 : -1;
            }
            SamplerParams clean = p;
            clean.penalty_last_n = 0;
            clean.penalty_repeat = 1.0f;
            clean.penalty_freq = 0.0f;
            clean.penalty_present = 0.0f;
            std::vector<int> want((size_t) NT2), bare((size_t) NT2);
            for (int t = 0; t < NT2; ++t) {
                const std::vector<float> row = row_of(l, NV2, t);
                const std::vector<int> h(hist.begin() + (size_t) t * 4, hist.begin() + (size_t) (t + 1) * 4);
                want[(size_t) t] = sampled_reference(row, h, p, t);
                bare[(size_t) t] = sampled_reference(row, h, clean, t);
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "(penalised picks %d/%d, clean %d/%d)", want[0], want[1], bare[0],
                          bare[1]);
            verdict("sampled penalties are observable", want[0] != bare[0] || want[1] != bare[1], detail);
            if (want[0] != bare[0] || want[1] != bare[1])
                run("sampled chain: penalties + top_k/p", l, NT2, p, want, hist, 4);
        }

        {
            const int NV3 = 8, NT3 = 2;
            std::vector<float> l((size_t) NV3 * NT3, -8.0f);
            for (int t = 0; t < NT3; ++t) {
                float* row = l.data() + (size_t) t * NV3;
                row[0] = 4.0f;
                row[1] = 3.5f;
                row[2] = 3.2f;
                row[3] = 3.1f;
                row[4] = 2.0f;
            }
            SamplerParams base;
            base.top_k = 6;
            base.top_p = 1.0f;
            base.temperature = 0.9f;
            base.seed = 77;
            int c0 = 0, c05 = 0, c09 = 0;
            for (int t = 0; t < NT3; ++t) {
                const std::vector<float> row = row_of(l, NV3, t);
                SamplerParams q = base;
                q.min_p = 0.0f;
                c0 += sampled_cut(row, {}, q);
                q.min_p = 0.5f;
                c05 += sampled_cut(row, {}, q);
                q.min_p = 0.9f;
                c09 += sampled_cut(row, {}, q);
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "(survivors: min_p 0 -> %d, 0.5 -> %d, 0.9 -> %d)", c0, c05, c09);
            verdict("min_p cut is observable", c0 > c05 && c05 > c09 && c09 >= NT3, detail);
            for (float mp : {0.0f, 0.5f, 0.9f}) {
                SamplerParams q = base;
                q.min_p = mp;
                std::vector<int> want((size_t) NT3);
                for (int t = 0; t < NT3; ++t) want[(size_t) t] = sampled_reference(row_of(l, NV3, t), {}, q, t);
                char name[64];
                std::snprintf(name, sizeof name, "sampled chain: min_p=%.1f", (double) mp);
                run(name, l, NT3, q, want);
            }
        }

        {
            SamplerParams p;
            p.top_k = 0;
            p.top_p = 1.0f;
            p.temperature = 1.0f;
            p.greedy = true;
            p.penalty_last_n = 4;
            p.penalty_repeat = 3.0f;
            p.penalty_freq = 0.3f;
            p.penalty_present = 0.5f;
            const int NV4 = 8;
            std::vector<float> l((size_t) NV4, -8.0f);
            l[0] = 6.0f;
            l[3] = 6.5f;
            std::vector<int> hist = {0, 0, 0, 0, 3, 3, 3, 3};
            auto pick_clamped = [&](bool clamp) {
                int best = 0;
                float bv = 0;
                bool first = true;
                const int n = clamp ? 4 : 8;
                for (int v = 0; v < NV4; ++v) {
                    int c = 0;
                    for (int i = 0; i < n; ++i)
                        if (hist[(size_t) (8 - n + i)] == v) ++c;
                    float logit = l[(size_t) v];
                    if (c > 0) {
                        logit = logit <= 0.0f ? logit * p.penalty_repeat : logit / p.penalty_repeat;
                        logit -= (float) c * p.penalty_freq + p.penalty_present;
                    }
                    if (first || logit > bv) {
                        bv = logit;
                        best = v;
                        first = false;
                    }
                }
                return best;
            };
            const int want = pick_clamped(true), full = pick_clamped(false);
            char detail[64];
            std::snprintf(detail, sizeof detail, "(clamped pick %d, full-history pick %d)", want, full);
            verdict("penalty window clamp is observable", want != full, detail);
            if (want != full) run("penalty window: tail only", l, 1, p, {want}, hist, 8);
        }

        {
            const int NT9 = 8000;
            SamplerParams p;
            p.top_k = 2;
            p.top_p = 1.0f;
            p.min_p = 0.0f;
            p.temperature = 0.7f;
            p.seed = 53;
            p.penalty_last_n = 1;
            p.penalty_repeat = 1.0f;
            p.penalty_present = 1.5f;
            std::vector<float> l((size_t) 2 * NT9, 0.0f);
            std::vector<int> hist((size_t) NT9, 0);
            std::vector<int> want((size_t) NT9);
            int zeros = 0;
            for (int t = 0; t < NT9; ++t) {
                want[(size_t) t] = sampled_reference({0.0f, 0.0f}, {0}, p, t);
                zeros += want[(size_t) t] == 0;
            }
            const double expect = 1.0 / (1.0 + std::exp(1.5 / 0.7));
            const double share = (double) zeros / NT9;
            char detail[96];
            std::snprintf(detail, sizeof detail, "(token 0 drawn %.4f of %d, expected %.4f)", share, NT9, expect);
            verdict("one penalties stage", std::fabs(share - expect) < 0.014, detail);
            run("sampled chain: one penalties stage", l, NT9, p, want, hist, 1);
        }

        {
            const int NT10 = 256;
            SamplerParams p;
            p.top_k = 4;
            p.top_p = 0.75f;
            p.min_p = 0.3f;
            p.temperature = 1.0f;
            p.seed = 10;
            const float lp[4] = {std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};
            std::vector<float> row = {lp[0], lp[1], lp[2], lp[3], -30.0f, -30.0f, -30.0f, -30.0f};
            std::vector<float> l;
            for (int t = 0; t < NT10; ++t) l.insert(l.end(), row.begin(), row.end());
            std::vector<int> want((size_t) NT10);
            int twos = 0;
            for (int t = 0; t < NT10; ++t) {
                want[(size_t) t] = sampled_reference(row, {}, p, t);
                twos += want[(size_t) t] == 2;
            }
            char detail[48];
            std::snprintf(detail, sizeof detail, "(token 2 drawn %d of %d times)", twos, NT10);
            verdict("top_p before min_p is observable", twos > 0, detail);
            run("sampled chain: top_p then min_p", l, NT10, p, want);
        }

        {
            std::mt19937 rng(11);
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) NV * NT);
            for (auto& v : l) v = g(rng);
            std::vector<int> hist((size_t) NT * 8, -1);
            for (int t = 0; t < NT; ++t) hist[(size_t) t * 8] = 3;
            SamplerParams p;
            p.top_k = 20;
            p.top_p = 0.95f;
            p.temperature = 0.8f;
            p.seed = 5;
            std::vector<int> want((size_t) NT);
            for (int t = 0; t < NT; ++t) want[(size_t) t] = sampled_reference(row_of(l, NV, t), {}, p, t);
            run("stale history, last_n=0 (sampled)", l, NT, p, want, hist, 8);
            SamplerParams gp = p;
            gp.greedy = true;
            gp.top_k = 0;
            gp.top_p = 1.0f;
            std::vector<int> gwant((size_t) NT);
            for (int t = 0; t < NT; ++t) gwant[(size_t) t] = reference_pick(row_of(l, NV, t), gp, false);
            run("stale history, last_n=0 (greedy)", l, NT, gp, gwant, hist, 8);
        }

        {
            auto greedy_pen = [](const std::vector<float>& l, const std::vector<int>& hist, const SamplerParams& p) {
                int best = 0;
                float bv = 0;
                bool first = true;
                for (int v = 0; v < (int) l.size(); ++v) {
                    const float s = penalise(l[(size_t) v], count_of(hist, v), p);
                    if (first || s > bv) {
                        bv = s;
                        best = v;
                        first = false;
                    }
                }
                return best;
            };
            std::mt19937 rng(12);
            std::normal_distribution<float> g(0.0f, 1.0f);
            const int TAIL = 5000;
            std::vector<int32_t> tail((size_t) TAIL);
            for (auto& v : tail) v = (int32_t) (rng() % NV);
            int observable = 0, rows_checked = 0;
            for (int T : {1, 2, 4, 8}) {
                for (int H : {1, 64, 1024, 4096}) {
                    std::vector<int32_t> window((size_t) T);
                    for (int t = 0; t < T; ++t) window[(size_t) t] = (int32_t) (100 + 37 * t);
                    std::vector<int32_t> rows((size_t) T * H);
                    penalty_rows(tail.data(), TAIL, window.data(), T, H, rows.data());
                    std::vector<float> l((size_t) T * NV);
                    for (auto& v : l) v = g(rng);
                    for (int t = 0; t < T; ++t) l[(size_t) t * NV + window[(size_t) t]] = 5.0f;
                    SamplerParams gp;
                    gp.greedy = true;
                    gp.temperature = 0.0f;
                    gp.top_k = 0;
                    gp.top_p = 1.0f;
                    gp.penalty_last_n = H;
                    gp.penalty_present = 4.0f;
                    SamplerParams sp;
                    sp.top_k = 20;
                    sp.top_p = 0.9f;
                    sp.temperature = 0.7f;
                    sp.seed = 1000 + (uint64_t) H;
                    sp.counter = 77;
                    sp.penalty_last_n = H;
                    sp.penalty_present = 4.0f;
                    sp.penalty_repeat = 1.1f;
                    std::vector<int> gwant((size_t) T), swant((size_t) T), rows_int(rows.begin(), rows.end());
                    for (int t = 0; t < T; ++t) {
                        const std::vector<float> lr = row_of(l, NV, t);
                        const std::vector<int> own(rows.begin() + (size_t) t * H, rows.begin() + (size_t) (t + 1) * H);
                        const std::vector<int> row0(rows.begin(), rows.begin() + H);
                        gwant[(size_t) t] = greedy_pen(lr, own, gp);
                        swant[(size_t) t] = sampled_reference(lr, own, sp, t);
                        if (t > 0 && greedy_pen(lr, row0, gp) != gwant[(size_t) t]) ++observable;
                        ++rows_checked;
                    }
                    char name[64];
                    std::snprintf(name, sizeof name, "per-row history T=%d H=%d greedy", T, H);
                    run(name, l, T, gp, gwant, rows_int, H);
                    std::snprintf(name, sizeof name, "per-row history T=%d H=%d sampled", T, H);
                    run(name, l, T, sp, swant, rows_int, H);
                }
            }
            char detail[64];
            std::snprintf(detail, sizeof detail, "(%d drafted rows differ, of %d)", observable, rows_checked);
            verdict("per-row histories are observable", observable > 0, detail);
        }

        {
            std::mt19937 rng(13);
            std::normal_distribution<float> g(0.0f, 1.0f);
            const int H = 16;
            std::vector<float> l((size_t) NV * NT);
            for (auto& v : l) v = g(rng);
            std::vector<int> hist((size_t) NT * H, -1), valid((size_t) NT * H, -1);
            const int junk[] = {NV, NV + 1000, 0x7fffffff, -5, 1 << 20};
            for (int t = 0; t < NT; ++t) {
                for (int j = 0; j < H; ++j) {
                    const bool bogus = j % 3 == 0;
                    const int v = bogus ? junk[(size_t) (j / 3) % 5] : (int) (rng() % NV);
                    hist[(size_t) t * H + j] = v;
                    if (!bogus) valid[(size_t) t * H + j] = v;
                }
                l[(size_t) t * NV + hist[(size_t) t * H + 1]] = 4.0f;
            }
            SamplerParams gp;
            gp.greedy = true;
            gp.temperature = 0.0f;
            gp.top_k = 0;
            gp.top_p = 1.0f;
            gp.penalty_last_n = H;
            gp.penalty_present = 3.0f;
            SamplerParams sp = gp;
            sp.greedy = false;
            sp.temperature = 0.8f;
            sp.top_k = 20;
            sp.top_p = 0.95f;
            sp.seed = 13;
            std::vector<int> gwant((size_t) NT), swant((size_t) NT);
            for (int t = 0; t < NT; ++t) {
                const std::vector<float> lr = row_of(l, NV, t);
                const std::vector<int> ok(valid.begin() + (size_t) t * H, valid.begin() + (size_t) (t + 1) * H);
                gwant[(size_t) t] = 0;
                float bv = 0;
                bool first = true;
                for (int v = 0; v < NV; ++v) {
                    const float s = penalise(lr[(size_t) v], count_of(ok, v), gp);
                    if (first || s > bv) {
                        bv = s;
                        gwant[(size_t) t] = v;
                        first = false;
                    }
                }
                swant[(size_t) t] = sampled_reference(lr, ok, sp, t);
            }
            run("out-of-vocab history ids (greedy)", l, NT, gp, gwant, hist, H);
            run("out-of-vocab history ids (sampled)", l, NT, sp, swant, hist, H);
        }

        {
            std::mt19937 rng(14);
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) NV * NT);
            for (auto& v : l) v = g(rng) * 0.3f;
            SamplerParams p64;
            p64.top_k = 64;
            p64.top_p = 1.0f;
            p64.temperature = 1.5f;
            p64.seed = 14;
            std::vector<int> want((size_t) NT);
            for (int t = 0; t < NT; ++t) want[(size_t) t] = sampled_reference(row_of(l, NV, t), {}, p64, t);
            run("sampled top_k=64", l, NT, p64, want);
            SamplerParams p0 = p64;
            p0.top_k = 0;
            run("sampled top_k=0 means 64", l, NT, p0, want);
            SamplerParams p100 = p64;
            p100.top_k = 100;
            run("sampled top_k=100 means 64", l, NT, p100, want);
            SamplerParams pneg = p64;
            pneg.top_k = -3;
            run("sampled top_k=-3 means 64", l, NT, pneg, want);

            std::vector<float> tiny = {0.2f, 1.5f, -0.4f, 0.7f};
            SamplerParams pt = p64;
            pt.top_k = 64;
            const int tiny_want = sampled_reference(tiny, {}, pt, 0);
            run("top_k wider than the vocabulary", tiny, 1, pt, {tiny_want});
        }

        {
            int wrong = 0, cases = 0;
            for (int n_tail : {0, 1, 5, 63, 64, 65, 300}) {
                for (int T : {1, 3, 8}) {
                    for (int h : {1, 4, 64, 100}) {
                        std::vector<int32_t> tail((size_t) n_tail), window((size_t) T);
                        for (int i = 0; i < n_tail; ++i) tail[(size_t) i] = 1000 + i;
                        for (int i = 0; i < T; ++i) window[(size_t) i] = 5000 + i;
                        std::vector<int32_t> rows((size_t) T * h, 12345);
                        penalty_rows(tail.data(), n_tail, window.data(), T, h, rows.data());
                        for (int t = 0; t < T; ++t) {
                            std::vector<int32_t> seq(tail);
                            seq.insert(seq.end(), window.begin(), window.begin() + t + 1);
                            std::vector<int32_t> expect((size_t) h, -1);
                            const int take = (int) std::min<size_t>((size_t) h, seq.size());
                            for (int j = 0; j < take; ++j)
                                expect[(size_t) (h - take + j)] = seq[seq.size() - take + j];
                            if (!std::equal(expect.begin(), expect.end(), rows.begin() + (size_t) t * h)) ++wrong;
                            ++cases;
                        }
                        std::vector<int32_t> old((size_t) h, -1);
                        const int take0 = (int) std::min<int64_t>(h, (int64_t) n_tail + 1);
                        for (int j = 0; j < take0 - 1; ++j)
                            old[(size_t) (h - take0 + j)] = tail[(size_t) (n_tail - (take0 - 1) + j)];
                        old[(size_t) (h - 1)] = window[0];
                        if (!std::equal(old.begin(), old.end(), rows.begin())) ++wrong;
                        ++cases;
                    }
                }
            }
            line_match("penalty_rows layout", wrong, cases, "");
        }

        {
            constexpr int count = 32, vocab = 16;
            std::vector<float> uniform((size_t) count * vocab, 0.0f);
            Bytes input = upload(uniform);
            Bytes output((uint64_t) count * 4);
            SamplerParams p;
            p.top_k = vocab;
            p.top_p = 1.0f;
            p.seed = 123;
            p.counter = (uint64_t(1) << 32) + 7;
            sample_tokens(input, count, vocab, nullptr, 0, p, output);
            std::vector<int> batch(count), singles(count), repeated(count);
            std::memcpy(batch.data(), output.data(), batch.size() * 4);
            for (int i = 0; i < count; ++i) {
                SamplerParams one = p;
                one.counter += (uint64_t) i;
                sample_tokens(input, 1, vocab, nullptr, 0, one, output);
                singles[(size_t) i] = ((const int*) output.data())[0];
            }
            sample_tokens(input, count, vocab, nullptr, 0, p, output);
            std::memcpy(repeated.data(), output.data(), repeated.size() * 4);
            bool varies = false;
            for (int i = 1; i < count; ++i) varies = varies || batch[(size_t) i] != batch[0];
            verdict("draw counter is the row index", batch == singles && batch == repeated && varies, "");
        }

        {
            const float inf = std::numeric_limits<float>::infinity();
            const float qnan = std::numeric_limits<float>::quiet_NaN();
            int wrong = 0, probes = 0, sentinels = 0;
            for (int nv : {1000, 248320}) {
                const int T = 4, H = 64;
                std::mt19937 rng((unsigned) (1600 + nv));
                std::normal_distribution<float> g(0.0f, 1.0f);
                std::vector<float> l((size_t) nv * T);
                for (auto& v : l) v = std::floor(g(rng) * 4.0f) / 4.0f;
                auto spread = [&](int j) { return (int) (((int64_t) j * 7919 + 13) % nv); };
                float* r0 = l.data();
                for (int j = 0; j < 320; ++j) r0[spread(j)] = j < 20 ? 6.25f : 6.0f;
                r0[nv / 2] = inf;
                r0[nv - 1] = inf;
                float* r1 = l.data() + (size_t) nv;
                for (int v = 0; v < nv; ++v) {
                    r1[v] = -std::fabs(r1[v]);
                    if (r1[v] == 0.0f) r1[v] = (v & 1) ? 0.0f : -0.0f;
                }
                r1[3] = inf;
                float* r2 = l.data() + (size_t) 2 * nv;
                for (int v = 0; v < nv; ++v) r2[v] = -inf;
                for (int j = 0; j < 10; ++j) r2[spread(j + 400)] = j < 5 ? 1.0f : 0.5f;
                r2[spread(500)] = qnan;
                r2[spread(501)] = qnan;
                r2[spread(502)] = inf;
                float* r3 = l.data() + (size_t) 3 * nv;
                std::vector<int> hist((size_t) T * H, -1);
                int* h3 = hist.data() + (size_t) 3 * H;
                const int edges[] = {0, 1023, 1024, 4095, 4096, 8191, 8192, 12345, nv / 2 + 1, nv - 2};
                int hn = 0;
                for (int e : edges) {
                    if (e < nv && e != nv / 2) {
                        h3[hn++] = e;
                        if (hn % 3 == 0) h3[hn++] = e;
                        r3[e] = 9.0f;
                    }
                }
                h3[hn++] = nv;
                h3[hn++] = nv + 77;
                h3[hn++] = -5;
                for (int j = 0; j < 30; ++j) r3[spread(j + 600)] = j < 15 ? 3.75f : 3.5f;
                r3[nv / 2] = inf;

                SamplerParams base;
                base.top_p = 1.0f;
                base.min_p = 0.0f;
                base.temperature = 0.8f;
                base.seed = 16;
                base.penalty_last_n = H;
                base.penalty_repeat = 2.0f;
                base.penalty_freq = 0.25f;
                base.penalty_present = 0.5f;
                const int kmax = sampled_k(64, nv);
                std::vector<SelList> lists;
                for (int t = 0; t < T; ++t)
                    lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, H), base, kmax));
                Bytes d_l = upload(l);
                Bytes d_h = upload(hist);
                for (int top_k : {0, 1, 11, 12, 64, 100, -3}) {
                    for (float top_p : {0.5f, 1.0f}) {
                        SamplerParams sp = base;
                        sp.top_k = top_k;
                        sp.top_p = top_p;
                        sp.counter = (uint64_t) (top_k + (top_p < 1.0f ? 100 : 0));
                        const int k = sampled_k(sp.top_k, nv);
                        Bytes d_o((uint64_t) T * 4);
                        std::fill((int*) d_o.data(), (int*) d_o.data() + T, -1);
                        sample_tokens(d_l, T, nv, &d_h, H, sp, d_o);
                        const int* got = (const int*) d_o.data();
                        for (int t = 0; t < T; ++t) {
                            const int want = lists[(size_t) t].ids[(size_t) (k - 1)];
                            sentinels += (t == 2 && k > 11);
                            ++probes;
                            if (got[t] != want) {
                                if (wrong < 8)
                                    std::printf("    n_vocab %d row %d top_k %d top_p %.2f: want %d got %d\n", nv, t,
                                                sp.top_k, (double) top_p, want, got[t]);
                                ++wrong;
                            }
                        }
                    }
                }
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "(%d of %d positions differ; %d sentinel positions)", wrong, probes,
                          sentinels);
            verdict("top_k list under ties", wrong == 0 && sentinels > 0, detail);
        }

        {
            const int T = 8, H = 32, nv = 512;
            std::mt19937 rng(1700);
            std::normal_distribution<float> g(0.0f, 1.5f);
            std::vector<float> l((size_t) nv * T);
            for (auto& v : l) v = std::floor(g(rng) * 2.0f) / 2.0f;
            std::vector<int> hist((size_t) T * H);
            for (int t = 0; t < T; ++t) {
                const float* row = l.data() + (size_t) t * nv;
                const float mx = *std::max_element(row, row + nv);
                std::vector<int> head;
                for (int v = 0; v < nv; ++v)
                    if (row[v] >= mx - 1.0f) head.push_back(v);
                for (int j = 0; j < H; ++j)
                    hist[(size_t) t * H + j] = (j & 1) ? (int) (rng() % (unsigned) nv)
                                                       : head[(size_t) (rng() % (unsigned) head.size())];
            }
            int wrong = 0, draws = 0, tied = 0;
            Bytes d_l = upload(l);
            Bytes d_h = upload(hist);
            int config = 0;
            for (int pen = 0; pen < 2; ++pen) {
                SamplerParams base;
                base.temperature = 2.5f;
                base.seed = 17;
                base.penalty_last_n = pen ? H : 0;
                base.penalty_repeat = 1.25f;
                base.penalty_freq = 0.25f;
                base.penalty_present = 0.5f;
                const int kmax = sampled_k(64, nv);
                std::vector<SelList> lists;
                for (int t = 0; t < T; ++t)
                    lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv,
                                                  window_of(hist, H, t, base.penalty_last_n), base, kmax));
                for (int top_k : {1, 20, 64}) {
                    for (float top_p : {0.9f, 1.0f}) {
                        for (float min_p : {0.0f, 0.05f}) {
                            SamplerParams sp = base;
                            sp.top_k = top_k;
                            sp.top_p = top_p;
                            sp.min_p = min_p;
                            sp.counter = (uint64_t) (1000 * ++config);
                            const int k = sampled_k(top_k, nv);
                            Bytes d_o((uint64_t) T * 4);
                            sample_tokens(d_l, T, nv, pen ? &d_h : nullptr, pen ? H : 0, sp, d_o);
                            const int* got = (const int*) d_o.data();
                            for (int t = 0; t < T; ++t) {
                                const int want = mirror_pick(lists[(size_t) t], k, sp, t);
                                ++draws;
                                int same = 0;
                                for (int i = 0; i < k; ++i) {
                                    if (lists[(size_t) t].ids[(size_t) i] != want) continue;
                                    for (int j = 0; j < k; ++j)
                                        same += lists[(size_t) t].logit[(size_t) j] ==
                                                lists[(size_t) t].logit[(size_t) i];
                                    break;
                                }
                                tied += same > 1;
                                if (got[t] != want) {
                                    if (wrong < 8)
                                        std::printf("    row %d top_k %d top_p %.2f min_p %.2f pen %d: want %d got %d\n",
                                                    t, top_k, (double) top_p, (double) min_p, pen, want, got[t]);
                                    ++wrong;
                                }
                            }
                        }
                    }
                }
            }
            char detail[80];
            std::snprintf(detail, sizeof detail, "(%d of %d draws differ; %d picks tie with another kept token)",
                          wrong, draws, tied);
            verdict("sampled draws under ties", wrong == 0 && tied > 0, detail);
        }

        {
            const int nv = 248321;
            std::vector<float> l((size_t) nv, -1.0f);
            l[5] = 4.0f;
            l[9] = 3.5f;
            std::vector<int> hist = {5, -1, nv + 3};
            SamplerParams p;
            p.greedy = true;
            p.temperature = 0.0f;
            p.top_k = 0;
            p.top_p = 1.0f;
            p.penalty_last_n = 3;
            p.penalty_present = 3.0f;
            p.penalty_repeat = 1.0f;
            const int want = 9;
            run("penalties past the bitmap", l, 1, p, {want}, hist, 3);
        }

        verdict("a negative count is refused",
                throws_metal([] {
                    Bytes l(16), o(16);
                    sample_tokens(l, -1, 8, nullptr, 0, SamplerParams{}, o);
                }),
                "");
        verdict("penalties without a history are refused",
                throws_metal([] {
                    Bytes l(64), o(16);
                    SamplerParams p;
                    p.penalty_last_n = 4;
                    sample_tokens(l, 1, 8, nullptr, 0, p, o);
                }),
                "");
        {
            Bytes l(64), o(16);
            int* p = (int*) o.data();
            p[0] = 7;
            sample_tokens(l, 0, 512, nullptr, 0, SamplerParams{}, o);
            verdict("zero tokens leaves the id buffer", p[0] == 7, "");
        }

        std::printf("\nsampler: %d failures\n", g_fail);
        if (g_fail) return 1;
    } catch (const MetalError& e) {
        std::fprintf(stderr, "Metal error: %s\n", e.what());
        return 2;
    }
    std::printf("sampler_parity OK\n");
    return 0;
}
