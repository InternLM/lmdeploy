#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

using namespace turbomind;

template<class T>
void print_vecs(const T* data, int m, int k, std::string msg, int width = 4)
{
    if (!msg.empty()) {
        std::cout << msg << ":\n";
    }
    for (int mm = 0; mm < m; ++mm) {
        for (int kk = 0; kk < k; ++kk) {
            std::cout << std::setw(width) << data[mm * k + kk];
        }
        std::cout << "\n";
    }
}

template<class T>
void diff_vecs(const T* data, const T* refs, int m, int k, std::string msg)
{
    if (!msg.empty()) {
        std::cout << msg << ": [" << m << ", " << k << "]\n";
    }
    for (int mm = 0; mm < m; ++mm) {
        std::cout << "m=" << mm << ": ";
        for (int kk = 0; kk < k; ++kk) {
            const auto& x = data[mm * k + kk];
            const auto& y = refs[mm * k + kk];
            if (x != y) {
                std::cout << kk << "(" << x << ", " << y << ") ";
            }
        }
        std::cout << "\n";
    }
}

RNG& gRNG()
{
    static RNG inst{};
    return inst;
}

using thrust::universal_vector;

void moe_gate_ref(int                            tokens,
                  int                            expert_num,
                  int                            experts_per_token,
                  const universal_vector<float>& logits,
                  universal_vector<int>&         offsets,
                  universal_vector<int>&         eids,
                  universal_vector<int>&         f2n,
                  universal_vector<int>&         en2f,
                  universal_vector<float>&       scales)
{
    std::vector<int> eid_range(expert_num);
    std::iota(eid_range.begin(), eid_range.end(), 0);

    for (int t = 0; t < tokens; ++t) {
        const float* logit   = logits.data().get() + expert_num * t;
        const float  max_val = *std::max_element(logit, logit + expert_num);
        if constexpr (0) {
            std::vector<float> probs(logit, logit + expert_num);
            float              sum = 0;
            for (auto& p : probs) {
                p = std::exp(p - max_val);
                sum += p;
            }
            for (auto& p : probs) {
                p /= sum;
            }
            std::vector<int> idxs = eid_range;
            // Had to use stable sort since there is no `std::stable_nth_element`
            std::stable_sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
                return probs[i] > probs[j];
            });
            // Recover natural order in top-k
            std::sort(idxs.begin(), idxs.begin() + experts_per_token);
            idxs.resize(experts_per_token);
            sum = 0;
            for (int e = 0; e < experts_per_token; ++e) {
                eids[e * tokens + t] = idxs[e];
                sum += probs[idxs[e]];
            }
            for (int e = 0; e < experts_per_token; ++e) {
                scales[e * tokens + t] = probs[idxs[e]] / sum;
            }
        }
        else {
            std::vector<int> idxs = eid_range;
            // Had to use stable sort since there is no `std::stable_nth_element`
            std::stable_sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
                return logit[i] > logit[j];
            });
            // Recover natural order in top-k
            std::sort(idxs.begin(), idxs.begin() + experts_per_token);
            idxs.resize(experts_per_token);
            std::vector<float> probs(experts_per_token);
            float              sum = 0;
            for (int e = 0; e < experts_per_token; ++e) {
                eids[e * tokens + t] = idxs[e];
                probs[e]             = std::exp(logit[idxs[e]] - max_val);
                sum += probs[e];
            }
            for (int e = 0; e < experts_per_token; ++e) {
                scales[e * tokens + t] = probs[e] / sum;
            }
        }
    }

    // f2en
    std::vector<int> f2en(eids.size());
    std::iota(f2en.begin(), f2en.end(), 0);

    std::stable_sort(f2en.begin(), f2en.end(), [&](int i, int j) {  //
        if (eids[i] != eids[j]) {
            return eids[i] < eids[j];
        }
        return i % tokens < j % tokens;
    });

    std::fill_n(offsets.begin(), offsets.size(), 0);
    std::vector<int> accum(expert_num);

    for (size_t i = 0; i < f2en.size(); ++i) {
        f2n[i]        = f2en[i] % tokens;
        en2f[f2en[i]] = i;
        ++accum[eids[i]];
    }

    for (size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] = offsets[i - 1] + accum[i - 1];
    }
}

void mask2eids(universal_vector<int8_t>& masks, universal_vector<int>& eids, int tokens, int expert_num)
{
    const int tokens_padded = masks.size() / expert_num;
    // std::cout << eids.size() << std::endl;
    for (int e = 0; e < expert_num; ++e) {
        for (int t = 0; t < tokens_padded; ++t) {
            if (auto v = masks[e * tokens_padded + t]; v >= 0) {
                // if (v >= 2 || t >= 8193) {
                //     std::cerr << "FUCK " << v << " " << t << std::endl;
                // }
                eids[v * tokens + t] = e;
            }
        }
    }
}

struct Tiling {
    int  output_dims;
    int  input_dims;
    int3 cta_tile;
};

bool test_moe_gate(int                     tokens,  //
                   int                     expert_num,
                   int                     experts_per_token,
                   gemm::Tape&             tape,
                   const Tiling&           tiling,
                   universal_vector<float> logits = {})
{
    if (logits.empty()) {
        logits.resize(tokens * expert_num);
        gRNG().GenerateUniform(logits.data().get(), logits.size());
    }
    assert(logits.size() == tokens * expert_num);

    const int tokens_padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    // const int max_coords    = get_max_coords(tokens, expert_num, experts_per_token, tiling);

    universal_vector<int>    offsets(expert_num + 1);
    universal_vector<int>    accum(expert_num * kMoeGateMaxTiles);
    universal_vector<int8_t> masks(expert_num * tokens_padded);
    universal_vector<int>    eids(experts_per_token * tokens);
    universal_vector<int>    f2n(experts_per_token * tokens);
    universal_vector<int>    f2E(experts_per_token * tokens);
    universal_vector<int>    en2f(experts_per_token * tokens);
    universal_vector<float>  scales(experts_per_token * tokens);
    // universal_vector<int2>  coords(max_coords);
    // thrust::fill(coords.begin(), coords.end(), int2{-1, 0});

    auto offsets_ref = offsets;
    auto eids_ref    = eids;
    auto f2n_ref     = f2n;
    auto en2f_ref    = en2f;
    auto scales_ref  = scales;

    moe_gate_ref(tokens, expert_num, experts_per_token, logits, offsets_ref, eids_ref, f2n_ref, en2f_ref, scales_ref);

    cudaMemPrefetchAsync(f2n.data().get(), sizeof(int) * f2n.size(), 0);
    cudaMemPrefetchAsync(f2E.data().get(), sizeof(int) * f2E.size(), 0);
    cudaMemPrefetchAsync(en2f.data().get(), sizeof(int) * en2f.size(), 0);
    cudaMemPrefetchAsync(offsets.data().get(), sizeof(int) * offsets.size(), 0);
    cudaMemPrefetchAsync(scales.data().get(), sizeof(float) * scales.size(), 0);
    cudaMemPrefetchAsync(logits.data().get(), sizeof(float) * logits.size(), 0);

    bool softmax = true;

    if (1) {
        invokeMoeSoftmaxMaskTopKGroups(logits.data().get(), tokens, expert_num, expert_num / 8, 8, nullptr);
        softmax = false;
    }

    for (int i = 0; i < 1; ++i) {
        gemm::CacheFlushing::flush();
        cudaMemset(accum.data().get(), 0, sizeof(int) * accum.size());
        cudaMemset(masks.data().get(), -1, sizeof(int8_t) * masks.size());
        invokeMoeGate_V2(f2n.data().get(),
                         f2E.data().get(),
                         en2f.data().get(),
                         offsets.data().get(),
                         scales.data().get(),
                         masks.data().get(),
                         accum.data().get(),
                         logits.data().get(),
                         tokens,
                         tokens_padded,
                         expert_num,
                         experts_per_token,
                         softmax,
                         false,
                         1.f,
                         nullptr);
    }

    // invokeMoeTiling(coords.data().get(), offsets.data().get(), expert_num, coords.size(), &tiling, 1, 0);

    // gemm::scheduleGemmMoe(tape,
    //                       offsets.data().get(),
    //                       tokens,
    //                       experts_per_token,
    //                       expert_num,
    //                       tiling.output_dims,
    //                       tiling.input_dims,
    //                       tiling.cta_tile,
    //                       tiling.cta_tile.z,
    //                       1,
    //                       0,
    //                       0);

    if (auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::abort();
    }

    // print_vecs(masks.data().get(), expert_num, tokens_padded, "masks");
    mask2eids(masks, eids, tokens, expert_num);

    bool success = true;

    // success = offsets == offsets_ref && eids == eids_ref && f2n == f2n_ref && en2f == en2f_ref;

    if (offsets != offsets_ref) {
        std::cerr << "offset\n";
        success = false;
    }
    if (eids != eids_ref) {
        std::cerr << "eids\n";
        success = false;
    }
    if (f2n != f2n_ref) {
        std::cerr << "f2n\n";
        success = false;
    }
    if (en2f != en2f_ref) {
        std::cerr << "en2f\n";
        success = false;
    }

    // print_vecs(logits.data().get(), tokens, expert_num, "logits", 12);

    if (!success && 1) {

        diff_vecs(eids.data().get(), eids_ref.data().get(), experts_per_token, tokens, "eids");

        print_vecs(offsets_ref.data().get(), 1, expert_num + 1, "offsets_ref");
        print_vecs(offsets.data().get(), 1, expert_num + 1, "offsets");

        print_vecs(eids_ref.data().get(), experts_per_token, tokens, "eids_ref");
        print_vecs(eids.data().get(), experts_per_token, tokens, "eids");

        print_vecs(f2n_ref.data().get(), 1, experts_per_token * tokens, "f2n_ref");
        print_vecs(f2n.data().get(), 1, experts_per_token * tokens, "f2n");

        print_vecs(en2f_ref.data().get(), experts_per_token, tokens, "en2f_ref");
        print_vecs(en2f.data().get(), experts_per_token, tokens, "en2f");

        print_vecs(scales_ref.data().get(), experts_per_token, tokens, "scales_ref", 12);
        print_vecs(scales.data().get(), experts_per_token, tokens, "scales", 12);

        for (int i = 0; i < tokens; ++i) {
            float sum = 0;
            for (int j = 0; j < experts_per_token; ++j) {
                sum += scales[j * tokens + i];
            }
            std::cout << sum << " ";
        }
        std::cout << "\n";

        // print_vecs(accum.data().get(), expert_num, 1, "accum");

        // print_vecs(coords.data().get(), 1, max_coords, "coords");

        // thrust::host_vector<int4> tile_offsets(tape.max_ctas);
        // std::cout << tape.max_ctas << std::endl;
        // cudaMemcpy(tile_offsets.data(), tape.tile_offsets, sizeof(int4) * tile_offsets.size(),
        // cudaMemcpyDefault); cudaDeviceSynchronize();

        // std::cout << "coords:\n";
        // int last = -1;
        // for (int i = 0; i < tape.max_ctas; ++i) {
        //     auto& c = tile_offsets[i];
        //     if (last >= 0 && c.w != last) {
        //         std::cout << "\n";
        //     }
        //     if (c.w == -1) {
        //         std::cout << i << "\n";
        //         break;
        //     }
        //     last = c.w;
        //     std::stringstream ss;
        //     ss << c.x << "," << c.y;
        //     std::cout << std::setw(6) << ss.str();
        // }
        // std::cout << "\n";
    }

    return success;
}

int main()
{
    gemm::Tape       tape{};
    constexpr Tiling tiling{14336, 128, {128, 128, 32}};

    // test_moe_gate(32768 * 4, 60, 4, tape, tiling);
    // test_moe_gate(32768, 64, 8, tape, tiling);
    // test_moe_gate(8, 60, 4, tape, tiling);

    test_moe_gate(16, 160, 6, tape, tiling);

    return 0;

    for (int i = 1; i < 16384; ++i) {
        // std::cerr << i << std::endl;
        auto success = test_moe_gate(i, 8, 2, tape, tiling);
        if (!success) {
            std::cerr << i << std::endl;
            // std::abort();
        }
        // break;
    }
}
