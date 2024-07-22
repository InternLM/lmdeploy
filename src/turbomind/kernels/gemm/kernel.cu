// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

namespace turbomind::gemm {

std::vector<std::pair<int, float>> Kernel::Estimate(int   m,
                                                    int   n,
                                                    int   k,
                                                    int   max_split_k,
                                                    int   sm_count,
                                                    int   max_wave_count,
                                                    int   top_k,
                                                    float bytes_per_second,
                                                    float fma_per_second)
{
    const int tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    // std::cout << tiled_shape_m << " " << tiled_shape_n << " " << chunk_cnt_k << std::endl;

    const int     cta_m  = desc_.cta_tile.x;
    const int     cta_n  = desc_.cta_tile.y;
    const int64_t cta_mn = cta_m * cta_n;

    const int tiled_shape_mn = tiled_shape_m * tiled_shape_n;
    // const int concurrency    = sm_count * std::min(2, max_active_ctas_);
    // const int   concurrency    = sm_count;
    const int   concurrency    = sm_count * max_active_ctas_;
    const float wave_per_split = float(tiled_shape_mn) / float(concurrency);

    //                     cost    volume   waves  split-k
    std::vector<std::tuple<float, int64_t, float, int>> estimations;

    for (int splits = 1; splits <= max_split_k; ++splits) {
        const float waves      = wave_per_split * splits;
        const int   wave_count = (int)std::ceil(waves);

        if (splits > 1 && wave_count > max_wave_count) {
            break;
        }

        const int64_t gemm_size_k = ceil_div(chunk_cnt_k, splits) * chunk_size_k_;

        const int64_t fma_volume   = cta_mn * gemm_size_k;  // cta volume
        const int64_t ldg_volume   = (cta_m * 2 + cta_n / 2) * gemm_size_k;
        const int64_t split_volume = 2 * cta_mn * 4;  // 2 for W/R, 4 for float

        // (CTA_M * CTA_N) * ceil(K / S) * ceil(S * TILED_M * TILED_N / C) * C
        const float fma_cost = fma_volume * wave_count * concurrency / fma_per_second;

        // (CTA_M + CTA_N) * ceil(K / S) * ceil(S * TILED_M * TILED_N / C) * C
        const float ldg_cost = ldg_volume * wave_count * concurrency / bytes_per_second;

        // (CTA_M * CTA_N) * ceil(S * TILED_M * TILED_N / C) * C
        // const float split_cost = splits > 1 ? split_volume * wave_count * concurrency / bytes_per_second : 0;

        // (CTA_M * CTA_N) * (S - 1) * ceil(TILED_M * TILED_N / C) * C
        const float split_cost =
            split_volume * (splits - 1) * std::ceil(wave_per_split) * concurrency / bytes_per_second;

        // (CTA_M * CTA_N) * ceil(TILED_M * TILED_N / C) * C
        const float output_volume = cta_mn * 2;
        const float output_cost   = output_volume * std::ceil(wave_per_split) * concurrency / bytes_per_second;

        const float epi_cost = split_cost + output_cost;

        // Non-perfect latency hiding
        float cost = std::pow(fma_cost + ldg_cost, 0.2) * std::pow(std::max(fma_cost, ldg_cost), 0.8) + epi_cost;

        // std::cout << splits << " waves=" << waves << " fma=" << fma_cost * 1e3 << " ldg=" << ldg_cost * 1e3
        //           << " spk=" << split_cost * 1e3 << " cost=" << cost * 1e3 << std::endl;

        estimations.emplace_back(cost, 0, waves, splits);

        float w = (float)tiled_shape_mn / concurrency;
        if (splits == 1 && std::ceil(w) - w < 0.05) {
            break;
        }
    }

    for (auto i = (int)estimations.size() - 1; i >= 1; --i) {
        if (std::get<0>(estimations[i]) == std::get<0>(estimations[i - 1])) {
            std::get<0>(estimations[i]) = -1;
        }
    }
    estimations.erase(
        std::remove_if(estimations.begin(), estimations.end(), [](auto e) { return std::get<0>(e) == -1; }),
        estimations.end());

    top_k = std::min<int>(top_k, estimations.size());

    std::vector<int> idxs(estimations.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::partial_sort(idxs.begin(), idxs.begin() + top_k, idxs.end(), [&](int i, int j) {  //
        return estimations[i] < estimations[j];
    });

    std::vector<std::pair<int, float>> ret;
    ret.reserve(top_k);

    for (int i = 0; i < top_k; ++i) {
        auto& [cost, volume, waves, splits] = estimations[idxs[i]];
        if (i == 0) {
            // std::cout << "* " << cost * 1e3 << " " << volume << " " << waves << " " << splits << std::endl;
        }
        ret.emplace_back(splits, cost);
    }

    return ret;
}

bool Kernel::is_feasible(const GemmDesc& desc) const noexcept
{
    constexpr bool debug = false;

    if constexpr (debug)
        printf("S\n");

    if (!is_arch_compatible(desc_.arch, desc.arch)) {
        return false;
    }

    if constexpr (debug)
        printf("S0\n");

    if (std::tie(desc.order_a, desc.order_b, desc.order_c) != std::tie(desc_.order_a, desc_.order_b, desc_.order_c)) {
        return false;
    }

    if constexpr (debug)
        printf("A\n");

    if (std::tie(desc.type_a, desc.type_b, desc.type_c) != std::tie(desc_.type_a, desc_.type_b, desc_.type_c)) {
        return false;
    }

    if constexpr (debug) {
        printf("B\n");
        printf("%X %X %X %X\n", desc.pack_a, desc_.pack_a, desc.pack_u, desc_.pack_u);
    }

    if (std::tie(desc.pack_a, desc.pack_u) != std::tie(desc_.pack_a, desc_.pack_u)) {
        return false;
    }

    if constexpr (debug) {
        printf("C\n");
        printf("%X %X %X %X\n", desc.pack_b, desc_.pack_b, desc.pack_v, desc_.pack_v);
    }

    if (std::tie(desc.pack_b, desc.pack_v) != std::tie(desc_.pack_b, desc_.pack_v)) {
        return false;
    }

    if constexpr (debug)
        printf("D\n");

    if (desc.quant_a.type != desc_.quant_a.type || desc.quant_a.group_size != desc_.quant_a.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("E\n");

    if (desc.quant_b.type != desc_.quant_b.type || desc.quant_b.group_size != desc_.quant_b.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("F\n");

    if (desc.m % desc_.align.x || desc.n % desc_.align.y || desc.k % desc_.align.z) {
        return false;
    }

    return true;
}

std::string Kernel::GetName() const
{
    std::stringstream ss;

    ss << "sm" << desc_.arch / 10;
    ss << "_" << to_string(desc_.type_a);  //
    if ((int)desc_.quant_a.type) {
        ss << "g" << desc_.quant_a.group_size;
    }
    ss << "_" << to_string(desc_.type_b);  //
    if ((int)desc_.quant_b.type) {
        ss << "g" << desc_.quant_b.group_size;
    }
    ss << "_" << to_string(desc_.type_c);
    ss << "_"                                                                            //
       << (desc_.order_a == kColMajor ? 'n' : 't')                                       //
       << (desc_.order_b == kColMajor ? 'n' : 't')                                       //
       << (desc_.order_c == kColMajor ? 'n' : 't');                                      //
    ss << "_" << desc_.cta_tile.x << "x" << desc_.cta_tile.y << "x" << desc_.cta_tile.z  //
       << "_" << desc_.stages                                                            //
       << "_" << to_string(desc_.op_class)                                               //
       << "_" << desc_.mma_tile.x << "x" << desc_.mma_tile.y << "x" << desc_.mma_tile.z  //
       << "_c" << desc_.c_tile.x << "x" << desc_.c_tile.y                                //
       << "_a" << desc_.align.x << "x" << desc_.align.y << "x" << desc_.align.z          //
       << "_" << desc_.policy_a << desc_.policy_b;

    return ss.str();
}

}  // namespace turbomind::gemm
