// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/kernel.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace turbomind::gemm {

std::vector<std::pair<int, int64_t>>
Kernel::EstimateSplits(int m, int n, int k, int max_split_k, int sm_count, int max_wave_count, int top_k)
{
    const int tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    std::cout << tiled_shape_m << " " << tiled_shape_n << " " << chunk_cnt_k << std::endl;

    const int64_t cta_mn = desc_.cta_tile.x * desc_.cta_tile.y;

    const float wave_per_split = float(tiled_shape_m * tiled_shape_n) / float(sm_count * max_active_ctas_);

    //                     cost    volume   waves  split-k
    std::vector<std::tuple<int64_t, int64_t, float, int>> estimations;

    for (int splits = 1; splits <= max_split_k; ++splits) {
        const float waves      = wave_per_split * splits;
        const int   wave_count = (int)std::ceil(waves);

        if (splits > 1 && wave_count > max_wave_count) {
            break;
        }

        const int gemm_size_k = ceil_div(chunk_cnt_k, splits) * chunk_size_k_;

        const int64_t volume = cta_mn * gemm_size_k;  // cta volume
        const int64_t cost   = volume * wave_count;

        std::cout << cost << " " << volume << " " << waves << " " << splits << std::endl;

        estimations.emplace_back(cost, volume, waves, splits);
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

    std::vector<std::pair<int, int64_t>> ret;
    ret.reserve(top_k);

    for (int i = 0; i < top_k; ++i) {
        auto& [cost, volume, waves, splits] = estimations[idxs[i]];
        if (i == 0) {
            std::cout << "* " << cost << " " << volume << " " << waves << " " << splits << std::endl;
        }
        ret.emplace_back(splits, cost);
    }

    return ret;
}

bool Kernel::is_feasible(const GemmDesc& desc) const noexcept
{
    // printf("S\n");

    if (std::tie(desc.order_a, desc.order_b, desc.order_c) != std::tie(desc_.order_a, desc_.order_b, desc_.order_c)) {
        return false;
    }

    // printf("A\n");

    if (std::tie(desc.type_a, desc.type_b, desc.type_c) != std::tie(desc_.type_a, desc_.type_b, desc_.type_c)) {
        return false;
    }

    // printf("B\n");

    if (desc.quant_b.type != desc_.quant_b.type || desc.quant_b.group_size != desc_.quant_b.group_size) {
        return false;
    }

    // printf("C\n");

    if (desc.k % desc_.cta_tile.z) {
        return false;
    }

    // printf("D\n");

    if (desc.n % 8 != 0) {
        return false;
    }

    // printf("E\n");

    if (desc_.align_m && desc.m % desc_.cta_tile.x) {
        return false;
    }

    // printf("F\n");

    if (desc_.align_n && desc.n % desc_.cta_tile.y) {
        return false;
    }

    // printf("G\n");

    return true;
}

std::string Kernel::GetName() const
{
    std::stringstream ss;

    ss << "gemm_"                                                                           //
       << to_string(desc_.type_a) << "_"                                                    //
       << to_string(desc_.type_b) << "_"                                                    //
       << to_string(desc_.type_c) << "_"                                                    //
       << desc_.cta_tile.x << "x" << desc_.cta_tile.y << "x" << desc_.cta_tile.z << "_"     //
       << desc_.warp_tile.x << "x" << desc_.warp_tile.y << "x" << desc_.warp_tile.z << "_"  //
       << desc_.stages << "_"                                                               //
       << (desc_.align_m ? "a" : "n") << (desc_.align_n ? "a" : "n");

    return ss.str();
}

}  // namespace turbomind::gemm