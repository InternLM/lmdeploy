// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/kernel.h"
#include <iostream>

namespace turbomind::gemm {

std::pair<int, int64_t> Kernel::FindSplitCount(int m, int n, int k, int max_split_k, int sm_count, int max_wave_count)
{
    const int tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    std::cout << tiled_shape_m << " " << tiled_shape_n << " " << chunk_cnt_k << std::endl;

    const int64_t cta_mn = desc_.cta_tile.x * desc_.cta_tile.y;

    const float wave_per_split = float(tiled_shape_m * tiled_shape_n) / float(sm_count * max_active_ctas_);

    //          cost    volume   waves  split-k
    std::tuple<int64_t, int64_t, float, int> best{std::numeric_limits<int64_t>::max(), -1, 0.f, 0};

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
        if (cost < std::get<0>(best)) {
            best = std::tie(cost, volume, waves, splits);
        }
    }

    auto [cost, volume, waves, splits] = best;

    std::cout << "* " << cost << " " << volume << " " << waves << " " << splits << std::endl;

    return {splits, cost};
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

    ss << "gemm_"                                                                        //
       << to_string(desc_.type_a) << "_"                                                 //
       << to_string(desc_.type_b) << "_"                                                 //
       << to_string(desc_.type_c) << "_"                                                 //
       << desc_.cta_tile.x << "x" << desc_.cta_tile.y << "x" << desc_.cta_tile.z << "_"  //
       << desc_.stages;

    return ss.str();
}

}  // namespace turbomind::gemm