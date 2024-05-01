// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/kernel.h"
#include <iostream>

namespace turbomind::gemm {

std::pair<int, int64_t> Kernel::FindSplitCount(int m, int n, int k, int max_split_k, int sm_count, int max_wave_count)
{
    const int tiled_shape_m = ceil_div(m, cta_tile_size_.x);
    const int tiled_shape_n = ceil_div(n, cta_tile_size_.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    std::cout << tiled_shape_m << " " << tiled_shape_n << " " << chunk_cnt_k << std::endl;

    const int64_t cta_mn = cta_tile_size_.x * cta_tile_size_.y;

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

    if (std::tie(desc.layout_A, desc.layout_B, desc.layout_C) != std::tie(layout_A_, layout_B_, layout_C_)) {
        return false;
    }

    // printf("A\n");

    if (std::tie(desc.type_A, desc.type_B, desc.type_C) != std::tie(type_A_, type_B_, type_C_)) {
        return false;
    }

    // printf("B\n");

    if (desc.quant_type != quant_type_) {
        return false;
    }

    // printf("C\n");

    if (desc.k % cta_tile_size_.z) {
        return false;
    }

    // printf("D\n");

    if (desc.n % 8 != 0) {
        return false;
    }

    // printf("E\n");

    if (align_m_ && desc.m % cta_tile_size_.x) {
        return false;
    }

    // printf("F\n");

    if (align_n_ && desc.n % cta_tile_size_.y) {
        return false;
    }

    // printf("G\n");

    return true;
}

std::string Kernel::GetName() const
{
    std::stringstream ss;

    ss << "gemm_" << to_string(type_A_) << "_" << to_string(type_B_) << "_" << to_string(type_C_) << "_"
       << cta_tile_size_.x << "x" << cta_tile_size_.y << "x" << cta_tile_size_.z << "_" << stages_;

    return ss.str();
}

}  // namespace turbomind::gemm