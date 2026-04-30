// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"
#include "reduce.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "utils.h"

namespace turbomind {

template<class Kernel>
[[nodiscard]] cudaError_t invokeAttention(const typename Kernel::ParamType& params, int sm_count, int max_active_ctas)
{
    static const size_t kSmemSize = sizeof(typename Kernel::SharedStorage);

    if constexpr (1) {

        [[maybe_unused]] static const int _ = [&] {
            // std::cout << __PRETTY_FUNCTION__ << std::endl;
            // std::cout << "GmemMap:\n";
            // Print(typename Kernel::Impl::ThreadMapKV{});
            // std::cout << "\nDynamic smem size: " << kSmemSize << "\n";
            return 0;
        }();
    }

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    static const auto kernel_func = &attention_kernel<Kernel>;

    const int max_cp_k_len    = cdiv(params.max_k_len, (int)params.cp_size);
    const int tile_count      = cdiv(std::min(max_cp_k_len, params.window_size), Kernel::CTA_S);
    const int max_split_count = std::min(params.max_split_k, tile_count);

    typename Kernel::CtaMap cta_map{
        params.max_q_len, params.batch_size, params.num_heads, Kernel::CTA_Q, Kernel::CTA_H, 1};

    // grid shape when split cnt = 1
    dim3 grid = cta_map.get_grid_shape();

    const int grid_size = grid.x * grid.y * grid.z;
    const int split_cnt = GetSplitCount(max_split_count, grid_size, max_active_ctas, sm_count, 8);

    // printf("max split cnt: %d, split cnt: %d\n", max_split_count, split_cnt);

    // adjust split cnt and update grid shape
    cta_map.set_split_cnt(split_cnt);
    grid = cta_map.get_grid_shape();

    auto cache_iter_factory = CreateCacheIterFactory<typename Kernel::CacheIteratorFactory>::apply(params);

    const int q_group_size = params.num_heads / params.num_kv_heads;

    kernel_func<<<grid, block, kSmemSize, params.stream>>>(params,
                                                           cache_iter_factory,
                                                           cta_map,
                                                           q_group_size,
                                                           1,            // q_head_per_cta
                                                           q_group_size  // cta_per_q_group
    );

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        return err;
    }

    if (params.cp_fn) {
        params.cp_fn(params.cp_fn_ctx);
    }

    if (split_cnt > 1 || params.cp_size > 1) {
        TM_CUDA_CHECK(attention::invokeReduceV3<Kernel::kHeadDim>(
            params.out + params.offset_q * params.num_heads * Kernel::kHeadDim,
            params.partial_ML,
            params.partial_O,
            split_cnt > 1 ? params.split_cnt : nullptr,
            params.max_split_k,
            split_cnt,
            params.cp_size,
            params.cp_rank,
            params.token_num,
            params.num_heads,
            params.inv_sqrt_dh,
            params.stream));
    }

    return cudaSuccess;
}

}  // namespace turbomind
