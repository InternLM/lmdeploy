// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"
#include "reduce_template.h"
#include "utils.h"

namespace turbomind {

template<class Kernel>
void invokeAttention(const typename Kernel::ParamType& params)
{
    static const size_t kSmemSize = sizeof(typename Kernel::SharedStorage);

    if constexpr (0) {
        [[maybe_unused]] static const int _ = [&] {
            std::cout << "GmemMap:\n";
            Print(typename Kernel::Impl::ThreadMapKV{});
            std::cout << "\nDynamic smem size: " << kSmemSize << "\n";
            return 0;
        }();
    }

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    static const auto kernel_func = &attention_kernel<Kernel>;

    thread_local const int2 caps = [&] {
        int device_id{};
        cudaGetDevice(&device_id);
        int sm_count{};
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        int max_active_ctas{};
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_ctas, kernel_func, block.x, kSmemSize);
        return int2{sm_count, max_active_ctas};
    }();

    const int tile_count      = (params.max_k_len + Kernel::CTA_S - 1) / Kernel::CTA_S;
    const int max_split_count = std::min(params.max_split_k, tile_count);

    typename Kernel::CtaMap cta_map{
        params.max_q_len, params.batch_size, params.num_heads, Kernel::CTA_Q, Kernel::CTA_H, 1};

    // grid shape when split cnt = 1
    dim3 grid = cta_map.get_grid_shape();

    const int grid_size = grid.x * grid.y * grid.z;
    const int split_cnt = GetSplitCount(max_split_count, grid_size, caps.y, caps.x, 8);

    // adjust split cnt and update grid shape
    cta_map.set_split_cnt(split_cnt);
    grid = cta_map.get_grid_shape();

    auto err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    if (err) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    kernel_func<<<grid, block, kSmemSize, params.stream>>>(params, cta_map);

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    if (Kernel::need_separate_reduce(split_cnt)) {
        attention::dispatchReduce<Kernel::kHeadDim>(params.out,
                                                    params.partial_M,
                                                    params.partial_L,
                                                    params.partial_O,
                                                    params.split_cnt,
                                                    params.max_split_k,
                                                    split_cnt,
                                                    params.token_num,
                                                    params.num_heads,
                                                    params.inv_sqrt_dh,
                                                    params.stream);
    }
}

}  // namespace turbomind