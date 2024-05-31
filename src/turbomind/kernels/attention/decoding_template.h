// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"
#include "reduce.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "utils.h"
namespace turbomind {

template<class Kernel>
bool invokeDecoding(const typename Kernel::ParamType& params)
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

    const int tile_count      = (params.max_k_len + Kernel::CTA_S - 1) / Kernel::CTA_S;
    const int max_split_count = std::min(params.max_split_k, tile_count);

    using CtaMap = typename Kernel::CtaMap;

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    auto kernel_func = &attention_kernel<Kernel>;

    thread_local const int2 caps = [&] {
        auto err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
        if (err) {
            std::cout << cudaGetErrorString(err) << "\n";
            std::abort();
        }
        int device_id{};
        cudaGetDevice(&device_id);
        int sm_count{};
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        int max_active_ctas{};
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_ctas, kernel_func, block.x, kSmemSize);
        return int2{sm_count, max_active_ctas};
    }();

    const int q_group_size   = params.num_heads / params.num_kv_heads;
    const int q_head_per_cta = std::min(q_group_size, Kernel::CTA_H);

    // cta needed to process one query group
    const int cta_per_q_group = (q_group_size + q_head_per_cta - 1) / q_head_per_cta;

    // std::cout << "CTA_H: " << Kernel::CTA_H << ", head_per_cta: " << q_head_per_cta
    //           << ", cta_per_q_group: " << cta_per_q_group << "\n";

    dim3 grid = CtaMap::get_grid_shape(params.num_kv_heads, params.batch_size, 1, cta_per_q_group);

    const int grid_size = grid.x * grid.y * grid.z;
    const int split_cnt = GetSplitCount(max_split_count, grid_size, caps.y, caps.x, 4);

    grid = CtaMap::get_grid_shape(params.num_kv_heads, params.batch_size, split_cnt, cta_per_q_group);

    auto err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    if (err) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    // Print(typename Kernel::Impl::ThreadMapKVp{});

    auto cache_iter_factory = CreateCacheIterFactory<typename Kernel::CacheIteratorFactory>::apply(params);

    kernel_func<<<grid, block, kSmemSize, params.stream>>>(
        params, cache_iter_factory, CtaMap{}, q_group_size, q_head_per_cta, cta_per_q_group);

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    if (Kernel::need_separate_reduce(split_cnt)) {
        attention::invokeReduce<Kernel::kHeadDim>(params.out,
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

    return true;
}

}  // namespace turbomind
