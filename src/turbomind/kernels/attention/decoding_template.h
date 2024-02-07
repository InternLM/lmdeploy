// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"
#include "reduce_template.h"
#include "src/turbomind/kernels/attention/thread_map.h"

namespace turbomind {

template<class Kernel>
void invokeDecoding(const typename Kernel::ParamType& params)
{
    static const size_t kDynamicSmemSize = sizeof(typename Kernel::SharedStorage);

    if constexpr (0) {
        [[maybe_unused]] static const int _ = [&] {
            std::cout << "GmemMap:\n";
            Print(typename Kernel::Impl::ThreadMapKV{});
            std::cout << "\nDynamic smem size: " << kDynamicSmemSize << "\n";
            return 0;
        }();
    }

    const int tile_count      = (params.max_k_len + Kernel::CTA_S - 1) / Kernel::CTA_S;
    const int max_split_count = std::min(params.max_split_k, tile_count);

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    using CtaMap = typename Kernel::CtaMap;

    dim3 grid = CtaMap::get_grid_shape(params.num_heads, params.batch_size, max_split_count, Kernel::CTA_H);

    auto err =
        cudaFuncSetAttribute(attention_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemSize);
    if (err) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    attention_kernel<Kernel><<<grid, block, kDynamicSmemSize, params.stream>>>(params);

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    if (max_split_count > 32) {
        using Reduce = typename Kernel::SeparateReduce;
        attention::invokeReduce<Reduce>(params.out,
                                        params.partial_M,
                                        params.partial_L,
                                        params.partial_O,
                                        params.locks,
                                        params.split_cnt,
                                        params.max_split_k,
                                        max_split_count,
                                        params.token_num,
                                        params.num_heads,
                                        params.inv_sqrt_dh,
                                        params.stream);
    }
}

}  // namespace turbomind