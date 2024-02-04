// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"

namespace turbomind {

template<class Kernel>
void invokeAttention(const typename Kernel::ParamType& params)
{
    static const size_t kDynamicSmemSize = sizeof(typename Kernel::SharedStorage);

    // [[maybe_unused]] static const int _ = [&] {
    //     std::cout << "GmemMap:\n";
    //     Print(typename Kernel::Impl::ThreadMapKV{});
    //     std::cout << "\nDynamic smem size: " << kDynamicSmemSize << "\n";
    //     return 0;
    // }();

    // const int slice_count = (params.max_seq_len + Attn::kSliceLen - 1) / Attn::kSliceLen;
    // const int max_split_k = std::min(params.max_split_k, std::max(1, slice_count));

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    using CtaMap = typename Kernel::CtaMap;

    dim3 grid =
        CtaMap::get_grid_shape(params.num_heads, params.batch_size, params.max_q_len, Kernel::CTA_H, Kernel::CTA_Q);

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
}

}  // namespace turbomind