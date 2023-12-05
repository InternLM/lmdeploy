// Copyright (c) OpenMMLab. All rights reserved.

#include "decoder_multihead_attention_template.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <iostream>

namespace turbomind {

namespace {

template<typename MHAType>
bool Print(size_t dynamic_smem_size)
{
    using MapKv = typename MHAType::MapKv;

    std::cout << "     warps: " << MapKv::kWarpCount << "\n";
    std::cout << "     shape: (" << MapKv::kC << ", " << MapKv::kS << ")\n";
    std::cout << "    access: (" << MapKv::kAccessC << ", " << 1 << ")\n";
    std::cout << "warpThread: (" << MapKv::kWarpThreadC << ", " << MapKv::kWarpThreadS << ")\n";
    std::cout << "warpAccess: (" << MapKv::kWarpAccessC << ", " << MapKv::kWarpAccessS << ")\n";
    std::cout << "  warpIter: (" << MapKv::kWarpIterC << ", " << MapKv::kWarpIterS << ")\n";
    std::cout << "      warp: (" << MapKv::kWarpC << ", " << MapKv::kWarpS << ")\n";
    std::cout << "      iter: (" << MapKv::kIterC << ", " << MapKv::kIterS << ")\n";
    std::cout << " footprint: (" << MapKv::kFootprintC << ", " << MapKv::kFootprintS << ")\n";
    std::cout << "     delta: (" << MapKv::kDeltaC << ", " << MapKv::kDeltaS << ")\n";
    std::cout << "dynamic smem size: " << dynamic_smem_size << "\n";

    return true;
}

}  // namespace

template<typename T, typename Tkv, int HeadDim, int HeadPerCta>
void invokeDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    auto invoke = [&](auto* type) {
        using Attn = std::remove_reference_t<decltype(*type)>;

        static const size_t kDynSmemSize = Attn::GetDynamicSmemSize();

        // [[maybe_unused]] static const bool _ = Print<Attn>(kDynSmemSize);

        const int slice_count = (params.max_seq_len + Attn::kSliceLen - 1) / Attn::kSliceLen;
        const int max_split_k = std::min(params.max_split_k, std::max(1, slice_count));

        dim3 block(Attn::kWarpCount * WARP_SIZE);
        dim3 grid(params.num_heads / HeadPerCta, params.batch_size, max_split_k);

        // if (params.layer_offset == 0) {
        //     std::cout << "max_split_k' = " << max_split_k << ", arch = " << params.arch << "\n";
        // }

        cudaFuncSetAttribute(
            decoder_multihead_attention<Attn>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmemSize);

        decoder_multihead_attention<Attn><<<grid, block, kDynSmemSize, params.stream>>>(params);

        if (max_split_k > 1) {
            dim3 grid(params.num_heads, params.batch_size);
            decoder_multihead_attention_reduce<Attn><<<grid, block, 0, params.stream>>>(params);
        }
    };

    if (params.arch >= 80) {
        // DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 2048, 6>;  // 64k

        using Type = DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 1024, 5, true>;
        invoke((Type*)0);
    }
    else {
        // DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 2048, 3>; // 34k
        // DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 64, HeadDim, 2048, 3>;  // 34k

        using Type = DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 64, HeadDim, 1024, 3, true>;
        invoke((Type*)0);
    }
}

template<typename T>
void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    static constexpr int HeadDim = 128;

    FT_CHECK(params.size_per_head == HeadDim);

    if constexpr (std::is_same_v<T, half>) {
        if (params.quant_policy & QuantPolicy::kCacheKVInt8) {
            invokeDecoderMultiheadAttention<T, int8_t, HeadDim, 1>(params);
            return;
        }

        int group_size = params.num_heads / params.num_kv_heads;

        if (0) {}
        // else if (group_size % 8 == 0) {
        //     invokeDecoderMultiheadAttention<T, T, HeadDim, 8>(params);
        // }
        else if (group_size % 4 == 0) {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 4>(params);
        }
        else if (group_size % 2 == 0) {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 2>(params);
        }
        else {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 1>(params);
        }
    }

#ifdef ENABLE_BF16
    if constexpr(std::is_same_v<T, __nv_bfloat16>) {
        int group_size = params.num_heads / params.num_kv_heads;
        if (group_size % 4 == 0) {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 4>(params);
        }
        else if (group_size % 2 == 0) {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 2>(params);
        }
        else {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 1>(params);
        }
    }
#endif // ENABLE_BF16
}

template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<half>& params);
template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<float>& params);
#ifdef ENABLE_BF16
template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<__nv_bfloat16>& params);
#endif // ENABLE_BF16
}  // namespace turbomind
