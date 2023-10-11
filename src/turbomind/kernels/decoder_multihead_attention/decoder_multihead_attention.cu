#include "decoder_multihead_attention_template.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include <iostream>

namespace turbomind {

template<typename MHAType>
bool Dump()
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

    return true;
}

template<typename T, typename Tkv, int HeadDim, int HeadPerCta>
void invokeDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    // 2048_32x6 ~ 64k smem
    using MHAType = DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 2048, 6>;

    [[maybe_unused]] static const bool init = Dump<MHAType>();

    dim3 block(MHAType::kWarpCount * WARP_SIZE);
    dim3 grid(params.num_heads / HeadPerCta, params.batch_size);

    static const size_t kDynSmemSize = MHAType::GetDynamicSmemSize();
    // std::cout << "dynamic shared memory size: " << kDynamicSmemSize << "\n";

    cudaFuncSetAttribute(
        decoder_multihead_attention<MHAType>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmemSize);

    decoder_multihead_attention<MHAType><<<grid, block, kDynSmemSize>>>(params);
}

template<typename T>
void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    static constexpr int HeadDim = 128;

    FT_CHECK(params.size_per_head == HeadDim);

    if constexpr (std::is_same_v<T, half>) {

        //     int group_size = params.num_heads / params.num_kv_heads;

        //     if (group_size % 8 == 0) {
        //         invokeDecoderMultiheadAttention<T, HeadDim, 8>(params);
        //     }
        //     else if (group_size % 4 == 0) {
        //         invokeDecoderMultiheadAttention<T, HeadDim, 4>(params);
        //     }
        //     else if (group_size % 2 == 0) {
        //         invokeDecoderMultiheadAttention<T, HeadDim, 2>(params);
        //     }
        //     else {
        //         invokeDecoderMultiheadAttention<T, HeadDim, 1>(params);
        //     }
        // }
        // else {
        if (params.quant_policy & QuantPolicy::kCacheKVInt8) {
            invokeDecoderMultiheadAttention<T, int8_t, HeadDim, 1>(params);
        }
        else {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 1>(params);
        }
    }
}

template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<half>& params);
template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<float>& params);

}  // namespace turbomind
