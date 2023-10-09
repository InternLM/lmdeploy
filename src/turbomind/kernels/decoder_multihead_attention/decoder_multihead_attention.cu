#include "decoder_multihead_attention_template.h"

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

template<typename T, int HeadDim, int HeadPerCta>
void InvokeDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    using MHAType = DecoderMultiHeadAttentionKernel<T, HeadPerCta, HeadDim, 16, HeadDim, 2048, 6>;

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

        int group_size = params.num_heads / params.num_kv_heads;

        if (group_size % 8 == 0) {
            InvokeDecoderMultiheadAttention<T, HeadDim, 8>(params);
        }
        else if (group_size % 4 == 0) {
            InvokeDecoderMultiheadAttention<T, HeadDim, 4>(params);
        }
        else if (group_size % 2 == 0) {
            InvokeDecoderMultiheadAttention<T, HeadDim, 2>(params);
        }
        else {
            InvokeDecoderMultiheadAttention<T, HeadDim, 1>(params);
        }
    }
    else {
        InvokeDecoderMultiheadAttention<T, HeadDim, 1>(params);
    }
}

template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<half>& params);
template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<float>& params);

}  // namespace turbomind
