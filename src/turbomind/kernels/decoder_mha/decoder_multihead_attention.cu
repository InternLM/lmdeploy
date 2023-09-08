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

template<typename T, int HeadDim>
void LaunchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    using MHAType = DecoderMultiHeadAttentionKernel<T, 1, HeadDim, 16, HeadDim, 1024, 5>;

    [[maybe_unused]] static const bool init = Dump<MHAType>();

    dim3 block(MHAType::kWarpCount * WARP_SIZE);
    dim3 grid(params.num_kv_heads, params.batch_size);

    const size_t kDynamicSmemSize = MHAType::GetDynamicSmemSize(params.max_timestep);
    std::cout << "dynamic shared memory size: " << kDynamicSmemSize << "\n";

    cudaFuncSetAttribute(
        decoder_multihead_attention<MHAType>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemSize);

    decoder_multihead_attention<MHAType><<<grid, block, kDynamicSmemSize>>>(params);
}

template void LaunchDecoderMultiheadAttention<half, 128>(const DecoderMultiHeadAttentionParams<half>& params);

}  // namespace turbomind
