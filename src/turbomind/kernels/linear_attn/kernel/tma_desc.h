#pragma once

#include "src/turbomind/kernels/core/smem.h"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace turbomind::linear_attn::delta_rule {

CUtensorMap MakeTmaDesc(void*              global_address,
                        CUtensorMapDataType data_type,
                        uint32_t            rank,
                        const uint64_t*     global_dim,
                        const uint64_t*     global_stride,
                        const uint32_t*     box_dim,
                        CUtensorMapSwizzle  swizzle);

__device__ __forceinline__ void CopyTmaDescriptor(CUtensorMap* dst, const CUtensorMap* src, int lane, int lanes)
{
    constexpr int kWords = sizeof(CUtensorMap) / sizeof(uint2);
    auto*       dst_words = reinterpret_cast<uint2*>(dst);
    const auto* src_words = reinterpret_cast<const uint2*>(src);
    for (int word = lane; word < kWords; word += lanes) {
        dst_words[word] = src_words[word];
    }
}

__device__ __forceinline__ void ReplaceTmaAddress(CUtensorMap* desc, const void* global_address)
{
    const uint32_t smem_ptr = cast_smem_ptr_to_uint(desc);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :
                 : "r"(smem_ptr), "l"(global_address));
}

__device__ __forceinline__ void PublishTmaDescriptor(CUtensorMap* gmem_desc, CUtensorMap* smem_desc)
{
    const uint64_t gmem_ptr = reinterpret_cast<uint64_t>(gmem_desc);
    const uint32_t smem_ptr = cast_smem_ptr_to_uint(smem_desc);
    asm volatile(
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
        :
        : "l"(gmem_ptr), "r"(smem_ptr));
}

}  // namespace turbomind::linear_attn::delta_rule
