#pragma once

#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void ConvertLinearToBlocks(
    const T* src, T** dst_block_ptrs, int dst_block_size, int heads, int dims, int seq_len, cudaStream_t st);

template<typename T>
void ConvertBlocksToLinear(
    const T** src_block_ptrs, T* dst, int src_block_size, int heads, int dims, int seq_len, cudaStream_t st);

template<typename T>
void ConvertBlocksToBlocks(const T**    src_block_ptrs,
                           T**          dst_block_ptrs,
                           int          src_block_size,
                           int          dst_block_size,
                           int          heads,
                           int          dims,
                           int          seq_len,
                           cudaStream_t st);

}  // namespace turbomind