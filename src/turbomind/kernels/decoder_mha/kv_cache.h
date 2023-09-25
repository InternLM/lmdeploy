#pragma once

#include <cuda_runtime.h>

namespace turbomind {

// template<typename T>
// void ConvertLinearToBlocks(
//     const T* src, T** dst_block_ptrs, int dst_block_size, int heads, int dims, int seq_len, cudaStream_t st);

// template<typename T>
// void ConvertBlocksToLinear(
//     const T** src_block_ptrs, T* dst, int src_block_size, int heads, int dims, int seq_len, cudaStream_t st);

// template<typename T>
// void ConvertBlocksToBlocks(const T**    src_block_ptrs,
//                            T**          dst_block_ptrs,
//                            int          src_block_size,
//                            int          dst_block_size,
//                            int          heads,
//                            int          dims,
//                            int          seq_len,
//                            cudaStream_t st);

template<typename T>
void ConvertLinearToBlocks(const T*     src,
                           T**          dst_block_ptrs,
                           const int*   dst_cu_block_cnts,
                           const int*   seq_lens,
                           int          src_seq_len,
                           int          dst_block_len,
                           int          head_num,
                           int          head_dim,
                           int          batch_size,
                           cudaStream_t st);

template<typename T>
void ConvertBlocksToLinear(const T**    src_block_ptrs,
                           T*           dst,
                           const int*   src_cu_block_cnts,
                           const int*   seq_lens,
                           int          src_block_len,
                           int          dst_max_seq_len,
                           int          head_num,
                           int          head_dim,
                           int          batch_size,
                           cudaStream_t st);

}  // namespace turbomind