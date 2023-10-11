#pragma once

#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void ConvertLinearToBlocks(const T*     src,
                           T**          dst_block_ptrs,
                           const int*   dst_cu_block_cnts,
                           const int*   seq_lens,
                           int          dst_offset,
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
                           int          src_offset,
                           int          src_block_len,
                           int          dst_max_seq_len,
                           int          head_num,
                           int          head_dim,
                           int          batch_size,
                           cudaStream_t st);

void ConvertKvCacheBlocksToLinear(const void** src_k_block_ptrs,
                                  const void** src_v_block_ptrs,
                                  void**       dst_k_ptrs,
                                  void**       dst_v_ptrs,
                                  const int*   src_cu_block_cnts,
                                  const int*   seq_lens,
                                  int          src_offset,
                                  int          src_block_len,
                                  int          dst_block_len,  // max{seq_lens}
                                  int          head_num,
                                  int          head_dim,
                                  int          batch_size,
                                  int          elem_bits,
                                  cudaStream_t st);

template<typename T>
void ConvertKvCacheBlocksToLinear2(const void** src_k_block_ptrs,
                                   const void** src_v_block_ptrs,
                                   T**          dst_k_ptrs,
                                   T**          dst_v_ptrs,
                                   const int*   src_cu_block_cnts,
                                   const int*   seq_lens,
                                   int          src_offset,
                                   int          src_block_len,
                                   int          dst_block_len,
                                   int          head_num,
                                   int          head_dim,
                                   int          batch_size,
                                   int          quant_policy,
                                   const float* kv_params,
                                   cudaStream_t st);

}  // namespace turbomind