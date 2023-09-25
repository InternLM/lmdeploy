#include "../gemm_s_f16/common.h"
// #include "cute/tensor.hpp"
#include <cuda_fp16.h>
#include <type_traits>

namespace turbomind {

// [S/x, H, x, D] <-> [S/y, H, y, D]

template<typename T, typename SrcBlockLen, typename DstBlockLen, typename HeadDim>
__inline__ __device__ void ConvertBlockSize(const T** __restrict__ src_block_ptrs,
                                            T** __restrict__ dst_block_ptrs,
                                            const int* __restrict__ src_cu_block_cnts,
                                            const int* __restrict__ dst_cu_block_cnts,
                                            const int* __restrict__ seq_lens,
                                            SrcBlockLen src_block_len,
                                            DstBlockLen dst_block_len,
                                            HeadDim     head_dim)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    const int hi = blockIdx.y;
    const int bi = blockIdx.z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int di  = idx * kVecSize % head_dim;
    const int si  = idx * kVecSize / head_dim;

    if (si >= seq_lens[bi]) {
        return;
    }

    // compute indices into src
    int src_block_index  = si / src_block_len + src_cu_block_cnts[bi];
    int src_block_offset = hi * src_block_len * head_dim + si % src_block_len * head_dim + di;

    // compute indices into dst
    int dst_block_index  = si / dst_block_len + dst_cu_block_cnts[bi];
    int dst_block_offset = hi * dst_block_len * head_dim + si % dst_block_len * head_dim + di;

    // printf("%d %d\n", src_block_index, dst_block_index);

    const T* __restrict__ src_block = src_block_ptrs[src_block_index];
    T* __restrict__ dst_block       = dst_block_ptrs[dst_block_index];

    uint4 data = __ldg(reinterpret_cast<const uint4*>(src_block + src_block_offset));

    *reinterpret_cast<uint4*>(dst_block + dst_block_offset) = data;
}

// static inline size_t get_helper_smem_size(int batch_size)
// {
//     return (sizeof(void*) + sizeof(int)) * batch_size;
// }

template<typename T>
__global__ void LinearToBlocksKernel(const T*   src,
                                     T**        dst_block_ptrs,
                                     const int* dst_cu_block_cnts,
                                     const int* seq_lens,
                                     int        src_block_len,
                                     int        dst_block_len,
                                     int        head_num,
                                     int        head_dim,
                                     int        batch_size)
{
    extern __shared__ void* smem[];

    const T** src_block_ptrs    = (const T**)smem;
    int*      src_cu_block_cnts = (int*)(src_block_ptrs + batch_size);

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        src_cu_block_cnts[i] = i;
        src_block_ptrs[i]    = src + blockIdx.z * head_num * src_block_len * head_dim;
    }

    __syncthreads();

    ConvertBlockSize(src_block_ptrs,
                     dst_block_ptrs,
                     src_cu_block_cnts,
                     dst_cu_block_cnts,
                     seq_lens,
                     src_block_len,
                     dst_block_len,
                     head_dim);
}

template<typename T>
void ConvertLinearToBlocks(const T*     src,
                           T**          dst_block_ptrs,
                           const int*   dst_cu_block_cnts,
                           const int*   seq_lens,
                           int          src_max_len,
                           int          dst_block_len,
                           int          head_num,
                           int          head_dim,
                           int          batch_size,
                           cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    constexpr int threads = 128;
    const dim3    blocks((src_max_len * head_dim / kVecSize + threads - 1) / threads, head_num, batch_size);

    const auto smem_sz = (sizeof(void*) + sizeof(int)) * batch_size;

    auto fn = [&](auto head_dim) {
        LinearToBlocksKernel<<<blocks, threads, smem_sz, st>>>(src,
                                                               dst_block_ptrs,
                                                               dst_cu_block_cnts,
                                                               seq_lens,
                                                               src_max_len,
                                                               dst_block_len,
                                                               head_num,
                                                               head_dim,
                                                               batch_size);
    };

    switch (head_dim) {
        case 128:
            fn(std::integral_constant<int, 128>{});
            break;
        default:
            fn(head_dim);
    }
}

template void ConvertLinearToBlocks(const half*  src,
                                    half**       dst_block_ptrs,
                                    const int*   dst_cu_block_cnts,
                                    const int*   seq_lens,
                                    int          src_seq_len,
                                    int          dst_block_len,
                                    int          head_num,
                                    int          head_dim,
                                    int          batch_size,
                                    cudaStream_t st);

template<typename T, typename HeadDim>
__global__ void BlocksToLinearKernel(const T**  src_block_ptrs,
                                     T*         dst,
                                     const int* src_cu_block_cnts,
                                     const int* seq_lens,
                                     int        src_block_len,
                                     int        dst_block_len,
                                     int        head_num,
                                     HeadDim    head_dim,
                                     int        batch_size)
{
    extern __shared__ void* smem[];

    T**  dst_block_ptrs    = (T**)smem;
    int* dst_cu_block_cnts = (int*)(dst_block_ptrs + batch_size);

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        dst_cu_block_cnts[i] = i;
        dst_block_ptrs[i]    = dst + blockIdx.z * head_num * dst_block_len * head_dim;
    }

    __syncthreads();

    ConvertBlockSize(src_block_ptrs,
                     dst_block_ptrs,
                     src_cu_block_cnts,
                     dst_cu_block_cnts,
                     seq_lens,
                     src_block_len,
                     dst_block_len,
                     head_dim);
}

template<typename T>
void ConvertBlocksToLinear(const T**    src_block_ptrs,
                           T*           dst,
                           const int*   src_cu_block_cnts,
                           const int*   seq_lens,
                           int          src_block_len,
                           int          dst_max_len,
                           int          head_num,
                           int          head_dim,
                           int          batch_size,
                           cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    constexpr int threads = 256;
    const dim3    blocks((dst_max_len * head_dim / kVecSize + threads - 1) / threads, head_num, batch_size);

    const auto smem_sz = (sizeof(void*) + sizeof(int)) * batch_size;

    auto fn = [&](auto head_dim) {
        BlocksToLinearKernel<<<blocks, threads, smem_sz, st>>>(src_block_ptrs,
                                                               dst,
                                                               src_cu_block_cnts,
                                                               seq_lens,
                                                               std::integral_constant<int, 128>{},
                                                               dst_max_len,
                                                               head_num,
                                                               head_dim,
                                                               batch_size);
    };

    switch (head_dim) {
        case 128:
            fn(std::integral_constant<int, 128>{});
            break;
        default:
            fn(head_dim);
    }
}

template void ConvertBlocksToLinear(const half** src_block_ptrs,
                                    half*        dst,
                                    const int*   src_cu_block_cnts,
                                    const int*   seq_lens,
                                    int          src_block_len,
                                    int          dst_max_seq_len,
                                    int          head_num,
                                    int          head_dim,
                                    int          batch_size,
                                    cudaStream_t st);

template<typename T, typename SrcBlockLen, typename DstBlockLen, typename HeadDim>
__global__ void KvCacheBlocksToLinearKernel(const T**   src_k_block_ptrs,
                                            const T**   src_v_block_ptrs,
                                            T**         dst_k_ptrs,
                                            T**         dst_v_ptrs,
                                            const int*  src_cu_block_cnts,
                                            const int*  seq_lens,
                                            SrcBlockLen src_block_len,
                                            DstBlockLen dst_block_len,
                                            int         head_num,
                                            HeadDim     head_dim,
                                            int         batch_size)
{
    extern __shared__ int dst_cu_block_cnts[];

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        dst_cu_block_cnts[i] = i;
    }

    __syncthreads();

    ConvertBlockSize(src_k_block_ptrs,
                     dst_k_ptrs,
                     src_cu_block_cnts,
                     dst_cu_block_cnts,
                     seq_lens,
                     src_block_len,
                     dst_block_len,
                     head_dim);

    ConvertBlockSize(src_v_block_ptrs,
                     dst_v_ptrs,
                     src_cu_block_cnts,
                     dst_cu_block_cnts,
                     seq_lens,
                     src_block_len,
                     dst_block_len,
                     head_dim);
}

template<typename T>
void ConvertKvCacheBlocksToLinear(const T**    src_k_block_ptrs,
                                  const T**    src_v_block_ptrs,
                                  T**          dst_k_ptrs,
                                  T**          dst_v_ptrs,
                                  const int*   src_cu_block_cnts,
                                  const int*   seq_lens,
                                  int          src_block_len,
                                  int          dst_block_len,
                                  int          head_num,
                                  int          head_dim,
                                  int          batch_size,
                                  cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    constexpr int threads = 256;
    const dim3    blocks((dst_block_len * head_dim / kVecSize + threads - 1) / threads, head_num, batch_size);

    const auto smem_sz = sizeof(int) * batch_size;

    auto fn = [&](auto head_dim) {
        KvCacheBlocksToLinearKernel<<<blocks, threads, smem_sz, st>>>(src_k_block_ptrs,
                                                                      src_v_block_ptrs,
                                                                      dst_k_ptrs,
                                                                      dst_v_ptrs,
                                                                      src_cu_block_cnts,
                                                                      seq_lens,
                                                                      src_block_len,
                                                                      dst_block_len,
                                                                      head_num,
                                                                      head_dim,
                                                                      batch_size);
    };

    switch (head_dim) {
        case 128:
            fn(std::integral_constant<int, 128>{});
            break;
        default:
            fn(head_dim);
    }
}

template void ConvertKvCacheBlocksToLinear(const half** src_k_block_ptrs,
                                           const half** src_v_block_ptrs,
                                           half**       dst_k_ptrs,
                                           half**       dst_v_ptrs,
                                           const int*   src_cu_block_cnts,
                                           const int*   seq_lens,
                                           int          src_block_len,
                                           int          dst_block_len,
                                           int          head_num,
                                           int          head_dim,
                                           int          batch_size,
                                           cudaStream_t st);

template void ConvertKvCacheBlocksToLinear(const float** src_k_block_ptrs,
                                           const float** src_v_block_ptrs,
                                           float**       dst_k_ptrs,
                                           float**       dst_v_ptrs,
                                           const int*    src_cu_block_cnts,
                                           const int*    seq_lens,
                                           int           src_block_len,
                                           int           dst_block_len,
                                           int           head_num,
                                           int           head_dim,
                                           int           batch_size,
                                           cudaStream_t  st);

}  // namespace turbomind
