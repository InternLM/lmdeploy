#include "../gemm_s_f16/common.h"
// #include "cute/tensor.hpp"
#include <cuda_fp16.h>

namespace turbomind {

// [S/x, H, x, D] <-> [S/y, H, y, D]

template<typename T>
__device__ void ConvertBlockSize(const T** src_block_ptrs,
                                 T**       dst_block_ptrs,
                                 int       src_block_size,
                                 int       dst_block_size,
                                 int       heads,
                                 int       dims,
                                 int       seq_len)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    size_t count = (size_t)heads * seq_len * dims;

    for (size_t i = (threadIdx.x + blockIdx.x * blockDim.x) * kVecSize; i < count;
         i += blockDim.x * gridDim.x * kVecSize) {
        // get coords from [H, S, D]
        int di = i % dims;
        int ii = i / dims;

        int si = ii % seq_len;
        int hi = ii / seq_len;

        // compute indices into src
        int src_block_index  = si / src_block_size;
        int src_block_offset = hi * src_block_size * dims + si % src_block_size * dims + di;

        // compute indices into dst
        int dst_block_index  = si / dst_block_size;
        int dst_block_offset = hi * dst_block_size * dims + si % dst_block_size * dims + di;

        const T* src_block = src_block_ptrs[src_block_index];
        T*       dst_block = dst_block_ptrs[dst_block_index];

        uint4 data = __ldg(reinterpret_cast<const uint4*>(src_block + src_block_offset));

        *reinterpret_cast<uint4*>(dst_block + dst_block_offset) = data;
    }
}

template<typename T>
__global__ void
LinearToBlocksKernel(const T* src, T** dst_block_ptrs, int dst_block_size, int heads, int dims, int seq_len)
{
    __shared__ const T* src_block_ptr[1];

    if (threadIdx.x == 0) {
        src_block_ptr[0] = src;
    }

    __syncthreads();

    ConvertBlockSize(src_block_ptr, dst_block_ptrs, seq_len, dst_block_size, heads, dims, seq_len);
}

template<typename T>
__global__ void
BlocksToLinearKernel(const T** src_block_ptrs, T* dst, int src_block_size, int heads, int dims, int seq_len)
{
    __shared__ T* dst_block_ptr[1];

    if (threadIdx.x == 0) {
        dst_block_ptr[0] = dst;
    }

    __syncthreads();

    ConvertBlockSize(src_block_ptrs, dst_block_ptr, src_block_size, seq_len, heads, dims, seq_len);
}

template<typename T>
__global__ void BlocksToBlocksKernel(const T** src_block_ptrs,
                                     T**       dst_block_ptrs,
                                     int       src_block_size,
                                     int       dst_block_size,
                                     int       heads,
                                     int       dims,
                                     int       seq_len)
{
    ConvertBlockSize(src_block_ptrs, dst_block_ptrs, src_block_size, dst_block_size, heads, dims, seq_len);
}

template<typename T>
void ConvertLinearToBlocks(
    const T* src, T** dst_block_ptrs, int dst_block_size, int heads, int dims, int seq_len, cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    int threads = 512;
    int blocks  = std::min(512, (heads * seq_len * dims / kVecSize + threads - 1) / threads);

    LinearToBlocksKernel<<<blocks, threads, 0, st>>>(src, dst_block_ptrs, dst_block_size, heads, dims, seq_len);
}

template<typename T>
void ConvertBlocksToLinear(
    const T** src_block_ptrs, T* dst, int src_block_size, int heads, int dims, int seq_len, cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    int threads = 256;
    int blocks  = (heads * seq_len * dims / kVecSize + threads - 1) / threads;

    BlocksToLinearKernel<<<blocks, threads, 0, st>>>(src_block_ptrs, dst, src_block_size, heads, dims, seq_len);
}

template<typename T>
void ConvertBlocksToBlocks(const T**    src_block_ptrs,
                           T**          dst_block_ptrs,
                           int          src_block_size,
                           int          dst_block_size,
                           int          heads,
                           int          dims,
                           int          seq_len,
                           cudaStream_t st)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    int threads = 512;
    int blocks  = std::min(512, (heads * seq_len * dims / kVecSize + threads - 1) / threads);

    BlocksToBlocksKernel<<<blocks, threads, 0, st>>>(
        src_block_ptrs, dst_block_ptrs, src_block_size, dst_block_size, heads, dims, seq_len);
}

template void ConvertLinearToBlocks(
    const half* src, half** dst_block_ptrs, int dst_block_size, int heads, int dims, int seq_len, cudaStream_t st);

template void ConvertBlocksToLinear(
    const half** src_block_ptrs, half* dst, int src_block_size, int heads, int dims, int seq_len, cudaStream_t st);

template void ConvertBlocksToBlocks(const half** src_block_ptrs,
                                    half**       dst_block_ptrs,
                                    int          src_block_size,
                                    int          dst_block_size,
                                    int          heads,
                                    int          dims,
                                    int          seq_len,
                                    cudaStream_t st);

}  // namespace turbomind
