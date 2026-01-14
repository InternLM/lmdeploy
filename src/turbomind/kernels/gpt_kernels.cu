/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<class T, int vec_size>
__global__ void
embeddingLookupKernel(T* dst, int dst_stride, const T* src, int src_stride, const int* ids, int num, int dim)
{
    const int ti = blockIdx.x;

    const int64_t idx = ids[ti];

    src += idx * src_stride;
    dst += ti * dst_stride;

    for (int di = threadIdx.x * vec_size; di < dim; di += blockDim.x * vec_size) {
        Array<T, vec_size> vec;
        Ldg(vec, &src[di]);
        Store(&dst[di], vec);
    }
}

void invokeEmbeddingLookup(Ref<Tensor>         out_,
                           const Buffer_<int>& token_ids,
                           const Tensor&       embedding_table,
                           cudaStream_t        st)
{
    auto& out = out_.get();

    TM_CHECK_EQ(out.shape(0), token_ids.size());
    TM_CHECK_EQ(out.shape(1), embedding_table.shape(1));

    int num, dim;
    std::tie(num, dim) = out.shapes(0, 1);

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        TM_CHECK(dim % vec_size == 0) << dim << " " << vec_size;
        const int threads = std::min(dim / vec_size, 1024);
        const int blocks  = num;
        TM_CHECK(out_.get());
        TM_CHECK(token_ids);
        TM_CHECK(embedding_table);
        embeddingLookupKernel<T, vec_size><<<blocks, threads, 0, st>>>((T*)out.raw_data(),
                                                                       out.stride(0),
                                                                       (const T*)embedding_table.raw_data(),
                                                                       embedding_table.stride(0),
                                                                       token_ids.data(),
                                                                       num,
                                                                       dim);
    };

    if (byte_size(out.dtype()) == byte_size<uint16_t>()) {
        return invoke(uint16_t{});
    }
    TM_CHECK(0) << "not implemented";
}

// TODO Add half2 implementation
template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2) {
        const int input_dim2_index = index % dim2;
        index                      = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;

        out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index] =
            in[input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index];
    }
}

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
}

template void
invokeTransposeAxis01(float* out, float* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(half* out, half* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(int* out, int* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(uint16_t* out, uint16_t* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(uint8_t* out, uint8_t* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeTransposeAxis01(
    __nv_bfloat16* out, __nv_bfloat16* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);
#endif

template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1)
{
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1) {
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        const int in_offset        = in_skipping_dim1 == nullptr ? 0 : in_skipping_dim1[input_dim1_index] * dim1;

        out[input_dim1_index * dim0 + input_dim0_index] = in[in_offset + input_dim0_index * dim1 + input_dim1_index];
    }
}

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, in_skipping_dim1, dim0, dim1);
}

template void invokeTransposeAxis01(
    int* out, int* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

template<int TILE_DIM, int BLOCK_ROWS, class T>
__global__ void transpose_2d_kernel(T* __restrict__ dst, const T* __restrict__ src, int rows, int cols, bool swap_xy)
{
    __shared__ T smem[TILE_DIM][TILE_DIM + 1];

    const int block_idx_x = swap_xy ? blockIdx.y : blockIdx.x;
    const int block_idx_y = swap_xy ? blockIdx.x : blockIdx.y;

    {
        const int j = block_idx_x * TILE_DIM + threadIdx.x;
        const int i = block_idx_y * TILE_DIM + threadIdx.y;

#pragma unroll
        for (int y = 0; y < TILE_DIM; y += BLOCK_ROWS) {
            if (i + y < rows && j < cols) {
                smem[threadIdx.y + y][threadIdx.x] = src[(i + y) * cols + j];
            }
        }
    }

    __syncthreads();

    {
        const int j = block_idx_y * TILE_DIM + threadIdx.x;
        const int i = block_idx_x * TILE_DIM + threadIdx.y;

#pragma unroll
        for (int y = 0; y < TILE_DIM; y += BLOCK_ROWS) {
            if (i + y < cols && j < rows) {
                dst[(i + y) * rows + j] = smem[threadIdx.x][threadIdx.y + y];
            }
        }
    }
}

template<class T>
void invokeTranspose2D_(T* dst, const T* src, int rows, int cols, cudaStream_t st)
{
    constexpr int TILE_DIM   = 32;  // warp size
    constexpr int BLOCK_ROWS = 8;

    const dim3 block(TILE_DIM, BLOCK_ROWS);

    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,  //
              (rows + TILE_DIM - 1) / TILE_DIM);
    bool swap_xy = false;

    if (grid.y > 65535) {  // max dim for grid.y
        std::swap(grid.x, grid.y);
        swap_xy = true;
    }

    transpose_2d_kernel<TILE_DIM, BLOCK_ROWS><<<grid, block, 0, st>>>(dst, src, rows, cols, swap_xy);
}

template void invokeTranspose2D_(uint32_t*, const uint32_t*, int, int, cudaStream_t);

}  // namespace turbomind
