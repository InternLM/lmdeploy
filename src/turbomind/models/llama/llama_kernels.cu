// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/dispatch.h"

namespace turbomind {

__global__ void gatherOutput(int*       output_ids,
                             const int* ids,
                             const int* context_length,
                             int        max_context_len,
                             int        max_gen_step,
                             int        max_output_len,
                             int        batch_size)
{
    const int batch_id    = blockIdx.x;
    const int context_len = context_length[batch_id];
    output_ids += batch_id * max_output_len;
    for (int src_idx = threadIdx.x; src_idx < max_gen_step; src_idx += blockDim.x) {
        // skip padding for src
        if (context_len <= src_idx && src_idx < max_context_len) {
            continue;
        }
        // skip padding for dst
        const int dst_idx = src_idx < context_len ? src_idx : src_idx - (max_context_len - context_len);
        if (dst_idx < max_output_len) {
            output_ids[dst_idx] = ids[src_idx * batch_size + batch_id];
        }
    }
}

void invokeGatherOutput(int*         output_ids,
                        const int*   ids,
                        const int*   context_length,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        cudaStream_t stream)
{
    int block_size = 128;
    int grid_size  = batch_size;
    gatherOutput<<<grid_size, block_size, 0, stream>>>(
        output_ids, ids, context_length, max_context_len, max_gen_step, max_output_len, batch_size);
}

__global__ void updateOutput(int**      request_output_ids_ptrs,
                             int**      request_seqlen_ptrs,
                             const int* output_ids,
                             const int* sequence_lengths,
                             const int* request_output_ids_lens,
                             int        max_session_len,
                             bool       token_generated)
{
    const int batch_id = blockIdx.x;

    auto request_output_ids = request_output_ids_ptrs[batch_id];
    auto request_seqlen     = request_seqlen_ptrs[batch_id];

    output_ids += max_session_len * batch_id;

    const int seqlen     = sequence_lengths[batch_id] + (int)token_generated;
    const int output_len = min(seqlen, request_output_ids_lens[batch_id]);

    for (int i = threadIdx.x; i < output_len; i += blockDim.x) {
        request_output_ids[i] = output_ids[i];
    }

    *request_seqlen = seqlen;
}

void invokeUpdateOutput(int**        request_output_ids_ptrs,
                        int**        request_seqlen_ptrs,
                        const int*   output_ids,
                        const int*   sequence_lengths,
                        const int*   request_output_ids_lens,
                        int          max_session_len,
                        bool         token_generated,
                        int          batch_size,
                        cudaStream_t stream)
{
    constexpr int block_size = 128;
    const int     grid_size  = batch_size;

    updateOutput<<<grid_size, block_size, 0, stream>>>(request_output_ids_ptrs,
                                                       request_seqlen_ptrs,
                                                       output_ids,
                                                       sequence_lengths,
                                                       request_output_ids_lens,
                                                       max_session_len,
                                                       token_generated);
}

template<int BLOCK_DIM>
__global__ void compactOutputIds(
    int* cu_output_ids, const int* output_ids, const int* sequence_lengths, int session_len, bool token_generated)
{
    typedef cub::BlockReduce<int, BLOCK_DIM>     BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int batch_idx = blockIdx.x;

    int end   = (batch_idx + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;  // align to BLOCK_DIM boundary
    int count = 0;
    for (int i = threadIdx.x; i < end; i += blockDim.x) {
        int x = threadIdx.x < batch_idx ? sequence_lengths[threadIdx.x] : 0;
        count += BlockReduce(temp_storage).Sum(x);
        // https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
        __syncthreads();
    }

    __shared__ int offset;

    if (threadIdx.x == 0) {
        offset = count;
    }

    __syncthreads();

    auto dst = cu_output_ids + offset;

    const int seq_len = sequence_lengths[batch_idx];

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        dst[i] = output_ids[batch_idx * session_len + i];
    }
}

void invokeCompactOutputIds(int*         cu_output_ids,
                            const int*   output_ids,
                            const int*   sequence_lengths,
                            int          max_session_len,
                            bool         token_generated,
                            int          batch_size,
                            cudaStream_t stream)
{
    constexpr int BLOCK_DIM = 128;
    compactOutputIds<BLOCK_DIM><<<batch_size, BLOCK_DIM, 0, stream>>>(
        cu_output_ids, output_ids, sequence_lengths, max_session_len, token_generated);
}

template<int N, int C>
struct IndexedCopyParam {
    Array<void*, N> src_ptr;
    Array<void*, N> dst_ptr;
    Array<int, N>   stride;
    Array<int, C>   src_idx;
    Array<int, C>   dst_idx;
    int             max_stride;
};

template<class T, int N, int C>
__global__ void indexedCopy(IndexedCopyParam<N, C> param)
{
    const int bi = blockIdx.x;
    const int si = param.src_idx[bi];
    const int di = param.dst_idx[bi];
    for (int i = threadIdx.x; i < param.max_stride; i += blockDim.x) {
        PRAGMA_UNROLL
        for (int k = 0; k < N; ++k) {
            if (i < param.stride[k]) {
                *((T*)param.dst_ptr[k] + param.stride[k] * di + i) =
                    *((const T*)param.src_ptr[k] + param.stride[k] * si + i);
            }
        }
    }
}

template<class T, int N>
void invokeIndexedCopyImpl(void**       h_src_ptr,
                           void**       h_dst_ptr,
                           const int*   h_elem_sz,
                           const int*   h_src_idx,
                           const int*   h_dst_idx,
                           int          count,
                           cudaStream_t st)
{
    dispatch(  // dispatch for num of copy operations
        std::integer_sequence<int, 4, 8, 16, 32, 64, 128, 256>{},
        [&](auto C) { return count <= C; },
        [&](auto C) {
            // maximum parameter size: sm<70: 4kB, sm>=70: 32kB
            static_assert(sizeof(IndexedCopyParam<N, C>) <= 4096);
            IndexedCopyParam<N, C> param{};
            std::copy_n(h_src_ptr, N, param.src_ptr.data());
            std::copy_n(h_dst_ptr, N, param.dst_ptr.data());
            std::transform(h_elem_sz, h_elem_sz + N, param.stride.data(), [](int size) {
                // Basic alignment check
                FT_CHECK_WITH_INFO(size % sizeof(T) == 0, fmtstr("misalignment: %d %% %d", size, (int)sizeof(T)));
                return size / sizeof(T);
            });
            param.max_stride = *std::max_element(param.stride.begin(), param.stride.end());
            auto copy_idx    = [](const int* src, int offset, int n, auto dst) {
                return src ? (void)std::copy_n(src + offset, n, dst) : std::iota(dst, dst + n, offset);
            };
            for (int c = 0; c < count; c += C) {
                int batch_size = std::min(count - c, (int)C);
                copy_idx(h_src_idx, c, batch_size, param.src_idx.data());
                copy_idx(h_dst_idx, c, batch_size, param.dst_idx.data());
                indexedCopy<T><<<batch_size, 128, 0, st>>>(param);
            }
        });
}

void invokeIndexedCopy(void**       h_src_ptr,
                       void**       h_dst_ptr,
                       const int*   h_elem_sz,
                       const int*   h_src_idx,
                       const int*   h_dst_idx,
                       int          count,
                       int          n_copys,
                       cudaStream_t st)
{
    auto success = dispatch(std::integer_sequence<int, 1, 2, 3, 4>{}, [&](auto N) {
        if (N == n_copys) {
            invokeIndexedCopyImpl<uint32_t, N>(h_src_ptr, h_dst_ptr, h_elem_sz, h_src_idx, h_dst_idx, count, st);
            return true;
        }
        return false;
    });
    FT_CHECK(success);
}

__global__ void padLastTokenIds(int* token_ids, const int* context_length, int max_context_len, int batch_size)
{
    for (int bi = threadIdx.x; bi < batch_size; bi += blockDim.x) {
        token_ids[(max_context_len - 1) * batch_size + bi] = token_ids[(context_length[bi] - 1) * batch_size + bi];
    }
}

void invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream)
{
    padLastTokenIds<<<1, 512, 0, stream>>>(token_ids, context_length, max_context_len, batch_size);
}

template<typename T>
__global__ void getFeatureOfLastToken(T* output, const T* input, const int* cu_seqlens, int dims)
{
    int bi = blockIdx.x;
    int ti = cu_seqlens[bi + 1] - 1;
    for (int i = threadIdx.x; i < dims; i += blockDim.x) {
        output[dims * bi + i] = input[dims * ti + i];
    }
}

void invokeGetFeatureOfLastToken(
    uint16_t* output, const uint16_t* input, const int* cu_seqlens, int dims, int batch_size, cudaStream_t stream)
{
    getFeatureOfLastToken<<<batch_size, 256, 0, stream>>>(output, input, cu_seqlens, dims);
}

template<class T, int C>
struct BatchedCopyParam {
    Array<T*, C>  src_ptr;
    Array<T*, C>  dst_ptr;
    Array<int, C> size;
    int           count;
};

template<int kThrPerCpy, class T, int C>
__global__ void batchedCopy(BatchedCopyParam<T, C> param)
{
    const int ti = threadIdx.x + blockIdx.x * blockDim.x;
    const int bi = ti / kThrPerCpy;
    if (bi >= param.count) {
        return;
    }
    const T* __restrict__ src = param.src_ptr[bi];
    T* __restrict__ dst       = param.dst_ptr[bi];
    int size                  = param.size[bi];
    for (int i = ti % kThrPerCpy; i < size; i += kThrPerCpy) {
        dst[i] = src[i];
    }
}

// MSVC does not like CUDA kernel launch inside nested lambdas
template<class P>
struct BatchedCopyLauncher {
    int          max_size;
    int          count;
    const P*     params;
    cudaStream_t st;

    template<int S>
    void operator()(std::integral_constant<int, S>) const
    {
        constexpr int threads         = 128;
        constexpr int items_per_block = threads / S;
        const int     blocks          = (count + items_per_block - 1) / items_per_block;
        batchedCopy<S><<<blocks, threads, 0, st>>>(*params);
    }
};

void invokeBatchedCopy(void** src_ptr, void** dst_ptr, int* size, int count, cudaStream_t st)
{
    dispatch(
        std::integer_sequence<int, 1, 8, 32, 128>{},
        [&](auto C) { return count <= C; },
        [&](auto C) {
            using T = uint32_t;
            BatchedCopyParam<T, C> params{};
            // TODO: on CUDA 12.1 and sm_70+ this can be 32K
            static_assert(sizeof(params) <= 4096);
            for (int c = 0; c < count; c += C) {
                const int bsz = std::min<int>(count - c, C);
                params.count  = bsz;
                for (int i = 0; i < bsz; ++i) {
                    params.src_ptr[i] = (T*)src_ptr[c + i];
                    params.dst_ptr[i] = (T*)dst_ptr[c + i];
                    FT_CHECK(size[c + i] % sizeof(T) == 0);
                    params.size[i] = size[c + i] / sizeof(T);
                }
                const int max_size = *std::max_element(params.size.begin(), params.size.end());
                dispatch(
                    std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64, 128>{},
                    [&](auto S) { return max_size <= S; },
                    BatchedCopyLauncher<BatchedCopyParam<T, C>>{max_size, count, &params, st});
            }
        });
}

template<typename T>
__global__ void maskOutput(T* output, const int* mask, int dim)
{
    int batch_idx = blockIdx.x;
    output += dim * batch_idx;
    int masked = mask[batch_idx];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = (masked) ? output[i] : T();
    }
}

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream)
{
    maskOutput<<<batch_size, 1024, 0, stream>>>(output, mask, dim);
}

#ifdef ENABLE_FP32
template void invokeMask(float* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#endif
template void invokeMask(half* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMask(__nv_bfloat16* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#endif

template<typename T, int vec_size>
__global__ void castFloat2D(const T* input, float* output, int channels)
{
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    const int bi = blockIdx.y;
    input += (size_t)bi * channels;
    output += (size_t)bi * channels;

    const int step = gridDim.x * blockDim.x * vec_size;

    for (int i = vi * vec_size; i < channels; i += step) {
        Array<T, vec_size> src;

        if constexpr (sizeof(src) >= sizeof(uint)) {
            Load(src, input + i);
        }
        else {
            PRAGMA_UNROLL
            for (int j = 0; j < vec_size; ++j) {
                src[j] = input[i + j];
            }
        }

        auto dst = cast<float>(src);

        // store
        Store(output + i, dst);
    }
}

void invokeCastFloat2D(const core::Tensor& src, core::Tensor& dst, cudaStream_t stream)
{
    TM_CHECK(src.is_contiguous());
    TM_CHECK(dst.is_contiguous());
    TM_CHECK(src.shape() == dst.shape());

    auto batch_size = src.shape(0);
    auto channels   = src.shape(1);

    auto invoke = [&](auto t, auto vec_size) {
        using T                      = decltype(t);
        constexpr int threads        = 256;
        const int     blocks_per_tok = (channels + threads * vec_size - 1) / (threads * vec_size);
        const dim3    blocks(blocks_per_tok, batch_size);
        castFloat2D<T, vec_size.value><<<blocks, threads, 0, stream>>>(  //
            src.data<T>(),
            dst.data<float>(),
            channels);
    };

    auto dispatch_t = [&](auto vec_size) {
        switch (src.dtype()) {
            case kFloat32:
                return invoke(float{}, vec_size);
                break;
            case kFloat16:
                return invoke(half{}, vec_size);
                break;
#ifdef ENABLE_BF16
            case kBfloat16:
                return invoke(__nv_bfloat16{}, vec_size);
                break;
#endif
            default:
                TM_UNREACHABLE;
        }
    };

    if (channels % 4 == 0) {
        return dispatch_t(std::integral_constant<int, 4>{});
    }
    else if (channels % 2 == 0) {
        return dispatch_t(std::integral_constant<int, 2>{});
    }
    else {
        return dispatch_t(std::integral_constant<int, 1>{});
    }
}

template<class T>
__global__ void CollectHiddenStates_Kernel(const T* src, const int* idxs, T* dst, int dim)
{
    const int bi = blockIdx.x;
    const int ti = idxs[bi];

    if (ti < 0) {
        return;
    }

    src += ti * dim;
    dst += bi * dim;

    for (int di = threadIdx.x; di < dim; di += blockDim.x) {
        dst[di] = src[di];
    }
}

void CollectHiddenStates(const Tensor& src, const Buffer_<int>& idxs, Ref<Tensor> dst, cudaStream_t st)
{
    const auto stride = byte_size(src.dtype(), src.stride(0));

    auto invoke = [&](auto t) {
        using T           = decltype(t);
        const int dim     = stride / sizeof(T);
        const int threads = round_up(min(dim, 1024), WARP_SIZE);
        const int blocks  = idxs.size();
        CollectHiddenStates_Kernel<<<blocks, threads, 0, st>>>(
            (const T*)src.raw_data(), idxs.data(), (T*)dst.get().raw_data(), dim);
    };

    if (stride % sizeof(uint4) == 0) {
        invoke(uint4{});
    }
    else if (stride % sizeof(uint2) == 0) {
        invoke(uint2{});
    }
    else if (stride % sizeof(uint1) == 0) {
        invoke(uint1{});
    }
    else if (stride % sizeof(ushort) == 0) {
        invoke(ushort{});
    }
    else {
        TM_CHECK(0) << "unsupported byte stride: " << stride;
    }
}

template<int BLOCK_DIM, int MAX_COUNT>
__global__ void
BatchPrefixSumKernel(Array<const int*, MAX_COUNT> srcs, Array<int, MAX_COUNT> ns, Array<int*, MAX_COUNT> dsts)
{
    const int  bi  = blockIdx.x;
    const int* src = srcs[bi];
    int*       dst = dsts[bi];
    const int  n   = ns[bi];

    using BlockScan = cub::BlockScan<int, BLOCK_DIM>;

    __shared__ typename BlockScan::TempStorage temp_storage;

    int prefix{};
    for (int i = threadIdx.x; i < round_up(n, BLOCK_DIM); i += BLOCK_DIM) {
        if (i >= BLOCK_DIM) {
            __syncthreads();
        }
        int data = i < n ? src[i] : 0;
        int sum{};
        BlockScan{temp_storage}.ExclusiveSum(data, data, sum);
        if (i < n) {
            dst[i] = prefix + data;
        }
        prefix += sum;
    }

    if (threadIdx.x == 0) {
        dst[n] = prefix;
    }
}

void BatchPrefixSum(const int** srcs, const int* ns, int** dsts, int count, cudaStream_t st)
{
    constexpr int max_count = 1;

    Array<const int*, max_count> p_srcs{};
    Array<int*, max_count>       p_dsts{};
    Array<int, max_count>        p_ns{};

    for (int i = 0; i < count; ++i) {
        p_srcs[i] = srcs[i];
        p_dsts[i] = dsts[i];
        p_ns[i]   = ns[i];
    }

    TM_CHECK_LE(count, max_count);

    constexpr int block = 256;
    const int     grid  = count;

    BatchPrefixSumKernel<block><<<grid, block, 0, st>>>(p_srcs, p_ns, p_dsts);
}

__global__ void AppendTokenIdsKernel(int** token_ids_ptrs, const int* output_ids, const int* positions, int batch_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < batch_size) {
        int* token_ids = token_ids_ptrs[i];
        int  pos       = positions[i];
        token_ids[pos] = output_ids[i];
    }
}

void AppendTokenIds(
    int** token_ids_ptrs, const int* output_ids, const int* positions, int batch_size, cudaStream_t stream)
{
    constexpr int block = 128;
    const int     grid  = cdiv(batch_size, block);
    AppendTokenIdsKernel<<<grid, block, 0, stream>>>(token_ids_ptrs, output_ids, positions, batch_size);
}

}  // namespace turbomind
