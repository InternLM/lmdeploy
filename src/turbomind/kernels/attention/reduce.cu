// Copyright (c) OpenMMLab. All rights reserved.

#include "cutlass/fast_math.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <type_traits>

namespace turbomind::attention {

int next_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

template<int CTA_K, int HeadDim, int WarpCnt, bool First, class T>
__global__ void reduce_output(T*           out,
                              const float* partial_ML,
                              float*       partial_O,
                              const int*   split_cnt_,
                              int          max_split_cnt,
                              int          query_num,
                              int          head_num,
                              float        exp_scale,
                              int          stride_k,
                              int          offset_k)
{
    __shared__ float s_out[WarpCnt][HeadDim];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int head_idx  = ReduceCtaMap::head_idx();
    const int query_idx = ReduceCtaMap::query_idx();
    const int chunk_idx = ReduceCtaMap::split_idx();

    offset_k *= chunk_idx;
    const int split_cnt = (split_cnt_ != nullptr) ? split_cnt_[query_idx] : 1;
    if (offset_k >= split_cnt) {  // out of bound
        return;
    }

    // HeadDim / WARP_SIZE
    // 128     -> 4
    // 64, 192 -> 2
    constexpr int kVecSize = HeadDim % 128 == 0 ? 4 : 2;

    using Map = RakedThreadMap<HeadDim, WarpCnt, kVecSize, WarpCnt, WARP_SIZE>;
    static_assert(Map::kIterS == 1);

    constexpr int C = Map::kIterC;

    using Vec = Array<float, kVecSize>;

    Vec accu_O[C]{};
    Vec frag_O[C];

    const int2 d = Map::get_offset(warp_id, lane_id);

    auto for_each = [&](auto fn) {
        const int ki = d.y;
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            const int di = d.x + c * Map::kDeltaC;
            fn(c, ki, di);
        }
    };

    PRAGMA_UNROLL
    for (int k = 0; k < CTA_K; k += WarpCnt) {
        for_each([&](int c, int ki, int di) {
            using namespace ops;
            ki += k;
            const int  split_idx = offset_k + stride_k * ki;
            const bool mask      = split_idx < split_cnt;
            const int  index     = (query_idx * head_num + head_idx) * max_split_cnt + split_idx;
            const int  offset    = index * HeadDim + di;
            if (mask) {
                Load(frag_O[c], &partial_O[offset]);
                accu_O[c] = accu_O[c] + frag_O[c] * (First ? partial_ML[index * 2] : 1.0f);
            }
        });
    }

    for_each([&](int c, int ki, int di) {
        Store(&s_out[ki][di], accu_O[c]);  //
    });

    PRAGMA_UNROLL
    for (int w = WarpCnt / 2; w > 0; w /= 2) {
        __syncthreads();
        for_each([&](int c, int ki, int di) {
            using namespace ops;
            if (ki < w) {
                (Vec&)s_out[ki][di] = (Vec&)s_out[ki][di] + (Vec&)s_out[w + ki][di];
            }
        });
    }

    for_each([&](int c, int ki, int di) {
        if (ki == 0) {
            if (gridDim.z == 1) {
                const int offset = (query_idx * head_num + head_idx) * HeadDim + di;
                Store(&out[offset], cast<T>((Vec&)s_out[ki][di]));
            }
            else {
                const int offset = ((query_idx * head_num + head_idx) * max_split_cnt + offset_k) * HeadDim + di;
                Store(&partial_O[offset], (Vec&)s_out[ki][di]);
            }
        }
    });
}

template<int HeadDim, class T>
void invokeReduceOutput(T*           out,
                        const float* partial_ML,  // scale
                        float*       partial_O,
                        const int*   split_cnt,
                        int          partial_len,
                        int          max_split_cnt,
                        int          query_num,
                        int          head_num,
                        float        exp_scale,
                        cudaStream_t stream)
{
    constexpr int CTA_K = 32;  // warp size

    auto invoke = [&](auto is_first, int stride_k) {
        constexpr int kWarpCnt = 4;
        const dim3    block    = kWarpCnt * WARP_SIZE;
        const dim3    grid     = ReduceCtaMap::get_grid_shape(query_num, head_num, max_split_cnt, CTA_K);

        static constexpr size_t kSmemSize = sizeof(float) * kWarpCnt * HeadDim;
        static_assert(kSmemSize < (48 << 10));

        reduce_output<CTA_K, HeadDim, kWarpCnt, is_first><<<grid, block, kSmemSize, stream>>>(  //
            out,
            partial_ML,
            partial_O,
            split_cnt,
            partial_len,
            query_num,
            head_num,
            exp_scale,
            stride_k,
            stride_k * CTA_K);

        sync_check_cuda_error();
    };

    int stride_k = 1;

    invoke(std::true_type{}, stride_k);
    while (max_split_cnt > CTA_K) {
        max_split_cnt = (max_split_cnt + CTA_K - 1) / CTA_K;
        stride_k *= CTA_K;
        invoke(std::false_type{}, stride_k);
    }
}

template<int N>
__global__ void reduce_ML(float*              partial_ML,  // cp, q, h, k, 2
                          const int*          split_cnt_,
                          int                 max_split_cnt,
                          int                 query_num,
                          cutlass::FastDivmod head_num,
                          float               exp_scale,
                          int                 cp_size,
                          int                 dim0)
{
    constexpr int kIterWarp = N / WARP_SIZE;

    float frag_M[kIterWarp];
    float frag_L[kIterWarp];

    int qh = blockIdx.x * blockDim.y + threadIdx.y;
    if (qh >= query_num * head_num) {
        return;
    }

    const int split_k   = split_cnt_ != nullptr ? split_cnt_[head_num.div(qh)] : 1;
    const int split_cnt = cp_size * split_k;

    float block_M = -std::numeric_limits<float>::infinity();
    float block_L = 0.f;

    PRAGMA_UNROLL
    for (int i = 0; i < kIterWarp; ++i) {
        int  ki    = threadIdx.x + i * WARP_SIZE;
        int  index = (qh * max_split_cnt + ki) * 2;
        bool mask  = ki < split_cnt;

        if (mask && dim0 > 0) {  // handle cp case
            int cp_i = ki / split_k;
            ki       = ki % split_k;
            index    = cp_i * dim0 + (qh * max_split_cnt + ki) * 2;
        }

        frag_M[i] = mask ? partial_ML[index] : -std::numeric_limits<float>::infinity();
        frag_L[i] = mask ? partial_ML[index + 1] : 0.f;
        block_M   = max(block_M, frag_M[i]);
    }

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    PRAGMA_UNROLL
    for (int i = 0; i < kIterWarp; ++i) {
        block_L += (frag_M[i] == -std::numeric_limits<float>::infinity()) ?
                       0.0f :
                       exp2f((frag_M[i] - block_M) * exp_scale) * frag_L[i];
    }

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    PRAGMA_UNROLL
    for (int i = 0; i < kIterWarp; ++i) {
        int  ki    = threadIdx.x + i * WARP_SIZE;
        int  index = (qh * max_split_cnt + ki) * 2;
        bool mask  = ki < split_cnt;

        if (dim0 > 0) {  // handle cp case
            int cp_i = ki / split_k;
            ki       = ki % split_k;
            index    = cp_i * dim0 + (qh * max_split_cnt + ki) * 2;
        }

        float scale = (frag_M[i] == -std::numeric_limits<float>::infinity()) ?
                          0.0f :
                          exp2f((frag_M[i] - block_M) * exp_scale) / block_L;
        if (mask) {
            partial_ML[index] = scale;  // save scale to M
        }
    }
}

void invokeReduceML(float*       partial_ML,
                    const int*   split_cnt,
                    int          partial_len,
                    int          max_split_cnt,
                    int          cp_size,
                    int          cp_rank,
                    int          query_num,
                    int          head_num,
                    float        exp_scale,
                    cudaStream_t stream)
{
    max_split_cnt *= cp_size;
    TM_CHECK(max_split_cnt > 1);

    const int  warp_cnt = 4;
    const dim3 block(WARP_SIZE, warp_cnt);
    const dim3 grid((query_num * head_num + warp_cnt - 1) / warp_cnt);

    const int dim0 = cp_size > 1 ? query_num * head_num * partial_len * 2 : 0;
    partial_ML -= cp_rank * dim0;  // begin address of cp_rank0

    int n = max(next_power_of_two(max_split_cnt), WARP_SIZE);
    switch (n) {
#define LAUNCH_REDUCE_ML(n)                                                                                            \
    case n:                                                                                                            \
        reduce_ML<n><<<grid, block, 0, stream>>>(                                                                      \
            partial_ML, split_cnt, partial_len, query_num, cutlass::FastDivmod(head_num), exp_scale, cp_size, dim0);   \
        break;

        LAUNCH_REDUCE_ML(32);
        LAUNCH_REDUCE_ML(64);
        LAUNCH_REDUCE_ML(128);
        LAUNCH_REDUCE_ML(256);
        LAUNCH_REDUCE_ML(512);
        LAUNCH_REDUCE_ML(1024);
        default:
            TM_CHECK(false) << "reduce_ML does not support max_split_cnt = " << max_split_cnt;
#undef LAUNCH_REDUCE_ML
    }

    sync_check_cuda_error();
}

template<int HeadDim, class T>
void invokeReduceV2(T*           out,
                    float*       partial_ML,
                    float*       partial_O,
                    const int*   split_cnt,
                    int          partial_len,
                    int          max_split_cnt,
                    int          cp_size,
                    int          cp_rank,
                    int          query_num,
                    int          head_num,
                    float        exp_scale,
                    cudaStream_t stream)
{
    invokeReduceML(partial_ML,  //
                   split_cnt,
                   partial_len,
                   max_split_cnt,
                   cp_size,
                   cp_rank,
                   query_num,
                   head_num,
                   exp_scale,
                   stream);

    invokeReduceOutput<HeadDim>(out,  //
                                partial_ML,
                                partial_O,
                                split_cnt,
                                partial_len,
                                max_split_cnt,
                                query_num,
                                head_num,
                                exp_scale,
                                stream);
}

#define INSTANTIATE_invokeReduceV2(dim, type)                                                                          \
    template void invokeReduceV2<dim>(type * out,                                                                      \
                                      float*       partial_ML,                                                         \
                                      float*       partial_O,                                                          \
                                      const int*   split_cnt,                                                          \
                                      int          partial_len,                                                        \
                                      int          max_split_cnt,                                                      \
                                      int          cp_size,                                                            \
                                      int          cp_rank,                                                            \
                                      int          query_num,                                                          \
                                      int          head_num,                                                           \
                                      float        exp_scale,                                                          \
                                      cudaStream_t stream);

INSTANTIATE_invokeReduceV2(64, half);
INSTANTIATE_invokeReduceV2(128, half);
INSTANTIATE_invokeReduceV2(192, half);

#if ENABLE_BF16
INSTANTIATE_invokeReduceV2(64, nv_bfloat16);
INSTANTIATE_invokeReduceV2(128, nv_bfloat16);
INSTANTIATE_invokeReduceV2(192, nv_bfloat16);
#endif

}  // namespace turbomind::attention
