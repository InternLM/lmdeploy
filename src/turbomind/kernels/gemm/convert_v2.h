// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<int M_, int K_, int Pack_M, class Operand, class Td, class Converter>
struct ConvertOperand {
    using Ts         = typename Operand::Dtype;
    using SmemLayout = typename Operand::SmemLayout;
    using SmemCopy   = typename Operand::SmemCopy;
    using GmemIter   = typename Operand::GmemIter;

    static constexpr int M = M_;
    static constexpr int K = K_;

    using Accessor = SmemAccessor<Ts, SmemLayout>;

    static constexpr auto kOrderS = Operand::kOrder;

    using PtrD = get_pointer_type<Td>;

    static constexpr int COPY_M = cs2mk<kOrderS>(SmemCopy::C, SmemCopy::S).x;
    static constexpr int COPY_K = cs2mk<kOrderS>(SmemCopy::C, SmemCopy::S).y;

    static constexpr int ITER_K   = K / COPY_K;
    static constexpr int WARP_CNT = M / COPY_M;

    struct Param {
        int       m;
        int       k;
        const Ts* src;
        int       lds;
        PtrD      dst;
        int       ldd;
    };

    using SharedStorage = Array<Ts, SmemLayout::kSize>;

    template<class T, int N, int M>
    static constexpr int get_fragment_size(Array<T, N> (&)[M])
    {
        return N;
    }

    template<class T, int N, int M>
    static constexpr int get_fragment_num(Array<T, N> (&)[M])
    {
        return M;
    }

    __device__ constexpr int2 _mk2cs(int m, int k)
    {
        return mk2cs<kOrderS>(m, k);
    }

    __device__ void operator()(const Param& param, char* smem_buf)
    {
        Ts* smem = (Ts*)smem_buf;

        const int cta_cnt_m = ceil_div(param.m, M);
        const int cta_cnt_k = ceil_div(param.k, K);

        const int cta_idx_m = blockIdx.x;

        const int cta_offset_m = cta_idx_m * M;

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_offset_m = warp_id * COPY_M;

        GmemIter gmem{(Ts*)param.src, param.lds, {cta_offset_m, 0}, {0, K}, {M, K}};

        gmem.smem_data_ = smem;

        gmem.ClearSmem();
        __syncthreads();

        Converter converter{};

        typename SmemCopy::Frag data;

        constexpr int kFragSize = get_fragment_size(data);
        constexpr int kFragNum  = get_fragment_num(data);
        constexpr int kPackSize = kFragSize * Pack_M;

        const int pack_cnt_k = cta_cnt_k * ITER_K;
        const int pack_cnt_m = cta_cnt_m * WARP_CNT * kFragNum;

        for (int cta_idx_k = 0; cta_idx_k < cta_cnt_k; ++cta_idx_k) {

            gmem.Prefetch(true);
            gmem.Advance();
            __pipeline_commit();

            __pipeline_wait_prior(0);
            __syncthreads();

            PRAGMA_UNROLL
            for (int k = 0; k < ITER_K; ++k) {

                // Load from smem as we are doing GEMMs
                SmemCopy::copy(Accessor{smem}, data, _mk2cs(warp_offset_m, k * COPY_K));

                PRAGMA_UNROLL
                for (int m = 0; m < kFragNum; m += Pack_M) {
                    // Convert and pack rmem data
                    Array<Td, kPackSize> packed = converter((Array<Ts, kPackSize>&)data[m]);

                    // Logical pack coords
                    const int pack_idx_k = cta_idx_k * ITER_K + k;
                    const int pack_idx_m = ((cta_idx_m * WARP_CNT + warp_id) * kFragNum + m) / Pack_M;

                    // Linear pack index
                    const int pack_index = cs2idx(_mk2cs(pack_idx_m, pack_idx_k),  //
                                                  _mk2cs(pack_cnt_m, pack_cnt_k).x);

                    // Store in [pack_id, lane_id]
                    Store(param.dst + (pack_index * WARP_SIZE + lane_id) * kPackSize, packed);
                }
            }

            __syncthreads();
        }
    }
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void convert_kernel(typename Kernel::Param param)
{
    Kernel kernel;
    kernel(param, smem_buf);
}

}  // namespace turbomind::gemm