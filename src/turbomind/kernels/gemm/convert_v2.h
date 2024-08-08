// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <cuda_pipeline_primitives.h>

template<class T>
__device__ void print_type(T)
{
    if (threadIdx.x == 0) {
        printf("%s\n", __PRETTY_FUNCTION__);
    }
}

namespace turbomind::gemm {

template<int M_, int K_, int Pack_M, class Operand_, class Td, class Converter>
struct ConvertOperand {

    static constexpr int M = M_;
    static constexpr int K = K_;

    using Operand = MakeOperand<Operand_, IteratorSm70<cache_policy::Default>, M_, K_, 1>;

    using Ts         = typename Operand::Dtype;
    using SmemLayout = typename Operand::SmemLayout;
    using GmemIter   = typename Operand::GmemIter;

    using Atom = typename Operand::SmemCopyAtom;

    using SmemCopy = SmemCopy<Operand, M_ / Atom::M, K_ / Atom::K, Atom::M, Atom::K>;

    using Accessor = SmemAccessor<Ts, SmemLayout>;

    static constexpr auto kOrderS = Operand::kOrder;

    static constexpr int ITER_K = ceil_div(K, Atom::K);

    /// TODO: generailize this
    static constexpr int WARP_CNT = 1;

    using PtrD = get_pointer_type<Td>;

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
        const int residue_m    = min(M, param.m - cta_offset_m);

        const int warp_id = threadIdx.x / WARP_SIZE;

        const int warp_offset_m = 0;

        Converter converter{};

        typename SmemCopy::Frag data;

        constexpr int kFragSize = get_fragment_size(data);
        constexpr int kFragNum  = get_fragment_num(data);
        constexpr int kPackSize = kFragSize * Pack_M;

        const int pack_cnt_k = ceil_div(param.k, Atom::K);
        const int pack_cnt_m = ceil_div(param.m, Atom::M * Pack_M);

        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            // printf("m=%d, k=%d, lds = %d\n", param.m, param.k, param.lds);
            // printf(
            //     "CTA_M=%d, CTA_K=%d, cta_cnt_m=%d, cta_cnt_k=%d, cta_idx_m=%d, ITER_K=%d, pack_cnt_m=%d,
            //     pack_cnt_k=%d\n", M_, K_, cta_cnt_m, cta_cnt_k, cta_idx_m, ITER_K, pack_cnt_m, pack_cnt_k);
            // printf("frag_size=%d, frag_num=%d, pack_size=%d\n", kFragSize, kFragNum, kPackSize);
        }

        const int cta_offset_k = (cta_cnt_k - 1) * K;
        const int residue_k    = min(K, param.k - cta_offset_k);

        // Handle residue k first
        GmemIter gmem{(Ts*)param.src, param.lds, {cta_offset_m, cta_offset_k}, {residue_m, residue_k}};

        gmem.smem_data_ = smem;
        gmem.ClearSmem();

        __syncthreads();

        gmem.Prefetch(true);

        // Rest full k tiles
        gmem            = GmemIter{(Ts*)param.src, param.lds, {cta_offset_m, 0}, {residue_m, K}};
        gmem.smem_data_ = smem;

        SmemCopy smem_copy({warp_offset_m, 0});

        // last, 0, 1, 2, 3, ..., last - 1
        int cta_idx_k = cta_cnt_k - 1;

        for (int k_stage = 0; k_stage < cta_cnt_k; ++k_stage) {
            __syncthreads();

            PRAGMA_UNROLL
            for (int k = 0; k < ITER_K; ++k) {
                // Assuming `SmemCopy` is a warp-level operation
                // Load from smem as we are doing GEMMs
                // SmemCopy::copy(smem, data, int2{warp_offset_m, 0}, k);
                smem_copy(smem, data, k);

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

                    auto [unique_id, repeat_id] = Atom::unique(threadIdx.x, pack_index);

                    // Store in [pack_id, lane_id], static cast is needed to decay SubBytePtr<T> to T*
                    auto dst_ptr = static_cast<Td*>(param.dst + unique_id * kPackSize);

                    if (pack_idx_m < pack_cnt_m && pack_idx_k < pack_cnt_k && repeat_id == 0) {
                        Store(dst_ptr, packed);
                    }
                }
            }

            __syncthreads();

            if (k_stage == cta_cnt_k - 1) {
                break;
            }

            gmem.Prefetch(true);
            gmem.Advance();

            cta_idx_k = k_stage;
        }
    }

    __device__ void print(...) {}

    __device__ void print(Array<uint32_t, 2> _x)
    {
        auto& x = (const Array<half, 4>&)_x;
        printf("tidx=%d, %f %f %f %f\n", (int)threadIdx.x, (float)x[0], (float)x[1], (float)x[2], (float)x[3]);
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
