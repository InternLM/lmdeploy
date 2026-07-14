// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_pipeline_primitives.h>

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/kernels/attention/quantization.h"

#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/format.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

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

    using Operand = MakeOperand<Operand_, IteratorSm70<Striding::kFlat, cache_policy::Default>, M_, K_, 1>;

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
        int         m;
        int         k;
        MatrixParam src;
        MatrixParam dst;
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

        const auto mat_S = resolve<Ts, Striding::kFlat>(param.src, 0);
        const auto mat_D = resolve<Td, Striding::kFlat>(param.dst, 0);

        // Handle residue k first
        GmemIter gmem{mat_S, {cta_offset_m, cta_offset_k}, {residue_m, residue_k}};

        gmem.smem_data_ = smem;
        gmem.ClearSmem();

        __syncthreads();

        // gmem.Prefetch(true);

        typename GmemIter::Fragments fragments{};
        gmem.Fetch(fragments, true);
        gmem.Store(fragments);

        // Rest full k tiles
        gmem            = GmemIter{mat_S, {cta_offset_m, 0}, {residue_m, K}};
        gmem.smem_data_ = smem;

        SmemCopy smem_copy({warp_offset_m, 0});

        // last, 0, 1, 2, 3, ..., last - 1
        int cta_idx_k = cta_cnt_k - 1;

        get_pointer_type<Td> mat_D_ptr{(Td*)mat_D.ptr.ptr};

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
                    auto dst_ptr = static_cast<Td*>(mat_D_ptr + unique_id * kPackSize);

                    if (pack_idx_m < pack_cnt_m && pack_idx_k < pack_cnt_k && repeat_id == 0) {
                        Store(dst_ptr, packed);
                    }
                }
            }

            __syncthreads();

            if (k_stage == cta_cnt_k - 1) {
                break;
            }

            // gmem.Prefetch(true);
            gmem.Fetch(fragments, true);
            gmem.Store(fragments);
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

constexpr bool is_AB(Op_Tag op)
{
    if (op == OPERAND_A || op == OPERAND_B) {
        return true;
    }
    else {
        return false;
    }
}

constexpr bool is_UV(Op_Tag op)
{
    return !is_AB(op);
}

template<class Dtype>
constexpr int unit_size(basic_type<Dtype>)
{
    return 1;
}

constexpr int unit_size(basic_type<uint8_t>)
{
    return 4;
}

constexpr int unit_size(basic_type<uint4_t>)
{
    return 8;
}

// MMA     : H_16816, H_1688, H_884, H_SIMT
// Operand : A, B, U, V
// Order   : row, col
// Dtype   : u16, u8, u4 (u6, u3)
// PackNum : 1, 2, 4

template<class Operand, class Dtype_, int PackNum>
struct Config {
    static constexpr int CTA_M = 64;
    static constexpr int CTA_K = 32;

    static constexpr int BLOCK_SIZE = 32;

    using Stype = typename Operand::Dtype;
    using Dtype = Dtype_;

    using Kernel = ConvertOperand<CTA_M, CTA_K, PackNum, Operand, Dtype, Converter<Stype, Dtype>>;
};

template<class Config>
void Convert_v2_Impl(const void* S, const MatrixLayout& Sdesc, void* D, const MatrixLayout& Ddesc, cudaStream_t stream)
{
    using Kernel = typename Config::Kernel;
    using Stype  = typename Config::Stype;
    using Dtype  = typename Config::Dtype;

    constexpr int CTA_M = Config::CTA_M;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);

    if (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(convert_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    typename Kernel::Param param{Sdesc.rows, Sdesc.cols, to_param((void*)S, Sdesc), to_param((void*)D, Ddesc)};

    constexpr int threads = Config::BLOCK_SIZE;
    const int     blocks  = ceil_div(Sdesc.rows, CTA_M);

    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // std::cout << __PRETTY_FUNCTION__ << "\nThreadMap:\n";
    // Print(typename Kernel::GmemIter::ThreadMap{});

    convert_kernel<Kernel><<<blocks, threads, kSmemSize, stream>>>(param);
}

}  // namespace turbomind::gemm
