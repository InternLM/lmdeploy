// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

#include "src/turbomind/kernels/gemm/tma.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::gemm {

// PFN_cuTensorMapEncodeTiled
// get_

// CUtensorMap make_2d_tma_desc()

template<class Gemm>
class KernelImplSm90: public Kernel {
public:
    // import frequently used constants
    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;

    // using Impl = typename Gemm::Impl;

    // using OpA = typename Gemm::OperandA;
    // using OpB = typename Gemm::OperandB;
    // using OpU = typename Gemm::OperandU;
    // using OpV = typename Gemm::OperandV;

    KernelImplSm90()
    {
        desc_.order_a = kRowMajor;  // m, k
        desc_.order_b = kColMajor;  // k, n
        desc_.order_c = kColMajor;

        desc_.type_a = data_type_v<typename Gemm::Ta>;
        desc_.type_b = data_type_v<typename Gemm::Tb>;
        desc_.type_c = data_type_v<typename Gemm::Tc>;

        // using IterA = typename OpA::GmemIter;
        // using IterB = typename OpB::GmemIter;

        desc_.striding_a = {};  // IterA::kMode;
        desc_.striding_b = {};  // IterB::kMode;
        desc_.striding_c = {};  // Gemm::Epilogue::kMode;

        desc_.pack_a = {};  // OpA::kPack;
        desc_.pack_b = {};  // OpB::kPack;
        desc_.pack_u = {};  // OpU::kPack;
        desc_.pack_v = {};  // OpV::kPack;

        desc_.quant_a = QuantDesc{};
        desc_.quant_b = QuantDesc{};

        // if constexpr (OpU::SmemLayout::kSize > 1) {
        //     desc_.quant_a = QuantDesc{QuantType::kDefault, OpU::kGroupSize};
        // }

        // if constexpr (OpV::SmemLayout::kSize > 1) {
        //     desc_.quant_b = QuantDesc{QuantType::kDefault, OpV::kGroupSize};
        // }

        desc_.cta_tile = {CTA_M, CTA_N, CTA_K};
        desc_.mma_tile = {CTA_M / Gemm::MMA_M, CTA_N / Gemm::MMA_N, CTA_N / Gemm::MMA_N};
        chunk_size_k_  = Gemm::CTA_K;

        desc_.align.x = 1;  // OpA::kOrder == kColMajor ? IterA::ThreadMap::kAccessC : 1;
        desc_.align.y = 1;  // OpB::kOrder == kColMajor ? IterB::ThreadMap::kAccessC : 1;
        desc_.align.z = 1;  // Gemm::CTA_K;

        desc_.policy_a = 0;               // (int)IterA::Policy::kEvictPolicy;
        desc_.policy_b = 0;               // (int)IterB::Policy::kEvictPolicy;
        desc_.c_tile   = {CTA_M, CTA_N};  // {Gemm::Epilogue::TM, Gemm::Epilogue::TN};
        desc_.op_class = OpClass::kGMMA_s64n16;

        smem_size_ = sizeof(typename Gemm::SharedStorage);

        desc_.stages  = Gemm::Stages;
        desc_.split_k = 1;      // Gemm::kSplitK;
        desc_.sched   = false;  // Gemm::kDynamicSched;

        desc_.arch = Gemm::Arch::value;

        // using Sched = typename Gemm::Sched;

        // auto func = gemm_kernel<Gemm, GemmParam, EpilogueParam, Sched>;

        auto func = gemm_kernel_sm90<Gemm>;

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&desc_.max_active_ctas, func, Gemm::CTA_SIZE, smem_size_);

        cudaFuncGetAttributes(&desc_.attr, func);

        name_ = GetName();
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& _Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& _Bdesc,
               const void*         V,
               const MatrixLayout& _Vdesc,
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) const override
    {
        using Sched = typename Gemm::Scheduler;

        MatrixLayout Adesc = _Adesc;

        [[maybe_unused]] const int m = Ddesc.rows;
        [[maybe_unused]] const int n = Ddesc.cols;
        [[maybe_unused]] const int k = Adesc.cols;

        auto transpose = [](MatrixLayout x) {
            std::swap(x.rows, x.cols);
            x.order = gemm::transpose(x.order);
            return x;
        };

        MatrixLayout Bdesc = transpose(_Bdesc);
        // MatrixLayout Vdesc = transpose(_Vdesc);

        auto sched = [&] {
            if constexpr (0) {
                LaunchSpec spec{(Kernel*)this};
                spec.splits  = splits;
                spec.swizzle = swizzle;
                return Sched{operation.context->Schedule(spec)};
            }
            else {
                const int chunk_cnt = ceil_div(k, CTA_K);
                // Limit splits by num of chunks to avoid chaos
                splits = std::min(chunk_cnt, splits);

                const int2 tiles = get_tiled_shape(m, n, CTA_M, CTA_N);
                const int4 shape{m, n, k, 1};

                if (splits > 1) {
                    splits = FixSplits(shape, tiles, splits, workspace);
                }

                swizzle = Sched::get_log_tile(tiles, 1 << swizzle);

                return Sched{shape, tiles, splits, swizzle, CTA_K, CTA_K};
            }
        }();

        // using Ta = typename Gemm::Ta;
        // using Tb = typename Gemm::Tb;
        using Tc = typename Gemm::Tc;

        auto tm_a = make_2d_tma_desc((void*)A, Adesc.type, m, k, CTA_M, CTA_K, kRowMajor, CU_TENSOR_MAP_SWIZZLE_128B);
        auto tm_b = make_2d_tma_desc((void*)B, Bdesc.type, k, n, CTA_K, CTA_N, kColMajor, CU_TENSOR_MAP_SWIZZLE_128B);
        auto tm_c = make_2d_tma_desc((void*)C, Cdesc.type, m, n, CTA_M, CTA_N, kColMajor, CU_TENSOR_MAP_SWIZZLE_NONE);

        const auto grid  = sched.get_grid_shape();
        const auto block = Gemm::CTA_SIZE;

        

        cudaLaunchAttribute attrs[1];
        attrs[0].val.clusterDim.x = 1;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;

        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = block;
        config.dynamicSmemBytes = smem_size_;
        config.stream           = stream;
        config.attrs            = attrs;

        auto func = gemm_kernel_sm90<Gemm>;

        check_cuda_error(cudaLaunchKernelEx(&config, func, tm_a, tm_b, tm_c, (Tc*)C, Cdesc.ld, sched));

        // gemm_kernel_sm90<Gemm><<<grid, block, smem_size_, stream>>>(tm_a, tm_b, tm_c, (Tc*)C, Cdesc.ld, sched);

        // if constexpr (0) {
        //     [[maybe_unused]] static const int _ = [] {
        //         std::cout << "A:\n";
        //         Print(typename Gemm::OperandA::GmemIter::ThreadMap{});
        //         std::cout << "\nB:\n";
        //         Print(typename Gemm::OperandB::GmemIter::ThreadMap{});
        //         if constexpr (!std::is_same_v<Ta, Tc>) {
        //             std::cout << "\nU:\n";
        //             Print(typename Gemm::OperandU::GmemIter::ThreadMap{});
        //         }
        //         if constexpr (!std::is_same_v<Tb, Tc>) {
        //             std::cout << "\nV:\n";
        //             Print(typename Gemm::OperandV::GmemIter::ThreadMap{});
        //         }
        //         printf("warp count: %d\n", Impl::WARPS);
        //         Print_(typename Gemm::Impl::MMA_Map{});

        //         printf("C:\n");
        //         Print(typename Gemm::Epilogue::Map{});

        //         std::cout << "Smem for mainloop: " << sizeof(Gemm::SharedStorage::mainloop) << "\n";
        //         std::cout << "Smem for epilogue: " << sizeof(Gemm::SharedStorage::epilogue) << "\n";

        //         return 0;
        //     }();
        // }

        // const bool silu_act = ((int)operation.epilogue & (int)Epilogue::kGatedSilu);

        // MatrixLayout Pdesc = Ddesc;
        // Pdesc.ld           = mk2cs<Gemm::kOrderC>(Pdesc.rows, Pdesc.cols).x;

        // MatrixCombination_v3 combin_mat{to_param((void*)C, Cdesc), alpha, beta};

        // EpilogueParam epilogue{to_param((void*)D, Ddesc),
        //                        to_param((void*)workspace.partials, Pdesc),
        //                        (int*)workspace.barriers,
        //                        combin_mat,
        //                        silu_act};

        // std::cout << Adesc.offsets << " " << Adesc.idxs << "\n";

        // GemmParam param{
        //     to_param((void*)A, Adesc),
        //     to_param((void*)B, Bdesc),
        //     to_param((void*)U, Udesc),
        //     to_param((void*)V, Vdesc),
        // };

        // std::cout << grid.x << " " << grid.y << " " << grid.z << "\n";

        // gemm_kernel<Gemm><<<grid, block, smem_size_, stream>>>(param, epilogue, sched);

        return 0;
    }

    std::array<size_t, 2> GetWorkspaceSizesV2(const int4& shape, int tiles, int splits) const
    {
        static constexpr bool kSerial = true;

        const auto& [m, n, _, num] = shape;

        size_t barriers_size = sizeof(int) * tiles;
        size_t partials_size = sizeof(float) * m * n * num;

        if constexpr (!kSerial) {
            barriers_size *= splits;
            partials_size *= splits;
        }

        return {barriers_size, partials_size};
    }

    int GetMaxSplits(const int4& shape, int64_t tiles, size_t bsize, size_t psize) const override
    {
        if (!Gemm::kSplitK) {
            return 1;
        }

        const auto& [m, n, k, _] = shape;

        const auto& [a, b] = GetWorkspaceSizesV2(shape, tiles, 1);

        if (bsize >= a && psize >= b) {
            // Serial split-k requires workspace for 1 split only
            // But it can't exceed num of k chunks
            return cdiv(k, Gemm::kChunkSizeK);
        }
        else {
            return 1;
        }
    }

    int GetSwizzle(int m, int n, int k, int splits, int swizzle) const override
    {
        using Map        = typename Gemm::Scheduler;
        const auto tiles = get_tiled_shape(m, n, CTA_M, CTA_N);
        return Map::get_log_tile(tiles, 1 << swizzle);
    }

    int FixSplits(const int4& shape, int2 tiled_mn, int splits, Workspace& ws) const
    {
        const int tiles            = tiled_mn.x * tiled_mn.y;
        const auto& [bsize, psize] = GetWorkspaceSizesV2(shape, tiles, splits);

        if (ws.barriers_size < bsize || ws.partials_size < psize) {
            const int max_splits       = GetMaxSplits(shape, tiles, ws.barriers_size, ws.partials_size);
            const auto& [m, n, k, num] = shape;
            fprintf(
                stderr,
                "Problem size (%d, %d, %d), workspace size too small (%d, %d) vs required (%d, %d) for %d splits. Force `splits` = %d\n",
                m,
                n,
                k,
                (int)ws.barriers_size,
                (int)ws.partials_size,
                (int)bsize,
                (int)psize,
                splits,
                max_splits);
            splits = max_splits;
        }

        return splits;
    }
};

}  // namespace turbomind::gemm