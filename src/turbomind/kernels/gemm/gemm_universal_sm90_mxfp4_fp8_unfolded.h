// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/iterator_sm90.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gemm/sm90_mxfp4_fp8_traits.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

// Unfolded SM90 E4M3-K128 x MXFP4-K32 mainloop for dense kFlat GEMM with a
// plain BF16 output. Public (M,N,K) is (tokens,output,K), while WGMMA sees
// packed weight as RS A and activation as descriptor B, hence the hardware
// tile is (output,tokens,K).
template<class Config_, int Stages_, Order Raster, int MmaN>
struct GemmUniversalSm90MxFp4Fp8Unfolded {
    using Tile                                = typename Config_::Tile;
    using Groups                              = typename Config_::Groups;
    using RegisterConfig                      = typename Config_::RegisterConfig;
    using WGLayout                            = cute::Layout<cute::Shape<cute::Int<Groups::M>, cute::Int<Groups::N>>>;
    using Arch                                = Sm90;
    using Format                              = Sm90MxFp4Fp8UnfoldedFormat;
    static constexpr Order    kRasterOrder    = Raster;
    static constexpr bool     is_grouped_gemm = false;
    static constexpr bool     kSupportsFusedSilu = false;
    static constexpr Striding kStridingA         = Striding::kFlat;
    static constexpr Striding kStridingB         = Striding::kFlat;
    static constexpr Striding kStridingC         = Striding::kFlat;
    static constexpr int      kMulticastA        = 1;
    static constexpr int      kMulticastB        = 1;
    static constexpr int      kMulticastU        = 1;
    static constexpr int      kClusterSize       = 1;

    static constexpr int TILE_M           = Tile::M;
    static constexpr int TILE_N           = Tile::N;
    static constexpr int TILE_K           = 128;
    static constexpr int Stages           = Stages_;
    static constexpr int kGroupSize       = 32;
    static constexpr int kOutputFragmentN = 64;
    using Ta                              = __nv_fp8_e4m3;
    using Tb                              = fp4_e2m1_t;
    using Tv                              = typename Format::QparamType;
    using Tc                              = nv_bfloat16;
    using Tu                              = float;

    static constexpr int kComputeTileN = TILE_N;
    using Traits                       = GmmaMxFp4Fp8UnfoldedTraits<kComputeTileN, TILE_M, Stages, WGLayout, MmaN>;
    using TiledMma                     = typename Traits::TiledMma;
    using WgTiledMma                   = typename Traits::WgTiledMma;
    using ElementMmaA                  = typename Traits::ElementA;
    using AtomLayoutMNK                = typename Traits::AtomLayoutMNK;
    static constexpr int kAtomM        = Traits::kAtomM;
    static constexpr int kAtomN        = Traits::kAtomN;
    static constexpr int kRestM        = Traits::kRestM;
    static constexpr int kRestN        = Traits::kRestN;
    static_assert(kRestN >= 1 && kRestN <= 4, "registered peak tiles use at most four residual-N atoms");
    static constexpr int kOpM            = Traits::kOpM;
    static constexpr int kOpN            = Traits::kOpN;
    static constexpr int kOpK            = Traits::kOpK;
    static constexpr int kMathWarpGroups = Traits::kMathWarpgroups;
    static_assert(kMathWarpGroups == 2, "SM90 MXFP4 x FP8 kernels use two math warpgroups");
    static constexpr int WARPGROUPS         = kMathWarpGroups;
    static constexpr int WARPGROUP_SIZE     = 128;
    static constexpr int kMathThreads       = WARPGROUP_SIZE * kMathWarpGroups;
    static constexpr int CTA_SIZE           = kMathThreads + WARPGROUP_SIZE;
    static constexpr int kEpilogueBarrierId = 1;
    static constexpr int kProducerBarrierId = 8;
    static_assert(kEpilogueBarrierId + WARPGROUPS <= kProducerBarrierId);
    static constexpr int kProducerRegs = RegisterConfig::Producer;
    static constexpr int kMathRegs     = RegisterConfig::Math;
    static_assert(kProducerRegs % 8 == 0 && kMathRegs % 8 == 0);
    static_assert(kMathWarpGroups != 1 || kProducerRegs + kMathRegs <= 512);
    static_assert(kMathWarpGroups != 2 || kProducerRegs + 2 * kMathRegs <= 504);
    static_assert(kMathWarpGroups != 3 || kProducerRegs + 3 * kMathRegs <= 512);

    using Cluster          = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;
    using ClusterShape     = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;
    using Scheduler        = TileScheduler<Raster, Cluster, true, true, TILE_M, TILE_N, Stages, false>;
    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using MainloopState    = typename MainloopPipeline::PipelineState;
    using PipelineStorage  = typename MainloopPipeline::SharedStorage;

    using PackedRecordLayout                      = decltype(Traits::packed_layout_a_mk());
    static constexpr int kPackedElementsPerRecord = cute::cosize_v<PackedRecordLayout>;
    static constexpr int kPackedInnerWords =
        kPackedElementsPerRecord * cute::sizeof_bits_v<cute::uint4_t> / cute::sizeof_bits_v<uint32_t>;
    static constexpr int kPackedTmaInnerWords  = 256;
    static constexpr int kPackedTmaParts       = kPackedInnerWords / kPackedTmaInnerWords;
    static constexpr int kOutputFragments      = TILE_N / kOutputFragmentN;
    static constexpr int kK32FragmentsPerStage = Traits::kKBlocksPerStage;
    using QparamRecordLayout = cute::Layout<cute::Shape<cute::Int<kOutputFragmentN>, cute::Int<kK32FragmentsPerStage>>,
                                            cute::Stride<cute::Int<kK32FragmentsPerStage>, cute::_1>>;
    using QparamSmemLogicalLayout = decltype(
        cute::tile_to_shape(QparamRecordLayout{},
                            cute::Shape<cute::Int<TILE_N>, cute::Int<kK32FragmentsPerStage>, cute::Int<Stages>>{},
                            cute::Step<cute::_1, cute::_2, cute::_3>{}));
    static constexpr int kQparamValuesPerFragment   = kOutputFragmentN * kK32FragmentsPerStage;
    static constexpr int kQparamTmaInnerValues      = 128;
    static constexpr int kQparamTmaParts            = kQparamValuesPerFragment / kQparamTmaInnerValues;
    static constexpr int kPackedWordsStage          = kPackedInnerWords * kOutputFragments * kK32FragmentsPerStage;
    static constexpr int kPackedWeightStageBytes    = kPackedWordsStage * sizeof(uint32_t);
    static constexpr int kQparamValuesStage         = kQparamValuesPerFragment * kOutputFragments;
    static constexpr int kMxScaleStageBytes         = kQparamValuesStage * sizeof(Tv);
    static constexpr int kActivationStageBytes      = TILE_M * TILE_K * sizeof(Ta);
    static constexpr int kAlignmentU                = 16 / sizeof(float);
    static constexpr int kBoxU                      = TILE_M;
    static constexpr int kUStageStride              = round_up<int>(kBoxU, 128);
    static constexpr int kActivationScaleStageBytes = kBoxU * sizeof(float);
    static_assert(kPackedElementsPerRecord == kOutputFragmentN * kOpK);
    static_assert(kPackedWeightStageBytes == TILE_N * TILE_K / 2);
    static_assert(cute::cosize_v<QparamSmemLogicalLayout> * sizeof(Tv) == kMxScaleStageBytes * Stages);
    static_assert(kPackedTmaParts == 1);
    static_assert(kQparamTmaParts == 2);
    static_assert(kMxScaleStageBytes == TILE_N * kK32FragmentsPerStage * sizeof(Tv));

    // WGMMA C and STSM use (output, token) mode order.  Store one
    // GMMA-compatible token strip at a time so the register-to-shared copy is
    // derived from the MMA object instead of reconstructing its TV mapping.
    static constexpr int  kWgM            = TILE_M / kAtomN;
    static constexpr bool kSplitEpiM      = kAtomN == 2;
    static constexpr int  kEpiN           = 64 * kAtomM;
    static constexpr int  kWgMLowBit      = kWgM & -kWgM;
    static constexpr int  kEpiM           = kWgMLowBit < 32 ? kWgMLowBit : 32;
    static constexpr int  kTmaStoreN      = 64;
    static constexpr int  kTmaStoreM      = kEpiM;
    static constexpr int  kTmaStoreCountN = kEpiN / kTmaStoreN;
    static constexpr int  kEpiStripsM     = kWgM / kEpiM;
    static constexpr int  kEpiStripsN     = kComputeTileN / kEpiN;
    static constexpr int  kEpiPlanes      = kSplitEpiM ? kAtomN : 1;
    static constexpr int  kEpiThreads     = kSplitEpiM ? WARPGROUP_SIZE : kMathThreads;
    static constexpr int  kFragmentSize   = kEpiM * kEpiN / kEpiThreads;
    static constexpr int  kSwizzleC       = 128;
    static_assert(TILE_M % kEpiM == 0 && kWgM % kEpiM == 0);
    static_assert(kComputeTileN % kEpiN == 0 && kEpiN % kTmaStoreN == 0);
    static_assert(kFragmentSize == 16);
    using SmemLayoutAtomD = decltype(
        gmma_ss_smem_selector<cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::Int<kEpiN>, cute::Int<kEpiM>>());
    using SmemLayoutD =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::Int<kEpiPlanes>{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x8_STSM_T;

    using PackedCtaTile    = cute::Shape<cute::Int<kPackedTmaInnerWords>,
                                      cute::Int<kPackedTmaParts * kOutputFragments>,
                                      cute::Int<kK32FragmentsPerStage>>;
    using PackedSmemLayout = decltype(cute::make_layout(PackedCtaTile{}));
    using PackedPipelineLayout =
        cute::Layout<cute::Shape<cute::Int<cute::cosize_v<PackedSmemLayout>>, cute::Int<Stages>>>;
    using QparamCtaTile =
        cute::Shape<cute::Int<kQparamTmaInnerValues>, cute::Int<kQparamTmaParts * kOutputFragments>, cute::_1>;
    using QparamSmemLayout = decltype(cute::make_layout(QparamCtaTile{}));
    using QparamPipelineLayout =
        cute::Layout<cute::Shape<cute::Int<cute::cosize_v<QparamSmemLayout>>, cute::Int<Stages>>>;
    using SmemLayoutB       = typename Traits::SmemLayoutB;
    using SmemLayoutB_2D    = typename Traits::SmemLayoutB_2D;
    using ActivationCtaTile = cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_K>>;

    static auto MakeTmaActivation(void* ptr, int m, int k, int ld)
    {
        auto gmem =
            cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<Ta*>(ptr)),
                              cute::make_layout(cute::make_shape(m, k), cute::make_stride(int64_t{ld}, cute::_1{})));
        return cute::make_tma_copy<cutlass::float_e4m3_t>(
            cute::SM90_TMA_LOAD{}, gmem, SmemLayoutB_2D{}, ActivationCtaTile{}, cute::_1{});
    }
    using TmaActivation = decltype(MakeTmaActivation(nullptr, TILE_M, TILE_K, TILE_K));

    static auto MakeTmaPacked(void* ptr, int n, int k)
    {
        const int out_fragments = n / kOutputFragmentN;
        const int k32_fragments = k / kOpK;
        auto      layout        = cute::make_layout(
            cute::make_shape(cute::Int<kPackedTmaInnerWords>{}, out_fragments * kPackedTmaParts, k32_fragments),
            cute::make_stride(
                cute::_1{}, cute::Int<kPackedTmaInnerWords>{}, int64_t{kPackedInnerWords} * out_fragments));
        auto gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint32_t*>(ptr)), layout);
        return cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem, PackedSmemLayout{}, PackedCtaTile{}, cute::_1{});
    }
    using TmaPacked = decltype(MakeTmaPacked(nullptr, TILE_N, TILE_K));

    static auto MakeTmaQparam(void* ptr, int n, int k)
    {
        const int out_fragments = n / kOutputFragmentN;
        const int k128_groups   = k / TILE_K;
        auto      layout        = cute::make_layout(
            cute::make_shape(cute::Int<kQparamTmaInnerValues>{}, out_fragments * kQparamTmaParts, k128_groups),
            cute::make_stride(
                cute::_1{}, cute::Int<kQparamTmaInnerValues>{}, int64_t{kQparamValuesPerFragment} * out_fragments));
        auto gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<Tv*>(ptr)), layout);
        return cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem, QparamSmemLayout{}, QparamCtaTile{}, cute::_1{});
    }
    using TmaQparam = decltype(MakeTmaQparam(nullptr, TILE_N, TILE_K));
    static_assert(cute::size(typename TmaPacked::Traits::SrcLayout{}) == kPackedWeightStageBytes * 8);
    static_assert(cute::size(typename TmaQparam::Traits::SrcLayout{}) == kMxScaleStageBytes * 8);

    struct alignas(1024) SharedStorage {
        cute::array_aligned<uint32_t, kPackedWordsStage * Stages, 128> A;
        cute::array_aligned<Ta, cute::cosize_v<SmemLayoutB>, 128>      B;
        cute::array_aligned<Tv, kQparamValuesStage * Stages, 128>      V;
        cute::array_aligned<float, kUStageStride * Stages, 128>        U;
        PipelineStorage                                                pipeline;
        typename Scheduler::Storage                                    sched;
    };
    static constexpr int kOutputOffset  = round_up<int>(sizeof(SharedStorage), 1024);
    static constexpr int kEpilogueBytes = cute::cosize_v<SmemLayoutD> * sizeof(nv_bfloat16);
    static constexpr int kSmemSize      = kOutputOffset + kEpilogueBytes;
    static_assert(cute::cosize_v<SmemLayoutB> == Stages * TILE_M * TILE_K);
    static_assert(kSmemSize <= (228 << 10));

    __device__ void operator()(const TmaActivation& tm_a,
                               const TmaPacked&     tm_b,
                               const TmaQparam&     tm_v,
                               const CUtensorMap&   tm_u,
                               const CUtensorMap&   tm_c,
                               const MatrixParam&   param_A,
                               const MatrixParam&   param_B,
                               const MatrixParam&   param_V,
                               const MatrixParam&   param_U,
                               const MatrixParam&   param_C,
                               const MatrixParam&   param_W,
                               bool                 fuse_silu,
                               Scheduler            sched,
                               CUtensorMap*         tensormap_buf,
                               char*                smem_buf)
    {
        assert(!fuse_silu);
        (void)param_A;
        (void)param_B;
        (void)param_V;
        (void)param_U;
        (void)param_C;
        (void)param_W;
        (void)tensormap_buf;
        SharedStorage& storage    = *reinterpret_cast<SharedStorage*>(smem_buf);
        const int      wg_idx     = cutlass::canonical_warp_group_idx();
        const int      warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const int      lane       = threadIdx.x % 32;
        if (threadIdx.x == 0) {
            sched.init_dyanmic(storage.sched, kClusterSize * (kMathWarpGroups * 4 + 1));
        }
        typename MainloopPipeline::Params params{};
        params.transaction_bytes =
            kPackedWeightStageBytes + kMxScaleStageBytes + kActivationStageBytes + kActivationScaleStageBytes;
        params.num_consumers     = kMathThreads;
        params.num_producers     = 1;
        params.initializing_warp = 0;
        if (wg_idx == kMathWarpGroups) {
            params.role      = warp_in_wg == 0 ? MainloopPipeline::ThreadCategory::Producer :
                                                 MainloopPipeline::ThreadCategory::NonParticipant;
            params.is_leader = warp_in_wg == 0 && lane == 0;
        }
        else {
            params.role      = MainloopPipeline::ThreadCategory::Consumer;
            params.is_leader = 0;
        }
        MainloopPipeline pipeline(storage.pipeline, params, ClusterShape{});
        if (threadIdx.x == 0) {
            cutlass::arch::fence_view_async_shared();
        }
        __syncthreads();
        if (wg_idx == kMathWarpGroups) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();
            run_producer_tma(tm_a, tm_b, tm_v, tm_u, sched, storage, pipeline);
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
            run_consumer(tm_c, sched, storage, pipeline);
        }
    }

private:
    __device__ static void run_producer_tma(const TmaActivation& tm_a,
                                            const TmaPacked&     tm_b,
                                            const TmaQparam&     tm_v,
                                            const CUtensorMap&   tm_u,
                                            Scheduler            sched,
                                            SharedStorage&       storage,
                                            MainloopPipeline&    pipeline)
    {
        const int  warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const bool cta_0      = cute::block_id_in_cluster().x == 0;
        if (warp_in_wg == 0) {
            MainloopState write_state   = cutlass::make_producer_start_state<MainloopPipeline>();
            auto          sched_state   = sched.init_consumer(storage.sched);
            const bool    elected       = cute::elect_one_sync();
            const int     k_iters       = sched.k_iters_;
            const int     out_fragments = sched.gemm_shape().y / kOutputFragmentN;
            const int     out_tiles     = (out_fragments + kOutputFragments - 1) / kOutputFragments;
            auto          packed_gmem   = tm_b.get_tma_tensor(cute::make_shape(
                cute::Int<kPackedTmaInnerWords>{}, out_fragments * kPackedTmaParts, k_iters * kK32FragmentsPerStage));
            auto          packed_tiles  = cute::flat_divide(packed_gmem, PackedCtaTile{});
            auto          packed_cta    = tm_b.get_slice(0);
            auto          packed_part   = packed_cta.partition_S(packed_tiles);
            auto          packed_src    = cute::group_modes<1, cute::rank(packed_part)>(packed_part);
            auto          qparam_gmem   = tm_v.get_tma_tensor(
                cute::make_shape(cute::Int<kQparamTmaInnerValues>{}, out_fragments * kQparamTmaParts, k_iters));
            auto qparam_tiles    = cute::flat_divide(qparam_gmem, QparamCtaTile{});
            auto qparam_cta      = tm_v.get_slice(0);
            auto qparam_part     = qparam_cta.partition_S(qparam_tiles);
            auto qparam_src      = cute::group_modes<1, cute::rank(qparam_part)>(qparam_part);
            auto sPackedPipeline = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), PackedPipelineLayout{});
            auto sQparamPipeline = cute::make_tensor(cute::make_smem_ptr(storage.V.data()), QparamPipelineLayout{});
            auto sActivation     = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
            auto activation_gmem = tm_a.get_tma_tensor(cute::make_shape(sched.gemm_shape().x, k_iters * TILE_K));
            auto activation_cta  = tm_a.get_slice(0);
            auto sScale =
                cute::make_tensor(cute::make_smem_ptr(storage.U.data()),
                                  cute::make_layout(cute::Shape<cute::Int<kUStageStride>, cute::Int<Stages>>{}));
            typename Scheduler::Tile* tile;
            while (sched_state.acquire(tile)) {
                if (tile->is_valid_cluster && elected) {
                    const cute::TmaDescriptor* Adesc        = tm_a.get_tma_descriptor();
                    const CUtensorMap*         Udesc        = &tm_u;
                    const cute::TmaDescriptor* Bdesc        = tm_b.get_tma_descriptor();
                    const cute::TmaDescriptor* Vdesc        = tm_v.get_tma_descriptor();
                    const int                  out_fragment = tile->offset_n / kOutputFragmentN;
                    const int                  out_tile     = out_fragment / kOutputFragments;
                    GmemIteratorSm90<1>        gmem_U{Udesc, {tile->offset_m, 0}, {0, 1}};
                    for (int k_tile = 0; k_tile < k_iters; ++k_tile) {
                        pipeline.producer_acquire(write_state);
                        auto*     bar             = pipeline.producer_get_barrier(write_state);
                        const int stage           = write_state.index();
                        auto      packed_stage    = sPackedPipeline(cute::_, stage);
                        auto      packed_smem     = cute::make_tensor(packed_stage.data(), PackedSmemLayout{});
                        auto      packed_dst_part = packed_cta.partition_D(packed_smem);
                        auto      packed_dst      = cute::group_modes<1, cute::rank(packed_dst_part)>(packed_dst_part);
                        cute::copy(tm_b.with(Bdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                                   packed_src(cute::_, out_tile + out_tiles * k_tile),
                                   packed_dst(cute::_, 0));
                        auto qparam_stage    = sQparamPipeline(cute::_, stage);
                        auto qparam_smem     = cute::make_tensor(qparam_stage.data(), QparamSmemLayout{});
                        auto qparam_dst_part = qparam_cta.partition_D(qparam_smem);
                        auto qparam_dst      = cute::group_modes<1, cute::rank(qparam_dst_part)>(qparam_dst_part);
                        cute::copy(tm_v.with(Vdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                                   qparam_src(cute::_, out_tile + out_tiles * k_tile),
                                   qparam_dst(cute::_, 0));
                        auto activation_tile = cute::local_tile(
                            activation_gmem, ActivationCtaTile{}, cute::make_coord(tile->offset_m / TILE_M, k_tile));
                        auto activation_src   = activation_cta.partition_S(activation_tile);
                        auto activation_stage = sActivation(cute::_, cute::_, stage);
                        auto activation_dst   = activation_cta.partition_D(activation_stage);
                        cute::copy(tm_a.with(Adesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                                   activation_src,
                                   activation_dst);
                        gmem_U.Step(bar, &sScale(cute::Int<0>{}, stage), 0);
                        ++write_state;
                    }
                }
                if constexpr (Scheduler::is_dynamic) {
                    if (cta_0) {
                        named_barrier_arrive_unaligned(WARP_SIZE * 2, 8);
                    }
                }
                sched_state.release();
            }
            sched_state.release();
            if (elected) {
                pipeline.producer_tail(write_state);
            }
        }
        else if (warp_in_wg == 1 && cta_0) {
            auto state = sched.init_producer(storage.sched);
            while (state.next()) {
                if constexpr (Scheduler::is_dynamic) {
                    named_barrier_arrive_and_wait_unaligned(WARP_SIZE * 2, 8);
                }
            }
            sched.tail(state);
        }
    }

    __device__ static void
    run_consumer(const CUtensorMap& tm_c, Scheduler sched, SharedStorage& storage, MainloopPipeline& pipeline)
    {
        const int  mma_tid = threadIdx.x;
        TiledMma   tiled_mma;
        auto       thr_mma    = tiled_mma.get_thread_slice(mma_tid);
        const auto thr_vmnk   = thr_mma.thr_vmnk_;
        const int  local_tid  = cute::get<0>(thr_vmnk);
        const int  wg_m       = cute::get<1>(thr_vmnk);
        const int  wg_n       = cute::get<2>(thr_vmnk);
        const int  wg_idx     = AtomLayoutMNK{}(wg_m, wg_n, cute::Int<0>{});
        auto       cC         = cute::make_identity_tensor(cute::Shape<cute::Int<kComputeTileN>, cute::Int<TILE_M>>{});
        auto       tCcC       = thr_mma.partition_C(cC);
        auto       cA         = cute::make_identity_tensor(cute::Shape<cute::Int<kComputeTileN>, cute::Int<TILE_K>>{});
        auto       tAcA       = thr_mma.partition_A(cA);
        auto       dummy_sA   = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ElementMmaA*>(storage.A.data())),
                                          typename Traits::DummySmemLayoutA{});
        auto       dummy_tCsA = thr_mma.partition_A(dummy_sA);
        auto       tCrA       = thr_mma.make_fragment_A(dummy_tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
        auto       sB         = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
        auto       sScale     = cute::make_tensor(cute::make_smem_ptr(storage.U.data()),
                                        cute::make_layout(cute::Shape<cute::Int<kUStageStride>, cute::Int<Stages>>{}));
        auto warp_group_thread_layout = cute::make_layout(cute::Int<kMathWarpGroups>{}, cute::Int<WARPGROUP_SIZE>{});
        auto wg_mma                   = tiled_mma.get_slice(warp_group_thread_layout(wg_idx));
        auto tCrB                     = wg_mma.make_fragment_B(wg_mma.partition_B(sB));
        static_assert(cute::size<0>(tCrA) == 16);
        static_assert(cute::rank(tCrA) == 3);
        static_assert(cute::size<1>(tCrA) == kRestM);
        static_assert(cute::size<2>(tCrA) == kK32FragmentsPerStage);
        static_assert(cute::size<2>(decltype(tCrB){}) == 4);
        static_assert(cute::size<1>(decltype(tCrB){}) == kRestN);

        WgTiledMma    wg_tiled_mma;
        auto          wg_thr_mma  = wg_tiled_mma.get_thread_slice(local_tid);
        auto          layout_a_tv = wg_tiled_mma.get_layoutA_TV();
        constexpr int kPackedWordsPerThread =
            cute::size<1>(decltype(layout_a_tv){}) * cute::sizeof_bits_v<cute::uint4_t> / cute::sizeof_bits_v<uint32_t>;
        static_assert(kPackedWordsPerThread == 2);
        auto packed_copy     = cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<uint32_t>, uint32_t>{},
                                                 cute::make_layout(cute::Int<cute::size<0>(decltype(layout_a_tv){})>{}),
                                                 cute::make_layout(cute::Int<kPackedWordsPerThread>{}));
        auto packed_thr_copy = packed_copy.get_thread_slice(local_tid);
        auto sPackedPipeline = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), PackedPipelineLayout{});

        constexpr int kQparamGroups = Traits::kKBlocksPerStage;
        auto          sQparam = cute::make_tensor(cute::make_smem_ptr(storage.V.data()), QparamSmemLogicalLayout{});

        auto run = [&] {
            MainloopState             read_state{};
            auto                      sched_state = sched.init_consumer(storage.sched);
            typename Scheduler::Tile* tile;
            sched_state.acquire(tile);
            while (tile->alive) {
                if (tile->is_valid_cta) {
                    auto accum = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    cute::clear(accum);
                    for (int k_tile = 0; k_tile < sched.k_iters_; ++k_tile) {
                        auto token = pipeline.consumer_try_wait(read_state);
                        pipeline.consumer_wait(read_state, token);
                        const int     stage        = read_state.index();
                        constexpr int u_pad        = 0;
                        auto          packed_stage = sPackedPipeline(cute::_, stage);
                        auto          sPacked      = cute::make_tensor(packed_stage.data(), PackedSmemLayout{});
                        auto          tAsPacked    = packed_thr_copy.partition_S(sPacked);

                        auto sQparamStage = sQparam(cute::_, cute::_, stage);
                        {
                            // Scratch fragments use the same CuTe C ownership as
                            // the WA mainloop; the issue schedule below alternates
                            // two of them across residual output atoms.
                            auto scratch = cute::make_fragment_like(accum(cute::_, cute::Int<0>{}, cute::Int<0>{}));
                            // The C value mode is (v0,v1,v2).  GMMA ownership
                            // makes the activation scale independent of v1 and
                            // the per-row weight scale independent of v0/v2.
                            // Derive both projections from the MMA partition,
                            // as in the WA FP8 mainloop, and load each distinct
                            // scale once while the stage is resident.
                            auto c_value_shape = cute::shape<0>(scratch);
                            static_assert(cute::rank(c_value_shape) == 3);
                            constexpr int  kCValue0                 = cute::size<0>(decltype(c_value_shape){});
                            constexpr int  kCValue1                 = cute::size<1>(decltype(c_value_shape){});
                            constexpr int  kCValue2                 = cute::size<2>(decltype(c_value_shape){});
                            constexpr bool kPreloadActivationScales = kOpN == 128 && kRestM == 1 && kRestN == 1;
                            auto           activation_scales        = [&] {
                                if constexpr (kPreloadActivationScales) {
                                    return cute::make_tensor<float>(cute::make_layout(
                                        cute::Shape<cute::Int<kCValue0>, cute::Int<kCValue2>, cute::Int<kRestN>>{}));
                                }
                                else {
                                    return cute::make_tensor<float>(cute::make_layout(cute::_1{}));
                                }
                            }();
                            if constexpr (kPreloadActivationScales) {
                                cute::for_each(cute::make_seq<kRestN>{}, [&](auto rest_n) {
                                    cute::for_each(cute::make_seq<kCValue2>{}, [&](auto v2) {
                                        cute::for_each(cute::make_seq<kCValue0>{}, [&](auto v0) {
                                            const auto coord =
                                                tCcC(cute::make_coord(v0, cute::Int<0>{}, v2), cute::Int<0>{}, rest_n);
                                            activation_scales(v0, v2, rest_n) =
                                                sScale(u_pad + cute::get<1>(coord), stage);
                                        });
                                    });
                                });
                            }
                            cute::for_each(cute::make_seq<kRestM>{}, [&](auto rest_m) {
                                auto       a_atom = tCrA(cute::_, rest_m, cute::_);
                                const auto row_lo_coord =
                                    tAcA(cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}),
                                         rest_m,
                                         cute::Int<0>{});
                                const auto output_fragment_coord = cute::idx2crd(
                                    cute::get<0>(row_lo_coord),
                                    cute::make_shape(cute::Int<kOutputFragmentN>{}, cute::Int<kOutputFragments>{}));
                                const int output_fragment = cute::get<1>(output_fragment_coord);
                                CUTE_UNROLL
                                for (int kb = 0; kb < kQparamGroups; ++kb) {
                                    auto packed_regs = cute::make_fragment_like(
                                        tAsPacked(cute::_, cute::Int<0>{}, output_fragment, kb));
                                    cute::copy(packed_copy,
                                               tAsPacked(cute::_, cute::Int<0>{}, output_fragment, kb),
                                               packed_regs);
                                    auto packed_words = cute::coalesce(packed_regs);
                                    auto a_words      = cute::recast<uint32_t>(a_atom(cute::_, kb));
                                    detail::unpack_e2m1x8_to_e4m3x4x2(
                                        packed_words(cute::Int<0>{}),
                                        a_words(cute::make_coord(
                                            cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}))),
                                        a_words(cute::make_coord(
                                            cute::make_coord(cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}))));
                                    detail::unpack_e2m1x8_to_e4m3x4x2(
                                        packed_words(cute::Int<1>{}),
                                        a_words(cute::make_coord(
                                            cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<1>{}))),
                                        a_words(cute::make_coord(
                                            cute::make_coord(cute::Int<0>{}, cute::Int<1>{}, cute::Int<1>{}))));
                                }
                            });

                            auto issue = [&](auto kb, auto rest_m, auto rest_n, auto& frag) {
                                cute::clear(frag);
                                cute::warpgroup_fence_operand(frag);
                                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                                cute::warpgroup_arrive();
                                cute::gemm(
                                    tiled_mma, tCrA(cute::_, rest_m, kb), tCrB(cute::_, rest_n, kb, stage), frag);
                                cute::warpgroup_commit_batch();
                            };
                            auto scale = [&](auto kb, auto rest_m, auto rest_n, auto& src) {
                                cute::warpgroup_fence_operand(src);
                                auto dst = accum(cute::_, rest_m, rest_n);
                                static_assert(kCValue1 == 2);
                                const auto weight_coord0 =
                                    tCcC(cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}),
                                         rest_m,
                                         cute::Int<0>{});
                                const auto weight_coord1 =
                                    tCcC(cute::make_coord(cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}),
                                         rest_m,
                                         cute::Int<0>{});
                                const uint8_t weight_exponent0 = sQparamStage(cute::get<0>(weight_coord0), kb);
                                const uint8_t weight_exponent1 = sQparamStage(cute::get<0>(weight_coord1), kb);
                                if constexpr (kPreloadActivationScales) {
                                    CUTE_UNROLL
                                    for (int v2 = 0; v2 < kCValue2; ++v2) {
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < kCValue1; ++v1) {
                                            CUTE_UNROLL
                                            for (int v0 = 0; v0 < kCValue0; ++v0) {
                                                const auto value_coord = cute::make_coord(v0, v1, v2);
                                                const auto c_coord     = cute::make_coord(value_coord);
                                                dst(c_coord)           = fmaf(src(c_coord),
                                                                    inject_ue8m0_exponent(
                                                                        activation_scales(v0, v2, rest_n),
                                                                        v1 == 0 ? weight_exponent0 : weight_exponent1),
                                                                    dst(c_coord));
                                            }
                                        }
                                    }
                                }
                                else {
                                    CUTE_UNROLL
                                    for (int v2 = 0; v2 < kCValue2; ++v2) {
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < kCValue1; ++v1) {
                                            CUTE_UNROLL
                                            for (int v0 = 0; v0 < kCValue0; ++v0) {
                                                const auto  value_coord = cute::make_coord(v0, v1, v2);
                                                const auto  c_coord     = cute::make_coord(value_coord);
                                                const auto  coord       = tCcC(value_coord, rest_m, rest_n);
                                                const float as          = sScale(u_pad + cute::get<1>(coord), stage);
                                                dst(c_coord) =
                                                    fmaf(src(c_coord),
                                                         inject_ue8m0_exponent(
                                                             as, v1 == 0 ? weight_exponent0 : weight_exponent1),
                                                         dst(c_coord));
                                            }
                                        }
                                    }
                                }
                            };

                            cute::warpgroup_fence_operand(tCrA);
                            cute::for_each(cute::make_seq<kQparamGroups>{}, [&](auto kb) {
                                cute::for_each(cute::make_seq<kRestM>{}, [&](auto rest_m) {
                                    cute::for_each(cute::make_seq<kRestN>{}, [&](auto rest_n) {
                                        issue(kb, rest_m, rest_n, scratch);
                                        cute::warpgroup_wait<0>();
                                        scale(kb, rest_m, rest_n, scratch);
                                    });
                                });
                            });
                            cute::warpgroup_fence_operand(tCrA);
                        }
                        pipeline.consumer_release(read_state);
                        ++read_state;
                    }
                    {
                        nv_bfloat16* smem_C =
                            reinterpret_cast<nv_bfloat16*>(reinterpret_cast<char*>(&storage) + kOutputOffset);
                        auto sD = cute::as_position_independent_swizzle_tensor(
                            cute::make_tensor(cute::make_smem_ptr(smem_C), SmemLayoutD{}));
                        using EpiTiledMma = std::conditional_t<kSplitEpiM, WgTiledMma, TiledMma>;
                        EpiTiledMma epi_tiled_mma;
                        auto        tiled_copy_c = cute::make_tiled_copy_C_atom(CopyAtomC{}, epi_tiled_mma);
                        auto        tiled_r2s =
                            cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, cutlass::bfloat16_t>{}, tiled_copy_c);
                        auto thr_r2s  = tiled_r2s.get_slice(kSplitEpiM ? local_tid : mma_tid);
                        auto tRS_rAcc = thr_r2s.retile_S(accum);
                        auto tRS_rD   = cute::make_tensor<cutlass::bfloat16_t>(
                            cute::make_layout(cute::take<0, 3>(cute::shape(thr_r2s.partition_S(sD)))));
                        auto tRS_rAcc_frg = cute::recast<cutlass::Array<float, kFragmentSize>>(tRS_rAcc);
                        auto tRS_rD_frg   = cute::recast<cutlass::Array<cutlass::bfloat16_t, kFragmentSize>>(tRS_rD);
                        constexpr int kMmaTileN =
                            cute::size<0>(typename Traits::TileShape{}) / cute::size<1>(decltype(tRS_rAcc){});
                        constexpr int kMmaTileM = (kSplitEpiM ? kWgM : cute::size<1>(typename Traits::TileShape{}))
                                                  / cute::size<2>(decltype(tRS_rAcc){});
                        static_assert(kMmaTileN == kEpiN);
                        static_assert(kMmaTileM % kEpiM == 0);
                        auto r2s_value_layout =
                            cute::make_layout(cute::make_shape(cute::size(tRS_rD_frg), cute::Int<kMmaTileM / kEpiM>{}));
                        auto epi_pass_layout =
                            cute::make_layout(cute::make_shape(cute::Int<kEpiStripsM>{}, cute::Int<kEpiStripsN>{}),
                                              cute::make_stride(cute::Int<kEpiStripsN>{}, cute::_1{}));
                        auto store_offset_m_layout = cute::make_layout(
                            cute::make_shape(cute::Int<kEpiPlanes>{}, cute::Int<kEpiStripsM>{}, cute::_1{}),
                            cute::make_stride(cute::Int<kWgM>{}, cute::Int<kEpiM>{}, cute::Int<kTmaStoreM>{}));
                        auto store_offset_n_layout =
                            cute::make_layout(cute::make_shape(cute::Int<kEpiStripsN>{}, cute::Int<kTmaStoreCountN>{}),
                                              cute::make_stride(cute::Int<kEpiN>{}, cute::Int<kTmaStoreN>{}));
                        auto store_tile_layout =
                            cute::make_layout(cute::make_shape(cute::Int<kTmaStoreCountN>{}, cute::Int<kEpiPlanes>{}));
                        constexpr int kTmaStoreWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
                        const int     tma_store_warp      = int(threadIdx.x) / WARP_SIZE;
                        const bool    tma_store_leader    = cute::elect_one_sync();
                        const int     store_wg            = tma_store_warp / kTmaStoreWarpsPerWg;
                        static_assert(kEpiPlanes * kTmaStoreCountN == WARPGROUPS);
                        auto epi_synchronize = [&] {
                            if constexpr (kSplitEpiM) {
                                const int barrier_id = kEpilogueBarrierId + store_wg;
                                named_barrier_arrive_and_wait(WARPGROUP_SIZE, barrier_id);
                            }
                            else {
                                named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            }
                        };

                        CUTE_UNROLL
                        for (int epi_m = 0; epi_m < kEpiStripsM; ++epi_m) {
                            CUTE_UNROLL
                            for (int epi_n = 0; epi_n < kEpiStripsN; ++epi_n) {
                                const int epi_pass = epi_pass_layout(epi_m, epi_n);
                                if (tma_store_leader) {
                                    cute::tma_store_wait<0>();
                                }
                                epi_synchronize();

                                const int mma_n        = epi_n;
                                const int mma_m        = (epi_m * kEpiM) / kMmaTileM;
                                const int epi_m_in_mma = epi_m % (kMmaTileM / kEpiM);
                                const int r2s_v        = r2s_value_layout(0, epi_m_in_mma);
                                const int epi_plane    = kSplitEpiM ? store_wg : 0;
                                auto      sD_epi       = sD(cute::_, cute::_, epi_plane);
                                CUTE_UNROLL
                                for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                    cutlass::Array<cutlass::bfloat16_t, kFragmentSize> dst;
                                    auto src = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                    CUTE_UNROLL
                                    for (int j = 0; j < kFragmentSize; ++j) {
                                        dst[j] = cutlass::bfloat16_t(src[j]);
                                    }
                                    tRS_rD_frg(epi_v) = dst;
                                }

                                auto tRS_sD = thr_r2s.partition_D(sD_epi);
                                cute::copy(tiled_r2s, tRS_rD, tRS_sD);
                                cutlass::arch::fence_view_async_shared();
                                cute::tma_store_fence();
                                epi_synchronize();

                                if (tma_store_leader) {
                                    const int warp_in_wg = tma_store_warp % kTmaStoreWarpsPerWg;
                                    const int first_warp = epi_pass % kTmaStoreWarpsPerWg;
                                    const int tma_m =
                                        (warp_in_wg + kTmaStoreWarpsPerWg - first_warp) % kTmaStoreWarpsPerWg;
                                    if (tma_m == 0) {
                                        const auto store_tile_coord = store_tile_layout.get_flat_coord(store_wg);
                                        const int  tma_n            = cute::get<0>(store_tile_coord);
                                        const int  store_plane      = cute::get<1>(store_tile_coord);
                                        const int  store_m =
                                            tile->offset_m + store_offset_m_layout(store_plane, epi_m, 0);
                                        const int store_n = tile->offset_n + store_offset_n_layout(epi_n, tma_n);
                                        auto      sD_tma  = cute::local_tile(
                                            sD_epi,
                                            cute::make_shape(cute::Int<kTmaStoreN>{}, cute::Int<kTmaStoreM>{}),
                                            cute::make_coord(tma_n, 0));
                                        cute::SM90_TMA_STORE::copy(
                                            &tm_c, cute::raw_pointer_cast(sD_tma.data()), store_n, store_m);
                                    }
                                    cute::tma_store_arrive();
                                }
                            }
                        }
                    }
                }
                else if (tile->is_valid_cluster) {
                    for (int k_tile = 0; k_tile < sched.k_iters_; ++k_tile) {
                        pipeline.consumer_wait(read_state);
                        pipeline.consumer_release(read_state);
                        ++read_state;
                    }
                }
                sched_state.release();
                sched_state.acquire(tile);
            }
            sched_state.release();
            if (cute::elect_one_sync()) {
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(kMathThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        };

        run();
    }
};

}  // namespace turbomind::gemm
