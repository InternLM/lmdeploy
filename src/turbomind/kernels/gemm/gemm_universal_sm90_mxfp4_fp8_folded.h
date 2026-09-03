// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <numeric>
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
namespace detail {

template<int ValuesPerThread>
struct FoldedEpiStsmAtoms {
    static_assert(ValuesPerThread >= 8);
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x8_STSM_T;
};

template<>
struct FoldedEpiStsmAtoms<4> {
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x2_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x4_STSM_T;
};

template<int Multicast, int BoxM, int BoxK, class Element>
__device__ __forceinline__ void load_mxfp4_folded_tma(const cute::TmaDescriptor* desc,
                                                      uint64_t*                  barrier,
                                                      Element*                   smem,
                                                      int                        coord_k,
                                                      int                        coord_m,
                                                      uint16_t                   multicast_mask,
                                                      uint64_t cache_hint =
                                                          uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL))
{
    constexpr int kNumBits = BoxM * BoxK * int(cute::sizeof_bits_v<Element>);
    constexpr int kNumVals = BoxM * BoxK;
    using Aux = cute::AuxTmaParams<cute::Stride<cute::_1, cute::_1>,
                                   cute::Layout<cute::Shape<cute::_1>>,
                                   cute::Swizzle<0, 4, 3>>;
    auto gmem = cute::make_tensor(
        cute::make_inttuple_iter(coord_k, coord_m), cute::Layout<cute::Int<kNumVals>>{});
    auto dst = cute::make_tensor(
        cute::make_smem_ptr(smem), cute::Layout<cute::Int<kNumVals>>{});
    if constexpr (Multicast > 1) {
        using Traits = cute::Copy_Traits<
            cute::SM90_TMA_LOAD_MULTICAST, cute::Int<kNumBits>, Aux>;
        using Atom = cute::Copy_Atom<Traits, Element>;
        Atom atom{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(
            atom.with(desc,
                      *barrier,
                      multicast_mask,
                      cute::TMA::CacheHintSm90(cache_hint)),
            gmem,
            dst);
    }
    else {
        using Traits = cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::Int<kNumBits>, Aux>;
        using Atom = cute::Copy_Atom<Traits, Element>;
        Atom atom{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(atom.with(desc, *barrier, 0, cute::TMA::CacheHintSm90(cache_hint)),
                   gmem,
                   dst);
    }
}

// Packed weights/qparams are nonlinear byte streams.  Grouped conversion
// publishes one StridedPtr per expert, so routing offsets must not be applied.
__device__ __forceinline__ StridedPtr resolve_mxfp4_folded_group_ptr(const MatrixParam& param, int group_idx)
{
    StridedPtr ptr{param.ptr, param.stride};
    if (ptr.stride == 0) {
        reinterpret_cast<uint4&>(ptr) = __ldg(reinterpret_cast<const uint4*>(param.ptr) + group_idx);
    }
    return ptr;
}

__device__ __forceinline__ void copy_mxfp4_folded_tma_desc(CUtensorMap* dst, const CUtensorMap* src, int lane)
{
    constexpr int kWords = int(sizeof(CUtensorMap) / sizeof(uint2));
    if (lane < kWords) {
        reinterpret_cast<uint2*>(dst)[lane] = reinterpret_cast<const uint2*>(src)[lane];
    }
}

__device__ __forceinline__ void replace_mxfp4_folded_tma_addr_dim(
    CUtensorMap* desc,
    void* global_addr,
    int dim_idx,
    int dim,
    uint64_t stride_bytes)
{
    const uint32_t smem_addr = cast_smem_ptr_to_uint(desc);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :
                 : "r"(smem_addr), "l"(global_addr));
    if (dim >= 0) {
        if (dim_idx == 0) {
            asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;"
                         :
                         : "r"(smem_addr), "r"(dim));
        }
        else {
            asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;"
                         :
                         : "r"(smem_addr), "r"(dim));
        }
    }
    if (stride_bytes) {
        replace_tma_global_stride(desc, stride_bytes);
    }
}

__device__ __forceinline__ void publish_mxfp4_folded_tma_desc(CUtensorMap* dst, CUtensorMap* src)
{
    const uint32_t smem_addr = cast_smem_ptr_to_uint(src);
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned "
                 "[%0], [%1], 128;"
                 :
                 : "l"(dst), "r"(smem_addr));
}

template<int N>
__device__ __forceinline__ void
rebase_publish_mxfp4_folded_tma_descs(CUtensorMap*                 dst,
                                      CUtensorMap*                 smem,
                                      Array<const CUtensorMap*, N> templates,
                                      Array<void*, N>              addrs,
                                      Array<int, N>                dims,
                                      Array<int, N>                dim_idxs,
                                      Array<uint64_t, N>           strides,
                                      int                          lane)
{
    CUTE_UNROLL
    for (int i = 0; i < N; ++i) {
        copy_mxfp4_folded_tma_desc(&smem[i], templates[i], lane);
    }
    __syncwarp();
    if (lane == 0) {
        CUTE_UNROLL
        for (int i = 0; i < N; ++i) {
            replace_mxfp4_folded_tma_addr_dim(
                &smem[i], addrs[i], dim_idxs[i], dims[i], strides[i]);
        }
    }
    __syncwarp();
    CUTE_UNROLL
    for (int i = 0; i < N; ++i) {
        publish_mxfp4_folded_tma_desc(&dst[i], &smem[i]);
    }
    __syncwarp();
}

}  // namespace detail

template<int kAlignmentU, Striding kStridingA>
__global__ void __launch_bounds__(32, 1)
prepare_moe_tma_descs_sm90_mxfp4_fp8_folded(const __grid_constant__ CUtensorMap tm_a,
                                     const __grid_constant__ CUtensorMap tm_b,
                                     const __grid_constant__ CUtensorMap tm_v_shift,
                                     const __grid_constant__ CUtensorMap tm_u,
                                     const __grid_constant__ CUtensorMap tm_c,
                                     MatrixParam                         param_A,
                                     MatrixParam                         param_B,
                                     MatrixParam                         param_V,
                                     MatrixParam                         param_U,
                                     MatrixParam                         param_C,
                                     bool                                fuse_silu,
                                     CUtensorMap*                        out,
                                     int*                                offsets,
                                     int                                 M_total)
{
    static_assert(kStridingA == Striding::kBlocked || kStridingA == Striding::kIndexed);
    const int g = int(blockIdx.x);
    const int lane = int(threadIdx.x) & 31;
    const int m0 = param_A.offsets ? __ldg(param_A.offsets + g) : 0;
    const int m1 = param_A.offsets ? __ldg(param_A.offsets + g + 1) : M_total;
    const int M = m1 - m0;
    const int M_desc = M > 0 ? M : 1;
    if (lane == 0) {
        offsets[g] = m0;
        if (g + 1 == gridDim.x) {
            offsets[g + 1] = m1;
        }
    }

    const auto b = detail::resolve_mxfp4_folded_group_ptr(param_B, g);
    const auto v = detail::resolve_mxfp4_folded_group_ptr(param_V, g);
    if constexpr (kStridingA == Striding::kBlocked) {
        constexpr int kNum = 5;
        __shared__ __align__(128) CUtensorMap smem[kNum];
        const auto a = resolve<__nv_fp8_e4m3, Striding::kBlocked>(param_A, g);
        const int beg_u = m0 / kAlignmentU * kAlignmentU;
        const int end_u = round_up(m1, kAlignmentU);
        const auto c_bf16 = resolve<nv_bfloat16, Striding::kBlocked>(param_C, g);
        const auto c_fp8 = resolve<__nv_fp8_e4m3, Striding::kBlocked>(param_C, g);
        Array<const CUtensorMap*, kNum> templates{&tm_a, &tm_b, &tm_v_shift, &tm_u, &tm_c};
        Array<void*, kNum> addrs{a.ptr.ptr,
                                 b.ptr,
                                 v.ptr,
                                 static_cast<float*>(param_U.ptr) + beg_u,
                                 fuse_silu ? c_fp8.ptr.ptr : c_bf16.ptr.ptr};
        Array<int, kNum> dims{M_desc, -1, -1, end_u - beg_u, M_desc};
        Array<int, kNum> dim_idxs{1, 1, 1, 0, 1};
        Array<uint64_t, kNum> strides{
            uint64_t(a.ptr.stride) * sizeof(__nv_fp8_e4m3),
            0,
            0,
            0,
            uint64_t(fuse_silu ? c_fp8.ptr.stride : c_bf16.ptr.stride)
                * (fuse_silu ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16))};
        detail::rebase_publish_mxfp4_folded_tma_descs<kNum>(
            out + g * kNum, smem, templates, addrs, dims, dim_idxs, strides, lane);
    }
    else {
        constexpr int kNum = 3;
        __shared__ __align__(128) CUtensorMap smem[kNum];
        const auto c_bf16 = resolve<nv_bfloat16, Striding::kBlocked>(param_C, g);
        const auto c_fp8 = resolve<__nv_fp8_e4m3, Striding::kBlocked>(param_C, g);
        Array<const CUtensorMap*, kNum> templates{&tm_b, &tm_v_shift, &tm_c};
        Array<void*, kNum> addrs{b.ptr, v.ptr, fuse_silu ? c_fp8.ptr.ptr : c_bf16.ptr.ptr};
        Array<int, kNum> dims{-1, -1, M_desc};
        Array<int, kNum> dim_idxs{1, 1, 1};
        Array<uint64_t, kNum> strides{
            0,
            0,
            uint64_t(fuse_silu ? c_fp8.ptr.stride : c_bf16.ptr.stride)
                * (fuse_silu ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16))};
        detail::rebase_publish_mxfp4_folded_tma_descs<kNum>(
            out + g * kNum, smem, templates, addrs, dims, dim_idxs, strides, lane);
    }
}

// Native SM90 E4M3-K128 x MXFP4-K32 folded mainloop. Public (M,N,K) is
// (tokens,output,K), while WGMMA sees packed weight as RS A and activation as
// descriptor B, hence the hardware tile is (output,tokens,K).
template<Order    Raster,
         int      MulticastA,
         int      MulticastB,
         bool     Grouped,
         Striding StridingA,
         int      TileM,
         int      TileN,
         int      StageCount,
         class    WGLayout,
         int      MmaN,
         int      ProducerRegsTma,
         int      MathRegsTma,
         int      ProducerRegsIndexed,
         int      MathRegsIndexed,
         int      EpilogueTileM,
         int      EpilogueTileN,
         int      EpilogueStages,
         bool     SupportsFusedSilu = false>
struct GemmUniversalSm90MxFp4Fp8Folded {
    using Arch = Sm90;
    using Format = Sm90MxFp4Fp8FoldedFormat;
    static constexpr Order kRasterOrder = Raster;
    static constexpr bool is_grouped_gemm = Grouped;
    static constexpr bool kSupportsFusedSilu = SupportsFusedSilu;
    static constexpr Striding kStridingA = StridingA;
    static constexpr Striding kStridingB = Grouped ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC = Grouped ? Striding::kBlocked : Striding::kFlat;
    static constexpr bool kIndexedGather = StridingA == Striding::kIndexed;
    static constexpr int kMulticastA = MulticastA;
    static constexpr int kMulticastB = MulticastB;
    static constexpr int kMulticastU = Grouped ? 1 : kMulticastA;
    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static_assert(Grouped == (StridingA != Striding::kFlat));
    static_assert(!SupportsFusedSilu || !Grouped || StridingA == Striding::kIndexed,
                  "grouped fused SiLU requires indexed gate/up activations");
    static_assert(kMulticastA == 1 || kMulticastA == 2);
    static_assert(kMulticastB == 1 || kMulticastB == 2);
    static_assert(kClusterSize <= 2);
    static_assert(kClusterSize == 1 || (!Grouped && StridingA == Striding::kFlat));

    static constexpr int TILE_M = TileM;
    static constexpr int TILE_N = TileN;
    static constexpr int TILE_K = 128;
    static constexpr int Stages = StageCount;
    static constexpr int kGroupSize = 32;
    static constexpr int kOutputFragmentN = 64;
    static constexpr int kGroupScaleTableCount = 7;
    using Ta = __nv_fp8_e4m3;
    using Tb = fp4_e2m1_t;
    using Tv = uint8_t;
    using Tc = nv_bfloat16;
    using Tu = float;

    static constexpr int kComputeTileN = kSupportsFusedSilu ? TILE_N / 2 : TILE_N;
    using Traits = GmmaMxFp4Fp8FoldedTraits<kComputeTileN, TILE_M, Stages, WGLayout, MmaN>;
    using TiledMma = typename Traits::TiledMma;
    using WgTiledMma = typename Traits::WgTiledMma;
    using ElementMmaA = typename Traits::ElementA;
    using AtomLayoutMNK = typename Traits::AtomLayoutMNK;
    static constexpr int kAtomM = Traits::kAtomM;
    static constexpr int kAtomN = Traits::kAtomN;
    static constexpr int kRestM = Traits::kRestM;
    static constexpr int kRestN = Traits::kRestN;
    static_assert(kRestN >= 1 && kRestN <= 8,
                  "registered WG1x2 tiles use at most eight residual-N atoms");
    static constexpr int kOpM = Traits::kOpM;
    static constexpr int kOpN = Traits::kOpN;
    static constexpr int kOpK = Traits::kOpK;
    static constexpr int kMathWarpGroups = Traits::kMathWarpgroups;
    static_assert(kMathWarpGroups == 2, "SM90 MXFP4 x FP8 kernels use two math warpgroups");
    static constexpr int WARPGROUPS = kMathWarpGroups;
    static constexpr int WARPGROUP_SIZE = 128;
    static constexpr int kMathThreads = WARPGROUP_SIZE * kMathWarpGroups;
    static constexpr int CTA_SIZE = kMathThreads + WARPGROUP_SIZE;
    static constexpr int kEpilogueBarrierId = 1;
    static constexpr int kProducerBarrierId = 8;
    static_assert(kEpilogueBarrierId + WARPGROUPS <= kProducerBarrierId);
    using GatherCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>, Ta>;
    static constexpr int kGatherVec = GatherCopyAtom::NumValSrc;
    static constexpr int kGatherThreadsK = TILE_K / kGatherVec;
    static constexpr int kGatherVectors = TILE_M * kGatherThreadsK;
    static constexpr int kGatherSlots = cute::ceil_div(kGatherVectors, WARPGROUP_SIZE);
    static_assert(kGatherVec * int(sizeof(Ta)) == 16);

    static constexpr int kProducerRegsTma = ProducerRegsTma;
    static constexpr int kMathRegsTma = MathRegsTma;
    static constexpr int kProducerRegsIndexed = ProducerRegsIndexed;
    static constexpr int kMathRegsIndexed = MathRegsIndexed;
    static_assert(kProducerRegsTma % 8 == 0 && kMathRegsTma % 8 == 0);
    static_assert(kMathWarpGroups != 1 || kProducerRegsTma + kMathRegsTma <= 512);
    static_assert(kMathWarpGroups != 2 || kProducerRegsTma + 2 * kMathRegsTma <= 504);
    static_assert(kMathWarpGroups != 3 || kProducerRegsTma + 3 * kMathRegsTma <= 512);

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;
    using ClusterShape = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;
    using Scheduler = TileScheduler<Raster, Cluster, true, true, TILE_M, TILE_N, Stages, Grouped>;
    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using MainloopState = typename MainloopPipeline::PipelineState;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;

    using PackedRecordLayout = decltype(Traits::packed_layout_a_mk());
    static constexpr int kPackedElementsPerRecord = cute::cosize_v<PackedRecordLayout>;
    static constexpr int kPackedInnerWords =
        kPackedElementsPerRecord * cute::sizeof_bits_v<cute::uint4_t>
        / cute::sizeof_bits_v<uint32_t>;
    static constexpr int kPackedTmaInnerWords = 256;
    static constexpr int kPackedTmaParts = kPackedInnerWords / kPackedTmaInnerWords;
    static constexpr int kOutputFragments = TILE_N / kOutputFragmentN;
    static constexpr int kK32FragmentsPerStage = Traits::kKBlocksPerStage;
    static constexpr int kQparamShiftValuesPerFragment = kOutputFragmentN * kK32FragmentsPerStage;
    static constexpr int kQparamBaseValuesPerFragment = 16;
    static constexpr int kPackedWordsStage =
        kPackedInnerWords * kOutputFragments * kK32FragmentsPerStage;
    static constexpr int kPackedWeightStageBytes = kPackedWordsStage * sizeof(uint32_t);
    static constexpr int kQparamShiftValuesStage = kQparamShiftValuesPerFragment * kOutputFragments;
    static constexpr int kQparamBaseValuesStage = kQparamBaseValuesPerFragment * kOutputFragments;
    static constexpr int kQparamShiftStageBytes = kQparamShiftValuesStage * sizeof(Tv);
    static constexpr int kMxScaleStageBytes =
        (kQparamShiftValuesStage + kQparamBaseValuesStage) * sizeof(Tv);
    static constexpr int kActivationStageBytes = TILE_M * TILE_K * sizeof(Ta);
    static constexpr int kAlignmentU = 16 / sizeof(float);
    static constexpr int kBoxU = TILE_M + (Grouped ? kAlignmentU : 0);
    static constexpr int kTmaCountM = cute::ceil_div(TILE_M / kMulticastA, 256);
    static constexpr int kTmaBoxM = TILE_M / (kMulticastA * kTmaCountM);
    static constexpr int kUStageStride = round_up<int>(kBoxU, 128);
    static constexpr int kActivationScaleStageBytes = kBoxU * sizeof(float);
    static constexpr bool kFusedTileShape = TILE_N == 256 && kAtomN == 1 && kRestN == 1
                                            && (kAtomM == 1 || kAtomM == 2);
    static constexpr int kFusedOutputN = TILE_N / 2;
    static constexpr int kFusedScratchElems = kSupportsFusedSilu ? 4 * kMathWarpGroups * TILE_M : 1;
    static constexpr int kSiluExchangeElems = kSupportsFusedSilu ? kFusedOutputN * TILE_M : 1;
    static_assert(!kSupportsFusedSilu || kFusedTileShape);
    static_assert(!kSupportsFusedSilu || kRestM == 2 / kAtomM);
    static_assert(kPackedElementsPerRecord == kOutputFragmentN * kOpK);
    static_assert(TILE_M % kMulticastA == 0);
    static_assert(kOutputFragments % kMulticastB == 0);
    static_assert(kBoxU % kMulticastU == 0);
    static_assert(TILE_M % (kMulticastA * kTmaCountM) == 0);
    static_assert(kTmaBoxM <= 256);
    static_assert(kMulticastU == 1 || kActivationScaleStageBytes / kMulticastU % 128 == 0);
    static_assert(kPackedWeightStageBytes == TILE_N * TILE_K / 2);
    static_assert(kPackedTmaParts == 1);
    static_assert(kQparamShiftValuesPerFragment == 256);
    static_assert(kQparamShiftValuesPerFragment + kQparamBaseValuesPerFragment == Format::kQparamValuesFragment);
    static_assert(kMxScaleStageBytes == kOutputFragments * 272);

    static constexpr int  kEpilogueTileM  = std::gcd(32, TILE_M);
    static constexpr int  kEpilogueTileN  = 128;
    static constexpr int  kEpilogueStages = EpilogueStages;
    static constexpr bool kSplitEpiM      = kAtomN == 2;
    static constexpr int  kWgM            = TILE_M / kAtomN;
    static constexpr int  kEpiN           = kEpilogueTileN;
    static constexpr int  kEpiM           = kEpilogueTileM;
    static constexpr int  kEpiPlanes      = kSplitEpiM ? kAtomN : 1;
    static constexpr int  kTmaStoreN      = 64;
    static constexpr int  kTmaStoreM      = kEpiM <= 256 ? kEpiM : 64;
    static constexpr int  kTmaStoreCountM = kEpiM / kTmaStoreM;
    static constexpr int  kEpiThreads     = kSplitEpiM ? WARPGROUP_SIZE : kMathThreads;
    static constexpr int  kFragmentSize   = kEpiM * kEpiN / kEpiThreads;
    static constexpr int  kEpiStripsM     = kWgM / kEpiM;
    static constexpr int  kEpiStripsN     = TILE_N / kEpiN;
    static constexpr int  kEpiPasses      = kEpiStripsM * kEpiStripsN;
    static_assert(TILE_M % kEpiM == 0);
    static_assert(TILE_N % kEpiN == 0);
    static_assert(kWgM % kEpiM == 0);
    static_assert(EpilogueTileM == kEpilogueTileM);
    static_assert(EpilogueTileN == kEpilogueTileN);
    static_assert(kEpiM % kTmaStoreM == 0);
    static_assert(kTmaStoreM <= 256);
    static_assert(kEpiN % kTmaStoreN == 0);
    static_assert(kFragmentSize >= 1);
    static_assert(kEpilogueStages == 1 || kEpilogueStages == 2);
    static_assert(1 <= kEpilogueStages && kEpilogueStages <= kEpiPasses);
    using SmemLayoutAtomD = decltype(
        gmma_ss_smem_selector<cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::Int<kEpiN>, cute::Int<kEpiM>>());
    using SmemLayoutDPlane =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::_1{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    static constexpr int kEpiStageElems = cute::cosize_v<SmemLayoutDPlane> * kEpiPlanes;
    static constexpr int kEpiSmemSlices = kEpiPlanes * kEpilogueStages;
    using SmemLayoutD                   = decltype(
        cute::tile_to_shape(SmemLayoutAtomD{},
                            cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::Int<kEpiSmemSlices>{}),
                            cute::Step<cute::_2, cute::_1, cute::_3>{}));
    using EpiStageLayout = cute::Layout<cute::Shape<cute::Int<kEpiPlanes>, cute::Int<kEpilogueStages>>,
                                        cute::Stride<cute::_1, cute::Int<kEpiPlanes>>>;
    static_assert(cute::cosize_v<SmemLayoutD> == kEpiStageElems * kEpilogueStages);
    static constexpr int kCValsPerThread = kAtomN == 2 ? TILE_M / 4 : TILE_M / 2;
    static_assert(kCValsPerThread >= 4);
    using CopyAtomC = typename detail::FoldedEpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyAtomC;
    using CopyOpR2S = typename detail::FoldedEpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyOpR2S;

    using PackedCtaTile = cute::Shape<cute::Int<kPackedTmaInnerWords>,
                                      cute::Int<kPackedTmaParts * kOutputFragments>,
                                      cute::Int<kK32FragmentsPerStage>>;
    using PackedSmemLayout = decltype(cute::make_layout(PackedCtaTile{}));
    using PackedPipelineLayout = cute::Layout<
        cute::Shape<cute::Int<cute::cosize_v<PackedSmemLayout>>, cute::Int<Stages>>>;
    using QparamShiftCtaTile = cute::Shape<cute::Int<kQparamShiftValuesPerFragment>,
                                           cute::Int<kOutputFragments>,
                                           cute::_1>;
    using QparamShiftSmemLayout = decltype(cute::make_layout(QparamShiftCtaTile{}));
    using QparamShiftPipelineLayout = cute::Layout<
        cute::Shape<cute::Int<cute::cosize_v<QparamShiftSmemLayout>>, cute::Int<Stages>>>;
    using QparamBaseCtaTile = cute::Shape<cute::Int<kQparamBaseValuesPerFragment>,
                                          cute::Int<kOutputFragments>,
                                          cute::_1>;
    using QparamBaseSmemLayout = decltype(cute::make_layout(QparamBaseCtaTile{}));
    using QparamBasePipelineLayout = cute::Layout<
        cute::Shape<cute::Int<cute::cosize_v<QparamBaseSmemLayout>>, cute::Int<Stages>>>;
    using SmemLayoutB = typename Traits::SmemLayoutB;
    using TmaActivation = CUtensorMap;

    static auto MakeTmaPacked(void* ptr, int n, int k)
    {
        const int out_fragments = n / kOutputFragmentN;
        const int k32_fragments = k / kOpK;
        auto layout = cute::make_layout(
            cute::make_shape(
                cute::Int<kPackedTmaInnerWords>{},
                out_fragments * kPackedTmaParts,
                k32_fragments),
            cute::make_stride(cute::_1{},
                              cute::Int<kPackedTmaInnerWords>{},
                              int64_t{kPackedInnerWords} * out_fragments));
        auto gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint32_t*>(ptr)), layout);
        using CopyOp = std::conditional_t<kMulticastB == 1,
                                          cute::SM90_TMA_LOAD,
                                          cute::SM90_TMA_LOAD_MULTICAST>;
        return cute::make_tma_copy(
            CopyOp{}, gmem, PackedSmemLayout{}, PackedCtaTile{}, cute::Int<kMulticastB>{});
    }
    using TmaPacked = decltype(MakeTmaPacked(nullptr, TILE_N, TILE_K));

    static auto MakeTmaQparamShift(void* ptr, int n, int k)
    {
        const int out_fragments = n / kOutputFragmentN;
        const int k128_groups = k / TILE_K;
        auto layout = cute::make_layout(
            cute::make_shape(cute::Int<kQparamShiftValuesPerFragment>{}, out_fragments, k128_groups),
            cute::make_stride(cute::_1{},
                              cute::Int<kQparamShiftValuesPerFragment>{},
                              int64_t{kQparamShiftValuesPerFragment} * out_fragments));
        auto gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<Tv*>(ptr)), layout);
        using CopyOp = std::conditional_t<kMulticastB == 1,
                                          cute::SM90_TMA_LOAD,
                                          cute::SM90_TMA_LOAD_MULTICAST>;
        return cute::make_tma_copy(
            CopyOp{}, gmem, QparamShiftSmemLayout{}, QparamShiftCtaTile{}, cute::Int<kMulticastB>{});
    }
    using TmaQparamShift = decltype(MakeTmaQparamShift(nullptr, TILE_N, TILE_K));
    using TmaQparam = TmaQparamShift;

    static TmaQparam MakeTmaQparam(void* ptr, int n, int k)
    {
        return MakeTmaQparamShift(ptr, n, k);
    }
    static_assert(cute::size(typename TmaPacked::Traits::SrcLayout{})
                  == kPackedWeightStageBytes * 8);
    static_assert(cute::size(typename TmaQparamShift::Traits::SrcLayout{})
                  == kQparamShiftValuesStage * sizeof(Tv) * 8);

    struct alignas(1024) SharedStorage {
        cute::array_aligned<uint32_t, kPackedWordsStage * Stages, 128> A;
        cute::array_aligned<Ta, cute::cosize_v<SmemLayoutB>, 128> B;
        cute::array_aligned<Tc, kSupportsFusedSilu ? 1 : cute::cosize_v<SmemLayoutD>, 1024> D;
        cute::array_aligned<Tv, kQparamShiftValuesStage * Stages, 128>                      V;
        cute::array_aligned<Tv, kQparamBaseValuesStage * Stages, 128>                       W;
        cute::array_aligned<float, kUStageStride * Stages, 128>                             U;
        cute::array_aligned<detail::E2m1E4m3ScaleTable, kGroupScaleTableCount, 8>           group_scale_tables;
        cute::array_aligned<float, kFusedScratchElems, 128>  fused_amax_scratch;
        cute::array_aligned<float, kSiluExchangeElems, 1024> silu_exchange;
        PipelineStorage                                      pipeline;
        typename Scheduler::Storage                          sched;
        StridedPtr                                                                          gather_A;
        const int*                                                                          gather_idxs;
        int                                                                                 gather_alive;
        int                                                                                 gather_k_iters;
        int                                                                                 gather_M_group;
        int gather_offset_m;
        int gather_group_idx;
        int gather_out_fragment;
    };
    static constexpr int kOutputOffset = round_up<int>(sizeof(SharedStorage), 1024);
    static constexpr int kFp8EpilogueBytes = TILE_M * kFusedOutputN * sizeof(__nv_fp8_e4m3);
    static constexpr int kEpilogueBytes = kSupportsFusedSilu ? kFp8EpilogueBytes : 0;
    static constexpr int kSmemSize = kOutputOffset + kEpilogueBytes;
    static_assert(cute::cosize_v<SmemLayoutB> == Stages * TILE_M * TILE_K);
    static_assert(kSmemSize <= (228 << 10));

    static constexpr int kDescA = 0;
    static constexpr int kDescB = kIndexedGather ? 0 : 1;
    static constexpr int kDescV = kDescB + 1;
    static constexpr int kDescU = kDescV + 1;
    static constexpr int kDescC = kIndexedGather ? kDescV + 1 : kDescU + 1;
    static constexpr int kTmaDescNum = Grouped ? kDescC + 1 : 1;

    static int* PrepareTmaDescs(const CUtensorMap& tm_a,
                                const CUtensorMap& tm_b,
                                const CUtensorMap& tm_v_shift,
                                const CUtensorMap& tm_u,
                                const CUtensorMap& tm_c,
                                const MatrixParam& param_A,
                                const MatrixParam& param_B,
                                const MatrixParam& param_V,
                                const MatrixParam& param_U,
                                const MatrixParam& param_C,
                                bool               fuse_silu,
                                CUtensorMap*       out,
                                int                num_groups,
                                int                M,
                                cudaStream_t       stream)
    {
        if constexpr (!Grouped) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_moe_tma_descs_sm90_mxfp4_fp8_folded<kAlignmentU, kStridingA>
            <<<num_groups, 32, 0, stream>>>(
                tm_a,
                tm_b,
                tm_v_shift,
                tm_u,
                tm_c,
                param_A,
                param_B,
                param_V,
                param_U,
                param_C,
                fuse_silu,
                out,
                offsets,
                M);
        return offsets;
    }

    __device__ void operator()(const TmaActivation& tm_a,
                               const TmaPacked& tm_b,
                               const TmaQparam& tm_v,
                               const CUtensorMap& tm_u,
                               const CUtensorMap& tm_c,
                               const MatrixParam& param_A,
                               const MatrixParam& param_B,
                               const MatrixParam& param_V,
                               const MatrixParam& param_U,
                               const MatrixParam& param_C,
                               const MatrixParam& param_W,
                               bool fuse_silu,
                               Scheduler sched,
                               CUtensorMap* tensormap_buf,
                               char* smem_buf)
    {
        if constexpr (kSupportsFusedSilu) {
            assert(fuse_silu);
        }
        else {
            assert(!fuse_silu);
        }
        (void)param_W;
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        const int wg_idx = cutlass::canonical_warp_group_idx();
        const int warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const int lane = threadIdx.x % 32;
        auto* group_scale_tables = storage.group_scale_tables.data();
        if (threadIdx.x < kGroupScaleTableCount) {
            group_scale_tables[threadIdx.x] =
                detail::make_e2m1_e4m3_scale_table(threadIdx.x);
        }
        if (threadIdx.x == 0) {
            sched.init_dyanmic(storage.sched, kClusterSize * (kMathWarpGroups * 4 + 1));
        }
        typename MainloopPipeline::Params params{};
        params.transaction_bytes = kIndexedGather ? kPackedWeightStageBytes + kQparamShiftStageBytes :
                                                    kPackedWeightStageBytes + kQparamShiftStageBytes
                                                        + kActivationStageBytes + kActivationScaleStageBytes;
        params.num_consumers = kMathThreads;
        params.num_producers = kIndexedGather ? 1 + WARPGROUP_SIZE : 2;
        params.initializing_warp = 0;
        if (wg_idx == kMathWarpGroups) {
            params.role = kIndexedGather ? MainloopPipeline::ThreadCategory::Producer :
                                           (warp_in_wg == 0 ? MainloopPipeline::ThreadCategory::Producer :
                                                              MainloopPipeline::ThreadCategory::NonParticipant);
            params.is_leader = warp_in_wg == 0 && lane == 0;
        }
        else {
            params.role = MainloopPipeline::ThreadCategory::Consumer;
            params.is_leader = 0;
        }
        MainloopPipeline pipeline(storage.pipeline, params, ClusterShape{});
        if (threadIdx.x == 0) {
            cutlass::arch::fence_view_async_shared();
        }
        if constexpr (kClusterSize > 1) {
            cute::cluster_sync();
        }
        else {
            __syncthreads();
        }
        if (wg_idx == kMathWarpGroups) {
            if constexpr (kIndexedGather) {
                cutlass::arch::warpgroup_reg_dealloc<kProducerRegsIndexed>();
                run_producer_indexed(
                    tm_b, tm_v, param_A, param_V, param_U, sched, tensormap_buf, storage, pipeline);
            }
            else {
                cutlass::arch::warpgroup_reg_dealloc<kProducerRegsTma>();
                run_producer_tma(
                    tm_a, tm_b, tm_v, tm_u, param_A, param_B, param_V, param_U, sched, tensormap_buf, storage, pipeline);
            }
        }
        else {
            if constexpr (kIndexedGather) {
                cutlass::arch::warpgroup_reg_alloc<kMathRegsIndexed>();
            }
            else {
                cutlass::arch::warpgroup_reg_alloc<kMathRegsTma>();
            }
            run_consumer(tm_c, param_C, param_W, fuse_silu, sched, tensormap_buf, storage, pipeline);
        }
    }

private:
    __device__ static void run_producer_indexed(const TmaPacked& tm_b,
                                                const TmaQparam& tm_v,
                                                const MatrixParam& param_A,
                                                const MatrixParam& param_V,
                                                const MatrixParam& param_U,
                                                Scheduler sched,
                                                CUtensorMap* tensormap_buf,
                                                SharedStorage& storage,
                                                MainloopPipeline& pipeline)
    {
        static_assert(kIndexedGather);
        const int warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const int lane = int(threadIdx.x) % WARP_SIZE;
        const int prod_tid = int(threadIdx.x) - kMathThreads;
        const bool cta_0 = cute::block_id_in_cluster().x == 0;
        MainloopState write_state = cutlass::make_producer_start_state<MainloopPipeline>();
        auto sched_state = sched.init_consumer(storage.sched);
        auto producer_sched_state = sched.init_producer(storage.sched);
        const bool elected = warp_in_wg == 0 ? cute::elect_one_sync() : false;
        const int k_iters = sched.k_iters_;
        const int out_fragments = sched.gemm_shape().y / kOutputFragmentN;
        const int out_tiles = (out_fragments + kOutputFragments - 1) / kOutputFragments;

        typename Scheduler::Tile* tile;
        while (true) {
            if (warp_in_wg == 0) {
                if (cta_0) {
                    (void)producer_sched_state.next();
                }
                const bool alive = sched_state.acquire(tile);
                if (lane == 0) {
                    MatrixData a{{param_A.ptr, param_A.stride}, param_A.idxs};
                    storage.gather_alive = alive ? 1 : 0;
                    storage.gather_k_iters = 0;
                    storage.gather_M_group = 0;
                    storage.gather_offset_m = 0;
                    storage.gather_group_idx = 0;
                    storage.gather_out_fragment = 0;
                    if (alive && tile->is_valid_cluster) {
                        a = resolve<Ta, Striding::kIndexed>(param_A, tile->group_idx);
                        storage.gather_k_iters = k_iters;
                        storage.gather_M_group = tile->m1 - tile->m0;
                        storage.gather_offset_m = tile->offset_m;
                        storage.gather_group_idx = tile->group_idx;
                        storage.gather_out_fragment = tile->offset_n / kOutputFragmentN;
                    }
                    storage.gather_A = a.ptr;
                    storage.gather_idxs = a.idxs;
                }
                __syncwarp();
            }

            named_barrier_arrive_and_wait(WARPGROUP_SIZE, kProducerBarrierId);
            if (storage.gather_alive == 0) {
                break;
            }

            const int tile_k_iters = storage.gather_k_iters;
            const int out_fragment = storage.gather_out_fragment;
            const int group_idx = storage.gather_group_idx;
            const int packed_m0 = storage.gather_offset_m;
            const int row_count = storage.gather_M_group - packed_m0;
            const Ta* act_gmem = static_cast<const Ta*>(storage.gather_A.ptr);
            const int ldA = storage.gather_A.stride;
            const int* idxs = storage.gather_idxs;
            const int group_m0 = sched.offsets_ ? __ldg(sched.offsets_ + group_idx) : 0;
            const int u_pad = group_m0 % kAlignmentU;
            const int activation_scale_ld = param_U.stride;
            const auto qparam_ptr = detail::resolve_mxfp4_folded_group_ptr(param_V, group_idx);
            const Tv* base_gmem = static_cast<const Tv*>(qparam_ptr.ptr)
                                  + size_t(out_fragments) * tile_k_iters
                                        * kQparamShiftValuesPerFragment;

            auto gAct = cute::make_tensor(
                cute::make_gmem_ptr(act_gmem),
                cute::make_shape(sched.gemm_shape().x, tile_k_iters * TILE_K),
                cute::make_stride(ldA, cute::_1{}));
            auto gScale = cute::make_tensor(
                cute::make_gmem_ptr(static_cast<const float*>(param_U.ptr)),
                cute::make_shape(sched.gemm_shape().x, tile_k_iters),
                cute::make_stride(cute::_1{}, activation_scale_ld));
            auto sPackedPipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.A.data()), PackedPipelineLayout{});
            auto sQparamShiftPipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.V.data()), QparamShiftPipelineLayout{});
            auto sQparamBasePipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.W.data()), QparamBasePipelineLayout{});
            auto sActivation = cute::make_tensor(
                cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
            auto sScale = cute::make_tensor(
                cute::make_smem_ptr(storage.U.data()),
                cute::make_layout(cute::Shape<cute::Int<kUStageStride>, cute::Int<Stages>>{}));

            constexpr int kLoadPasses = kSupportsFusedSilu ? 2 : 1;
            CUTE_UNROLL
            for (int load_pass = 0; load_pass < kLoadPasses; ++load_pass) {
            for (int k_tile = 0; k_tile < tile_k_iters; ++k_tile) {
                pipeline.producer_acquire(write_state);
                auto* bar = pipeline.producer_get_barrier(write_state);
                const int stage = write_state.index();
                if (warp_in_wg == 0 && elected) {
                    const cute::TmaDescriptor* Bdesc = tensormap_buf + group_idx * kTmaDescNum + kDescB;
                    const cute::TmaDescriptor* Vdesc = tensormap_buf + group_idx * kTmaDescNum + kDescV;
                    auto packed_gmem = tm_b.get_tma_tensor(
                        cute::make_shape(cute::Int<kPackedTmaInnerWords>{},
                                         out_fragments * kPackedTmaParts,
                                         k_iters * kK32FragmentsPerStage));
                    auto packed_tiles = cute::flat_divide(packed_gmem, PackedCtaTile{});
                    auto packed_cta = tm_b.get_slice(0);
                    auto packed_part = packed_cta.partition_S(packed_tiles);
                    auto packed_src = cute::group_modes<1, cute::rank(packed_part)>(packed_part);
                    auto packed_stage = sPackedPipeline(cute::_, stage);
                    auto packed_smem = cute::make_tensor(packed_stage.data(), PackedSmemLayout{});
                    auto packed_dst_part = packed_cta.partition_D(packed_smem);
                    auto packed_dst = cute::group_modes<1, cute::rank(packed_dst_part)>(packed_dst_part);
                    cute::copy(tm_b.with(Bdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                               packed_src(cute::_, out_fragment / kOutputFragments + out_tiles * k_tile),
                               packed_dst(cute::_, 0));

                    auto shift_gmem = tm_v.get_tma_tensor(
                        cute::make_shape(cute::Int<kQparamShiftValuesPerFragment>{}, out_fragments, k_iters));
                    auto shift_tiles = cute::flat_divide(shift_gmem, QparamShiftCtaTile{});
                    auto shift_cta = tm_v.get_slice(0);
                    auto shift_part = shift_cta.partition_S(shift_tiles);
                    auto shift_src = cute::group_modes<1, cute::rank(shift_part)>(shift_part);
                    auto shift_stage = sQparamShiftPipeline(cute::_, stage);
                    auto shift_smem = cute::make_tensor(shift_stage.data(), QparamShiftSmemLayout{});
                    auto shift_dst_part = shift_cta.partition_D(shift_smem);
                    auto shift_dst = cute::group_modes<1, cute::rank(shift_dst_part)>(shift_dst_part);
                    cute::copy(tm_v.with(Vdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                               shift_src(cute::_, out_fragment / kOutputFragments + out_tiles * k_tile),
                               shift_dst(cute::_, 0));

                    auto base_stage = sQparamBasePipeline(cute::_, stage);
                    CUTE_UNROLL
                    for (int fragment = 0; fragment < kOutputFragments; ++fragment) {
                        const int global_fragment = out_fragment + fragment;
                        const bool pred = global_fragment < out_fragments;
                        const Tv* src = base_gmem
                                        + size_t(k_tile * out_fragments + global_fragment)
                                              * kQparamBaseValuesPerFragment;
                        Tv* dst = &base_stage(fragment * kQparamBaseValuesPerFragment);
                        cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
                            *reinterpret_cast<const uint4*>(src), *reinterpret_cast<uint4*>(dst), pred);
                    }
                }

                auto sB_stage = sActivation(cute::_, cute::_, stage);
                CUTE_UNROLL
                for (int slot = 0; slot < kGatherSlots; ++slot) {
                    const int vector_idx = prod_tid + slot * WARPGROUP_SIZE;
                    const bool slot_valid = vector_idx < kGatherVectors;
                    const int tile_m = slot_valid ? vector_idx / kGatherThreadsK : 0;
                    const int tile_k_offset = slot_valid ? vector_idx % kGatherThreadsK * kGatherVec : 0;
                    const int packed_row = packed_m0 + tile_m;
                    const bool pred = slot_valid && tile_m < row_count;
                    const int source_row = (idxs && pred) ? __ldg(idxs + packed_row) : packed_row;
                    const Ta* src = &gAct(source_row, k_tile * TILE_K + tile_k_offset);
                    Ta* dst = &sB_stage(tile_m, tile_k_offset);
                    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(
                        *reinterpret_cast<const uint4*>(src), *reinterpret_cast<uint4*>(dst), pred);
                    if (slot_valid && tile_k_offset == 0) {
                        const float* scale_src = &gScale(source_row, k_tile);
                        float* scale_dst = &sScale(u_pad + tile_m, stage);
                        cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
                            *reinterpret_cast<const uint32_t*>(scale_src),
                            *reinterpret_cast<uint32_t*>(scale_dst),
                            pred);
                    }
                }
                cutlass::arch::cpasync_barrier_arrive_noinc(bar);
                ++write_state;
            }
            }
            if (warp_in_wg == 0) {
                sched_state.release();
            }
        }

        if (warp_in_wg == 0) {
            sched_state.release();
            if (cta_0) {
                sched.tail(producer_sched_state);
            }
            if (elected) {
                pipeline.producer_tail(write_state);
            }
        }
    }

    __device__ static void run_producer_tma(const TmaActivation& tm_a,
                                        const TmaPacked& tm_b,
                                        const TmaQparam& tm_v,
                                        const CUtensorMap& tm_u,
                                        const MatrixParam& param_A,
                                        const MatrixParam& param_B,
                                        const MatrixParam& param_V,
                                        const MatrixParam& param_U,
                                        Scheduler sched,
                                        CUtensorMap* tensormap_buf,
                                        SharedStorage& storage,
                                        MainloopPipeline& pipeline)
    {
        const int warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const bool cta_0 = cute::block_id_in_cluster().x == 0;
        Cluster cluster(cute::block_id_in_cluster().x);
        const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
        const uint16_t mask_A = cluster.mask_m();
        const uint16_t mask_B = cluster.mask_n();
        if (warp_in_wg == 0) {
            MainloopState write_state = cutlass::make_producer_start_state<MainloopPipeline>();
            auto sched_state = sched.init_consumer(storage.sched);
            const bool elected = cute::elect_one_sync();
            const int k_iters = sched.k_iters_;
            const int out_fragments = sched.gemm_shape().y / kOutputFragmentN;
            const int out_tiles = (out_fragments + kOutputFragments - 1) / kOutputFragments;
            auto packed_gmem = tm_b.get_tma_tensor(
                cute::make_shape(cute::Int<kPackedTmaInnerWords>{},
                                 out_fragments * kPackedTmaParts,
                                 k_iters * kK32FragmentsPerStage));
            auto packed_tiles = cute::flat_divide(packed_gmem, PackedCtaTile{});
            auto packed_cta = [&] {
                if constexpr (kMulticastB == 1) {
                    return tm_b.get_slice(cute::Int<0>{});
                }
                else {
                    return tm_b.get_slice(cluster.cta_m());
                }
            }();
            auto packed_part = packed_cta.partition_S(packed_tiles);
            auto packed_src = cute::group_modes<1, cute::rank(packed_part)>(packed_part);
            auto shift_gmem = tm_v.get_tma_tensor(
                cute::make_shape(cute::Int<kQparamShiftValuesPerFragment>{}, out_fragments, k_iters));
            auto shift_tiles = cute::flat_divide(shift_gmem, QparamShiftCtaTile{});
            auto shift_cta = [&] {
                if constexpr (kMulticastB == 1) {
                    return tm_v.get_slice(cute::Int<0>{});
                }
                else {
                    return tm_v.get_slice(cluster.cta_m());
                }
            }();
            auto shift_part = shift_cta.partition_S(shift_tiles);
            auto shift_src = cute::group_modes<1, cute::rank(shift_part)>(shift_part);
            auto sPackedPipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.A.data()), PackedPipelineLayout{});
            auto sQparamShiftPipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.V.data()), QparamShiftPipelineLayout{});
            auto sQparamBasePipeline = cute::make_tensor(
                cute::make_smem_ptr(storage.W.data()), QparamBasePipelineLayout{});
            auto sScale = cute::make_tensor(
                cute::make_smem_ptr(storage.U.data()),
                cute::make_layout(cute::Shape<cute::Int<kUStageStride>, cute::Int<Stages>>{}));
            typename Scheduler::Tile* tile;
            while (sched_state.acquire(tile)) {
                if (tile->is_valid_cluster && elected) {
                    const cute::TmaDescriptor* Adesc = &tm_a;
                    const CUtensorMap* Udesc = &tm_u;
                    const cute::TmaDescriptor* Bdesc = tm_b.get_tma_descriptor();
                    const cute::TmaDescriptor* Vdesc = tm_v.get_tma_descriptor();
                    if constexpr (Grouped) {
                        CUtensorMap* descs = tensormap_buf + tile->group_idx * kTmaDescNum;
                        Adesc = &descs[kDescA];
                        Bdesc = &descs[kDescB];
                        Vdesc = &descs[kDescV];
                        Udesc = &descs[kDescU];
                    }
                    const int out_fragment = tile->offset_n / kOutputFragmentN;
                    const int out_tile = out_fragment / kOutputFragments;
                    int group_idx = 0;
                    if constexpr (Grouped) {
                        group_idx = tile->group_idx;
                    }
                    const auto qparam_ptr = detail::resolve_mxfp4_folded_group_ptr(param_V, group_idx);
                    const Tv* base_gmem = static_cast<const Tv*>(qparam_ptr.ptr)
                                          + size_t(out_fragments) * k_iters
                                                * kQparamShiftValuesPerFragment;
                    constexpr int kLoadPasses = kSupportsFusedSilu ? 2 : 1;
                    CUTE_UNROLL
                    for (int load_pass = 0; load_pass < kLoadPasses; ++load_pass) {
                    GmemIteratorSm90<kMulticastU> gmem_U{
                        Udesc, {tile->offset_m + mc_offset_m, 0}, {0, 1}};
                    for (int k_tile = 0; k_tile < k_iters; ++k_tile) {
                        pipeline.producer_acquire(write_state);
                        auto* bar = pipeline.producer_get_barrier(write_state);
                        const int stage = write_state.index();
                        auto packed_stage = sPackedPipeline(cute::_, stage);
                        auto packed_smem = cute::make_tensor(packed_stage.data(), PackedSmemLayout{});
                        auto packed_dst_part = packed_cta.partition_D(packed_smem);
                        auto packed_dst = cute::group_modes<1, cute::rank(packed_dst_part)>(packed_dst_part);
                        if constexpr (kMulticastB == 1) {
                            cute::copy(tm_b.with(Bdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                       packed_src(cute::_, out_tile + out_tiles * k_tile),
                                       packed_dst(cute::_, 0));
                        }
                        else {
                            cute::copy(tm_b.with(Bdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                       packed_src(cute::_, out_tile + out_tiles * k_tile),
                                       packed_dst(cute::_, 0));
                        }
                        auto shift_stage = sQparamShiftPipeline(cute::_, stage);
                        auto shift_smem = cute::make_tensor(shift_stage.data(), QparamShiftSmemLayout{});
                        auto shift_dst_part = shift_cta.partition_D(shift_smem);
                        auto shift_dst = cute::group_modes<1, cute::rank(shift_dst_part)>(shift_dst_part);
                        if constexpr (kMulticastB == 1) {
                            cute::copy(tm_v.with(Vdesc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                                       shift_src(cute::_, out_tile + out_tiles * k_tile),
                                       shift_dst(cute::_, 0));
                        }
                        else {
                            cute::copy(tm_v.with(Vdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_LAST),
                                       shift_src(cute::_, out_tile + out_tiles * k_tile),
                                       shift_dst(cute::_, 0));
                        }
                        auto base_stage = sQparamBasePipeline(cute::_, stage);
                        CUTE_UNROLL
                        for (int fragment = 0; fragment < kOutputFragments; ++fragment) {
                            const int global_fragment = out_fragment + fragment;
                            const bool pred = global_fragment < out_fragments;
                            const Tv* src = base_gmem
                                            + size_t(k_tile * out_fragments + global_fragment)
                                                  * kQparamBaseValuesPerFragment;
                            Tv* dst = &base_stage(fragment * kQparamBaseValuesPerFragment);
                            cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
                                *reinterpret_cast<const uint4*>(src), *reinterpret_cast<uint4*>(dst), pred);
                        }
                        cutlass::arch::cpasync_barrier_arrive_noinc(bar);
                        CUTE_UNROLL
                        for (int tma_m = 0; tma_m < kTmaCountM; ++tma_m) {
                            detail::load_mxfp4_folded_tma<kMulticastA, kTmaBoxM, TILE_K>(
                                Adesc,
                                bar,
                                storage.B.data()
                                    + stage * TILE_M * TILE_K
                                    + (mc_offset_m + tma_m * kTmaBoxM) * TILE_K,
                                k_tile * TILE_K,
                                tile->offset_m + mc_offset_m + tma_m * kTmaBoxM,
                                mask_A);
                        }
                        if constexpr (kMulticastU == 1) {
                            gmem_U.Step(bar, &sScale(cute::Int<0>{}, stage), 0);
                        }
                        else {
                            gmem_U.Step(bar, &sScale(mc_offset_m, stage), mask_A);
                        }
                        ++write_state;
                    }
                    }
                }
                if constexpr (Scheduler::is_dynamic) {
                    if (cta_0) {
                        named_barrier_arrive_unaligned(WARP_SIZE * 2,
                                                        kProducerBarrierId);
                    }
                }
                sched_state.release();
            }
            (void)param_A;
            (void)param_B;
            (void)param_V;
            (void)param_U;
            sched_state.release();
            if (elected) {
                pipeline.producer_tail(write_state);
            }
        }
        else if (warp_in_wg == 1 && cta_0) {
            auto state = sched.init_producer(storage.sched);
            while (state.next()) {
                if constexpr (Scheduler::is_dynamic) {
                    named_barrier_arrive_and_wait_unaligned(WARP_SIZE * 2, kProducerBarrierId);
                }
            }
            sched.tail(state);
        }
    }

    template<class ThrR2S, class TiledR2S, class LayoutRD, class Accum>
    __device__ static void run_epilogue(const void*               Cdesc,
                                        typename Scheduler::Tile* tile,
                                        ThrR2S&                   thr_r2s,
                                        TiledR2S&                 tiled_r2s,
                                        const LayoutRD&           tRS_rD_layout,
                                        Accum&                    accum,
                                        SharedStorage&            storage,
                                        int                       tma_store_warp,
                                        bool                      tma_store_leader,
                                        int&                      epi_store_count)
    {
        auto          tRS_rAcc     = thr_r2s.retile_S(accum);
        auto          tRS_rD       = cute::make_tensor<cutlass::bfloat16_t>(tRS_rD_layout);
        auto          tRS_rAcc_frg = cute::recast<cutlass::Array<float, kFragmentSize>>(tRS_rAcc);
        auto          tRS_rD_frg   = cute::recast<cutlass::Array<cutlass::bfloat16_t, kFragmentSize>>(tRS_rD);
        constexpr int kMmaTileN    = cute::size<0>(typename Traits::TileShape{}) / cute::size<1>(decltype(tRS_rAcc){});
        constexpr int kMmaTileM =
            (kSplitEpiM ? kWgM : cute::size<1>(typename Traits::TileShape{})) / cute::size<2>(decltype(tRS_rAcc){});
        constexpr int kTmaStoreCountN = kEpiN / kTmaStoreN;
        (void)kMmaTileN;
        static_assert(kMmaTileM % kEpiM == 0);

        auto epi_synchronize = [&] {
            constexpr int kWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
            const int barrier_id = kEpilogueBarrierId + tma_store_warp / kWarpsPerWg;
            named_barrier_arrive_and_wait(WARPGROUP_SIZE, barrier_id);
        };
        auto epi_pass_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiStripsM>{}, cute::Int<kEpiStripsN>{}),
                                                 cute::make_stride(cute::Int<kEpiStripsN>{}, cute::_1{}));
        auto store_offset_m_layout = cute::make_layout(
            cute::make_shape(cute::Int<kEpiPlanes>{}, cute::Int<kEpiStripsM>{}, cute::Int<kTmaStoreCountM>{}),
            cute::make_stride(cute::Int < kSplitEpiM ? kWgM : 0 > {}, cute::Int<kEpiM>{}, cute::Int<kTmaStoreM>{}));
        auto sD = cute::as_position_independent_swizzle_tensor(
            cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));
        auto r2s_value_layout =
            cute::make_layout(cute::make_shape(cute::size(tRS_rD_frg), cute::Int<kMmaTileM / kEpiM>{}));
        constexpr int kTmaStoreWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
        const int     store_wg             = tma_store_warp / kTmaStoreWarpsPerWg;

        CUTE_UNROLL
        for (int epi_m = 0; epi_m < kEpiStripsM; ++epi_m) {
            CUTE_UNROLL
            for (int epi_n = 0; epi_n < kEpiStripsN; ++epi_n) {
                const int epi_pass       = epi_pass_layout(epi_m, epi_n);
                const int epi_stage      = epi_store_count % kEpilogueStages;
                const int mma_n          = epi_n;
                const int mma_m          = epi_m * kEpiM / kMmaTileM;
                const int epi_m_in_mma   = epi_m % (kMmaTileM / kEpiM);
                const int r2s_v          = r2s_value_layout(0, epi_m_in_mma);
                const int epi_smem_slice = EpiStageLayout{}(kSplitEpiM ? store_wg : 0, epi_stage);
                auto      sD_epi         = sD(cute::_, cute::_, epi_smem_slice);
                auto      tRS_sD         = thr_r2s.partition_D(sD_epi);

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

                if (tma_store_leader) {
                    cute::tma_store_wait<kEpilogueStages - 1>();
                }
                epi_synchronize();
                cute::copy(tiled_r2s, tRS_rD, tRS_sD);
                cutlass::arch::fence_view_async_shared();
                epi_synchronize();

                if (tma_store_leader) {
                    auto store_offset_n_layout =
                        cute::make_layout(cute::make_shape(cute::Int<kEpiStripsN>{}, cute::Int<kTmaStoreCountN>{}),
                                          cute::make_stride(cute::Int<kEpiN>{}, cute::Int<kTmaStoreN>{}));
                    static_assert(kEpiPlanes * kTmaStoreCountN == WARPGROUPS);
                    static_assert(kTmaStoreCountM <= kTmaStoreWarpsPerWg);
                    auto store_tile_layout =
                        cute::make_layout(cute::make_shape(cute::Int<kTmaStoreCountN>{}, cute::Int<kEpiPlanes>{}));
                    const int warp_in_store_wg = tma_store_warp % kTmaStoreWarpsPerWg;
                    const int first_warp = epi_pass * kTmaStoreCountM % kTmaStoreWarpsPerWg;
                    const int tma_m      = (warp_in_store_wg + kTmaStoreWarpsPerWg - first_warp) % kTmaStoreWarpsPerWg;
                    if (tma_m < kTmaStoreCountM) {
                        const auto store_tile_coord = store_tile_layout.get_flat_coord(store_wg);
                        const int  tma_n            = cute::get<0>(store_tile_coord);
                        const int  epi_plane        = cute::get<1>(store_tile_coord);
                        const int  store_m          = tile->offset_m + store_offset_m_layout(epi_plane, epi_m, tma_m);
                        const int  store_n          = tile->offset_n + store_offset_n_layout(epi_n, tma_n);
                        auto       sD_tma =
                            cute::local_tile(sD(cute::_, cute::_, EpiStageLayout{}(epi_plane, epi_stage)),
                                             cute::make_shape(cute::Int<kTmaStoreN>{}, cute::Int<kTmaStoreM>{}),
                                             cute::make_coord(tma_n, tma_m));
                        cute::SM90_TMA_STORE::copy(Cdesc, cute::raw_pointer_cast(sD_tma.data()), store_n, store_m);
                    }
                    cute::tma_store_arrive();
                }
                ++epi_store_count;
            }
        }
    }

    __device__ static void run_consumer(const CUtensorMap& tm_c,
                                        const MatrixParam& param_C,
                                        const MatrixParam& param_W,
                                        bool               fuse_silu,
                                        Scheduler          sched,
                                        CUtensorMap*       tensormap_buf,
                                        SharedStorage&     storage,
                                        MainloopPipeline&  pipeline)
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
        auto       tCrAFull   = thr_mma.make_fragment_A(dummy_tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
        auto       tCrAStorage = cute::make_fragment_like(tCrAFull(cute::_, cute::Int<0>{}, cute::_));
        auto       tCrA        = cute::make_tensor(
            tCrAStorage.data(),
            cute::make_layout(cute::layout<0>(tCrAFull), cute::make_layout(cute::_1{}), cute::layout<2>(tCrAFull)));
        auto        sB                      = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
        auto        sScale                  = cute::make_tensor(cute::make_smem_ptr(storage.U.data()),
                                        cute::make_layout(cute::Shape<cute::Int<kUStageStride>, cute::Int<Stages>>{}));
        const auto* group_scale_table_bytes = reinterpret_cast<const uint8_t*>(storage.group_scale_tables.data());
        auto warp_group_thread_layout = cute::make_layout(cute::Int<kMathWarpGroups>{}, cute::Int<WARPGROUP_SIZE>{});
        auto wg_mma                   = tiled_mma.get_slice(warp_group_thread_layout(wg_idx));
        auto tCrB                     = wg_mma.make_fragment_B(wg_mma.partition_B(sB));
        static_assert(cute::size<0>(tCrA) == 16);
        static_assert(cute::rank(tCrA) == 3);
        static_assert(cute::size<1>(tCrA) == 1);
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
        auto cCAtom          = cute::make_identity_tensor(cute::Shape<cute::Int<kOpM>, cute::Int<kOpN>>{});
        auto tCcCAtomRaw     = wg_thr_mma.partition_C(cCAtom);
        auto tCcToken        = cute::coalesce(
            tCcCAtomRaw(cute::make_coord(cute::_, cute::Int<0>{}, cute::_), cute::Int<0>{}, cute::Int<0>{}));
        auto       c_layout_tv    = wg_tiled_mma.get_layoutC_TV();
        const auto c_thread_coord = cute::idx2crd(local_tid, cute::shape<0>(c_layout_tv));
        const int  c_row_group    = cute::get<1>(c_thread_coord);
        const int  c_warp         = cute::get<2>(c_thread_coord);

        constexpr int kOutputPasses           = kSupportsFusedSilu ? 2 : 1;
        constexpr int kOutputFragmentsPerPass = kComputeTileN / kOutputFragmentN;
        auto          output_fragment_layout =
            cute::make_layout(cute::make_shape(cute::Int<kOutputFragmentsPerPass>{}, cute::Int<kOutputPasses>{}),
                              cute::make_stride(cute::_1{}, cute::Int<kOutputFragmentsPerPass>{}));
        auto sPackedPipeline = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), PackedPipelineLayout{});

        constexpr int kQparamGroups = Traits::kKBlocksPerStage;
        auto          sQparamShiftPipeline =
            cute::make_tensor(cute::make_smem_ptr(storage.V.data()), QparamShiftPipelineLayout{});
        auto sQparamBasePipeline = cute::make_tensor(cute::make_smem_ptr(storage.W.data()), QparamBasePipelineLayout{});
        auto sD                  = cute::as_position_independent_swizzle_tensor(
            cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));
        CopyAtomC copy_atom_c{};
        using EpiTiledMma = std::conditional_t<kSplitEpiM, WgTiledMma, TiledMma>;
        EpiTiledMma epi_tiled_mma;
        auto        tiled_copy_c = cute::make_tiled_copy_C_atom(copy_atom_c, epi_tiled_mma);
        auto tiled_r2s       = cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, cutlass::bfloat16_t>{}, tiled_copy_c);
        auto thr_r2s         = tiled_r2s.get_slice(kSplitEpiM ? local_tid : mma_tid);
        auto tRS_rD_layout   = cute::make_layout(cute::take<0, 3>(cute::shape(thr_r2s.partition_S(sD))));
        int  epi_store_count = 0;
        MainloopState             read_state{};
        auto                      sched_state = sched.init_consumer(storage.sched);
        typename Scheduler::Tile* tile;
        sched_state.acquire(tile);
        while (tile->alive) {
            if (tile->is_valid_cta) {
                CUTE_UNROLL
                for (int output_pass = 0; output_pass < kOutputPasses; ++output_pass) {
                    auto accum = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    cute::clear(accum);
                    for (int k_tile = 0; k_tile < sched.k_iters_; ++k_tile) {
                        auto token = pipeline.consumer_try_wait(read_state);
                        pipeline.consumer_wait(read_state, token);
                        const int stage = read_state.index();
                        int       u_pad = 0;
                        if constexpr (Grouped) {
                            u_pad = tile->m0 % kAlignmentU;
                        }
                        auto packed_stage       = sPackedPipeline(cute::_, stage);
                        auto sPacked            = cute::make_tensor(packed_stage.data(), PackedSmemLayout{});
                        auto tAsPacked          = packed_thr_copy.partition_S(sPacked);
                        auto qparam_shift_stage = sQparamShiftPipeline(cute::_, stage);
                        auto qparam_base_stage  = sQparamBasePipeline(cute::_, stage);
                        {
                            // Keep one residual output atom in flight.  This is
                            // the WA mainloop's low-RF schedule and lets an N256
                            // CTA reuse the activation stage without retaining a
                            // second 64x128 scratch fragment.
                            auto scratch = cute::make_fragment_like(accum(cute::_, cute::Int<0>{}, cute::Int<0>{}));
                            // The C value mode is (v0,v1,v2).  GMMA ownership
                            // makes the activation scale independent of v1 and
                            // the record-wide weight scale independent of v0/v1/v2.
                            // Derive both projections from the MMA partition,
                            // as in the WA FP8 mainloop, and load each distinct
                            // scale once while the stage is resident.
                            auto c_value_shape = cute::shape<0>(scratch);
                            static_assert(cute::rank(c_value_shape) == 3);
                            constexpr int kCValue0 = cute::size<0>(decltype(c_value_shape){});
                            constexpr int kCValue1 = cute::size<1>(decltype(c_value_shape){});
                            constexpr int kCValue2 = cute::size<2>(decltype(c_value_shape){});

                            cute::for_each(cute::make_seq<kRestM>{}, [&](auto rest_m) {
                                auto       a_atom = tCrA(cute::_, cute::Int<0>{}, cute::_);
                                const auto row_lo_coord =
                                    tAcA(cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}),
                                         rest_m,
                                         cute::Int<0>{});
                                const auto output_fragment_coord =
                                    cute::idx2crd(cute::get<0>(row_lo_coord),
                                                  cute::make_shape(cute::Int<kOutputFragmentN>{},
                                                                   cute::Int<kOutputFragmentsPerPass>{}));
                                const int output_fragment =
                                    output_fragment_layout(cute::get<1>(output_fragment_coord), output_pass);
                                const int8_t base_exponent = static_cast<int8_t>(
                                    qparam_base_stage(output_fragment * kQparamBaseValuesPerFragment));
                                CUTE_UNROLL
                                for (int kb = 0; kb < kQparamGroups; ++kb) {
                                    auto packed_regs = cute::make_fragment_like(
                                        tAsPacked(cute::_, cute::Int<0>{}, output_fragment, kb));
                                    cute::copy(packed_copy,
                                               tAsPacked(cute::_, cute::Int<0>{}, output_fragment, kb),
                                               packed_regs);
                                    auto packed_words = cute::coalesce(packed_regs);
                                    auto a_words      = cute::coalesce(cute::recast<uint32_t>(a_atom(cute::_, kb)));
                                    static_assert(cute::size(decltype(packed_words){}) == 2);
                                    static_assert(cute::size(decltype(a_words){}) == 4);
                                    // All four N-lanes in a WGMMA row quartet
                                    // consume the same packed row pair.
                                    const int shift_base = output_fragment * kQparamShiftValuesPerFragment;
                                    const int scale_pair_offset =
                                        shift_base + kb * kOutputFragmentN + 2 * (local_tid / 4);
                                    const uint16_t scale_bytes =
                                        *reinterpret_cast<const uint16_t*>(&qparam_shift_stage(scale_pair_offset));
                                    const uint2 table_lo = *reinterpret_cast<const uint2*>(
                                        group_scale_table_bytes + static_cast<uint8_t>(scale_bytes));
                                    const uint2 table_hi = *reinterpret_cast<const uint2*>(
                                        group_scale_table_bytes + static_cast<uint8_t>(scale_bytes >> 8));
                                    const detail::E2m1E4m3ScaleTables tables{
                                        table_lo.x, table_lo.y, table_hi.x, table_hi.y};
                                    detail::unpack_e2m1x16_to_e4m3x4x4(
                                        reinterpret_cast<const uint32_t*>(packed_words.data()),
                                        tables,
                                        reinterpret_cast<uint32_t*>(a_words.data()));
                                }
                                auto issue = [&](auto rest_n, auto& frag) {
                                    cute::clear(frag);
                                    cute::warpgroup_fence_operand(frag);
                                    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                                    cute::warpgroup_arrive();
                                    CUTE_UNROLL
                                    for (int kb = 0; kb < kQparamGroups; ++kb) {
                                        cute::gemm(
                                            tiled_mma, a_atom(cute::_, kb), tCrB(cute::_, rest_n, kb, stage), frag);
                                        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                                    }
                                    cute::warpgroup_commit_batch();
                                };
                                auto scale = [&](auto rest_n, auto& src) {
                                    cute::warpgroup_fence_operand(src);
                                    auto dst = accum(cute::_, rest_m, rest_n);
                                    // Load each distinct activation-scale projection
                                    // after wait<0>, when issue temporaries are dead.
                                    float activation_atom[kCValue0 * kCValue2];
                                    CUTE_UNROLL
                                    for (int v2 = 0; v2 < kCValue2; ++v2) {
                                        CUTE_UNROLL
                                        for (int v0 = 0; v0 < kCValue0; ++v0) {
                                            const auto coord =
                                                tCcC(cute::make_coord(v0, cute::Int<0>{}, v2), rest_m, rest_n);
                                            activation_atom[v0 + kCValue0 * v2] = inject_unbiased_exponent(
                                                sScale(u_pad + cute::get<1>(coord), stage), base_exponent);
                                        }
                                    }
                                    CUTE_UNROLL
                                    for (int v2 = 0; v2 < kCValue2; ++v2) {
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < kCValue1; ++v1) {
                                            CUTE_UNROLL
                                            for (int v0 = 0; v0 < kCValue0; ++v0) {
                                                const auto value_coord = cute::make_coord(v0, v1, v2);
                                                const auto c_coord     = cute::make_coord(value_coord);
                                                dst(c_coord) = fmaf(src(c_coord),
                                                                   activation_atom[v0 + kCValue0 * v2],
                                                                   dst(c_coord));
                                            }
                                        }
                                    }
                                };

                                cute::warpgroup_fence_operand(a_atom);
                                cute::for_each(cute::make_seq<kRestN>{}, [&](auto rest_n) {
                                    issue(rest_n, scratch);
                                    cute::warpgroup_wait<0>();
                                    scale(rest_n, scratch);
                                });
                                cute::warpgroup_fence_operand(a_atom);
                            });
                        }
                        pipeline.consumer_release(read_state);
                        ++read_state;
                    }
                    if constexpr (!kSupportsFusedSilu) {
                        const int   tma_store_warp   = mma_tid / WARP_SIZE;
                        const bool  tma_store_leader = cute::elect_one_sync();
                        const void* output_desc      = [&]() -> const CUtensorMap* {
                            if constexpr (Grouped) {
                                return tensormap_buf + tile->group_idx * kTmaDescNum + kDescC;
                            }
                            else {
                                return &tm_c;
                            }
                        }();
                        run_epilogue(output_desc,
                                     tile,
                                     thr_r2s,
                                     tiled_r2s,
                                     tRS_rD_layout,
                                     accum,
                                     storage,
                                     tma_store_warp,
                                     tma_store_leader,
                                     epi_store_count);
                    }
                    else {
                        static_assert(kRestN == 1 && kOpM == 64);
                        constexpr float kQmax     = 448.f;
                        constexpr int   kScales   = Traits::kActScalesPerAtom;
                        float*          exchange  = storage.silu_exchange.data();  // [gate OUT128, token]
                        auto            sExchange = cute::make_tensor(
                            cute::make_smem_ptr(exchange),
                            cute::make_layout(cute::Shape<cute::Int<kFusedOutputN>, cute::Int<TILE_M>>{}));
                        if (output_pass == 0) {
                            CUTE_UNROLL
                            for (int rm = 0; rm < kRestM; ++rm) {
                                auto C = accum(cute::_, rm, cute::Int<0>{});
                                CUTE_UNROLL
                                for (int c = 0; c < cute::size(C); ++c) {
                                    const auto coord                                    = tCcC(c, rm, cute::Int<0>{});
                                    sExchange(cute::get<0>(coord), cute::get<1>(coord)) = C(c);
                                }
                            }
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                        }
                        else {
                            if (threadIdx.x == 0) {
                                cute::tma_store_wait<0>();
                            }
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            float amax[kScales];
                            CUTE_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                amax[i] = 1e-8f;
                            }
                            auto c_value_shape = cute::shape<0>(accum);
                            static_assert(cute::rank(c_value_shape) == 3);
                            static_assert(cute::size<0>(c_value_shape) * cute::size<2>(c_value_shape) == kScales);
                            auto rAmax =
                                cute::make_tensor(cute::make_rmem_ptr(amax),
                                                  cute::make_layout(cute::make_shape(cute::size<0>(c_value_shape),
                                                                                     cute::size<2>(c_value_shape))));
                            CUTE_UNROLL
                            for (int rm = 0; rm < kRestM; ++rm) {
                                auto C = accum(cute::_, rm, cute::Int<0>{});
                                CUTE_UNROLL
                                for (int v2 = 0; v2 < cute::size<2>(c_value_shape); ++v2) {
                                    CUTE_UNROLL
                                    for (int v0 = 0; v0 < cute::size<0>(c_value_shape); ++v0) {
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < cute::size<1>(c_value_shape); ++v1) {
                                            const auto  value_coord = cute::make_coord(v0, v1, v2);
                                            const auto  coord       = tCcC(value_coord, rm, cute::Int<0>{});
                                            const float gate    = sExchange(cute::get<0>(coord), cute::get<1>(coord));
                                            const auto  c_coord = cute::make_coord(value_coord);
                                            C(c_coord)          = fdividef(gate, 1.f + expf(-gate)) * C(c_coord);
                                            rAmax(v0, v2)       = fmaxf(rAmax(v0, v2), fabsf(C(c_coord)));
                                        }
                                    }
                                }
                            }
                            auto          c_thread_shape       = cute::shape<0>(c_layout_tv);
                            constexpr int kCRowGroups          = cute::size<1>(decltype(c_thread_shape){});
                            constexpr int kCWarps              = cute::size<2>(decltype(c_thread_shape){});
                            constexpr int kCRowGroupLaneStride = cute::size<0>(decltype(c_thread_shape){});
                            static_assert(kCRowGroups == 8 && kCWarps == 4);
                            CUTE_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                CUTE_UNROLL
                                for (int group = 1; group < kCRowGroups; group *= 2) {
                                    amax[i] = fmaxf(
                                        amax[i], __shfl_xor_sync(0xffffffffu, amax[i], group * kCRowGroupLaneStride));
                                }
                            }
                            float* scratch = storage.fused_amax_scratch.data();
                            auto   sAmax   = cute::make_tensor(
                                cute::make_smem_ptr(scratch),
                                cute::make_layout(
                                    cute::Shape<cute::Int<TILE_M>, cute::Int<kCWarps>, cute::Int<kMathWarpGroups>>{}));
                            if (c_row_group == 0) {
                                CUTE_UNROLL
                                for (int i = 0; i < kScales; ++i) {
                                    const int token              = cute::get<1>(tCcToken(i));
                                    sAmax(token, c_warp, wg_idx) = amax[i];
                                }
                            }
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            CUTE_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                const int token = cute::get<1>(tCcToken(i));
                                float     value = sAmax(token, cute::Int<0>{}, wg_idx);
                                CUTE_UNROLL
                                for (int warp = 1; warp < kCWarps; ++warp) {
                                    value = fmaxf(value, sAmax(token, warp, wg_idx));
                                }
                                amax[i] = value;
                            }
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            if (c_row_group == 0 && c_warp == 0) {
                                CUTE_UNROLL
                                for (int i = 0; i < kScales; ++i) {
                                    const int token                      = cute::get<1>(tCcToken(i));
                                    sAmax(token, cute::Int<0>{}, wg_idx) = amax[i];
                                }
                            }
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            CUTE_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                const int token = cute::get<1>(tCcToken(i));
                                float     value = sAmax(token, cute::Int<0>{}, cute::Int<0>{});
                                CUTE_UNROLL
                                for (int wg = 1; wg < kMathWarpGroups; ++wg) {
                                    value = fmaxf(value, sAmax(token, cute::Int<0>{}, wg));
                                }
                                amax[i] = fmaxf(value, 1e-8f);
                            }
                            __nv_fp8_e4m3* smem_C =
                                reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<char*>(&storage) + kOutputOffset);
                            auto sC = cute::make_tensor(
                                cute::make_smem_ptr(smem_C),
                                cute::make_layout(cute::Shape<cute::Int<kFusedOutputN>, cute::Int<TILE_M>>{}));
                            CUTE_UNROLL
                            for (int rm = 0; rm < kRestM; ++rm) {
                                auto C = accum(cute::_, rm, cute::Int<0>{});
                                CUTE_UNROLL
                                for (int v2 = 0; v2 < cute::size<2>(c_value_shape); ++v2) {
                                    CUTE_UNROLL
                                    for (int v0 = 0; v0 < cute::size<0>(c_value_shape); ++v0) {
                                        const float inv = kQmax / rAmax(v0, v2);
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < cute::size<1>(c_value_shape); ++v1) {
                                            const auto value_coord = cute::make_coord(v0, v1, v2);
                                            const auto coord       = tCcC(value_coord, rm, cute::Int<0>{});
                                            sC(cute::get<0>(coord), cute::get<1>(coord)) =
                                                __nv_fp8_e4m3(C(cute::make_coord(value_coord)) * inv);
                                        }
                                    }
                                }
                            }
                            int group_m0 = 0;
                            int row_end  = sched.gemm_shape().x;
                            if constexpr (Grouped) {
                                group_m0 = tile->m0;
                                row_end  = tile->m1;
                            }
                            if (param_W.ptr && wg_idx == 0 && c_row_group == 0 && c_warp == 0) {
                                auto gW = cute::make_tensor(
                                    cute::make_gmem_ptr(static_cast<float*>(param_W.ptr)),
                                    cute::make_shape(row_end,
                                                     cute::ceil_div(sched.gemm_shape().y, cute::Int<TILE_N>{})),
                                    cute::make_stride(cute::_1{}, param_W.stride));
                                CUTE_UNROLL
                                for (int i = 0; i < kScales; ++i) {
                                    const int token      = cute::get<1>(tCcToken(i));
                                    const int global_row = group_m0 + tile->offset_m + token;
                                    if (global_row < row_end) {
                                        const int n_group       = tile->offset_n / TILE_N;
                                        gW(global_row, n_group) = amax[i] / kQmax;
                                    }
                                }
                            }
                            cute::tma_store_fence();
                            named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            if (threadIdx.x == 0) {
                                const CUtensorMap* output_desc = [&]() {
                                    if constexpr (Grouped) {
                                        return tensormap_buf + tile->group_idx * kTmaDescNum + kDescC;
                                    }
                                    else {
                                        return &tm_c;
                                    }
                                }();
                                cute::SM90_TMA_STORE::copy(output_desc, smem_C, tile->offset_n / 2, tile->offset_m);
                                cute::tma_store_arrive();
                            }
                            if constexpr (Grouped) {
                                if (threadIdx.x == 0) {
                                    cute::tma_store_wait<0>();
                                }
                                named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                            }
                        }
                    }
                }
            }
            else if (tile->is_valid_cluster) {
                constexpr int kOutputPasses = kSupportsFusedSilu ? 2 : 1;
                for (int k_tile = 0; k_tile < sched.k_iters_ * kOutputPasses; ++k_tile) {
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

        (void)param_C;
    }
};

}  // namespace turbomind::gemm
