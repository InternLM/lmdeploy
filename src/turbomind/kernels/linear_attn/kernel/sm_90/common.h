#pragma once

#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/linear_attn/kernel/tma_desc.h"
#include "src/turbomind/kernels/linear_attn/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/pointer_flagged.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kChunkSize = 64;
constexpr int kHeadDim   = 128;

constexpr int      kWideGdrBlockDv                = 128;
constexpr int      kFusedGdrBlockDv               = 64;
constexpr int      kCudaWarpThreads               = 32;
constexpr int      kFusedGdrRoleThreads           = 128;
constexpr int      kFusedGdrConsumerThreads       = 3 * kFusedGdrRoleThreads;
constexpr int      kFusedGdrThreads               = 4 * kFusedGdrRoleThreads;
constexpr size_t   kFusedGdrMaxDynamicSharedBytes = 227328 - 1024;
constexpr uint64_t kTmaNoCacheHint                = 0;

static_assert(kWideGdrBlockDv == 128);
static_assert(kFusedGdrBlockDv == 64);
static_assert(kFusedGdrThreads == 512);
static_assert(kFusedGdrConsumerThreads == 3 * kFusedGdrRoleThreads);
static_assert(kFusedGdrThreads == 4 * kFusedGdrRoleThreads);
static_assert(kFusedGdrRoleThreads % kCudaWarpThreads == 0);
static_assert(kHeadDim % kWideGdrBlockDv == 0);
static_assert(kHeadDim % kFusedGdrBlockDv == 0);

template<class StateT>
constexpr bool kFusedGdrValidStateT = std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>;

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaQkKLayout()
{
    return cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<Element>{},
                               cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kHeadDim>{}));
}

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaQkTransposeALayout()
{
    static_assert(sizeof(Element) == 2);
    // FlashQLA loads K as two DK/2 TMA boxes:
    //   offset(row, dk) = (dk / 64) * (64 * 64) + row * 64 + (dk % 64).
    // This is only a logical coordinate adapter for the same physical bytes:
    // TransposeALayout(dk, row) must equal QkKLayout(row, dk).
    return cute::composition(
        cute::Swizzle<3, 4, 3>{},
        cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
        cute::Layout<cute::Shape<cute::Shape<cute::Int<kHeadDim / 2>, cute::_2>,
                                 cute::Shape<cute::_8, cute::Int<kChunkSize / 8>>>,
                     cute::Stride<cute::Stride<cute::_1, cute::Int<(kHeadDim / 2) * kChunkSize>>,
                                  cute::Stride<cute::Int<kHeadDim / 2>, cute::Int<(kHeadDim / 2) * 8>>>>{});
}

static_assert(cute::cosize_v<decltype(FusedGdrGmmaQkTransposeALayout<cute::bfloat16_t>())> == kChunkSize * kHeadDim);
static_assert(FusedGdrGmmaQkTransposeALayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{})
              == FusedGdrGmmaQkKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{}));
static_assert(FusedGdrGmmaQkTransposeALayout<cute::bfloat16_t>()(cute::Int<64>{}, cute::Int<0>{})
              == FusedGdrGmmaQkKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<64>{}));
static_assert(FusedGdrGmmaQkTransposeALayout<cute::bfloat16_t>()(cute::Int<127>{}, cute::Int<63>{})
              == FusedGdrGmmaQkKLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<127>{}));

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaStateTLayout()
{
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == 32) {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW64_Atom<Element>{},
                                   cute::make_shape(cute::Int<BlockDv>{}, cute::Int<kHeadDim>{}));
    }
    else {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW128_Atom<Element>{},
                                   cute::make_shape(cute::Int<BlockDv>{}, cute::Int<kHeadDim>{}));
    }
}

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaStateRowLayout()
{
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == 32) {
        return cute::composition(
            cute::Swizzle<2, 4, 3>{},
            cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
            cute::Layout<
                cute::Shape<cute::Shape<cute::_8, cute::Int<kHeadDim / 8>>, cute::Shape<cute::Int<BlockDv>, cute::_1>>,
                cute::Stride<cute::Stride<cute::Int<BlockDv>, cute::Int<BlockDv * 8>>,
                             cute::Stride<cute::_1, cute::_0>>>{});
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return cute::composition(
            cute::Swizzle<3, 4, 3>{},
            cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
            cute::Layout<
                cute::Shape<cute::Shape<cute::_8, cute::Int<kHeadDim / 8>>, cute::Shape<cute::Int<BlockDv>, cute::_1>>,
                cute::Stride<cute::Stride<cute::Int<BlockDv>, cute::Int<BlockDv * 8>>,
                             cute::Stride<cute::_1, cute::_0>>>{});
    }
    else {
        return cute::composition(
            cute::Swizzle<3, 4, 3>{},
            cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
            cute::Layout<cute::Shape<cute::Shape<cute::_8, cute::Int<kHeadDim / 8>>,
                                     cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>>,
                         cute::Stride<cute::Stride<cute::Int<kFusedGdrBlockDv>, cute::Int<BlockDv * 8>>,
                                      cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * 8>>>>{});
    }
}

static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, 32>()(cute::Int<127>{}, cute::Int<31>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, 32>()(cute::Int<31>{}, cute::Int<127>{}));
static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<127>{}, cute::Int<63>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<127>{}));
static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<127>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<127>{}));

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaVdTLayout()
{
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == 32) {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW64_Atom<Element>{},
                                   cute::make_shape(cute::Int<BlockDv>{}, cute::Int<kChunkSize>{}));
    }
    else {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW128_Atom<Element>{},
                                   cute::make_shape(cute::Int<BlockDv>{}, cute::Int<kChunkSize>{}));
    }
}

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaVdRowLayout()
{
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == 32) {
        return cute::composition(cute::Swizzle<2, 4, 3>{},
                                 cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
                                 cute::Layout<cute::Shape<cute::Shape<cute::_8, cute::Int<kChunkSize / 8>>,
                                                          cute::Shape<cute::Int<BlockDv>, cute::_1>>,
                                              cute::Stride<cute::Stride<cute::Int<BlockDv>, cute::Int<BlockDv * 8>>,
                                                           cute::Stride<cute::_1, cute::_0>>>{});
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return cute::composition(cute::Swizzle<3, 4, 3>{},
                                 cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
                                 cute::Layout<cute::Shape<cute::Shape<cute::_8, cute::Int<kChunkSize / 8>>,
                                                          cute::Shape<cute::Int<BlockDv>, cute::_1>>,
                                              cute::Stride<cute::Stride<cute::Int<BlockDv>, cute::Int<BlockDv * 8>>,
                                                           cute::Stride<cute::_1, cute::_0>>>{});
    }
    else {
        return cute::composition(
            cute::Swizzle<3, 4, 3>{},
            cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
            cute::Layout<cute::Shape<cute::Shape<cute::_8, cute::Int<kChunkSize / 8>>,
                                     cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>>,
                         cute::Stride<cute::Stride<cute::Int<kFusedGdrBlockDv>, cute::Int<BlockDv * 8>>,
                                      cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * 8>>>>{});
    }
}

static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, 32>()(cute::Int<63>{}, cute::Int<31>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, 32>()(cute::Int<31>{}, cute::Int<63>{}));
static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<63>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<63>{}));
static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<63>{}, cute::Int<127>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<63>{}));

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVTLayout()
{
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == 32) {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kChunkSize>>,
                                              cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kChunkSize>>,
                                              cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
    }
    else {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>, cute::Int<kChunkSize>>,
                         cute::Stride<cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * kChunkSize>>,
                                      cute::Int<kFusedGdrBlockDv>>>{});
    }
}

static_assert(cute::cosize_v<decltype(FusedGdrSwizzledVTLayout<32>())> == 32 * kChunkSize);
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledVTLayout<kFusedGdrBlockDv>())> == kFusedGdrBlockDv * kChunkSize);
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledVTLayout<kWideGdrBlockDv>())> == kWideGdrBlockDv * kChunkSize);

template<class T>
struct FusedGdrMmaTraits;

template<>
struct FusedGdrMmaTraits<__nv_bfloat16> {
    using Element = cute::bfloat16_t;
    using Atom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

inline int CeilDiv(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

template<class StateT>
__device__ __forceinline__ StateT* GroupedStateBase(const int64_t* state_ptrs,
                                                    int            sequence,
                                                    int            value_head,
                                                    int            num_head_groups,
                                                    int            heads_per_block,
                                                    int64_t        state_layer_offset)
{
    const int     head_group = value_head / heads_per_block;
    const int     local_head = value_head % heads_per_block;
    const int64_t address    = state_ptrs[sequence * num_head_groups + head_group];
    return reinterpret_cast<StateT*>(static_cast<uintptr_t>(address)) + state_layer_offset
           + static_cast<int64_t>(local_head) * kHeadDim * kHeadDim;
}

__device__ __forceinline__ float FastExp(float value)
{
    return exp2f(value * 1.4426950408889634f);
}

template<bool FenceProxyAsyncShared = true,
         class TiledMma,
         class TA,
         class ALayout,
         class TB,
         class BLayout,
         class Accumulator>
__device__ __forceinline__ void FusedGdrGmmaSs(TiledMma&                        tiled_mma,
                                               uint32_t                         thread_idx,
                                               cute::Tensor<TA, ALayout> const& sA,
                                               cute::Tensor<TB, BLayout> const& sB,
                                               Accumulator&                     tCrC,
                                               cute::SM90::GMMA::ScaleOut       scale)
{
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto tCsA    = thr_mma.partition_A(sA);
    auto tCsB    = thr_mma.partition_B(sB);
    auto tCrA    = thr_mma.make_fragment_A(tCsA);
    auto tCrB    = thr_mma.make_fragment_B(tCsB);

    if constexpr (FenceProxyAsyncShared) {
        cutlass::arch::fence_view_async_shared();
    }
    cute::warpgroup_fence_operand(tCrC);
    cute::warpgroup_arrive();
    tiled_mma.accumulate_     = scale;
    constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrC);
        tiled_mma.accumulate_ = cute::SM90::GMMA::ScaleOut::One;
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(tCrC);
}

template<class T, class Element, class Fragment, class SmemTensor, class ThrMma>
__device__ __forceinline__ void FusedGdrStoreFragmentBf16Stsm(Fragment const&   fragment,
                                                              SmemTensor const& smem_tensor,
                                                              ThrMma const&     thr_mma,
                                                              int               role_tid)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    static_assert(std::is_same_v<Element, cute::bfloat16_t>);

    auto tCrPack = cute::make_fragment_like<Element>(fragment);
#pragma unroll
    for (int i = 0; i < cute::size(fragment); ++i) {
        tCrPack(i) = Element(__float2bfloat16(static_cast<float>(fragment(i))));
    }

    auto s_pack            = cute::as_position_independent_swizzle_tensor(smem_tensor);
    auto smem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, thr_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCsPack           = smem_thr_copy_C.partition_D(s_pack);
    auto tCrPackView       = smem_thr_copy_C.retile_S(tCrPack);
    cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
}

template<int BlockDv, class T, class TA, class ALayout, class TB, class BLayout, class StateFragment>
__device__ __forceinline__ void FusedGdrStateUpdateFragmentGmmaBf16Vd(uint32_t                         thread_idx,
                                                                      cute::Tensor<TA, ALayout> const& sA,
                                                                      cute::Tensor<TB, BLayout> const& sB,
                                                                      StateFragment&                   tCrState)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    static_assert(BlockDv == 32 || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    using Element    = typename FusedGdrMmaTraits<T>::Element;
    using InputTypeA = typename TA::value_type;
    using InputTypeB = typename TB::value_type;
    static_assert(std::is_same_v<InputTypeA, Element>);
    static_assert(std::is_same_v<InputTypeB, Element>);

    using TileShape = cute::Shape<cute::Int<64>, cute::Int<BlockDv>, cute::Int<64>>;
    using GmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                               Element,
                                                               float,
                                                               TileShape,
                                                               cute::SM90::GMMA::Major::MN,
                                                               cute::SM90::GMMA::Major::MN>());
    auto tiled_mma  = cute::make_tiled_mma(GmmaAtom{});
    auto thr_mma    = tiled_mma.get_thread_slice(thread_idx);

    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);

    auto tCrA = thr_mma.make_fragment_A(tCsA);
    auto tCrB = thr_mma.make_fragment_B(tCsB);

    cutlass::arch::fence_view_async_shared();
    cute::warpgroup_fence_operand(tCrState);
    cute::warpgroup_arrive();
    tiled_mma.accumulate_     = cute::SM90::GMMA::ScaleOut::One;
    constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrState);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(tCrState);
}

template<class StateFragment>
__device__ __forceinline__ void FusedGdrDecayStateFragment(StateFragment& tCrState, float decay)
{
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        tCrState(i) *= decay;
    }
}

template<class StateT, class StateFragment, class GlobalTensor, class ThrMma>
__device__ __forceinline__ void FusedGdrLoadStateFragmentGlobal(StateFragment&      tCrState,
                                                                GlobalTensor const& g_state,
                                                                ThrMma const&       thr_mma,
                                                                int                 role_tid)
{
    auto gmem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, StateT>{}, thr_mma);
    auto gmem_thr_copy_C   = gmem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCgState          = gmem_thr_copy_C.partition_S(g_state);
    if constexpr (std::is_same_v<StateT, float>) {
        auto tCrStateView = gmem_thr_copy_C.retile_D(tCrState);
        cute::copy(gmem_tiled_copy_C, tCgState, tCrStateView);
    }
    else {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        auto tCrPacked     = cute::make_fragment_like<StateT>(tCrState);
        auto tCrPackedView = gmem_thr_copy_C.retile_D(tCrPacked);
        cute::copy(gmem_tiled_copy_C, tCgState, tCrPackedView);
        cute::copy(tCrPacked, tCrState);
    }
}

template<class StateT, class StateFragment, class GlobalTensor, class ThrMma>
__device__ __forceinline__ void FusedGdrStoreStateFragmentGlobal(StateFragment const& tCrState,
                                                                 GlobalTensor const&  g_state,
                                                                 ThrMma const&        thr_mma,
                                                                 int                  role_tid)
{
    auto gmem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, StateT>{}, thr_mma);
    auto gmem_thr_copy_C   = gmem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCgState          = gmem_thr_copy_C.partition_D(g_state);
    if constexpr (std::is_same_v<StateT, float>) {
        auto tCrStateView = gmem_thr_copy_C.retile_S(tCrState);
        cute::copy(gmem_tiled_copy_C, tCrStateView, tCgState);
    }
    else {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        auto tCrPacked     = cute::make_fragment_like<StateT>(tCrState);
        auto tCrPackedView = gmem_thr_copy_C.retile_S(tCrPacked);
        cute::copy(tCrState, tCrPacked);
        cute::copy(gmem_tiled_copy_C, tCrPackedView, tCgState);
    }
}

static_assert(alignof(__nv_bfloat162) <= 16);
static_assert((kFusedGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kFusedGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((32 * sizeof(float)) % alignof(float2) == 0);
static_assert((32 * sizeof(float)) % sizeof(float2) == 0);
static_assert((kWideGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kWideGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kChunkSize * kChunkSize * sizeof(cute::bfloat16_t)) % alignof(__nv_bfloat162) == 0);

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
