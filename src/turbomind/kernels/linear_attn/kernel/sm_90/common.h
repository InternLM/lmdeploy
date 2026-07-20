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

constexpr int      kWideGdrBlockDv                   = 128;
constexpr int      kFusedGdrBlockDv                  = 64;
constexpr int      kContextParallelGdrBlockDv        = 32;
constexpr int      kFusedGdrHBlockDv                 = kWideGdrBlockDv;
constexpr int      kFusedGdrHMBlockDv                = kFusedGdrBlockDv;
constexpr int      kCorrectInitialStatesF32BlockDv   = 32;
constexpr int      kCorrectInitialStatesBf16BlockDv  = kCorrectInitialStatesF32BlockDv;
constexpr int      kCorrectInitialStatesMRowsPerTma  = 64;
constexpr int      kFusedGdrTmaDescCount             = 6;
constexpr int      kKktTmaDescCount                  = 4;
constexpr int      kFusedGdrHTmaDescCount            = 6;
constexpr int      kCorrectInitialStatesTmaDescCount = 4;
constexpr int      kTmaDescriptorBytes               = 128;
constexpr int      kCudaWarpThreads                  = 32;
constexpr int      kFusedGdrMmaThreads               = 128;
constexpr int      kFusedGdrRoleThreads              = 128;
constexpr int      kFusedGdrConsumerThreads          = 3 * kFusedGdrRoleThreads;
constexpr int      kFusedGdrProducerThreads          = kFusedGdrRoleThreads;
constexpr int      kFusedGdrConsumerStoreThreads     = kFusedGdrConsumerThreads + kCudaWarpThreads;
constexpr int      kFusedGdrHStateReadThreads        = 3 * kFusedGdrRoleThreads;
constexpr int      kFusedGdrGateVdReadThreads        = 3 * kFusedGdrRoleThreads;
constexpr int      kFusedGdrConsumerWgs              = kFusedGdrConsumerThreads / kFusedGdrRoleThreads;
constexpr int      kFusedGdrThreads                  = kFusedGdrConsumerThreads + kFusedGdrProducerThreads;
constexpr size_t   kFusedGdrSm90SharedBytes          = 227328;
constexpr size_t   kFusedGdrStaticSharedReserveBytes = 1024;
constexpr size_t   kFusedGdrMaxDynamicSharedBytes    = kFusedGdrSm90SharedBytes - kFusedGdrStaticSharedReserveBytes;
constexpr uint64_t kTmaNoCacheHint                   = 0;

static_assert(kWideGdrBlockDv == 128);
static_assert(kFusedGdrBlockDv == 64);
static_assert(kContextParallelGdrBlockDv == 32);
static_assert(kFusedGdrThreads == 512);
static_assert(kFusedGdrThreads >= kFusedGdrMmaThreads);
static_assert(kFusedGdrMmaThreads == 128);
static_assert(kFusedGdrRoleThreads == kFusedGdrMmaThreads);
static_assert(kFusedGdrConsumerThreads == 3 * kFusedGdrRoleThreads);
static_assert(kFusedGdrConsumerStoreThreads == kFusedGdrConsumerThreads + kCudaWarpThreads);
static_assert(kFusedGdrHStateReadThreads == 384);
static_assert(kFusedGdrGateVdReadThreads == 384);
static_assert(kFusedGdrConsumerWgs == 3);
static_assert(kFusedGdrProducerThreads == kFusedGdrRoleThreads);
static_assert(kFusedGdrThreads == 4 * kFusedGdrRoleThreads);
static_assert(kFusedGdrRoleThreads % kCudaWarpThreads == 0);
static_assert(kHeadDim % kWideGdrBlockDv == 0);
static_assert(kHeadDim % kFusedGdrBlockDv == 0);
static_assert(kHeadDim % kContextParallelGdrBlockDv == 0);
static_assert(kHeadDim % kCorrectInitialStatesF32BlockDv == 0);
static_assert(kHeadDim % kCorrectInitialStatesBf16BlockDv == 0);
static_assert(kCorrectInitialStatesBf16BlockDv == kCorrectInitialStatesF32BlockDv);
static_assert(kHeadDim % kCorrectInitialStatesMRowsPerTma == 0);
static_assert(sizeof(CUtensorMap) == kTmaDescriptorBytes);
static_assert(kFusedGdrTmaDescCount * sizeof(CUtensorMap) <= kFusedGdrStaticSharedReserveBytes);

template<class StateT>
constexpr bool kFusedGdrValidStateT = std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>;

template<int ChunkSize>
constexpr bool kSupportedGdrChunkSize = ChunkSize == 64;

CUTE_HOST_DEVICE constexpr auto FusedGdrQkLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Int<kHeadDim>>,
                                          cute::Stride<cute::Int<kHeadDim>, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrQkLayout())> == kChunkSize * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrQkTransposedLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kChunkSize>>,
                                          cute::Stride<cute::_1, cute::Int<kHeadDim>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrQkTransposedLayout())> == kChunkSize * kHeadDim);

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaQkLayout()
{
    return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW128_Atom<Element>{},
                               cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kHeadDim>{}));
}

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

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaSquareLayout()
{
    return cute::tile_to_shape(cute::SM90::GMMA::Layout_MN_SW128_Atom<Element>{},
                               cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kChunkSize>{}));
}

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaSquareKLayout()
{
    return cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<Element>{},
                               cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kChunkSize>{}));
}

template<class Element>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaSquareKRowMajorLayout()
{
    return cute::composition(cute::Swizzle<3, 4, 3>{},
                             cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
                             cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Int<kChunkSize>>,
                                          cute::Stride<cute::Int<kChunkSize>, cute::_1>>{});
}

static_assert(FusedGdrGmmaSquareKRowMajorLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{}));
static_assert(FusedGdrGmmaSquareKRowMajorLayout<cute::bfloat16_t>()(cute::Int<1>{}, cute::Int<0>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<1>{}, cute::Int<0>{}));
static_assert(FusedGdrGmmaSquareKRowMajorLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<1>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<1>{}));
static_assert(FusedGdrGmmaSquareKRowMajorLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<31>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<31>{}));
static_assert(FusedGdrGmmaSquareKRowMajorLayout<cute::bfloat16_t>()(cute::Int<9>{}, cute::Int<57>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<9>{}, cute::Int<57>{}));

static_assert(FusedGdrGmmaSquareLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{}));
static_assert(FusedGdrGmmaSquareLayout<cute::bfloat16_t>()(cute::Int<1>{}, cute::Int<0>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<1>{}));
static_assert(FusedGdrGmmaSquareLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<31>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<31>{}, cute::Int<63>{}));
static_assert(FusedGdrGmmaSquareLayout<cute::bfloat16_t>()(cute::Int<9>{}, cute::Int<57>{})
              == FusedGdrGmmaSquareKLayout<cute::bfloat16_t>()(cute::Int<57>{}, cute::Int<9>{}));

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaStateTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
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
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
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

static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, kContextParallelGdrBlockDv>()(cute::Int<127>{},
                                                                                         cute::Int<31>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, kContextParallelGdrBlockDv>()(cute::Int<31>{},
                                                                                          cute::Int<127>{}));
static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<127>{}, cute::Int<63>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<127>{}));
static_assert(FusedGdrGmmaStateRowLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<127>{})
              == FusedGdrGmmaStateTLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<127>{}));

template<class Element, int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrGmmaVdTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
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
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
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

static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, kContextParallelGdrBlockDv>()(cute::Int<63>{}, cute::Int<31>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, kContextParallelGdrBlockDv>()(cute::Int<31>{},
                                                                                       cute::Int<63>{}));
static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<63>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, kFusedGdrBlockDv>()(cute::Int<63>{}, cute::Int<63>{}));
static_assert(FusedGdrGmmaVdRowLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<63>{}, cute::Int<127>{})
              == FusedGdrGmmaVdTLayout<cute::bfloat16_t, kWideGdrBlockDv>()(cute::Int<127>{}, cute::Int<63>{}));

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledA64Layout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_64, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledA64Layout())> == kChunkSize * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledV128TLayout()
{
    return cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>, cute::Int<kChunkSize>>,
                     cute::Stride<cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * kChunkSize>>,
                                  cute::Int<kFusedGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledV128TLayout())> == kWideGdrBlockDv * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledV128RowLayout()
{
    return cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>>,
                     cute::Stride<cute::Int<kFusedGdrBlockDv>,
                                  cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * kChunkSize>>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledV128RowLayout())> == kWideGdrBlockDv * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState128TLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kWideGdrBlockDv>, cute::Int<kHeadDim>>,
                                          cute::Stride<cute::_1, cute::Int<kWideGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState128TLayout())> == kWideGdrBlockDv * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState128CLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kWideGdrBlockDv>>,
                                          cute::Stride<cute::Int<kWideGdrBlockDv>, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState128CLayout())> == kWideGdrBlockDv * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledV64TLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::Int<kChunkSize>>,
                                          cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledV64TLayout())> == kFusedGdrBlockDv * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState64TLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::Int<kHeadDim>>,
                                          cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState64TLayout())> == kFusedGdrBlockDv * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState64CLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kFusedGdrBlockDv>>,
                                          cute::Stride<cute::Int<kFusedGdrBlockDv>, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState64CLayout())> == kFusedGdrBlockDv * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledV32TLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kContextParallelGdrBlockDv>, cute::Int<kChunkSize>>,
                                          cute::Stride<cute::_1, cute::Int<kContextParallelGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledV32TLayout())> == kContextParallelGdrBlockDv * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledV32RowLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_64, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledV32RowLayout())> == kContextParallelGdrBlockDv * kChunkSize);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState32TLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kContextParallelGdrBlockDv>, cute::Int<kHeadDim>>,
                                          cute::Stride<cute::_1, cute::Int<kContextParallelGdrBlockDv>>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState32TLayout())> == kContextParallelGdrBlockDv * kHeadDim);

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledState32CLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kContextParallelGdrBlockDv>>,
                                          cute::Stride<cute::Int<kContextParallelGdrBlockDv>, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledState32CLayout())> == kContextParallelGdrBlockDv * kHeadDim);

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledV32TLayout();
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return FusedGdrSwizzledV64TLayout();
    }
    else {
        return FusedGdrSwizzledV128TLayout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVRowLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledV32RowLayout();
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return FusedGdrSwizzledA64Layout();
    }
    else {
        return FusedGdrSwizzledV128RowLayout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVdTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    return cute::composition(cute::Swizzle<3, 2, 3>{},
                             cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kChunkSize>>,
                                          cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVdRowLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    return cute::composition(cute::Swizzle<3, 2, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Int<BlockDv>>,
                                          cute::Stride<cute::Int<BlockDv>, cute::_1>>{});
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledStateTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledState32TLayout();
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return FusedGdrSwizzledState64TLayout();
    }
    else {
        return FusedGdrSwizzledState128TLayout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledStateCLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledState32CLayout();
    }
    else if constexpr (BlockDv == kFusedGdrBlockDv) {
        return FusedGdrSwizzledState64CLayout();
    }
    else {
        return FusedGdrSwizzledState128CLayout();
    }
}

// CUTLASS offsets user named-barrier IDs 0-7 by ReservedNamedBarrierCount,
// mapping them to SM90 hardware barrier IDs 8-15.
constexpr int kFusedGdrBarrierStateUpdate    = 1;
constexpr int kFusedGdrBarrierValueU         = 2;
constexpr int kFusedGdrBarrierVdReadDone     = 3;
constexpr int kFusedGdrBarrierOutputAg       = 4;
constexpr int kFusedGdrBarrierOutputLocal    = 5;
constexpr int kFusedGdrBarrierHStateReadDone = 6;
constexpr int kFusedGdrBarrierHVdReady       = 5;  // FusedGdrH does not use OutputLocal.
constexpr int kFusedGdrBarrierMVdReady       = 7;
// Dormant in the active SM90 runtime: retained for the uncalled state pack/unpack
// templates and FusedGdrStatePack* helpers below.
constexpr int kFusedGdrBarrierProducerStatePack  = 6;
constexpr int kFusedGdrBarrierProducerStateStore = 7;
constexpr int kFusedGdrBarrierStateToValue       = kFusedGdrBarrierProducerStatePack;
constexpr int kFusedGdrBarrierStateToOutput      = kFusedGdrBarrierProducerStateStore;
static_assert(kFusedGdrBarrierMVdReady + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
              < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

template<int BarrierId>
__device__ __forceinline__ void FusedGdrMmaSyncNamed()
{
    // Rendezvous the 128 threads in one logical WG. Async WGMMA completion is
    // separately owned by warpgroup_wait in the GMMA helpers.
    cutlass::arch::NamedBarrier::sync(kFusedGdrRoleThreads, BarrierId);
}

__device__ __forceinline__ void FusedGdrStateWaitForHReaders()
{
    // WG0 contributes 128 arrivals and waits until WGs 1 and 2 have completed
    // their h_shared WGMMA reads and contributed the other 256 arrivals.
    cutlass::arch::NamedBarrier::sync(kFusedGdrHStateReadThreads, kFusedGdrBarrierHStateReadDone);
}

__device__ __forceinline__ void FusedGdrHReaderArrive()
{
    cutlass::arch::NamedBarrier::arrive(kFusedGdrHStateReadThreads, kFusedGdrBarrierHStateReadDone);
}

__device__ __forceinline__ void FusedGdrValueWaitForGateVdReaders()
{
    // WG1 contributes 128 arrivals and waits until WG0 has consumed the
    // single-slot gate vectors and WG2 has drained the vd_shared WGMMA read.
    cutlass::arch::NamedBarrier::sync(kFusedGdrGateVdReadThreads, kFusedGdrBarrierVdReadDone);
}

__device__ __forceinline__ void FusedGdrStateGateReadArrive()
{
    cutlass::arch::NamedBarrier::arrive(kFusedGdrGateVdReadThreads, kFusedGdrBarrierVdReadDone);
}

__device__ __forceinline__ void FusedGdrOutputVdReadArrive()
{
    cutlass::arch::NamedBarrier::arrive(kFusedGdrGateVdReadThreads, kFusedGdrBarrierVdReadDone);
}

__device__ __forceinline__ void FusedGdrFenceProxyAsyncShared()
{
    // Release prior generic-proxy CTA shared writes to WGMMA's async proxy. This is
    // neither a thread rendezvous nor an async WGMMA completion wait.
    cutlass::arch::fence_view_async_shared();
}

// Dormant in the active SM90 runtime: these state-pack handoff helpers have no callers.
__device__ __forceinline__ void FusedGdrStatePackArrive()
{
    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kFusedGdrBarrierStateToValue);
    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kFusedGdrBarrierStateToOutput);
}

__device__ __forceinline__ void FusedGdrStatePackValueSync()
{
    cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kFusedGdrBarrierStateToValue);
}

__device__ __forceinline__ void FusedGdrStatePackOutputSync()
{
    cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kFusedGdrBarrierStateToOutput);
}

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

__device__ int CeilDivDevice(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

enum FusedGdrTmaDescIndex : int
{
    kFusedGdrQDesc = 0,
    kFusedGdrKDesc,
    kFusedGdrVDesc,
    kFusedGdrResolventDesc,
    kFusedGdrOutDesc,
    kFusedGdrStateDesc,
};

enum FusedGdrHTmaDescIndex : int
{
    kFusedGdrHKDesc = 0,
    kFusedGdrHVDesc,
    kFusedGdrHGDesc,
    kFusedGdrHResolventDesc,
    kFusedGdrHSegmentStateDesc,
    kFusedGdrHSegmentMDesc,
};

enum CorrectInitialStatesTmaDescIndex : int
{
    kCorrectInitialStatesCpStateDesc = 0,
    kCorrectInitialStatesSegmentStateDesc,
    kCorrectInitialStatesSegmentMDesc,
    kCorrectInitialStatesExternalStateDesc,
};

constexpr int kDirectFusedDvTiles = kHeadDim / kFusedGdrBlockDv;
constexpr int kFusedGdrHDvTiles   = kHeadDim / kFusedGdrHBlockDv;

constexpr int kFusedGdrDataDescCount  = kFusedGdrStateDesc;
constexpr int kFusedGdrStateDescCount = kFusedGdrTmaDescCount - kFusedGdrDataDescCount;

constexpr int kFusedGdrHDataDescCount                 = kFusedGdrHSegmentStateDesc;
constexpr int kFusedGdrHTensorDescCount               = kFusedGdrHTmaDescCount - kFusedGdrHDataDescCount;
constexpr int kContextParallelFusedGdrTensorDescCount = 1;
constexpr int kCorrectInitialStatesExternalDescCount =
    kCorrectInitialStatesTmaDescCount - kCorrectInitialStatesExternalStateDesc;

static_assert(kDirectFusedDvTiles == 2);
static_assert(kFusedGdrHDvTiles == 1);
static_assert(kFusedGdrDataDescCount + kFusedGdrStateDescCount == kFusedGdrTmaDescCount);
static_assert(kFusedGdrHDataDescCount == kFusedGdrHSegmentStateDesc);
static_assert(kFusedGdrHTensorDescCount == 2);
static_assert(kContextParallelFusedGdrTensorDescCount == 1);
static_assert(kCorrectInitialStatesExternalStateDesc + kCorrectInitialStatesExternalDescCount
              == kCorrectInitialStatesTmaDescCount);
static_assert(kFusedGdrStateDescCount == 1);
static_assert(kCorrectInitialStatesExternalDescCount == 1);
static_assert(kFusedGdrHSegmentStateDesc - kFusedGdrHDataDescCount == 0);
static_assert(kFusedGdrHSegmentMDesc - kFusedGdrHDataDescCount == 1);
static_assert(kCorrectInitialStatesCpStateDesc < kCorrectInitialStatesExternalStateDesc);
static_assert(kCorrectInitialStatesSegmentStateDesc < kCorrectInitialStatesExternalStateDesc);
static_assert(kCorrectInitialStatesSegmentMDesc < kCorrectInitialStatesExternalStateDesc);

template<class TensorMapPtr>
struct FusedGdrTmaDescriptorSlices {
    TensorMapPtr data{};
    TensorMapPtr state{};
};

template<class TensorMapPtr>
CUTE_HOST_DEVICE FusedGdrTmaDescriptorSlices<TensorMapPtr> MakeFusedGdrTmaDescriptorSlices(TensorMapPtr base,
                                                                                           int          sequence_num)
{
    FusedGdrTmaDescriptorSlices<TensorMapPtr> out{};
    out.data  = base;
    out.state = out.data + sequence_num * kFusedGdrDataDescCount;
    return out;
}

template<class TensorMapPtr>
struct ContextParallelFusedGdrTmaDescriptorSlices {
    TensorMapPtr data{};
    TensorMapPtr cp_state{};
};

template<class TensorMapPtr>
CUTE_HOST_DEVICE ContextParallelFusedGdrTmaDescriptorSlices<TensorMapPtr>
                 MakeContextParallelFusedGdrTmaDescriptorSlices(TensorMapPtr base, int sequence_num)
{
    ContextParallelFusedGdrTmaDescriptorSlices<TensorMapPtr> out{};
    out.data     = base;
    out.cp_state = out.data + sequence_num * kFusedGdrDataDescCount;
    return out;
}

template<class TensorMapPtr>
struct FusedGdrHTmaDescriptorSlices {
    TensorMapPtr data{};
    TensorMapPtr segment_state{};
    TensorMapPtr segment_m{};
};

template<class TensorMapPtr>
CUTE_HOST_DEVICE FusedGdrHTmaDescriptorSlices<TensorMapPtr> MakeFusedGdrHTmaDescriptorSlices(TensorMapPtr base,
                                                                                             int          sequence_num)
{
    FusedGdrHTmaDescriptorSlices<TensorMapPtr> out{};
    out.data          = base;
    out.segment_state = out.data + sequence_num * kFusedGdrHDataDescCount;
    out.segment_m     = out.segment_state + 1;
    return out;
}

template<class TensorMapPtr>
struct CorrectInitialStatesTmaDescriptorSlices {
    TensorMapPtr cp_state{};
    TensorMapPtr segment_state{};
    TensorMapPtr segment_m{};
    TensorMapPtr external_state{};
};

template<class TensorMapPtr>
CUTE_HOST_DEVICE CorrectInitialStatesTmaDescriptorSlices<TensorMapPtr>
                 MakeCorrectInitialStatesTmaDescriptorSlices(TensorMapPtr base)
{
    CorrectInitialStatesTmaDescriptorSlices<TensorMapPtr> out{};
    out.cp_state       = base + kCorrectInitialStatesCpStateDesc;
    out.segment_state  = base + kCorrectInitialStatesSegmentStateDesc;
    out.segment_m      = base + kCorrectInitialStatesSegmentMDesc;
    out.external_state = base + kCorrectInitialStatesExternalStateDesc;
    return out;
}

__device__ __forceinline__ int FusedGdrValueTmaCoord(int value_head, int dv0)
{
    return value_head * kHeadDim + dv0;
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

template<class StateT>
__device__ __forceinline__ float FusedGdrLoadState(const StateT* state, int64_t offset)
{
    return static_cast<float>(state[offset]);
}

template<>
__device__ __forceinline__ float FusedGdrLoadState<__nv_bfloat16>(const __nv_bfloat16* state, int64_t offset)
{
    return __bfloat162float(state[offset]);
}

template<class StateT>
__device__ __forceinline__ void FusedGdrStoreState(StateT* state, int64_t offset, float value)
{
    state[offset] = static_cast<StateT>(value);
}

template<>
__device__ __forceinline__ void FusedGdrStoreState<__nv_bfloat16>(__nv_bfloat16* state, int64_t offset, float value)
{
    state[offset] = __float2bfloat16(value);
}

template<class StateT, int BlockDv>
__device__ __forceinline__ constexpr int FusedGdrStateTmaBytes()
{
    return kHeadDim * BlockDv * static_cast<int>(sizeof(StateT));
}

// Dormant in the active SM90 runtime: the state TMA pack/unpack templates below
// have no SM90 callers and are retained without an active handoff contract.
template<class StateT, int BlockDv>
__device__ __forceinline__ void FusedGdrUnpackStateTma(float (&state_stage)[kHeadDim][BlockDv], int tid, int threads)
{
    if constexpr (!std::is_same_v<StateT, float>) {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        static_assert(BlockDv % 2 == 0);
        constexpr int kPairs = kHeadDim * BlockDv / 2;
        static_assert((kPairs & (kPairs - 1)) == 0);
        static_assert(alignof(decltype(state_stage)) >= alignof(__nv_bfloat162));
        auto* overlay = reinterpret_cast<__nv_bfloat162*>(&state_stage[0][0]);
        // The compact bf16 TMA payload aliases the low half of the float tile;
        // expand high-to-low so each phase overwrites only pairs already read.
        for (int begin = kPairs / 2; begin > 0; begin >>= 1) {
            for (int linear = begin + tid; linear < 2 * begin; linear += threads) {
                const int    element    = 2 * linear;
                const int    dk         = element / BlockDv;
                const int    dv         = element - dk * BlockDv;
                const float2 pair       = __bfloat1622float2(overlay[linear]);
                state_stage[dk][dv]     = pair.x;
                state_stage[dk][dv + 1] = pair.y;
            }
            cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        }
        if (tid == 0) {
            const float2 pair = __bfloat1622float2(overlay[0]);
            state_stage[0][0] = pair.x;
            state_stage[0][1] = pair.y;
        }
        cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
    }
}

template<class StateT, int BlockDv>
__device__ __forceinline__ void FusedGdrUnpackStateTma(float* state_stage, int tid, int threads)
{
    if constexpr (!std::is_same_v<StateT, float>) {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        static_assert(BlockDv % 2 == 0);
        constexpr int kPairs = kHeadDim * BlockDv / 2;
        static_assert((kPairs & (kPairs - 1)) == 0);
        static_assert(alignof(float) >= alignof(__nv_bfloat162));
        auto* overlay = reinterpret_cast<__nv_bfloat162*>(state_stage);
        // The compact bf16 TMA payload aliases the low half of the float tile;
        // expand high-to-low so each phase overwrites only pairs already read.
        for (int begin = kPairs / 2; begin > 0; begin >>= 1) {
            for (int linear = begin + tid; linear < 2 * begin; linear += threads) {
                const int    element     = 2 * linear;
                const float2 pair        = __bfloat1622float2(overlay[linear]);
                state_stage[element]     = pair.x;
                state_stage[element + 1] = pair.y;
            }
            cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        }
        if (tid == 0) {
            const float2 pair = __bfloat1622float2(overlay[0]);
            state_stage[0]    = pair.x;
            state_stage[1]    = pair.y;
        }
        cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
    }
}

template<class StateT, int BlockDv>
__device__ __forceinline__ void FusedGdrPackStateTma(float (&state_stage)[kHeadDim][BlockDv], int tid, int threads)
{
    if constexpr (!std::is_same_v<StateT, float>) {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        static_assert(BlockDv % 2 == 0);
        constexpr int kPairs = kHeadDim * BlockDv / 2;
        static_assert((kPairs & (kPairs - 1)) == 0);
        static_assert(alignof(decltype(state_stage)) >= alignof(__nv_bfloat162));
        auto* overlay = reinterpret_cast<__nv_bfloat162*>(&state_stage[0][0]);
        // Packing has the inverse dependency: write low-to-high so compact pairs
        // only overwrite float elements that earlier phases already consumed.
        if (tid == 0) {
            overlay[0] = __float22bfloat162_rn(make_float2(state_stage[0][0], state_stage[0][1]));
        }
        cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        for (int begin = 1; begin < kPairs; begin <<= 1) {
            for (int linear = begin + tid; linear < 2 * begin; linear += threads) {
                const int element = 2 * linear;
                const int dk      = element / BlockDv;
                const int dv      = element - dk * BlockDv;
                overlay[linear]   = __float22bfloat162_rn(make_float2(state_stage[dk][dv], state_stage[dk][dv + 1]));
            }
            cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        }
    }
}

template<class StateT, int BlockDv>
__device__ __forceinline__ void FusedGdrPackStateTma(float* state_stage, int tid, int threads)
{
    if constexpr (!std::is_same_v<StateT, float>) {
        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
        static_assert(BlockDv % 2 == 0);
        constexpr int kPairs = kHeadDim * BlockDv / 2;
        static_assert((kPairs & (kPairs - 1)) == 0);
        static_assert(alignof(float) >= alignof(__nv_bfloat162));
        auto* overlay = reinterpret_cast<__nv_bfloat162*>(state_stage);
        // Packing has the inverse dependency: write low-to-high so compact pairs
        // only overwrite float elements that earlier phases already consumed.
        if (tid == 0) {
            overlay[0] = __float22bfloat162_rn(make_float2(state_stage[0], state_stage[1]));
        }
        cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        for (int begin = 1; begin < kPairs; begin <<= 1) {
            for (int linear = begin + tid; linear < 2 * begin; linear += threads) {
                const int element = 2 * linear;
                overlay[linear]   = __float22bfloat162_rn(make_float2(state_stage[element], state_stage[element + 1]));
            }
            cutlass::arch::NamedBarrier::sync(threads, kFusedGdrBarrierProducerStatePack);
        }
    }
}

__device__ __forceinline__ float FastExp(float value)
{
    return exp2f(value * 1.4426950408889634f);
}

template<class T>
__device__ T CastFromFloat(float value);

template<>
__device__ __nv_bfloat16 CastFromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template<class T>
struct FusedGdrCastToElement {
    using Element = typename FusedGdrMmaTraits<T>::Element;

    __device__ __forceinline__ Element operator()(float value) const
    {
        return Element(CastFromFloat<T>(value));
    }
};

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
        FusedGdrFenceProxyAsyncShared();
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

template<class TiledMma, class AFragment, class TB, class BLayout, class Accumulator>
__device__ __forceinline__ void FusedGdrGmmaRs(TiledMma&                        tiled_mma,
                                               uint32_t                         thread_idx,
                                               AFragment const&                 tCrA,
                                               cute::Tensor<TB, BLayout> const& sB,
                                               Accumulator&                     tCrC,
                                               cute::SM90::GMMA::ScaleOut       scale)
{
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto tCsB    = thr_mma.partition_B(sB);
    auto tCrB    = thr_mma.make_fragment_B(tCsB);

    constexpr int K_BLOCK_MAX = cute::size<2>(AFragment{});
    static_assert(K_BLOCK_MAX == cute::size<2>(decltype(tCrB){}));

    FusedGdrFenceProxyAsyncShared();
    cute::warpgroup_fence_operand(tCrC);
    cute::warpgroup_arrive();
    tiled_mma.accumulate_ = scale;
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
        tCrPack(i) = Element(CastFromFloat<T>(static_cast<float>(fragment(i))));
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
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv || BlockDv == kWideGdrBlockDv);
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

    FusedGdrFenceProxyAsyncShared();
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

template<bool ApplyGate,
         class T,
         class... MmaArgs,
         class TA,
         class ALayout,
         class TB,
         class BLayout,
         class StateFragment>
__device__ __forceinline__ void FusedGdrStateUpdateFragmentFromVd(uint32_t                          thread_idx,
                                                                  cute::TiledMMA<MmaArgs...> const& tiled_mma,
                                                                  cute::Tensor<TA, ALayout> const&  sA,
                                                                  cute::Tensor<TB, BLayout> const&  sB,
                                                                  StateFragment&                    tCrState,
                                                                  int                               valid,
                                                                  const float*                      g,
                                                                  float                             last_g_exp)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    using InputTypeA   = typename TA::value_type;
    using InputTypeB   = typename TB::value_type;
    using ComputeTypeA = typename cute::TiledMMA<MmaArgs...>::ValTypeA;
    using ComputeTypeB = typename cute::TiledMMA<MmaArgs...>::ValTypeB;

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    auto tCrA  = thr_mma.partition_fragment_A(sA);
    auto tCrAi = cute::make_fragment_like<InputTypeA>(tCrA);
    auto tCrB  = thr_mma.partition_fragment_B(sB);
    auto tCrBi = cute::make_fragment_like<InputTypeB>(tCrB);

    auto smem_tiled_copy_A = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeA>{}, thr_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    auto tCsA              = smem_thr_copy_A.partition_S(sA);
    auto tCrAi_copy_view   = smem_thr_copy_A.retile_D(tCrAi);

    auto smem_tiled_copy_B = cute::make_tiled_copy_B(cute::Copy_Atom<cute::DefaultCopy, InputTypeB>{}, thr_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    auto tCsB              = smem_thr_copy_B.partition_S(sB);
    auto tCrBi_copy_view   = smem_thr_copy_B.retile_D(tCrBi);

    auto cB   = cute::make_identity_tensor(cute::shape(sB));
    auto cA   = cute::make_identity_tensor(cute::shape(sA));
    auto tCcA = thr_mma.partition_A(cA);
    auto tCcB = thr_mma.partition_B(cB);

    cute::copy(
        smem_tiled_copy_A, tCsA(cute::_, cute::_, cute::Int<0>{}), tCrAi_copy_view(cute::_, cute::_, cute::Int<0>{}));
    cute::copy(
        smem_tiled_copy_B, tCsB(cute::_, cute::_, cute::Int<0>{}), tCrBi_copy_view(cute::_, cute::_, cute::Int<0>{}));

    constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block < K_BLOCK_MAX - 1) {
            const int k_next = k_block + 1;
            cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, k_next), tCrAi_copy_view(cute::_, cute::_, k_next));
            cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, k_next), tCrBi_copy_view(cute::_, cute::_, k_next));
        }

#pragma unroll
        for (int n = 0; n < cute::size<1>(tCrB); ++n) {
#pragma unroll
            for (int i = 0; i < cute::size<0>(tCrB); ++i) {
                auto      coord = tCcB(i, n, k_block);
                const int row   = cute::get<1>(coord);
                float     value = 0.0f;
                if (row < valid) {
                    value = static_cast<float>(tCrBi(i, n, k_block));
                    if constexpr (ApplyGate) {
                        value *= last_g_exp * g[row];
                    }
                }
                tCrB(i, n, k_block) = ComputeTypeB(CastFromFloat<T>(value));
            }
        }

#pragma unroll
        for (int m = 0; m < cute::size<1>(tCrA); ++m) {
#pragma unroll
            for (int i = 0; i < cute::size<0>(tCrA); ++i) {
                auto      coord     = tCcA(i, m, k_block);
                auto      row_coord = cute::get<1>(coord);
                const int row       = cute::get<0>(row_coord) + 8 * cute::get<1>(row_coord);
                tCrA(i, m, k_block) =
                    row < valid ? ComputeTypeA(tCrAi(i, m, k_block)) : ComputeTypeA(CastFromFloat<T>(0.0f));
            }
        }
        cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrState);
    }
}

template<class T, class... MmaArgs, class TA, class ALayout, class TB, class BLayout, class StateFragment>
__device__ __forceinline__ void FusedGdrStateUpdateFragmentFromBf16Vd(uint32_t                          thread_idx,
                                                                      cute::TiledMMA<MmaArgs...> const& tiled_mma,
                                                                      cute::Tensor<TA, ALayout> const&  sA,
                                                                      cute::Tensor<TB, BLayout> const&  sB,
                                                                      StateFragment&                    tCrState,
                                                                      const float*                      g,
                                                                      float                             last_g_exp)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    using Element      = typename FusedGdrMmaTraits<T>::Element;
    using InputTypeA   = typename TA::value_type;
    using InputTypeB   = typename TB::value_type;
    using ComputeTypeB = typename cute::TiledMMA<MmaArgs...>::ValTypeB;
    static_assert(std::is_same_v<InputTypeB, Element>);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    auto tCrA  = thr_mma.partition_fragment_A(sA);
    auto tCrAi = cute::make_fragment_like<InputTypeA>(tCrA);
    auto tCrB  = thr_mma.partition_fragment_B(sB);
    auto tCrBi = cute::make_fragment_like<InputTypeB>(tCrB);

    auto smem_tiled_copy_A = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeA>{}, thr_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    auto tCsA              = smem_thr_copy_A.partition_S(sA);
    auto tCrAi_copy_view   = smem_thr_copy_A.retile_D(tCrAi);

    auto smem_tiled_copy_B = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeB>{}, thr_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    auto tCsB              = smem_thr_copy_B.partition_S(sB);
    auto tCrBi_copy_view   = smem_thr_copy_B.retile_D(tCrBi);

    auto cB   = cute::make_identity_tensor(cute::shape(sB));
    auto tCcB = thr_mma.partition_B(cB);

    cute::copy(
        smem_tiled_copy_A, tCsA(cute::_, cute::_, cute::Int<0>{}), tCrAi_copy_view(cute::_, cute::_, cute::Int<0>{}));
    cute::copy(
        smem_tiled_copy_B, tCsB(cute::_, cute::_, cute::Int<0>{}), tCrBi_copy_view(cute::_, cute::_, cute::Int<0>{}));

    constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block < K_BLOCK_MAX - 1) {
            const int k_next = k_block + 1;
            cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, k_next), tCrAi_copy_view(cute::_, cute::_, k_next));
            cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, k_next), tCrBi_copy_view(cute::_, cute::_, k_next));
        }

#pragma unroll
        for (int n = 0; n < cute::size<1>(tCrB); ++n) {
#pragma unroll
            for (int i = 0; i < cute::size<0>(tCrB); ++i) {
                auto        coord   = tCcB(i, n, k_block);
                const int   row     = cute::get<1>(coord);
                const float gate    = last_g_exp * g[row];
                const float value   = gate * static_cast<float>(tCrBi(i, n, k_block));
                tCrB(i, n, k_block) = ComputeTypeB(CastFromFloat<T>(value));
            }
        }

        cute::transform(tCrAi(cute::_, cute::_, k_block), tCrA(cute::_, cute::_, k_block), cute::identity{});
        cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrState);
    }
}

template<class T, class... MmaArgs, class TA, class ALayout, class TB, class BLayout, class StateFragment>
__device__ __forceinline__ void FusedGdrStateUpdateFragmentFromScaledVd(uint32_t                          thread_idx,
                                                                        cute::TiledMMA<MmaArgs...> const& tiled_mma,
                                                                        cute::Tensor<TA, ALayout> const&  sA,
                                                                        cute::Tensor<TB, BLayout> const&  sB,
                                                                        StateFragment&                    tCrState)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    using Element    = typename FusedGdrMmaTraits<T>::Element;
    using InputTypeA = typename TA::value_type;
    using InputTypeB = typename TB::value_type;
    static_assert(std::is_same_v<InputTypeB, Element>);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    auto tCrA  = thr_mma.partition_fragment_A(sA);
    auto tCrAi = cute::make_fragment_like<InputTypeA>(tCrA);
    auto tCrB  = thr_mma.partition_fragment_B(sB);
    auto tCrBi = cute::make_fragment_like<InputTypeB>(tCrB);

    auto smem_tiled_copy_A = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeA>{}, thr_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    auto tCsA              = smem_thr_copy_A.partition_S(sA);
    auto tCrAi_copy_view   = smem_thr_copy_A.retile_D(tCrAi);

    auto smem_tiled_copy_B = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeB>{}, thr_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    auto tCsB              = smem_thr_copy_B.partition_S(sB);
    auto tCrBi_copy_view   = smem_thr_copy_B.retile_D(tCrBi);

    cute::copy(
        smem_tiled_copy_A, tCsA(cute::_, cute::_, cute::Int<0>{}), tCrAi_copy_view(cute::_, cute::_, cute::Int<0>{}));
    cute::copy(
        smem_tiled_copy_B, tCsB(cute::_, cute::_, cute::Int<0>{}), tCrBi_copy_view(cute::_, cute::_, cute::Int<0>{}));

    constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block < K_BLOCK_MAX - 1) {
            const int k_next = k_block + 1;
            cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, k_next), tCrAi_copy_view(cute::_, cute::_, k_next));
            cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, k_next), tCrBi_copy_view(cute::_, cute::_, k_next));
        }

        cute::transform(tCrAi(cute::_, cute::_, k_block), tCrA(cute::_, cute::_, k_block), cute::identity{});
        cute::transform(tCrBi(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), cute::identity{});
        cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrState);
    }
}

template<class StateFragment, class CoordTensor, class StageTensor>
__device__ __forceinline__ void
FusedGdrLoadStateFragment(StateFragment& tCrState, CoordTensor const& tCcState, StageTensor const& s_state_stage)
{
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        auto      coord = tCcState(i);
        const int dk    = cute::get<0>(coord);
        const int dv    = cute::get<1>(coord);
        tCrState(i)     = s_state_stage(dk, dv);
    }
}

template<class StateFragment>
__device__ __forceinline__ void FusedGdrDecayStateFragment(StateFragment& tCrState, float decay)
{
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        tCrState(i) *= decay;
    }
}

template<class StateFragment, class CoordTensor, class StageTensor>
__device__ __forceinline__ void
FusedGdrStoreStateFragmentFloat(StateFragment const& tCrState, CoordTensor const& tCcState, StageTensor& s_state_stage)
{
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        auto      coord       = tCcState(i);
        const int dk          = cute::get<0>(coord);
        const int dv          = cute::get<1>(coord);
        s_state_stage(dk, dv) = tCrState(i);
    }
}

template<class StateFragment, class GlobalTensor, class ThrMma>
__device__ __forceinline__ void FusedGdrStoreStateFragmentFloatGlobal(StateFragment const& tCrState,
                                                                      GlobalTensor const&  g_state,
                                                                      ThrMma const&        thr_mma,
                                                                      int                  role_tid)
{
    auto gmem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, float>{}, thr_mma);
    auto gmem_thr_copy_C   = gmem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCgState          = gmem_thr_copy_C.partition_D(g_state);
    auto tCrStateView      = gmem_thr_copy_C.retile_S(tCrState);
    cute::copy(gmem_tiled_copy_C, tCrStateView, tCgState);
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

template<int BlockDv, class StateFragment, class CoordTensor>
__device__ __forceinline__ void FusedGdrStoreStateFragmentBf16Tma(StateFragment const& tCrState,
                                                                  CoordTensor const&   tCcState,
                                                                  __nv_bfloat16*       state_stage)
{
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        auto      coord                = tCcState(i);
        const int dk                   = cute::get<0>(coord);
        const int dv                   = cute::get<1>(coord);
        state_stage[dk * BlockDv + dv] = __float2bfloat16(static_cast<float>(tCrState(i)));
    }
}

template<class T, int BlockDv, class StateFragment, class CoordTensor>
__device__ __forceinline__ void FusedGdrStoreStateFragmentBf16(StateFragment const&                    tCrState,
                                                               CoordTensor const&                      tCcState,
                                                               typename FusedGdrMmaTraits<T>::Element* state_pack)
{
    using Element     = typename FusedGdrMmaTraits<T>::Element;
    auto s_state_pack = cute::make_tensor(cute::make_smem_ptr(state_pack), FusedGdrSwizzledStateTLayout<BlockDv>());

#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        auto      coord = tCcState(i);
        const int dk    = cute::get<0>(coord);
        const int dv    = cute::get<1>(coord);
        // WG1/WG2 consume state as (dv, dk) BF16 so their K@state and Q@state GEMMs keep TN operand layout.
        s_state_pack(dv, dk) = Element(CastFromFloat<T>(static_cast<float>(tCrState(i))));
    }
}

template<class T, int BlockDv, class StateFragment, class ThrMma>
__device__ __forceinline__ void FusedGdrStoreStateFragmentBf16Stsm(StateFragment const&                    tCrState,
                                                                   typename FusedGdrMmaTraits<T>::Element* state_pack,
                                                                   ThrMma const&                           thr_mma,
                                                                   int                                     role_tid)
{
    using Element = typename FusedGdrMmaTraits<T>::Element;
    static_assert(std::is_same_v<T, __nv_bfloat16>);

    // STSM writes the C fragment as (dk, dv); consumers read the same physical
    // BF16 bytes through StateTLayout as (dv, dk).
    auto s_state_pack = cute::as_position_independent_swizzle_tensor(
        cute::make_tensor(cute::make_smem_ptr(state_pack), FusedGdrSwizzledStateCLayout<BlockDv>()));
    auto tCrPack = cute::make_fragment_like<Element>(tCrState);
#pragma unroll
    for (int i = 0; i < cute::size(tCrState); ++i) {
        tCrPack(i) = Element(CastFromFloat<T>(static_cast<float>(tCrState(i))));
    }

    auto smem_tiled_copy_C = cute::make_tiled_copy_C_atom(cute::Copy_Atom<cute::SM90_U32x2_STSM_N, Element>{}, thr_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCsPack           = smem_thr_copy_C.partition_D(s_state_pack);
    auto tCrPackView       = smem_thr_copy_C.retile_S(tCrPack);
    cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
}

static_assert(alignof(__nv_bfloat162) <= 16);
static_assert((kFusedGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kFusedGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kContextParallelGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kContextParallelGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kWideGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kWideGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kChunkSize * kChunkSize * sizeof(cute::bfloat16_t)) % alignof(__nv_bfloat162) == 0);

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
