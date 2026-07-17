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
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int   kChunkSize = 64;
constexpr int   kHeadDim   = 128;
constexpr float kHeadScale = 0.08838834764831845f;

constexpr int      kFusedGdrBlockDv                            = 64;
constexpr int      kContextParallelGdrBlockDv                  = 32;
constexpr int      kCorrectInitialStatesF32BlockDv             = 4;
constexpr int      kCorrectInitialStatesBf16BlockDv            = kCorrectInitialStatesF32BlockDv;
constexpr int      kCorrectInitialStatesBf16ExternalTmaBlockDv = 8;
constexpr int      kCorrectInitialStatesKTile                  = 128;
constexpr int      kCorrectInitialStatesMRowsPerTma            = 32;
constexpr int      kFusedGdrTmaDescCount                       = 7;
constexpr int      kKktTmaDescCount                            = 2;
constexpr int      kFusedGdrHTmaDescCount                      = 6;
constexpr int      kCorrectInitialStatesTmaDescCount           = 4;
constexpr int      kTmaDescriptorBytes                         = 128;
constexpr int      kFusedGdrMmaThreads                         = 128;
constexpr int      kFusedGdrRoleThreads                        = 128;
constexpr int      kFusedGdrConsumerThreads                    = 3 * kFusedGdrRoleThreads;
constexpr int      kFusedGdrProducerThreads                    = kFusedGdrRoleThreads;
constexpr int      kFusedGdrThreads                            = kFusedGdrConsumerThreads + kFusedGdrProducerThreads;
constexpr int      kCudaWarpThreads                            = 32;
constexpr int      kFusedGdrGateRowsPerWarp                    = 8;
constexpr int      kFusedGdrGateWriterThreads = (kFusedGdrRoleThreads / kCudaWarpThreads) * kFusedGdrGateRowsPerWarp;
constexpr int      kFusedGdrGatePasses        = kChunkSize / kFusedGdrGateWriterThreads;
constexpr int      kFusedGdrStateRegisters    = 144;
constexpr int      kFusedGdrValueRegisters    = 144;
constexpr int      kFusedGdrOutputRegisters   = 160;
constexpr int      kFusedGdrProducerRegisters = 24;
constexpr size_t   kFusedGdrSm120SharedBytes  = 102400;
constexpr size_t   kFusedGdrStaticSharedReserveBytes = 1024;
constexpr size_t   kFusedGdrMaxDynamicSharedBytes    = kFusedGdrSm120SharedBytes - kFusedGdrStaticSharedReserveBytes;
constexpr uint64_t kTmaNoCacheHint                   = 0;

static_assert(kFusedGdrBlockDv == 64);
static_assert(kContextParallelGdrBlockDv == 32);
static_assert(kFusedGdrThreads == 512);
static_assert(kFusedGdrThreads >= kFusedGdrMmaThreads);
static_assert(kFusedGdrMmaThreads == 128);
static_assert(kFusedGdrRoleThreads == kFusedGdrMmaThreads);
static_assert(kFusedGdrConsumerThreads == 3 * kFusedGdrRoleThreads);
static_assert(kFusedGdrProducerThreads == kFusedGdrRoleThreads);
static_assert(kFusedGdrThreads == 4 * kFusedGdrRoleThreads);
static_assert(kFusedGdrRoleThreads % kCudaWarpThreads == 0);
static_assert(kFusedGdrGateRowsPerWarp <= kCudaWarpThreads);
static_assert(kChunkSize % kFusedGdrGateWriterThreads == 0);
static_assert(kFusedGdrGateWriterThreads * kFusedGdrGatePasses == kChunkSize);
static_assert(kHeadDim % kFusedGdrBlockDv == 0);
static_assert(kHeadDim % kContextParallelGdrBlockDv == 0);
static_assert(kHeadDim % kCorrectInitialStatesF32BlockDv == 0);
static_assert(kHeadDim % kCorrectInitialStatesBf16BlockDv == 0);
static_assert(kCorrectInitialStatesBf16BlockDv == kCorrectInitialStatesF32BlockDv);
static_assert(kCorrectInitialStatesBf16ExternalTmaBlockDv >= kCorrectInitialStatesBf16BlockDv);
static_assert(kCorrectInitialStatesBf16ExternalTmaBlockDv * static_cast<int>(sizeof(__nv_bfloat16)) >= 16);
static_assert(kCorrectInitialStatesKTile == kHeadDim);
static_assert(kHeadDim % kCorrectInitialStatesMRowsPerTma == 0);
static_assert(sizeof(CUtensorMap) == kTmaDescriptorBytes);
static_assert(kFusedGdrTmaDescCount * sizeof(CUtensorMap) <= kFusedGdrStaticSharedReserveBytes);

template<class StateT>
constexpr bool kFusedGdrValidStateT = std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>;

constexpr const char* kContextParallelSegmentUnsupportedMessage = "Fused GDR H requires fixed CP segment tensors";
constexpr const char* kGdrTargetUnsupportedMessage =
    "fused GDR forward supports only the SM120 bf16 chunked target shape "
    "(int32 q_offsets, bool finished mask, head_dim=128, chunk_size=32 or 64, Hv % Hq == 0)";

template<int ChunkSize>
constexpr bool kSupportedGdrChunkSize = ChunkSize == 32 || ChunkSize == 64;

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

CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledA64Layout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_64, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(FusedGdrSwizzledA64Layout())> == kChunkSize * kChunkSize);

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
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledV32TLayout();
    }
    else {
        return FusedGdrSwizzledV64TLayout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledVRowLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledV32RowLayout();
    }
    else {
        return FusedGdrSwizzledA64Layout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledStateTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledState32TLayout();
    }
    else {
        return FusedGdrSwizzledState64TLayout();
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto FusedGdrSwizzledStateCLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return FusedGdrSwizzledState32CLayout();
    }
    else {
        return FusedGdrSwizzledState64CLayout();
    }
}

constexpr int kFusedGdrBarrierConsumer           = 0;
constexpr int kFusedGdrBarrierStateUpdate        = 1;
constexpr int kFusedGdrBarrierValueU             = 2;
constexpr int kFusedGdrBarrierOutputState        = 3;
constexpr int kFusedGdrBarrierOutputP            = 4;
constexpr int kFusedGdrBarrierOutputLocal        = 5;
constexpr int kFusedGdrBarrierHStateReadDone     = 6;
constexpr int kFusedGdrBarrierHVdReady           = 5;  // FusedGdrH does not use OutputLocal.
constexpr int kFusedGdrBarrierMVdReady           = 7;
constexpr int kFusedGdrBarrierProducerStatePack  = 6;  // FusedGdrFwd does not use HStateReadDone.
constexpr int kFusedGdrBarrierProducerStateStore = 7;  // FusedGdrFwd does not use MVdReady.
constexpr int kSm120FusedGdrBarrierAgReady       = kFusedGdrBarrierOutputState;
constexpr int kSm120FusedGdrBarrierPackedVd      = kFusedGdrBarrierProducerStatePack;
constexpr int kFusedGdrBarrierStateToValue       = kFusedGdrBarrierProducerStatePack;
constexpr int kFusedGdrBarrierStateToOutput      = kFusedGdrBarrierProducerStateStore;
static_assert(kFusedGdrBarrierMVdReady + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
              < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

template<int BarrierId>
__device__ __forceinline__ void FusedGdrMmaSyncNamed()
{
    cutlass::arch::NamedBarrier::sync(kFusedGdrRoleThreads, BarrierId);
}

__device__ __forceinline__ void FusedGdrConsumerSync()
{
    cutlass::arch::NamedBarrier::sync(kFusedGdrConsumerThreads, kFusedGdrBarrierConsumer);
}

__device__ __forceinline__ void FusedGdrConsumerArrive()
{
    cutlass::arch::NamedBarrier::arrive(kFusedGdrConsumerThreads, kFusedGdrBarrierConsumer);
}

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

__device__ __forceinline__ void Sm120FusedGdrPackedVdSync()
{
    cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kSm120FusedGdrBarrierPackedVd);
}

__device__ __forceinline__ void Sm120FusedGdrAgReadyArrive()
{
    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kSm120FusedGdrBarrierAgReady);
}

__device__ __forceinline__ void Sm120FusedGdrAgReadySync()
{
    cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kSm120FusedGdrBarrierAgReady);
}

template<class T>
struct FusedGdrMmaTraits;

template<>
struct FusedGdrMmaTraits<__nv_bfloat16> {
    using Element = cute::bfloat16_t;
    using Atom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

template<class T>
constexpr CUtensorMapDataType FusedGdrTmaDataType()
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template<>
constexpr CUtensorMapDataType FusedGdrTmaDataType<float>()
{
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
}

inline CUtensorMapSwizzle FusedGdrBf16TmaSwizzle(int block_dv)
{
    if (block_dv == kContextParallelGdrBlockDv) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }
    if (block_dv == kFusedGdrBlockDv) {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
    return CU_TENSOR_MAP_SWIZZLE_128B;
}

template<int ChunkSize>
constexpr CUtensorMapSwizzle FusedGdrSquareTmaSwizzle()
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    if constexpr (ChunkSize == 32) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }
    else {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

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
    kFusedGdrGDesc,
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
constexpr int kFusedGdrHDvTiles   = kHeadDim / kContextParallelGdrBlockDv;

constexpr int kFusedGdrDataDescCount  = kFusedGdrStateDesc;
constexpr int kFusedGdrStateDescCount = kFusedGdrTmaDescCount - kFusedGdrDataDescCount;

constexpr int kFusedGdrHDataDescCount                 = kFusedGdrHSegmentStateDesc;
constexpr int kFusedGdrHTensorDescCount               = kFusedGdrHTmaDescCount - kFusedGdrHDataDescCount;
constexpr int kContextParallelFusedGdrTensorDescCount = 1;
constexpr int kCorrectInitialStatesExternalDescCount =
    kCorrectInitialStatesTmaDescCount - kCorrectInitialStatesExternalStateDesc;

constexpr int kFusedGdrGateTmaHeads = 4;

static_assert(kDirectFusedDvTiles == 2);
static_assert(kFusedGdrHDvTiles == 4);
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

template<class T>
struct StridedTensorBase {
    T*      ptr{};
    int64_t batch_stride{};
    int64_t token_stride{};
};

template<class T>
StridedTensorBase<T> MakeStridedTensorBase(core::Tensor& tensor)
{
    return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
}

template<class T>
StridedTensorBase<const T> MakeStridedTensorBase(const core::Tensor& tensor)
{
    return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
}

template<int Dim>
__device__ __forceinline__ void FusedGdrReplaceTmaAddressAndDim(CUtensorMap* desc, const void* global_address, int dim)
{
    const uint32_t smem_ptr = cast_smem_ptr_to_uint(desc);
    uint64_t       smem_ptr64;
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :
                 : "r"(smem_ptr), "l"(global_address));
    asm volatile("cvt.u64.u32 %0, %1;" : "=l"(smem_ptr64) : "r"(smem_ptr));
    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;"
                 :
                 : "l"(smem_ptr64), "n"(Dim), "r"(static_cast<uint32_t>(dim)));
}

template<int TokenAxis, class T>
static __device__ __forceinline__ void RebaseSequenceDescriptor(CUtensorMap*         output,
                                                                CUtensorMap*         scratch,
                                                                const CUtensorMap&   base_descriptor,
                                                                StridedTensorBase<T> tensor,
                                                                int                  physical_batch,
                                                                int                  local_token,
                                                                int                  sequence_len,
                                                                int                  lane)
{
    CopyTmaDescriptor(scratch, &base_descriptor, lane, 32);
    __syncwarp();
    if (lane == 0) {
        const int64_t element_offset = static_cast<int64_t>(physical_batch) * tensor.batch_stride
                                       + static_cast<int64_t>(local_token) * tensor.token_stride;
        FusedGdrReplaceTmaAddressAndDim<TokenAxis>(scratch, tensor.ptr + element_offset, sequence_len);
    }
    __syncwarp();
    PublishTmaDescriptor(output, scratch);
    __syncwarp();
}

template<int ChunkSize, class T>
__device__ __forceinline__ void FusedGdrBuildSequenceDataTmaDescriptors(CUtensorMap*               gmem_desc,
                                                                        CUtensorMap*               smem_desc,
                                                                        const CUtensorMap&         q_tma_desc,
                                                                        const CUtensorMap&         k_tma_desc,
                                                                        const CUtensorMap&         v_tma_desc,
                                                                        const CUtensorMap&         g_tma_desc,
                                                                        const CUtensorMap&         resolvent_tma_desc,
                                                                        const CUtensorMap&         out_tma_desc,
                                                                        StridedTensorBase<const T> q,
                                                                        StridedTensorBase<const T> k,
                                                                        StridedTensorBase<const T> v,
                                                                        StridedTensorBase<const float> g_cumsum,
                                                                        StridedTensorBase<const T>     resolvent,
                                                                        StridedTensorBase<T>           out,
                                                                        int                            tid,
                                                                        int                            seq_start,
                                                                        int                            local_seq_start,
                                                                        int                            physical_batch,
                                                                        int                            seq_len,
                                                                        int                            hq,
                                                                        int                            hv,
                                                                        int64_t                        gate_stride,
                                                                        int64_t gate_batch_stride)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const int lane_id = tid & 31;
    if (tid < 32) {
        RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrQDesc],
                                    &smem_desc[kFusedGdrQDesc],
                                    q_tma_desc,
                                    q,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrKDesc],
                                    &smem_desc[kFusedGdrKDesc],
                                    k_tma_desc,
                                    k,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrVDesc],
                                    &smem_desc[kFusedGdrVDesc],
                                    v_tma_desc,
                                    v,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<1>(&gmem_desc[kFusedGdrGDesc],
                                    &smem_desc[kFusedGdrGDesc],
                                    g_tma_desc,
                                    g_cumsum,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrResolventDesc],
                                    &smem_desc[kFusedGdrResolventDesc],
                                    resolvent_tma_desc,
                                    resolvent,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrOutDesc],
                                    &smem_desc[kFusedGdrOutDesc],
                                    out_tma_desc,
                                    out,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
    }
    __syncthreads();
    static_cast<void>(seq_start);
    static_cast<void>(hq);
    static_cast<void>(hv);
    static_cast<void>(gate_stride);
    static_cast<void>(gate_batch_stride);
}

template<class StateT>
__device__ __forceinline__ void FusedGdrBuildStateTmaDescriptor(
    CUtensorMap* gmem_desc, CUtensorMap* smem_desc, const CUtensorMap& state_tma_desc, const StateT* state, int tid)
{
    const int lane_id = tid & 31;
    if (tid < 32) {
        CopyTmaDescriptor(smem_desc, &state_tma_desc, lane_id, 32);
        __syncwarp();

        if (lane_id == 0) {
            ReplaceTmaAddress(smem_desc, state);
        }
        __syncwarp();

        PublishTmaDescriptor(gmem_desc, smem_desc);
    }
    __syncthreads();
}

__device__ __forceinline__ void FusedGdrAcquireAndPrefetchDataTmaDescriptors(const CUtensorMap* desc, int tid)
{
    if (tid < 32) {
        for (int idx = tid; idx < kFusedGdrDataDescCount; idx += 32) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&desc[idx]));
            cute::prefetch_tma_descriptor(&desc[idx]);
        }
    }
}

__device__ __forceinline__ void FusedGdrAcquireAndPrefetchStateTmaDescriptor(const CUtensorMap* desc, int tid)
{
    if (tid == 0) {
        cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(desc));
        cute::prefetch_tma_descriptor(desc);
    }
}

__device__ __forceinline__ int FusedGdrValueTmaCoord(int value_head, int dv0)
{
    return value_head * kHeadDim + dv0;
}

template<int ChunkSize>
__device__ __forceinline__ int FusedGdrResolventTmaCoord(int value_head)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    return value_head * ChunkSize;
}

__device__ __forceinline__ int FusedGdrResolventTmaCoord(int value_head)
{
    return FusedGdrResolventTmaCoord<kChunkSize>(value_head);
}

__device__ __forceinline__ int FusedGdrGateTmaCoord(int value_head)
{
    return (value_head / kFusedGdrGateTmaHeads) * kFusedGdrGateTmaHeads;
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

template<class Element>
__device__ __forceinline__ void FusedGdrStoreBf16Pair(Element* ptr, float2 values)
{
    static_assert(std::is_same_v<Element, cute::bfloat16_t>);
    uint32_t pack;
    reinterpret_cast<__nv_bfloat162*>(&pack)[0] = __float22bfloat162_rn(values);
    *reinterpret_cast<uint32_t*>(ptr)           = pack;
}

template<class T, class... MmaArgs, class TA, class ALayout, class TB, class BLayout, class StateFragment>
__device__ __forceinline__ void FusedGdrStateUpdateFragmentFromVd(uint32_t                          thread_idx,
                                                                  cute::TiledMMA<MmaArgs...> const& tiled_mma,
                                                                  cute::Tensor<TA, ALayout> const&  sA,
                                                                  cute::Tensor<TB, BLayout> const&  sB,
                                                                  StateFragment&                    tCrState,
                                                                  const float*                      g,
                                                                  float                             last_g_exp)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    using InputTypeA   = typename TA::value_type;
    using InputTypeB   = typename TB::value_type;
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

template<class T>
CUtensorMap MakeQkTmaDesc(const core::Tensor& tensor, const uint32_t (&box_dims)[5], CUtensorMapSwizzle swizzle)
{
    const uint64_t global_dims[5] = {
        64u,
        2u,
        static_cast<uint64_t>(tensor.shape(2)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[4] = {
        64u * sizeof(T),
        static_cast<uint64_t>(tensor.stride(2)) * sizeof(T),
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(T),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(T),
    };
    return MakeTmaDesc(
        const_cast<T*>(tensor.data<T>()), FusedGdrTmaDataType<T>(), 5, global_dims, global_strides, box_dims, swizzle);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrQkTmaDesc(const core::Tensor& tensor)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint32_t box_dims[5] = {64u, 2u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeQkTmaDesc<__nv_bfloat16>(tensor, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrValueTmaDesc(const core::Tensor& tensor, int block_dv)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(tensor.shape(2)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<__nv_bfloat16*>(tensor.data<__nv_bfloat16>()),
                       FusedGdrTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrBf16TmaSwizzle(block_dv));
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrOutputTmaDesc(core::Tensor& tensor, int block_dv)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(tensor.shape(2)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(tensor.data<__nv_bfloat16>(),
                       FusedGdrTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrBf16TmaSwizzle(block_dv));
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrResolventTmaDesc(const core::Tensor& tensor)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(ChunkSize),
        static_cast<uint64_t>(tensor.shape(2)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(ChunkSize), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<__nv_bfloat16*>(tensor.data<__nv_bfloat16>()),
                       FusedGdrTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrSquareTmaSwizzle<ChunkSize>());
}

template<int ChunkSize>
inline CUtensorMap MakeFusedGdrGateTmaDesc(const core::Tensor& tensor)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[3] = {
        static_cast<uint64_t>(tensor.stride(1)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[2] = {
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(float),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(float),
    };
    const uint32_t box_dims[3] = {4u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<float*>(tensor.data<float>()),
                       FusedGdrTmaDataType<float>(),
                       3,
                       global_dims,
                       global_strides,
                       box_dims,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class StateT>
CUtensorMap MakeFusedGdrStateTmaDesc(StateT* ptr, int batch, int hv, int block_dv)
{
    const uint64_t global_dim[2]    = {static_cast<uint64_t>(kHeadDim),
                                    static_cast<uint64_t>(batch) * static_cast<uint64_t>(hv)
                                        * static_cast<uint64_t>(kHeadDim)};
    const uint64_t global_stride[1] = {static_cast<uint64_t>(kHeadDim * sizeof(StateT))};
    const uint32_t box_dim[2]       = {static_cast<uint32_t>(block_dv), static_cast<uint32_t>(kHeadDim)};
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<StateT>(), 2, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class StateT>
CUtensorMap MakeContextParallelStateTmaDesc(StateT* ptr, int total_segments, int hv, int block_dv)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(StateT)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(StateT)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(StateT),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(block_dv),
        static_cast<uint32_t>(kHeadDim),
        1u,
        1u,
    };
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<StateT>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

inline CUtensorMap MakeCorrectInitialStatesSegmentMatrixTmaDesc(float* ptr, int total_segments, int hv)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(float)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(float)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(float),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(kHeadDim),
        static_cast<uint32_t>(kCorrectInitialStatesMRowsPerTma),
        1u,
        1u,
    };
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<float>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

inline CUtensorMap MakeFusedGdrHSegmentMatrixTmaDesc(float* ptr, int total_segments, int hv, int block_kk)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(float)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(float)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(float),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(block_kk),
        static_cast<uint32_t>(kHeadDim),
        1u,
        1u,
    };
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<float>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class StateT>
CUtensorMap MakeFusedGdrStateHeadTmaDesc(StateT* ptr, int block_dv)
{
    const uint64_t global_dim[2]    = {static_cast<uint64_t>(kHeadDim), static_cast<uint64_t>(kHeadDim)};
    const uint64_t global_stride[1] = {static_cast<uint64_t>(kHeadDim * sizeof(StateT))};
    const uint32_t box_dim[2]       = {static_cast<uint32_t>(block_dv), static_cast<uint32_t>(kHeadDim)};
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<StateT>(), 2, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class StateT>
CUtensorMap MakeFusedGdrStateTileTmaDesc(StateT* ptr, int batch, int hv, int block_dv, int rows)
{
    const uint64_t global_dim[2]    = {static_cast<uint64_t>(kHeadDim),
                                    static_cast<uint64_t>(batch) * static_cast<uint64_t>(hv)
                                        * static_cast<uint64_t>(kHeadDim)};
    const uint64_t global_stride[1] = {static_cast<uint64_t>(kHeadDim * sizeof(StateT))};
    const uint32_t box_dim[2]       = {static_cast<uint32_t>(block_dv), static_cast<uint32_t>(rows)};
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<StateT>(), 2, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class T, int BlockDv>
struct FusedGdrFwdSharedStorage {
    // Fused-forward lifetime arena: q becomes P/local, p becomes A/W then q_state, and vd becomes
    // state-pack/K@state/Vd/Vn.
    alignas(1024) T q[kChunkSize][kHeadDim];
    T     k[kChunkSize][kHeadDim];
    float gate_stage[2][kChunkSize][4];
    float g[kChunkSize];
    float g_exp[kChunkSize];
    alignas(16) cute::uint64_t state_tma_mbar;
    alignas(16) cute::uint64_t state_ready_mbar;
    cute::uint64_t early_ready_mbar;
    cute::uint64_t q_ready_mbar;
    cute::uint64_t k_ready_mbar;
    cute::uint64_t out_ready_bar;
    cute::uint64_t early_free_bar;
    cute::uint64_t compute_done_bar;
    cute::uint64_t vd_ready_bar;
    cute::uint64_t q_consumed_bar;
    cute::uint64_t packed_vd_ready_bar;
    alignas(1024) float vd[kChunkSize][BlockDv];
    float state_stage[kHeadDim][BlockDv];
    alignas(1024) float p[kChunkSize][kChunkSize];
};

template<class T, int BlockDv>
struct alignas(1024) FusedGdrHEarlyStage {
    alignas(1024) T a[kChunkSize][kChunkSize];
    alignas(1024) T v[kChunkSize][BlockDv];
    alignas(1024) float gate_stage[2][kChunkSize][4];
    alignas(1024) float g[kChunkSize];
    alignas(1024) float g_exp[kChunkSize];
};

template<class T, int BlockDv>
struct alignas(1024) FusedGdrHSharedStorage {
    alignas(1024) T k_stage[2][kChunkSize][kHeadDim];
    alignas(1024) FusedGdrHEarlyStage<T, BlockDv> early[2];
    alignas(1024) float vd[kChunkSize][BlockDv];
    alignas(16) cute::uint64_t state_ready_mbar;
    alignas(16) cute::uint64_t early_ready_mbar[2];
    alignas(16) cute::uint64_t early_free_bar[2];
    alignas(16) cute::uint64_t k_ready_mbar[2];
    alignas(16) cute::uint64_t k_free_bar[2];
    alignas(16) cute::uint64_t final_store_bar;
    alignas(16) cute::uint64_t h_pack_ready_bar;
    alignas(16) cute::uint64_t m_pack_ready_bar;
    alignas(16) cute::uint64_t scratch_free_bar;
    alignas(16) cute::uint64_t a_read_done_bar;
};

template<class T, int BlockDv>
constexpr size_t FusedGdrFwdSharedBytes()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    return sizeof(FusedGdrFwdSharedStorage<T, BlockDv>);
}

template<class T, int BlockDv>
constexpr size_t FusedGdrHSharedBytes()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    return sizeof(FusedGdrHSharedStorage<T, BlockDv>);
}

static_assert(FusedGdrFwdSharedBytes<__nv_bfloat16, kFusedGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);
static_assert(FusedGdrFwdSharedBytes<__nv_bfloat16, kContextParallelGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);
static_assert(FusedGdrHSharedBytes<__nv_bfloat16, kFusedGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);
static_assert(FusedGdrHSharedBytes<__nv_bfloat16, kContextParallelGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);

template<class T, int BlockDv>
constexpr size_t FusedGdrFwdPUpperOffset()
{
    using Storage = FusedGdrFwdSharedStorage<T, BlockDv>;
    return offsetof(Storage, p) + kChunkSize * kChunkSize * sizeof(T);
}

static_assert(FusedGdrFwdPUpperOffset<__nv_bfloat16, kFusedGdrBlockDv>() % alignof(__nv_bfloat162) == 0);
static_assert(FusedGdrFwdPUpperOffset<__nv_bfloat16, kContextParallelGdrBlockDv>() % alignof(__nv_bfloat162) == 0);

template<class T, int BlockDv>
constexpr size_t FusedGdrFwdVdOffset()
{
    using Storage = FusedGdrFwdSharedStorage<T, BlockDv>;
    return offsetof(Storage, vd);
}

static_assert(FusedGdrFwdVdOffset<__nv_bfloat16, kFusedGdrBlockDv>() % alignof(__nv_bfloat162) == 0);
static_assert(FusedGdrFwdVdOffset<__nv_bfloat16, kContextParallelGdrBlockDv>() % alignof(__nv_bfloat162) == 0);
static_assert(alignof(__nv_bfloat162) <= 16);
static_assert((kFusedGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kFusedGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kContextParallelGdrBlockDv * sizeof(float)) % alignof(float2) == 0);
static_assert((kContextParallelGdrBlockDv * sizeof(float)) % sizeof(float2) == 0);
static_assert((kChunkSize * kChunkSize * sizeof(cute::bfloat16_t)) % alignof(__nv_bfloat162) == 0);

inline bool CanUseFusedGdrFwd(const Problem& problem)
{
    return problem.arch == 1200 && problem.input_dtype == kBfloat16 && problem.batch == problem.sequence_num
           && problem.hv % problem.hq == 0 && problem.head_dim == kHeadDim && problem.chunk_size == kChunkSize;
}

template<int BlockDv>
struct CorrectInitialStatesSharedStorage {
    alignas(1024) float state[2][kHeadDim][BlockDv];
    alignas(1024) float m[kHeadDim][kCorrectInitialStatesKTile];
    alignas(16) cute::uint64_t tma_mbar;
};

static_assert(sizeof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesF32BlockDv>)
              <= kFusedGdrMaxDynamicSharedBytes);
static_assert(sizeof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesBf16BlockDv>)
              <= kFusedGdrMaxDynamicSharedBytes);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesF32BlockDv>, state) % 1024 == 0);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesBf16BlockDv>, state) % 1024 == 0);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesF32BlockDv>, m) % 1024 == 0);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesBf16BlockDv>, m) % 1024 == 0);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesF32BlockDv>, tma_mbar)
                  % alignof(cute::uint64_t)
              == 0);
static_assert(offsetof(CorrectInitialStatesSharedStorage<kCorrectInitialStatesBf16BlockDv>, tma_mbar)
                  % alignof(cute::uint64_t)
              == 0);

constexpr int kChunk32Size             = 32;
constexpr int kSm120FusedGdrGatePasses = kChunk32Size / kFusedGdrGateWriterThreads;

static_assert(kFusedGdrGateWriterThreads == kChunk32Size);
static_assert(kSm120FusedGdrGatePasses == 1);

CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrQkLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_32, cute::_128>, cute::Stride<cute::_128, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(Sm120FusedGdrQkLayout())> == kChunk32Size * kHeadDim);

CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrQkTransposedLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_128, cute::_32>, cute::Stride<cute::_1, cute::_128>>{});
}
static_assert(cute::cosize_v<decltype(Sm120FusedGdrQkTransposedLayout())> == kChunk32Size * kHeadDim);

CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrSquareLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
}
static_assert(cute::cosize_v<decltype(Sm120FusedGdrSquareLayout())> == kChunk32Size * kChunk32Size);

CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrPackedP128BRowLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_64, cute::_1>>{});
}

static constexpr int kChunk32PackedPOffset   = 0;
static constexpr int kChunk32PackedVdOffset  = 2 * kChunk32Size * kChunk32Size;
static constexpr int kChunk32QStageSlotElems = kChunk32Size * kHeadDim;
static_assert(cute::cosize_v<decltype(Sm120FusedGdrPackedP128BRowLayout())> <= kChunk32PackedVdOffset);

template<int BlockDv>
constexpr int Sm120FusedGdrPackedHandoffElems()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    return kChunk32PackedVdOffset + kChunk32Size * BlockDv;
}
static_assert(Sm120FusedGdrPackedHandoffElems<kFusedGdrBlockDv>() <= kChunk32QStageSlotElems);
static_assert(Sm120FusedGdrPackedHandoffElems<kContextParallelGdrBlockDv>() <= kChunk32QStageSlotElems);

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrVRowLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return Sm120FusedGdrSquareLayout();
    }
    else {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrVTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_1, cute::_32>>{});
    }
    else {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_32>, cute::Stride<cute::_1, cute::_64>>{});
    }
}

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrStateTLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kContextParallelGdrBlockDv>, cute::Int<kHeadDim>>,
                                              cute::Stride<cute::_1, cute::Int<kContextParallelGdrBlockDv>>>{});
    }
    else {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::Int<kHeadDim>>,
                                              cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv>>>{});
    }
}
static_assert(cute::cosize_v<decltype(
                  Sm120FusedGdrStateTLayout<kContextParallelGdrBlockDv>())> == kContextParallelGdrBlockDv * kHeadDim);
static_assert(cute::cosize_v<decltype(Sm120FusedGdrStateTLayout<kFusedGdrBlockDv>())> == kFusedGdrBlockDv * kHeadDim);

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto Sm120FusedGdrStateCLayout()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if constexpr (BlockDv == kContextParallelGdrBlockDv) {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kContextParallelGdrBlockDv>>,
                                              cute::Stride<cute::Int<kContextParallelGdrBlockDv>, cute::_1>>{});
    }
    else {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<kFusedGdrBlockDv>>,
                                              cute::Stride<cute::Int<kFusedGdrBlockDv>, cute::_1>>{});
    }
}
static_assert(cute::cosize_v<decltype(
                  Sm120FusedGdrStateCLayout<kContextParallelGdrBlockDv>())> == kContextParallelGdrBlockDv * kHeadDim);
static_assert(cute::cosize_v<decltype(Sm120FusedGdrStateCLayout<kFusedGdrBlockDv>())> == kFusedGdrBlockDv * kHeadDim);

template<class T, int BlockDv, class Fragment, class ThrMma>
__device__ __forceinline__ void Sm120FusedGdrStoreValueFragmentBf16Stsm(
    Fragment const& fragment, typename FusedGdrMmaTraits<T>::Element* value_pack, ThrMma const& thr_mma, int role_tid)
{
    using Element = typename FusedGdrMmaTraits<T>::Element;
    static_assert(std::is_same_v<T, __nv_bfloat16>);

    auto tCrPack = cute::make_fragment_like<Element>(fragment);
#pragma unroll
    for (int i = 0; i < cute::size(fragment); ++i) {
        tCrPack(i) = Element(CastFromFloat<T>(static_cast<float>(fragment(i))));
    }

    auto s_value_pack = cute::as_position_independent_swizzle_tensor(
        cute::make_tensor(cute::make_smem_ptr(value_pack), Sm120FusedGdrVRowLayout<BlockDv>()));
    auto smem_tiled_copy_C = cute::make_tiled_copy_C_atom(cute::Copy_Atom<cute::SM90_U32x2_STSM_N, Element>{}, thr_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
    auto tCsPack           = smem_thr_copy_C.partition_D(s_value_pack);
    auto tCrPackView       = smem_thr_copy_C.retile_S(tCrPack);
    cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
}

template<class T, int BlockDv, class StateFragment, class ThrMma>
__device__ __forceinline__ void
Sm120FusedGdrStoreStateFragmentBf16Stsm(StateFragment const&                    tCrState,
                                        typename FusedGdrMmaTraits<T>::Element* state_pack,
                                        ThrMma const&                           thr_mma,
                                        int                                     role_tid)
{
    using Element = typename FusedGdrMmaTraits<T>::Element;
    static_assert(std::is_same_v<T, __nv_bfloat16>);

    auto s_state_pack = cute::as_position_independent_swizzle_tensor(
        cute::make_tensor(cute::make_smem_ptr(state_pack), Sm120FusedGdrStateCLayout<BlockDv>()));
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

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
