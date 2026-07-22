// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py

#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>

#include <stdexcept>
#include <string>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class K, int ConsumerThreads, int ConsumerRegisters>
struct Sm90KktSolve {
    static constexpr int kChunkSize   = 64;
    static constexpr int kHeadDim     = 128;
    static constexpr int kHalfChunk   = kChunkSize / 2;
    static constexpr int kSharedAlign = 1024;
    // Measured setmaxnreg count for the 128-thread consumer warpgroup.
    static constexpr int kKktSolveRoleThreads = 128;
    static constexpr int kKktSolveConsumerWgs = 1;
    static constexpr int kTmaDescriptorBytes  = 128;

    static constexpr int kBlock16 = 16;
    // Rows per 16x16 fp32 tile including one padding row. Actual rows stay contiguous;
    // the extra row shifts successive tiles away from bank-aligned starts.
    static constexpr int kBlock16Stride = 17;

    static constexpr int kVec4Elems       = 4;
    static constexpr int kVec8Elems       = 8;
    static constexpr int kKTileSwizzleDim = 64;
    enum : int
    {
        kKTileTmaDim = kKTileSwizzleDim,
    };
    static constexpr int kA64LowerHalfOffset = kHalfChunk * kChunkSize;

    static constexpr int      kKTilePlaneElems = kChunkSize * kKTileSwizzleDim;
    static constexpr int      kA16iElems       = 4 * kBlock16Stride * kBlock16;
    static constexpr int      kA16oElems       = 2 * kBlock16Stride * kBlock16;
    static constexpr int      kMat32Elems      = 32 * 32;
    static constexpr uint64_t kTmaNoCacheHint  = 0;

    enum KktTmaDescIndex : int
    {
        kKktKDesc         = 0,
        kKktResolventDesc = 3,
        kKktTmaDescCount  = 4,
    };

    static_assert(kBlock16 == 16);
    static_assert(kHeadDim == 2 * kKTileTmaDim);
    static_assert(kChunkSize == 4 * kBlock16);
    static_assert(kA16iElems >= kMat32Elems);
    static_assert(sizeof(CUtensorMap) == kTmaDescriptorBytes);
    static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT solve only supports bfloat16 inputs");
    static_assert(ConsumerThreads == kKktSolveRoleThreads,
                  "KKT solve MMA and repack layouts currently require 128 consumer threads");
    static_assert(ConsumerThreads >= kChunkSize);
    static_assert(ConsumerThreads % 32 == 0);

    using Element = cute::bfloat16_t;

    struct __align__(16) Scratch
    {
        __align__(16) float a16i[kA16iElems];
        __align__(16) float a16o[kA16oElems];
        __align__(16) float a32i0[kMat32Elems];
        __align__(16) float a32i1[kMat32Elems];
        __align__(16) float a32o[kMat32Elems];
    };

    // Barrier contract:
    // Actors are the TMA-issuing consumer leader, the 128-thread consumer WG, and
    // the TMA load/store engines; TMA load completion releases bytes through the ready
    // mbarriers below.
    // - k_ready0/1 (count 1 plus expected bytes): the leader arms and issues each
    //   K-half load; TMA completion releases both halves to the consumer WG before
    //   asynchronous SM90 GMMA. Each is consumed once at phase 0.
    // - beta_stage is populated by 16-byte cp.async transactions. Each producer warp
    //   commits the same generation; a per-thread wait followed by named barrier 0
    //   acquires it for all 128 consumers. The same named barrier protects overwrite
    //   after the current beta values have been published into the solve scratch.
    // - named barrier 0 (128 consumers): reused across the payload-specific solve,
    //   repack, tail-GEMM, and output-store rendezvous in each group.
    // The consumer leader commits each output store and owns the final wait<0> drain.
    struct __align__(1024) SharedStorage
    {
        __align__(1024) Element k_tile[2 * kKTilePlaneElems];
        __align__(1024) float   beta_stage[kChunkSize * 4];
        __align__(16) Scratch   scratch[kKktSolveConsumerWgs];
        __align__(8) uint64_t   k_ready0;
        __align__(8) uint64_t   k_ready1;
    };

    static_assert(offsetof(SharedStorage, beta_stage) % kSharedAlign == 0);
    static_assert(offsetof(SharedStorage, scratch) % kSharedAlign == 0);
    static_assert(sizeof(SharedStorage) == 36 * 1024);

    static constexpr size_t SharedBytes()
    {
        static_assert(ConsumerThreads >= kChunkSize);
        static_assert(ConsumerThreads % 32 == 0);
        static_assert(alignof(SharedStorage) == kSharedAlign);
        return sizeof(SharedStorage);
    }

    static __device__ __forceinline__ int DenseTmaCoord(int batch_id, int total_tokens, int head_span, int coord)
    {
        return static_cast<int>(
            (static_cast<int64_t>(batch_id) * static_cast<int64_t>(total_tokens) * static_cast<int64_t>(head_span))
            + static_cast<int64_t>(coord));
    }

    static __device__ __forceinline__ int CeilDiv(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    static __device__ __forceinline__ void AcquireAndPrefetchTmaDescriptors(const CUtensorMap* desc, int tid)
    {
        if (tid == kKktKDesc || tid == kKktResolventDesc) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&desc[tid]));
            cute::prefetch_tma_descriptor(&desc[tid]);
        }
    }

    template<int BarrierId>
    static __device__ __forceinline__ void ConsumerWgSync()
    {
        static_assert(ConsumerThreads % 32 == 0);
        static_assert(BarrierId >= 0);
        static_assert(BarrierId < kKktSolveConsumerWgs);
        static_assert(BarrierId + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                      < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);
        cutlass::arch::NamedBarrier::sync(ConsumerThreads, BarrierId);
    }

    static __device__ __forceinline__ void PairMmaSync(int pair_id)
    {
        constexpr int kPairThreads      = ConsumerThreads / 2;
        constexpr int kFirstPairBarrier = 1;
        static_assert(ConsumerThreads == 2 * kPairThreads);
        static_assert(kFirstPairBarrier + 1 + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                      < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);
        cutlass::arch::NamedBarrier::sync(kPairThreads, kFirstPairBarrier + pair_id);
    }

    static CUTE_HOST_DEVICE constexpr auto BetaStageLayout()
    {
        return cute::composition(cute::Swizzle<3, 2, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_4>, cute::Stride<cute::_4, cute::_1>>{});
    }

    static_assert(cute::cosize_v<decltype(BetaStageLayout())> == kChunkSize * 4);

    static __device__ __forceinline__ void IssueBetaQuadAsync(float* __restrict__ beta_stage,
                                                              const float* __restrict__ beta,
                                                              int     role_tid,
                                                              int     valid,
                                                              int     beta_quad,
                                                              int     physical_batch,
                                                              int     local_token0,
                                                              int64_t beta_stride,
                                                              int64_t beta_batch_stride)
    {
        const int row_in_half = role_tid >> 2;
        const int col         = role_tid & 3;
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
            const int     row           = row_in_half + half * (kChunkSize / 2);
            const int     source_row    = row < valid ? row : valid - 1;
            const int64_t source_offset = static_cast<int64_t>(physical_batch) * beta_batch_stride
                                          + static_cast<int64_t>(local_token0 + source_row) * beta_stride
                                          + static_cast<int64_t>(beta_quad * 4 + col);
            const int destination_offset = static_cast<int>(BetaStageLayout()(row, col));
            cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
                beta[source_offset], beta_stage[destination_offset], row < valid);
        }

        // Every warp commits the same generation, keeping the wait contract uniform
        // across the full 128-thread consumer warpgroup.
        cute::cp_async_fence();
    }

    static __device__ __forceinline__ void WaitBetaQuadAsync()
    {
        cute::cp_async_wait<0>();
        ConsumerWgSync<0>();
    }

    template<int Tiles>
    static CUTE_HOST_DEVICE constexpr auto Block16Layout()
    {
        return cute::composition(
            cute::Swizzle<2, 2, 3>{},
            cute::make_layout(
                cute::make_shape(cute::Int<Tiles>{}, cute::Int<kBlock16Stride>{}, cute::Int<kBlock16>{}),
                cute::make_stride(cute::Int<kBlock16Stride * kBlock16>{}, cute::Int<kBlock16>{}, cute::Int<1>{})));
    }

    using PairMmaAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using PairMma16   = cute::TiledMMA<cute::MMA_Atom<PairMmaAtom>,
                                     cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>>,
                                     cute::Tile<cute::Underscore, cute::Int<kBlock16>, cute::Underscore>>;

    static_assert(cute::size(PairMma16{}) == ConsumerThreads / 2);

    static CUTE_HOST_DEVICE constexpr auto TailMmaALayout()
    {
        return cute::composition(cute::Swizzle<3, 2, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto TailMmaBLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 2>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto TailMmaBColMajorLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 2>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_1, cute::_32>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto Swizzled64Layout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
    }

    template<class Element>
    static CUTE_HOST_DEVICE constexpr auto GmmaKLayout()
    {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<Element>{},
                                   cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kHeadDim>{}));
    }

    template<class Element>
    static CUTE_HOST_DEVICE constexpr auto TmaTwoHalfKLayout()
    {
        static_assert(sizeof(Element) == 2);
        return cute::composition(
            cute::Swizzle<3, 4, 3>{},
            cute::smem_ptr_flag_bits<cute::sizeof_bits<Element>::value>{},
            cute::Layout<cute::Shape<cute::Shape<cute::Int<kKTileSwizzleDim>, cute::_2>,
                                     cute::Shape<cute::_8, cute::Int<kChunkSize / 8>>>,
                         cute::Stride<cute::Stride<cute::_1, cute::Int<kKTilePlaneElems>>,
                                      cute::Stride<cute::Int<kKTileSwizzleDim>, cute::Int<kKTileSwizzleDim * 8>>>>{});
    }

    static_assert(cute::cosize_v<decltype(GmmaKLayout<cute::bfloat16_t>())> == kChunkSize * kHeadDim);
    static_assert(cute::cosize_v<decltype(TmaTwoHalfKLayout<cute::bfloat16_t>())> == kChunkSize * kHeadDim);
    static_assert(TmaTwoHalfKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{})
                  == GmmaKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<0>{}));
    static_assert(TmaTwoHalfKLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<0>{})
                  == GmmaKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<63>{}));
    static_assert(TmaTwoHalfKLayout<cute::bfloat16_t>()(cute::Int<64>{}, cute::Int<0>{})
                  == GmmaKLayout<cute::bfloat16_t>()(cute::Int<0>{}, cute::Int<64>{}));
    static_assert(TmaTwoHalfKLayout<cute::bfloat16_t>()(cute::Int<127>{}, cute::Int<63>{})
                  == GmmaKLayout<cute::bfloat16_t>()(cute::Int<63>{}, cute::Int<127>{}));

    static CUTE_HOST_DEVICE constexpr auto Swizzled64Lower32Layout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::Int<kChunkSize>, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto A16SolveThreadLayout()
    {
        return cute::make_layout(cute::Shape<cute::Int<4>, cute::Int<kBlock16>>{},
                                 cute::make_stride(cute::Int<kBlock16>{}, cute::Int<1>{}));
    }

    static CUTE_HOST_DEVICE constexpr auto TailAccumulatorStoreLaneLayout()
    {
        return cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<4>{}),
                                 cute::make_stride(cute::Int<1>{}, cute::Int<8>{}));
    }

    static CUTE_HOST_DEVICE constexpr auto RepackUpperVec4ThreadLayout()
    {
        return cute::make_layout(
            cute::make_shape(
                cute::Int<2>{}, cute::make_shape(cute::Int<4>{}, cute::Int<4>{}), cute::Int<kHalfChunk / kVec4Elems>{}),
            cute::make_stride(cute::Int<128>{}, cute::make_stride(cute::Int<32>{}, cute::Int<8>{}), cute::Int<1>{}));
    }

    static CUTE_HOST_DEVICE constexpr auto RepackLowerDiagVec4ThreadLayout()
    {
        return cute::make_layout(
            cute::make_shape(cute::Int<2>{},
                             cute::Int<kBlock16 / kVec4Elems>{},
                             cute::Int<2>{},
                             cute::make_shape(cute::Int<2>{}, cute::Int<2>{}, cute::Int<2>{})),
            cute::make_stride(cute::Int<64>{},
                              cute::Int<1>{},
                              cute::Int<4>{},
                              cute::make_stride(cute::Int<8>{}, cute::Int<16>{}, cute::Int<32>{})));
    }

    static CUTE_HOST_DEVICE constexpr auto A64ZeroVec8ThreadLayout()
    {
        return cute::make_layout(
            cute::make_shape(cute::make_shape(cute::Int<8>{}, cute::Int<4>{}), cute::Int<kHalfChunk / kVec8Elems>{}),
            cute::make_stride(cute::make_stride(cute::Int<1>{}, cute::Int<32>{}), cute::Int<8>{}));
    }

    static_assert(cute::cosize_v<decltype(TailAccumulatorStoreLaneLayout())> == 32);
    static_assert(cute::cosize_v<decltype(RepackUpperVec4ThreadLayout())> == 2 * ConsumerThreads);
    static_assert(cute::cosize_v<decltype(RepackLowerDiagVec4ThreadLayout())> == ConsumerThreads);
    static_assert(cute::cosize_v<decltype(A64ZeroVec8ThreadLayout())> == ConsumerThreads);

    static __device__ __forceinline__ void StoreFloatPair(float* dst, float2 value)
    {
        using Pair = uint64_t;
        using Atom = cute::Copy_Atom<cute::UniversalCopy<Pair>, Pair>;

        Pair packed = static_cast<Pair>(__float_as_uint(value.x)) | (static_cast<Pair>(__float_as_uint(value.y)) << 32);
        auto src    = cute::make_tensor(cute::make_rmem_ptr(&packed),
                                     cute::Layout<cute::Shape<cute::_1>, cute::Stride<cute::_1>>{});
        auto out    = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Pair*>(dst)),
                                     cute::Layout<cute::Shape<cute::_1>, cute::Stride<cute::_1>>{});
        cute::copy(Atom{}, src, out);
    }

    static CUTE_HOST_DEVICE constexpr int Block16PairBaseOffset(int tile, int row, int col)
    {
        const int logical = tile * (kBlock16Stride * kBlock16) + row * kBlock16 + col;
        return static_cast<int>(cute::Swizzle<2, 2, 3>{}(logical));
    }

    template<int Stripe>
    static __device__ __forceinline__ void
    PublishDiagonalStripe(float const (&gram)[32], float beta0, float beta1, int t0, int t1, float* dst, int pair_base)
    {
        static_assert(Stripe >= 0 && Stripe < 8);
        constexpr int kFragment0  = 4 * Stripe;
        constexpr int kFragment1  = kFragment0 + 2;
        constexpr int kColumnHalf = Stripe & 1;

        const int col = 2 * t0 + 8 * kColumnHalf;

        float2 pair0 = make_float2(gram[kFragment0] * beta0, gram[kFragment0 + 1] * beta0);
        float2 pair1 = make_float2(gram[kFragment1] * beta1, gram[kFragment1 + 1] * beta1);

        pair0.x = t1 < col ? 0.0f : (t1 == col ? 1.0f : pair0.x);
        pair0.y = t1 < col + 1 ? 0.0f : (t1 == col + 1 ? 1.0f : pair0.y);

        const int row1 = t1 + 8;
        pair1.x        = row1 < col ? 0.0f : (row1 == col ? 1.0f : pair1.x);
        pair1.y        = row1 < col + 1 ? 0.0f : (row1 == col + 1 ? 1.0f : pair1.y);

        // Swizzle<2,2,3> leaves bit 7 untouched, so row-select 1 advances 128 floats.
        // Switching between the two eight-column halves toggles physical bit 3.
        const int offset0 = pair_base ^ (8 * kColumnHalf);
        StoreFloatPair(dst + offset0, pair0);
        StoreFloatPair(dst + offset0 + 128, pair1);
    }

    template<int Stripe>
    static __device__ __forceinline__ void
    PublishSubdiagonalStripe(float const (&gram)[32], float beta0, float beta1, float* dst, int pair_base)
    {
        static_assert(Stripe >= 0 && Stripe < 8);
        constexpr int kFragment0  = 4 * Stripe;
        constexpr int kFragment1  = kFragment0 + 2;
        constexpr int kColumnHalf = Stripe & 1;

        float2 pair0 = make_float2(gram[kFragment0] * beta0, gram[kFragment0 + 1] * beta0);
        float2 pair1 = make_float2(gram[kFragment1] * beta1, gram[kFragment1 + 1] * beta1);
        pair0.x      = -pair0.x;
        pair0.y      = -pair0.y;
        pair1.x      = -pair1.x;
        pair1.y      = -pair1.y;

        const int offset0 = pair_base ^ (8 * kColumnHalf);
        StoreFloatPair(dst + offset0, pair0);
        StoreFloatPair(dst + offset0 + 128, pair1);
    }

    template<int Stripe>
    static __device__ __forceinline__ void PublishLowerLeftStripe(
        float const (&gram)[32], float beta0, float beta1, float* dst, int row_word_base, int row_xor)
    {
        static_assert(Stripe >= 0 && Stripe < 4);
        constexpr int kFragment0 = 4 * Stripe;
        constexpr int kFragment1 = kFragment0 + 2;

        float2 pair0 = make_float2(gram[kFragment0] * beta0, gram[kFragment0 + 1] * beta0);
        float2 pair1 = make_float2(gram[kFragment1] * beta1, gram[kFragment1 + 1] * beta1);
        pair0.x      = -pair0.x;
        pair0.y      = -pair0.y;
        pair1.x      = -pair1.x;
        pair1.y      = -pair1.y;

        // TailMmaBLayout maps the pair to 32*row + 2*t0 + 8*(stripe XOR (row & 3)).
        // Row-select 1 advances eight rows, hence 256 physical floats.
        const int offset0 = row_word_base + 8 * (Stripe ^ row_xor);
        StoreFloatPair(dst + offset0, pair0);
        StoreFloatPair(dst + offset0 + 256, pair1);
    }

    template<class StoreElement>
    static __device__ __forceinline__ void StoreMmaFragment4(StoreElement* ptr, float4 vec)
    {
        static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT solve only stores bfloat16 fragments");
        static_assert(sizeof(StoreElement) * kVec4Elems == sizeof(uint2));
        // Callers pass column multiples of four under the swizzled layouts below. With the
        // kSharedAlign base, each four-element segment is 8-byte aligned for this uint2 store.
        uint2 pack;
        reinterpret_cast<__nv_bfloat162*>(&pack)[0] = __float22bfloat162_rn(make_float2(vec.x, vec.y));
        reinterpret_cast<__nv_bfloat162*>(&pack)[1] = __float22bfloat162_rn(make_float2(vec.z, vec.w));
        *reinterpret_cast<uint2*>(ptr)              = pack;
    }

    template<class StoreElement>
    static __device__ __forceinline__ void StoreMmaFragment2(StoreElement* ptr, float2 vec)
    {
        static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT solve only stores bfloat16 fragments");
        static_assert(sizeof(StoreElement) * 2 == sizeof(__nv_bfloat162));
        *reinterpret_cast<__nv_bfloat162*>(ptr) = __float22bfloat162_rn(vec);
    }

    struct Tf32LoadTransform {
        template<class T>
        __device__ __forceinline__ cute::tfloat32_t operator()(T const& value) const
        {
            const float raw = static_cast<float>(value);
            return cute::tfloat32_t::bitcast(__float_as_uint(raw));
        }
    };

    static constexpr int kThreads   = kKktSolveConsumerWgs * ConsumerThreads;
    static constexpr int kMinBlocks = 1;

    static __device__ __forceinline__ void Run(const float* __restrict__ beta,
                                               const int32_t* __restrict__ q_offsets,
                                               const bool* __restrict__ finished,
                                               CUtensorMap*   tma_desc_workspace,
                                               int            total_tokens,
                                               int            sequence_num,
                                               int            hq,
                                               int            hv,
                                               int64_t        beta_stride,
                                               int64_t        beta_batch_stride,
                                               int            groups_per_k_head,
                                               unsigned char* shared_raw)
    {
        const int     tx              = static_cast<int>(threadIdx.x);
        const int     qk_head         = static_cast<int>(blockIdx.x);
        int           local_chunk_id  = static_cast<int>(blockIdx.y);
        constexpr int batch_id        = 0;
        const int     value_head_base = qk_head * groups_per_k_head;
        int           token0          = 0;
        const int     wg_idx          = cutlass::canonical_warp_group_idx();
        const int     role_tid        = tx % ConsumerThreads;

        int sequence_id    = -1;
        int sequence_begin = 0;
        int sequence_end   = 0;
        for (int b = 0; b < sequence_num; ++b) {
            const int cur_start  = q_offsets[b];
            const int cur_end    = q_offsets[b + 1];
            const int cur_chunks = CeilDiv(cur_end - cur_start, kChunkSize);
            if (local_chunk_id < cur_chunks) {
                sequence_id    = b;
                sequence_begin = cur_start;
                sequence_end   = cur_end;
                break;
            }
            local_chunk_id -= cur_chunks;
        }
        if (sequence_id < 0) {
            return;
        }
        token0                         = local_chunk_id * kChunkSize;
        const int sequence_length      = sequence_end - sequence_begin;
        const int remaining            = sequence_length - token0;
        const int valid                = remaining < kChunkSize ? remaining : kChunkSize;
        const int physical_batch       = sequence_begin / total_tokens;
        const int local_sequence_begin = sequence_begin - physical_batch * total_tokens;
        const int local_beta_token0    = local_sequence_begin + token0;

        auto&     smem     = *reinterpret_cast<SharedStorage*>(shared_raw);
        uint64_t* k_ready0 = &smem.k_ready0;
        uint64_t* k_ready1 = &smem.k_ready1;

        using MmaElement       = Element;
        MmaElement* k_tile0    = smem.k_tile;
        MmaElement* k_tile1    = smem.k_tile + kKTilePlaneElems;
        float*      beta_stage = smem.beta_stage;
        const auto* gmem_desc  = tma_desc_workspace + sequence_id * kKktTmaDescCount;
        AcquireAndPrefetchTmaDescriptors(gmem_desc, tx);
        const CUtensorMap* k_desc         = &gmem_desc[kKktKDesc];
        const CUtensorMap* resolvent_desc = &gmem_desc[kKktResolventDesc];

        if (tx == 0) {
            cute::initialize_barrier(*k_ready0, 1);
            cute::initialize_barrier(*k_ready1, 1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();

            constexpr int ConsumerWg = 0;
            static_assert(ConsumerWg < kKktSolveConsumerWgs);

            if (ConsumerWg < groups_per_k_head) {

                auto& scratch   = smem.scratch[ConsumerWg];
                auto  s_k_gmma  = cute::make_tensor(cute::make_smem_ptr(smem.k_tile), GmmaKLayout<MmaElement>());
                auto  s_beta    = cute::make_tensor(cute::make_smem_ptr(smem.beta_stage), BetaStageLayout());
                auto  s_a16i    = cute::make_tensor(cute::make_smem_ptr(scratch.a16i), Block16Layout<4>());
                auto  s_a16o    = cute::make_tensor(cute::make_smem_ptr(scratch.a16o), Block16Layout<2>());
                auto  s_a32i0   = cute::make_tensor(cute::make_smem_ptr(scratch.a32i0), TailMmaBLayout());
                auto  s_a32i1   = cute::make_tensor(cute::make_smem_ptr(scratch.a32i1), TailMmaALayout());
                auto  s_a32o    = cute::make_tensor(cute::make_smem_ptr(scratch.a32o), TailMmaBLayout());
                auto  s_a32i0_n = cute::make_tensor(cute::make_smem_ptr(scratch.a32i0), TailMmaBColMajorLayout());
                auto  s_a32o_n  = cute::make_tensor(cute::make_smem_ptr(scratch.a32o), TailMmaBColMajorLayout());
                // After repack, a16i is dead; reuse it for the first tail GEMM output so that GEMM does not
                // read and write aliased views of a32o.
                auto s_a32tmp =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<float*>(scratch.a16i)), TailMmaALayout());

                float gram_fragment[32];

                using GmmaTileShape = cute::Shape<cute::Int<kChunkSize>, cute::Int<kChunkSize>, cute::Int<kHeadDim>>;
                using GmmaAtom      = decltype(cute::SM90::GMMA::ss_op_selector<MmaElement,
                                                                           MmaElement,
                                                                           float,
                                                                           GmmaTileShape,
                                                                           cute::SM90::GMMA::Major::K,
                                                                           cute::SM90::GMMA::Major::K>());
                auto gmma64         = cute::make_tiled_mma(GmmaAtom{});
                auto t_gram_fragment =
                    cute::make_tensor(cute::make_rmem_ptr(gram_fragment),
                                      cute::partition_shape_C(gmma64, cute::Shape<cute::_64, cute::_64>{}));

                constexpr int k_tma_box_rows  = kChunkSize;
                const int     first_beta_quad = value_head_base / 4;
                if (role_tid == 0) {
                    // Transaction bytes match the fixed descriptor box; TMA zero-fills OOB rows.
                    cute::set_barrier_transaction_bytes(*k_ready0, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                    // Keep raw TMA issue here for the same audited fast-codegen reason documented at descriptor
                    // creation.
                    cute::SM90_TMA_LOAD_5D::copy(
                        k_desc, k_ready0, kTmaNoCacheHint, k_tile0, 0, 0, qk_head, token0, batch_id);
                    cute::set_barrier_transaction_bytes(*k_ready1, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                    cute::SM90_TMA_LOAD_5D::copy(
                        k_desc, k_ready1, kTmaNoCacheHint, k_tile1, 0, 1, qk_head, token0, batch_id);
                }
                IssueBetaQuadAsync(beta_stage,
                                   beta,
                                   role_tid,
                                   valid,
                                   first_beta_quad,
                                   physical_batch,
                                   local_beta_token0,
                                   beta_stride,
                                   beta_batch_stride);
                // Acquire both K halves from TMA completion at phase 0; all expected
                // bytes are visible to the consumer WG before asynchronous SM90 GMMA.
                cute::wait_barrier(*k_ready0, 0);
                cute::wait_barrier(*k_ready1, 0);

                using namespace cute;
                auto thr_gmma = gmma64.get_thread_slice(role_tid);
                auto tCsK     = thr_gmma.partition_A(s_k_gmma);
                auto tCsKt    = thr_gmma.partition_B(s_k_gmma);
                auto tCrK     = thr_gmma.make_fragment_A(tCsK);
                auto tCrKt    = thr_gmma.make_fragment_B(tCsKt);

                cute::warpgroup_fence_operand(t_gram_fragment);
                cute::warpgroup_arrive();
                gmma64.accumulate_        = cute::SM90::GMMA::ScaleOut::Zero;
                constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrK){});
#pragma unroll
                for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                    cute::gemm(
                        gmma64, tCrK(cute::_, cute::_, k_block), tCrKt(cute::_, cute::_, k_block), t_gram_fragment);
                    gmma64.accumulate_ = cute::SM90::GMMA::ScaleOut::One;
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(t_gram_fragment);

                int loaded_beta_quad = -1;
                for (int group = ConsumerWg; group < groups_per_k_head; group += kKktSolveConsumerWgs) {
                    const int value_head = value_head_base + group;
                    const int beta_quad  = value_head / 4;
                    if (beta_quad != loaded_beta_quad) {
                        WaitBetaQuadAsync();
                        loaded_beta_quad = beta_quad;
                    }
                    const int   beta_lane = value_head & 3;
                    MmaElement* a64       = (group & 1) ? k_tile1 : k_tile0;
                    auto        s_a64     = cute::make_tensor(cute::make_smem_ptr(a64), Swizzled64Layout());
                    // The lower-left 32x32 output is stored through a rebased view of s_a64. Swizzle<3,3,3>
                    // mixes element-address bits 3..8; adding 32 * 64 elements changes only higher bits,
                    // so the lower-half view preserves the same swizzled row/column contract.
                    auto s_a64_tail =
                        cute::make_tensor(cute::make_smem_ptr(a64 + kA64LowerHalfOffset), Swizzled64Lower32Layout());
                    if (group >= 2) {
                        if (role_tid == 0) {
                            // Acquire completion of the older parity-matched output store
                            // before its shared a64 plane is reused by this group.
                            cute::tma_store_wait<1>();
                        }
                        // Rendezvous: gate all 128 consumers on the leader's output-store
                        // acquire before any thread reuses the shared output plane.
                        ConsumerWgSync<ConsumerWg>();
                    }

                    // CLayout_64x64 decomposes the warpgroup thread as (t0,t1,warp) and
                    // its accumulator values as adjacent-column pairs (row_select,stripe):
                    //   row = t1 + 16*warp + 8*row_select
                    //   col = 2*t0 + 8*stripe
                    //   fragment = 4*stripe + 2*row_select + pair_element
                    // Specializing once on the uniform warp id avoids retaining a distinct
                    // composed-layout pointer for every pair while preserving that CuTe TV map.
                    const int t0   = role_tid & 3;
                    const int t1   = (role_tid >> 2) & 7;
                    const int warp = role_tid >> 5;

                    const int   beta_row0 = 16 * warp + t1;
                    const float beta0     = s_beta(beta_row0, beta_lane);
                    const float beta1     = s_beta(beta_row0 + 8, beta_lane);

                    switch (warp) {
                        case 0: {
                            const int diag_base = Block16PairBaseOffset(0, t1, 2 * t0);
                            PublishDiagonalStripe<0>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            PublishDiagonalStripe<1>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            break;
                        }
                        case 1: {
                            const int subdiag_base = Block16PairBaseOffset(0, t1, 2 * t0);
                            PublishSubdiagonalStripe<0>(gram_fragment, beta0, beta1, scratch.a16o, subdiag_base);
                            PublishSubdiagonalStripe<1>(gram_fragment, beta0, beta1, scratch.a16o, subdiag_base);

                            const int diag_base = Block16PairBaseOffset(1, t1, 2 * t0);
                            PublishDiagonalStripe<2>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            PublishDiagonalStripe<3>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            break;
                        }
                        case 2: {
                            const int row32         = t1;
                            const int row_word_base = 32 * row32 + 2 * t0;
                            const int row_xor       = row32 & 3;

                            PublishLowerLeftStripe<0>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<1>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<2>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<3>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);

                            const int diag_base = Block16PairBaseOffset(2, t1, 2 * t0);
                            PublishDiagonalStripe<4>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            PublishDiagonalStripe<5>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            break;
                        }
                        default: {
                            const int row32         = 16 + t1;
                            const int row_word_base = 32 * row32 + 2 * t0;
                            const int row_xor       = row32 & 3;

                            PublishLowerLeftStripe<0>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<1>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<2>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);
                            PublishLowerLeftStripe<3>(
                                gram_fragment, beta0, beta1, scratch.a32o, row_word_base, row_xor);

                            const int subdiag_base = Block16PairBaseOffset(1, t1, 2 * t0);
                            PublishSubdiagonalStripe<4>(gram_fragment, beta0, beta1, scratch.a16o, subdiag_base);
                            PublishSubdiagonalStripe<5>(gram_fragment, beta0, beta1, scratch.a16o, subdiag_base);

                            const int diag_base = Block16PairBaseOffset(3, t1, 2 * t0);
                            PublishDiagonalStripe<6>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            PublishDiagonalStripe<7>(gram_fragment, beta0, beta1, t0, t1, scratch.a16i, diag_base);
                            break;
                        }
                    }
                    // Rendezvous: publish triangular-fragment writes from all 128
                    // consumers before the diagonal-solve readers acquire them.
                    ConsumerWgSync<ConsumerWg>();

                    // Once publication completes, every current beta value is dead.
                    // When the next head crosses a quad boundary, overwrite the single
                    // stage asynchronously under the solve and tail GEMMs below.
                    const int next_group = group + kKktSolveConsumerWgs;
                    if (next_group < groups_per_k_head) {
                        const int next_value_head = value_head_base + next_group;
                        const int next_beta_quad  = next_value_head / 4;
                        if (next_beta_quad != beta_quad) {
                            IssueBetaQuadAsync(beta_stage,
                                               beta,
                                               role_tid,
                                               valid,
                                               next_beta_quad,
                                               physical_batch,
                                               local_beta_token0,
                                               beta_stride,
                                               beta_batch_stride);
                        }
                    }

                    if (role_tid < kChunkSize) {
                        const auto solve_coord = A16SolveThreadLayout().get_hier_coord(role_tid);
                        const int  tile_id     = cute::get<0>(solve_coord);
                        const int  col         = cute::get<1>(solve_coord);

                        // Each 16-lane group owns one 16x16 diagonal tile, so lane `mid` carries L(row, mid).
                        float inv_col[kBlock16];
#pragma unroll
                        for (int row = 0; row < kBlock16; ++row) {
                            inv_col[row] = s_a16i(tile_id, row, col);
                        }

#pragma unroll
                        for (int row = 1; row < kBlock16; ++row) {
                            float acc = 0.0f;
#pragma unroll
                            for (int mid = 0; mid < row; ++mid) {
                                const float l_row_mid = __shfl_sync(0xffffffffu, inv_col[row], mid, kBlock16);
                                acc -= inv_col[mid] * l_row_mid;
                            }

                            if (col < row) {
                                inv_col[row] = acc;
                            }
                        }

#pragma unroll
                        for (int row = 1; row < kBlock16; ++row) {
                            if (col < row) {
                                s_a16i(tile_id, row, col) = inv_col[row];
                            }
                        }
                    }
                    // Rendezvous: publish inverse-diagonal writes before cross-block
                    // readers consume them to form the 32x32 tiles.
                    ConsumerWgSync<ConsumerWg>();

                    {
                        using namespace cute;

                        PairMma16 mma16;
                        const int pair_id   = role_tid >> 6;
                        const int pair_tid  = role_tid & 63;
                        const int pair_warp = pair_tid >> 5;
                        const int atom_lane = pair_tid & 31;
                        const int pair_t0   = atom_lane & 3;
                        const int pair_t1   = atom_lane >> 2;

                        auto pair_identity = make_identity_tensor(Shape<_16, _16>{});
                        auto pair_thr_mma  = mma16.get_thread_slice(pair_tid);
                        auto tAcA          = pair_thr_mma.partition_A(pair_identity);
                        auto tBcB          = pair_thr_mma.partition_B(pair_identity);

                        cute::tfloat32_t a_fragment[size(partition_shape_A(mma16, Shape<_16, _16>{}))];
                        cute::tfloat32_t b_fragment[size(partition_shape_B(mma16, Shape<_16, _16>{}))];
                        float            block_fragment[4];
                        auto tA = make_tensor(make_rmem_ptr(a_fragment), partition_shape_A(mma16, Shape<_16, _16>{}));
                        auto tB = make_tensor(make_rmem_ptr(b_fragment), partition_shape_B(mma16, Shape<_16, _16>{}));
                        auto t_block =
                            make_tensor(make_rmem_ptr(block_fragment), partition_shape_C(mma16, Shape<_16, _16>{}));

                        clear(t_block);
#pragma unroll
                        for (int i = 0; i < size(tA); ++i) {
                            const auto coord = tAcA(i);
                            tA(i) = Tf32LoadTransform{}(s_a16i(pair_id * 2 + 1, get<0>(coord), get<1>(coord)));
                        }
#pragma unroll
                        for (int i = 0; i < size(tB); ++i) {
                            const auto coord = tBcB(i);
                            tB(i)            = Tf32LoadTransform{}(s_a16o(pair_id, get<1>(coord), get<0>(coord)));
                        }
#pragma unroll
                        for (int k_block = 0; k_block < size<2>(tA); ++k_block) {
                            cute::gemm(mma16, tA(_, _, k_block), tB(_, _, k_block), t_block);
                        }

                        // Both products reuse a16o: wait for every first-product read before
                        // overwriting it, then publish the complete intermediate to the pair.
                        PairMmaSync(pair_id);
#pragma unroll
                        for (int row_select = 0; row_select < 2; ++row_select) {
                            const int    row           = pair_t1 + 8 * row_select;
                            const int    col           = 8 * pair_warp + 2 * pair_t0;
                            const int    fragment_base = 2 * row_select;
                            const float2 value =
                                make_float2(block_fragment[fragment_base], block_fragment[fragment_base + 1]);
                            StoreFloatPair(&s_a16o(pair_id, row, col), value);
                        }
                        PairMmaSync(pair_id);

                        clear(t_block);
#pragma unroll
                        for (int i = 0; i < size(tA); ++i) {
                            const auto coord = tAcA(i);
                            tA(i)            = Tf32LoadTransform{}(s_a16o(pair_id, get<0>(coord), get<1>(coord)));
                        }
#pragma unroll
                        for (int i = 0; i < size(tB); ++i) {
                            const auto coord = tBcB(i);
                            tB(i)            = Tf32LoadTransform{}(s_a16i(pair_id * 2, get<1>(coord), get<0>(coord)));
                        }
#pragma unroll
                        for (int k_block = 0; k_block < size<2>(tA); ++k_block) {
                            cute::gemm(mma16, tA(_, _, k_block), tB(_, _, k_block), t_block);
                        }

#pragma unroll
                        for (int row_select = 0; row_select < 2; ++row_select) {
                            const int    row           = pair_t1 + 8 * row_select;
                            const int    col           = 8 * pair_warp + 2 * pair_t0;
                            const int    fragment_base = 2 * row_select;
                            const float2 value =
                                make_float2(block_fragment[fragment_base], block_fragment[fragment_base + 1]);
                            float* fp32_dst =
                                pair_id == 0 ? &s_a32i0(kBlock16 + row, col) : &s_a32i1(kBlock16 + row, col);
                            StoreFloatPair(fp32_dst, value);
                            StoreMmaFragment2(&s_a64(pair_id * kHalfChunk + kBlock16 + row, pair_id * kHalfChunk + col),
                                              value);
                        }
                    }

                    {
#pragma unroll
                        for (int iter = 0; iter < 2; ++iter) {
                            const auto repack_coord =
                                RepackUpperVec4ThreadLayout().get_hier_coord(role_tid + iter * ConsumerThreads);
                            const int  js        = cute::get<0>(repack_coord);
                            const auto row_coord = cute::get<1>(repack_coord);
                            const int  row       = cute::get<0>(row_coord) + 4 * cute::get<1>(row_coord);
                            const int  col       = cute::get<2>(repack_coord) * kVec4Elems;
                            float4     value     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                            if (col < kBlock16) {
                                value = *reinterpret_cast<const float4*>(&s_a16i(js * 2, row, col));
                            }

                            auto* dst = (js == 0) ? &s_a32i0(row, col) : &s_a32i1(row, col);
                            dst[0]    = value.x;
                            dst[1]    = value.y;
                            dst[2]    = value.z;
                            dst[3]    = value.w;
                            StoreMmaFragment4(&s_a64(js * 32 + row, js * 32 + col), value);
                        }

                        const auto repack_coord = RepackLowerDiagVec4ThreadLayout().get_hier_coord(role_tid);
                        const int  js           = cute::get<0>(repack_coord);
                        const int  col          = kBlock16 + cute::get<1>(repack_coord) * kVec4Elems;
                        const int  row_select   = cute::get<2>(repack_coord);
                        const auto row_coord    = cute::get<3>(repack_coord);
                        const int  row_base =
                            2 * cute::get<0>(row_coord) + cute::get<1>(row_coord) + 8 * cute::get<2>(row_coord);
                        const int    row = kBlock16 + (row_base ^ (6 * row_select));
                        const float4 value =
                            *reinterpret_cast<const float4*>(&s_a16i(js * 2 + 1, row - kBlock16, col - kBlock16));

                        auto* dst = (js == 0) ? &s_a32i0(row, col) : &s_a32i1(row, col);
                        dst[0]    = value.x;
                        dst[1]    = value.y;
                        dst[2]    = value.z;
                        dst[3]    = value.w;
                        StoreMmaFragment4(&s_a64(js * 32 + row, js * 32 + col), value);
                    }
                    // Rendezvous: publish the repacked 32x32 tiles from all 128 consumers
                    // before the first tail GEMM reads them.
                    ConsumerWgSync<ConsumerWg>();

                    {
                        using namespace cute;
                        using MmaAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
                        using Mma32   = TiledMMA<MMA_Atom<MmaAtom>,
                                               Layout<Shape<_2, _2, _1>>,
                                               Tile<Underscore, Int<32>, Underscore>>;
                        Mma32 mma32;
                        float c_fragment[8];
                        auto  tC = make_tensor(make_rmem_ptr(c_fragment), partition_shape_C(mma32, Shape<_32, _32>{}));
                        cute::clear(tC);
                        cute::cooperative_gemm(role_tid,
                                               mma32,
                                               s_a32i1,
                                               s_a32o_n,
                                               tC,
                                               Tf32LoadTransform{},
                                               Tf32LoadTransform{},
                                               cute::SM75_U32x4_LDSM_N{},
                                               cute::DefaultCopy{});

                        const int  lane        = role_tid & 31;
                        const auto store_coord = TailAccumulatorStoreLaneLayout().get_hier_coord(lane);
                        const int  store_t1    = cute::get<0>(store_coord);
                        const int  store_t0    = cute::get<1>(store_coord);
                        const int  source_lane = store_t0 + 4 * store_t1;
                        const int  role_warp   = role_tid >> 5;
                        const int  mma_m       = role_warp & 1;
                        const int  mma_n       = role_warp >> 1;

#pragma unroll
                        for (int n_repeat = 0; n_repeat < 2; ++n_repeat) {
#pragma unroll
                            for (int row_select = 0; row_select < 2; ++row_select) {
                                const int    fragment_base = 4 * n_repeat + 2 * row_select;
                                const float2 pair =
                                    make_float2(__shfl_sync(0xffffffffu, c_fragment[fragment_base], source_lane),
                                                __shfl_sync(0xffffffffu, c_fragment[fragment_base + 1], source_lane));
                                const int row = 16 * mma_m + store_t1 + 8 * row_select;
                                const int col = 8 * mma_n + 2 * store_t0 + 16 * n_repeat;
                                StoreFloatPair(&s_a32tmp(row, col), pair);
                            }
                        }
                    }
                    // Rendezvous: publish the first tail-GEMM output before the second tail
                    // GEMM consumes it from the reused a16i scratch storage.
                    ConsumerWgSync<ConsumerWg>();

                    {
                        using namespace cute;
                        using MmaAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
                        using Mma32   = TiledMMA<MMA_Atom<MmaAtom>,
                                               Layout<Shape<_2, _2, _1>>,
                                               Tile<Underscore, Int<32>, Underscore>>;
                        Mma32 mma32;
                        float c_fragment[8];
                        auto  tC = make_tensor(make_rmem_ptr(c_fragment), partition_shape_C(mma32, Shape<_32, _32>{}));
                        auto  thr_mma = mma32.get_thread_slice(role_tid);
                        auto  tCsC    = thr_mma.partition_C(s_a64_tail);
                        cute::clear(tC);
                        cute::cooperative_gemm(role_tid,
                                               mma32,
                                               s_a32tmp,
                                               s_a32i0_n,
                                               tC,
                                               Tf32LoadTransform{},
                                               Tf32LoadTransform{},
                                               cute::SM75_U32x4_LDSM_N{},
                                               cute::DefaultCopy{});
                        cute::copy(tC, tCsC);
                    }

                    {
                        const auto zero_coord = A64ZeroVec8ThreadLayout().get_hier_coord(role_tid);
                        const auto row_coord  = cute::get<0>(zero_coord);
                        const int  row        = cute::get<0>(row_coord) + 8 * cute::get<1>(row_coord);
                        const int  col        = kHalfChunk + cute::get<1>(zero_coord) * kVec8Elems;
                        *reinterpret_cast<uint4*>(&s_a64(row, col)) = make_uint4(0, 0, 0, 0);
                    }
                    // Release each consumer's final a64 writes to the async proxy.
                    cute::tma_store_fence();
                    // Rendezvous: all 128 consumers finish a64 writes and proxy fences before
                    // the TMA-store leader reads the shared output.
                    ConsumerWgSync<ConsumerWg>();
                    if (role_tid == 0) {
                        // Release: issue the output TMA store and commit it to the store group.
                        cute::SM90_TMA_STORE_4D::copy(resolvent_desc, a64, 0, value_head, token0, batch_id);
                        cute::tma_store_arrive();
                    }
                    if (group + kKktSolveConsumerWgs < groups_per_k_head) {
                        // Rendezvous: the leader has issued and committed this store before
                        // peers advance to the next group and mutate shared output state.
                        ConsumerWgSync<ConsumerWg>();
                    }
                }
                if (role_tid == 0) {
                    // Acquire/drain all remaining output stores. The leader keeps the CTA
                    // alive until the final TMA store has completed.
                    cute::tma_store_wait<0>();
                }
            }
        }
    }
};

template<class K, int ConsumerThreads, int ConsumerRegisters>
__global__ void __launch_bounds__(Sm90KktSolve<K, ConsumerThreads, ConsumerRegisters>::kThreads,
                                  Sm90KktSolve<K, ConsumerThreads, ConsumerRegisters>::kMinBlocks)
    Sm90KktSolveKernel(const float* __restrict__ beta,
                       const int32_t* __restrict__ q_offsets,
                       const bool* __restrict__ finished,
                       CUtensorMap* tma_desc_workspace,
                       int          total_tokens,
                       int          sequence_num,
                       int          hq,
                       int          hv,
                       int64_t      beta_stride,
                       int64_t      beta_batch_stride,
                       int          groups_per_k_head)
{
#if __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000) {
        extern __shared__ __align__(1024) unsigned char shared_raw[];
        Sm90KktSolve<K, ConsumerThreads, ConsumerRegisters>::Run(beta,
                                                                 q_offsets,
                                                                 finished,
                                                                 tma_desc_workspace,
                                                                 total_tokens,
                                                                 sequence_num,
                                                                 hq,
                                                                 hv,
                                                                 beta_stride,
                                                                 beta_batch_stride,
                                                                 groups_per_k_head,
                                                                 shared_raw);
    }
#endif
}

template<class K, int ConsumerThreads = 128, int ConsumerRegisters = 160>
void LaunchKktSolveTyped(const K*            k_ptr,
                         const float*        beta_ptr,
                         const float*        g_cumsum_ptr,
                         const core::Tensor& q_offsets,
                         const core::Tensor& finished,
                         K*                  out_ptr,
                         void*               tma_desc_workspace,
                         const Problem&      problem,
                         cudaStream_t        stream)
{
    using Kernel = Sm90KktSolve<K, ConsumerThreads, ConsumerRegisters>;
    if (problem.total_chunks == 0) {
        return;
    }
    static_cast<void>(k_ptr);
    static_cast<void>(g_cumsum_ptr);
    static_cast<void>(out_ptr);

    const int    groups_per_k_head = problem.hv / problem.hq;
    const dim3   grid(problem.hq, problem.total_chunks, 1);
    const size_t shared_bytes = Kernel::SharedBytes();

    TM_CUDA_CHECK(cudaFuncSetAttribute(Sm90KktSolveKernel<K, ConsumerThreads, ConsumerRegisters>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(shared_bytes)));

    const int32_t* offsets_ptr    = q_offsets.data<int32_t>();
    const bool*    finished_ptr   = finished.data<bool>();
    auto*          desc_workspace = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);
    Sm90KktSolveKernel<K, ConsumerThreads, ConsumerRegisters>
        <<<grid, Kernel::kKktSolveConsumerWgs * ConsumerThreads, shared_bytes, stream>>>(beta_ptr,
                                                                                         offsets_ptr,
                                                                                         finished_ptr,
                                                                                         desc_workspace,
                                                                                         problem.token_num,
                                                                                         problem.sequence_num,
                                                                                         problem.hq,
                                                                                         problem.hv,
                                                                                         problem.beta_stride,
                                                                                         problem.beta_batch_stride,
                                                                                         groups_per_k_head);
    TM_CUDA_CHECK(cudaGetLastError());
}

void LaunchSm90KktSolveImpl(const core::Tensor& k,
                            const core::Tensor& beta,
                            const core::Tensor& q_offsets,
                            const core::Tensor* g_cumsum,
                            const core::Tensor& finished,
                            core::Tensor&       resolvent,
                            const Problem&      problem,
                            void*               tma_desc_workspace,
                            cudaStream_t        stream)
{
    const auto* k_ptr        = reinterpret_cast<const __nv_bfloat16*>(k.raw_data());
    const auto* g_cumsum_ptr = g_cumsum->data<float>();
    auto*       out_ptr      = reinterpret_cast<__nv_bfloat16*>(resolvent.raw_data());
    LaunchKktSolveTyped<__nv_bfloat16>(
        k_ptr, beta.data<float>(), g_cumsum_ptr, q_offsets, finished, out_ptr, tma_desc_workspace, problem, stream);
}

}  // namespace

namespace detail {

void LaunchSm90KktSolve(const core::Tensor& k,
                        const core::Tensor& beta,
                        const core::Tensor& q_offsets,
                        const core::Tensor* g_cumsum,
                        const core::Tensor& finished,
                        core::Tensor&       resolvent,
                        const Problem&      problem,
                        void*               tma_desc_workspace,
                        cudaStream_t        stream)
{
    LaunchSm90KktSolveImpl(k, beta, q_offsets, g_cumsum, finished, resolvent, problem, tma_desc_workspace, stream);
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
