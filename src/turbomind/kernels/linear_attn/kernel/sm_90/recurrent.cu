#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/linear_attn/kernel/tma_desc.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <new>
#include <type_traits>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/arch/barrier.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<int KernelBlockDv, class KernelStateT>
struct Sm90GdrRecurrent {
    static constexpr int BlockDv = KernelBlockDv;
    using StateT                 = KernelStateT;

    static constexpr int kHeadDim     = 128;
    static constexpr float kHeadScale = 0.08838834764831845f;

    // Strategy D: TMA raw bf16 into overlaid scratch, expand in a consumer
    // warp, and keep one shared input-state slot for every BlockDv.
    static constexpr int kRecurrentStages = 1;

    // Dv128 uses two independent Dv64-equivalent compute groups so its three
    // resident CTAs retain the same 24 compute warps as Dv64's six CTAs.
    static constexpr int kComputeGroupThreads         = 128;
    static constexpr int kComputeGroups               = BlockDv == 128 ? 2 : 1;
    static constexpr int kComputeBlockDv              = BlockDv / kComputeGroups;
    static constexpr int kRecurrentConsumerThreads    = kComputeGroups * kComputeGroupThreads;
    static constexpr int kRecurrentProducerThreads    = 32;
    static constexpr int kRecurrentThreads            = kRecurrentConsumerThreads + kRecurrentProducerThreads;
    // Preserve the accepted kernel's register-shaping launch bound while the
    // fused producer removes one synchronization-only warp from the CTA.
    static constexpr int kRecurrentLaunchBoundThreads = kRecurrentConsumerThreads + 64;
    static constexpr int kRecurrentConsumerWarps      = kRecurrentConsumerThreads / 32;
    static constexpr int kRecurrentConsumerWarpGroups = kComputeGroups;
    // This is a compile-time launch-bounds/resource-shaping constraint, not
    // the runtime CTA residency used by dispatch. Runtime residency comes
    // from cudaOccupancyMaxActiveBlocksPerMultiprocessor for each concrete
    // kernel and state type.
    static constexpr int kRecurrentMinBlocksPerSm = BlockDv == 128 ? 3 : 6;
    static constexpr int kRecurrentMaxDescriptorCtas = 792;
    static constexpr int kRecurrentSm90SharedBytes   = 227328;
    static constexpr int kRecurrentStaticSharedReserveBytes = 1024;
    static constexpr int kRecurrentMaxDynamicSharedBytes =
        kRecurrentSm90SharedBytes - kRecurrentStaticSharedReserveBytes;
    static constexpr float    kLog2e          = 1.4426950408889634f;
    static constexpr uint64_t kTmaNoCacheHint = 0;
    static constexpr int      kRecurrentBarrierPartial = 0;

    // Every global Dv value uses the Dv64 arithmetic graph: four sequential
    // Dk32 chains followed by the same four-way reduction. Dv32 assigns one
    // scalar Dv to every lane; Dv64/Dv128 assign one adjacent pair per lane.
    static constexpr int kCorrDvThreads        = 32;
    static constexpr int kCorrDvPerThread      = kComputeBlockDv / kCorrDvThreads;
    static constexpr int kDkSplit              = 4;
    static constexpr int kDkPerGroup           = kHeadDim / kDkSplit;
    static constexpr int kQkPairsPerGroup      = kDkPerGroup / 2;
    static constexpr int kUpdateDvPerThread    = 2;
    static constexpr int kUpdateDvThreads      = kComputeBlockDv / kUpdateDvPerThread;
    static constexpr int kUpdateDkWorkers      = kComputeGroupThreads / kUpdateDvThreads;
    static constexpr int kQkPairs              = kHeadDim / 2;
    static constexpr int kUpdatePairsPerWorker = kQkPairs / kUpdateDkWorkers;
    static constexpr int kBf16TmaAlignElems = 128 / static_cast<int>(sizeof(__nv_bfloat16));
    static constexpr int kValueSmemDv =
        ((BlockDv + kBf16TmaAlignElems - 1) / kBf16TmaAlignElems) * kBf16TmaAlignElems;

    static_assert(BlockDv == 32 || BlockDv == 64 || BlockDv == 128);
    static_assert(kRecurrentStages == 1);
    static_assert(kValueSmemDv >= BlockDv);
    static_assert((kValueSmemDv * static_cast<int>(sizeof(__nv_bfloat16))) % 128 == 0);
    static_assert(kHeadDim % BlockDv == 0);
    static_assert(kComputeGroupThreads == kCorrDvThreads * kDkSplit);
    static_assert(kComputeBlockDv % kCorrDvThreads == 0);
    static_assert(kCorrDvPerThread == 1 || kCorrDvPerThread == 2);
    static_assert(kHeadDim % kDkSplit == 0);
    static_assert(kDkPerGroup == 32);
    static_assert(kDkPerGroup % 2 == 0);
    static_assert(kQkPairsPerGroup == 16);
    static_assert(kComputeBlockDv % kUpdateDvPerThread == 0);
    static_assert(kComputeGroupThreads % kUpdateDvThreads == 0);
    static_assert(kQkPairs % kUpdateDkWorkers == 0);
    static_assert(kUpdatePairsPerWorker == 8 || kUpdatePairsPerWorker == 16);
    static_assert(kUpdateDkWorkers * kUpdateDvThreads == kComputeGroupThreads);
    static_assert(kRecurrentConsumerThreads % 32 == 0);
    static_assert(kRecurrentConsumerThreads % 128 == 0);
    static_assert(kRecurrentProducerThreads == 32);
    static_assert(kRecurrentThreads == (BlockDv == 128 ? 288 : 160));
    static_assert(kRecurrentLaunchBoundThreads == (BlockDv == 128 ? 320 : 192));
    static_assert(kRecurrentBarrierPartial + kComputeGroups - 1
                      + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

    struct __align__(16) QkPair {
        float values[4];
    };

    struct RawQk {
        __nv_bfloat16 q[kHeadDim];
        __nv_bfloat16 k[kHeadDim];
    };

    union __align__(128) QkStorage {
        RawQk  raw;
        QkPair packed[kHeadDim / 2];
    };

    union __align__(128) CorrScratch {
        __nv_bfloat16 v_raw[kValueSmemDv];
        float partial_corr[kComputeGroups][kDkSplit][kComputeBlockDv];
    };

    static CUTE_HOST_DEVICE constexpr auto QkSmemLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 0, 3>{},
            cute::Layout<cute::Shape<cute::Int<kDkSplit>, cute::Int<kQkPairsPerGroup>>,
                         cute::Stride<cute::Int<kQkPairsPerGroup>, cute::_1>>{});
    }

    static_assert(sizeof(QkPair) == 16);
    static_assert(sizeof(RawQk) == 512);
    static_assert(sizeof(QkStorage) == 1024);
    static_assert(sizeof(CorrScratch)
                  == kComputeGroups * kDkSplit * kComputeBlockDv * sizeof(float));
    static_assert(cute::cosize_v<decltype(QkSmemLayout())> == kHeadDim / 2);

    // Barrier contract (each barrier advances once per work tile):
    // Actors are one fused 32-thread producer warp, one or two 128-thread
    // compute WGs, and the TMA engine. compute_done makes every current-tile
    // shared lifetime except output and its offset dead. The producer then
    // issues the next state/auxiliary transfers before it copies the current
    // output and only publishes the next auxiliary inputs after that copy.
    // - aux_tma_ready (count 1 plus expected bytes): acquires the auxiliary
    //   TMA-load transaction.
    // - state_tma_ready (count 2 plus expected bytes): joins state TMA
    //   completion with preprocessing-warp completion, publishing both through
    //   the consumer's existing state wait.
    // - aux_ready (count 1, producer lane 0): releases raw Q/K/V and metadata
    //   after auxiliary TMA completes. Consumers preprocess them while the
    //   independent state TMA remains in flight.
    // - compute_done (count 4/8, one representative per compute warp): confirms
    //   every global state store is issued and releases output to the producer warp.
    // - named barriers 0/1 (128 consumers each): group-local partial-reduction
    //   rendezvous. Dv32/Dv64 use only barrier 0.
    struct __align__(1024) SharedStorage {
        __align__(1024) float state_in[kHeadDim][BlockDv];
        QkStorage qk;
        __align__(128) float v[BlockDv];
        __align__(128) __nv_bfloat16 out[kValueSmemDv];
        CorrScratch corr;
        __align__(128) float partial_sq[kComputeGroups][kDkSplit][kComputeBlockDv];
        __align__(16) float partial_kq[kDkSplit];
        __align__(16) float gate_beta[2];
        __align__(16) int tile_finished;
        __align__(16) int64_t tile_out_offset;
        __align__(8) uint64_t tile_state_address;
        __align__(8) cute::uint64_t aux_tma_ready;
        __align__(8) cute::uint64_t state_tma_ready;
        __align__(8) cute::uint64_t aux_ready;
        __align__(8) cute::uint64_t compute_done;
    };

    static constexpr size_t SharedBytes()
    {
        return sizeof(SharedStorage);
    }

    static_assert(BlockDv != 64 || SharedBytes() == 36864);
    static_assert(BlockDv != 32 || SharedBytes() == 19456);
    static_assert(BlockDv != 128 || SharedBytes() == 71680);
    static_assert(SharedBytes() <= kRecurrentMaxDynamicSharedBytes);
    static_assert(SharedBytes() * kRecurrentMinBlocksPerSm <= kRecurrentSm90SharedBytes);

    static constexpr int64_t MemoryBytesPerWorkTile()
    {
        constexpr int64_t qk_bytes = 2 * kHeadDim * sizeof(__nv_bfloat16);
        constexpr int64_t value_bytes = BlockDv * sizeof(__nv_bfloat16);
        constexpr int64_t output_bytes = BlockDv * sizeof(__nv_bfloat16);
        constexpr int64_t state_bytes =
            2 * kHeadDim * BlockDv * static_cast<int64_t>(sizeof(StateT));
        constexpr int64_t gate_bytes = 2 * sizeof(float);
        constexpr int64_t finished_bytes = sizeof(bool);  // Cached pointer/descriptor metadata is excluded.
        return qk_bytes + value_bytes + output_bytes + state_bytes + gate_bytes + finished_bytes;
    }

    template<class T>
    static constexpr CUtensorMapDataType TmaDataType()
    {
        static_assert(std::is_same_v<T, __nv_bfloat16>);
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }

    template<class T>
    static constexpr CUtensorMapDataType StateTmaDataType()
    {
        if constexpr (std::is_same_v<T, float>) {
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        }
        else {
            static_assert(std::is_same_v<T, __nv_bfloat16>);
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        }
    }

    template<class T>
    static CUtensorMap MakeQkTmaDesc(const core::Tensor& tensor)
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
        const uint32_t box_dims[5] = {
            64u,
            2u,
            1u,
            1u,
            1u,
        };
        return MakeTmaDesc(const_cast<T*>(tensor.data<T>()),
                           TmaDataType<T>(),
                           5,
                           global_dims,
                           global_strides,
                           box_dims,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template<class T>
    static CUtensorMap MakeValueTmaDesc(const core::Tensor& tensor, int block_dv)
    {
        const uint64_t global_dims[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(tensor.shape(2)),
            static_cast<uint64_t>(tensor.shape(1)),
            static_cast<uint64_t>(tensor.shape(0)),
        };
        const uint64_t global_strides[3] = {
            static_cast<uint64_t>(tensor.stride(2)) * sizeof(T),
            static_cast<uint64_t>(tensor.stride(1)) * sizeof(T),
            static_cast<uint64_t>(tensor.stride(0)) * sizeof(T),
        };
        const uint32_t box_dims[4] = {
            static_cast<uint32_t>(block_dv),
            1u,
            1u,
            1u,
        };
        return MakeTmaDesc(const_cast<T*>(tensor.data<T>()),
                           TmaDataType<T>(),
                           4,
                           global_dims,
                           global_strides,
                           box_dims,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    static CUtensorMap MakeStateTmaDesc(StateT* ptr, int layers, int hv, int block_dv)
    {
        const uint64_t global_dim[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(hv),
            static_cast<uint64_t>(layers),
        };
        const uint64_t global_stride[3] = {
            static_cast<uint64_t>(kHeadDim * sizeof(StateT)),
            static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(StateT)),
            static_cast<uint64_t>(hv * kHeadDim * kHeadDim * sizeof(StateT)),
        };
        const uint32_t box_dim[4] = {
            static_cast<uint32_t>(block_dv),
            static_cast<uint32_t>(kHeadDim),
            1u,
            1u,
        };
        return MakeTmaDesc(
            ptr, StateTmaDataType<StateT>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    static __device__ __forceinline__ int DecodeHeadTileBatch(int head_tile, int hv)
    {
        return head_tile / hv;
    }

    static __device__ __forceinline__ int DecodeHeadTileHead(int head_tile, int hv)
    {
        return head_tile % hv;
    }

    static __device__ __forceinline__ constexpr int StateTmaBytes()
    {
        return kHeadDim * BlockDv * static_cast<int>(sizeof(StateT));
    }

    static __device__ __forceinline__ void StoreQk128(QkPair* dst, float k0, float k1, float q0, float q1)
    {
        // Default initialization begins the packed union member's lifetime
        // without writing its trivially initialized float array.
        ::new (static_cast<void*>(dst)) QkPair;
        auto smem = cute::make_tensor(cute::make_smem_ptr(dst->values),
                                      cute::Layout<cute::Shape<cute::_4>, cute::Stride<cute::_1>>{});
        auto regs = cute::make_fragment_like(smem);
        regs(0)   = k0;
        regs(1)   = k1;
        regs(2)   = q0;
        regs(3)   = q1;
        cute::copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, float>{}, regs, smem);
    }

    static __device__ __forceinline__ float4 LoadQk128(const QkPair* src)
    {
        auto smem = cute::make_tensor(cute::make_smem_ptr(src->values),
                                      cute::Layout<cute::Shape<cute::_4>, cute::Stride<cute::_1>>{});
        auto regs = cute::make_fragment_like(smem);
        cute::copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, float>{}, smem, regs);
        return make_float4(regs(0), regs(1), regs(2), regs(3));
    }

    static __device__ __forceinline__ float2 LoadK64(const QkPair* src)
    {
        auto smem = cute::make_tensor(cute::make_smem_ptr(src->values),
                                      cute::Layout<cute::Shape<cute::_2>, cute::Stride<cute::_1>>{});
        auto regs = cute::make_fragment_like(smem);
        cute::copy(cute::Copy_Atom<cute::UniversalCopy<uint64_t>, float>{}, smem, regs);
        return make_float2(regs(0), regs(1));
    }

    struct IssuedTileMetadata {
        float    gate;
        float    beta;
        int      finished;
        int64_t  out_offset;
        uint64_t state_base_address;
        int64_t  state_element_offset;
    };

    static __device__ __forceinline__ void CommitTileMetadata(
        SharedStorage& smem, const IssuedTileMetadata& tile)
    {
        smem.gate_beta[0]       = tile.gate;
        smem.gate_beta[1]       = tile.beta;
        smem.tile_finished      = tile.finished;
        smem.tile_out_offset    = tile.out_offset;
        smem.tile_state_address =
            tile.state_base_address
            + static_cast<uint64_t>(tile.state_element_offset) * sizeof(StateT);
    }

    static __device__ __forceinline__ IssuedTileMetadata IssueTile(
        SharedStorage&                    smem,
        int                               work_tile,
        const int64_t* __restrict__        state_ptrs,
        const CUtensorMap* __restrict__    state_tma_descs,
        const CUtensorMap&                q_tma_desc,
        const CUtensorMap&                k_tma_desc,
        const CUtensorMap&                v_tma_desc,
        const float* __restrict__         g,
        const float* __restrict__         beta,
        const bool* __restrict__          finished,
        int                               hq,
        int                               hv,
        int64_t                           gate_batch_stride,
        int64_t                           out_batch_stride,
        int64_t                           out_head_stride,
        int                               num_head_groups,
        int                               heads_per_block,
        int64_t                           state_layer_offset,
        int                               state_layer)
    {
        constexpr int kDvTiles     = kHeadDim / BlockDv;
        constexpr int kValueBytes  = BlockDv * static_cast<int>(sizeof(__nv_bfloat16));
        constexpr int kQkBytes     = kHeadDim * static_cast<int>(sizeof(__nv_bfloat16));
        constexpr int kAuxTmaBytes = kValueBytes + 2 * kQkBytes;

        const int head_tile   = work_tile / kDvTiles;
        const int dv_tile     = work_tile - head_tile * kDvTiles;
        const int batch       = DecodeHeadTileBatch(head_tile, hv);
        const int value_head  = DecodeHeadTileHead(head_tile, hv);
        const int dv0         = dv_tile * BlockDv;
        const int qk_head     = value_head / (hv / hq);
        const int gate_quad   = value_head / 4;
        const int gate_lane   = value_head - gate_quad * 4;
        const int head_group  = value_head / heads_per_block;
        const int local_head  = value_head % heads_per_block;
        const int state_desc_index = batch * num_head_groups + head_group;
        const int gate_offset =
            static_cast<int>(static_cast<int64_t>(batch) * gate_batch_stride + gate_quad * 4);

        IssuedTileMetadata tile;
        tile.out_offset = static_cast<int64_t>(batch) * out_batch_stride
                          + static_cast<int64_t>(value_head) * out_head_stride + dv0;
        tile.state_element_offset = state_layer_offset
                                    + static_cast<int64_t>(local_head) * kHeadDim * kHeadDim
                                    + dv0;

        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.aux_tma_ready,
                                                                        kAuxTmaBytes);
        cute::SM90_TMA_LOAD_4D::copy(&v_tma_desc,
                                     &smem.aux_tma_ready,
                                     kTmaNoCacheHint,
                                     &smem.corr.v_raw[0],
                                     dv0,
                                     value_head,
                                     0,
                                     batch);
        cute::SM90_TMA_LOAD_5D::copy(&q_tma_desc,
                                     &smem.aux_tma_ready,
                                     kTmaNoCacheHint,
                                     &smem.qk.raw.q[0],
                                     0,
                                     0,
                                     qk_head,
                                     0,
                                     batch);
        cute::SM90_TMA_LOAD_5D::copy(&k_tma_desc,
                                     &smem.aux_tma_ready,
                                     kTmaNoCacheHint,
                                     &smem.qk.raw.k[0],
                                     0,
                                     0,
                                     qk_head,
                                     0,
                                     batch);

        tile.gate               = g[gate_offset + gate_lane];
        tile.beta               = beta[gate_offset + gate_lane];
        tile.finished           = static_cast<int>(finished[batch]);
        tile.state_base_address = static_cast<uint64_t>(state_ptrs[state_desc_index]);

        if (hv >= 32) {
            cute::prefetch_tma_descriptor(
                reinterpret_cast<const cute::TmaDescriptor*>(&state_tma_descs[state_desc_index]));
        }

        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.state_tma_ready,
                                                                        StateTmaBytes());
        cute::SM90_TMA_LOAD_4D::copy(&state_tma_descs[state_desc_index],
                                     &smem.state_tma_ready,
                                     hv >= 32 ? static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST) :
                                                kTmaNoCacheHint,
                                     reinterpret_cast<StateT*>(&smem.state_in[0][0]),
                                     dv0,
                                     0,
                                     local_head,
                                     state_layer);
        return tile;
    }

    static __device__ __forceinline__ float2 LoadStatePair(const SharedStorage& smem, int dk, int dv0)
    {
        if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
            const auto* state_ptr = reinterpret_cast<const __nv_bfloat162*>(&smem.state_in[0][0]);
            return __bfloat1622float2(state_ptr[(dk * BlockDv + dv0) / 2]);
        }
        else {
            static_assert(std::is_same_v<StateT, float>);
            return *reinterpret_cast<const float2*>(&smem.state_in[dk][dv0]);
        }
    }

    static __device__ __forceinline__ void StoreStatePair(StateT* dst, const float2& value)
    {
        if constexpr (std::is_same_v<StateT, float>) {
            auto src = cute::make_tensor(cute::make_rmem_ptr(&value.x),
                                         cute::Layout<cute::Shape<cute::_2>, cute::Stride<cute::_1>>{});
            auto out = cute::make_tensor(cute::make_gmem_ptr(dst),
                                         cute::Layout<cute::Shape<cute::_2>, cute::Stride<cute::_1>>{});
            cute::copy(cute::Copy_Atom<cute::UniversalCopy<uint64_t>, float>{}, src, out);
        }
        else {
            static_assert(std::is_same_v<StateT, __nv_bfloat16>);
            __nv_bfloat162 packed = __float22bfloat162_rn(value);
            auto src = cute::make_tensor(cute::make_rmem_ptr(reinterpret_cast<__nv_bfloat16*>(&packed)),
                                         cute::Layout<cute::Shape<cute::_2>, cute::Stride<cute::_1>>{});
            auto out = cute::make_tensor(cute::make_gmem_ptr(dst),
                                         cute::Layout<cute::Shape<cute::_2>, cute::Stride<cute::_1>>{});
            cute::copy(cute::Copy_Atom<cute::UniversalCopy<uint32_t>, __nv_bfloat16>{}, src, out);
        }
    }

    static __device__ __forceinline__ void UpdateAndStoreStatePair(const SharedStorage& smem,
                                                                   StateT*              state_tile,
                                                                   int                  dk,
                                                                   int                  dv0,
                                                                   float                decay,
                                                                   float                k_value,
                                                                   const float2&        delta)
    {
        float2 state_vec = LoadStatePair(smem, dk, dv0);
        state_vec.x = state_vec.x * decay + k_value * delta.x;
        state_vec.y = state_vec.y * decay + k_value * delta.y;
        StoreStatePair(state_tile + static_cast<int64_t>(dk) * kHeadDim + dv0, state_vec);
    }

    static __device__ __forceinline__ float LoadStateScalar(const SharedStorage& smem, int dk, int dv)
    {
        if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
            const float2 state_pair = LoadStatePair(smem, dk, dv & ~1);
            return (dv & 1) == 0 ? state_pair.x : state_pair.y;
        }
        else {
            static_assert(std::is_same_v<StateT, float>);
            return smem.state_in[dk][dv];
        }
    }

    static __device__ __forceinline__ void AccumulateScalar(const SharedStorage& smem,
                                                            int                  dk_group,
                                                            int                  dv,
                                                            float                decay,
                                                            float&               corr,
                                                            float&               sq)
    {
        constexpr auto qk_layout = QkSmemLayout();
#pragma unroll
        for (int pair = 0; pair < kQkPairsPerGroup; ++pair) {
            const float4 qk   = LoadQk128(&smem.qk.packed[qk_layout(dk_group, pair)]);
            int          dk   = dk_group * kDkPerGroup + 2 * pair;
            const float  state = LoadStateScalar(smem, dk, dv) * decay;
            corr += state * qk.x;
            sq += state * qk.z;

            ++dk;
            const float next_state = LoadStateScalar(smem, dk, dv) * decay;
            corr += next_state * qk.y;
            sq += next_state * qk.w;
        }
    }

    static __device__ __forceinline__ void AccumulatePair(const SharedStorage& smem,
                                                          int                  dk_group,
                                                          int                  dv0,
                                                          float                decay,
                                                          float&               corr0,
                                                          float&               corr1,
                                                          float&               sq0,
                                                          float&               sq1)
    {
        constexpr auto qk_layout = QkSmemLayout();
#pragma unroll
        for (int pair = 0; pair < kQkPairsPerGroup; ++pair) {
            const float4 qk = LoadQk128(&smem.qk.packed[qk_layout(dk_group, pair)]);
            int          dk = dk_group * kDkPerGroup + 2 * pair;
            float2       state_vec = LoadStatePair(smem, dk, dv0);
            const float  state0 = state_vec.x * decay;
            const float  state1 = state_vec.y * decay;
            corr0 += state0 * qk.x;
            corr1 += state1 * qk.x;
            sq0 += state0 * qk.z;
            sq1 += state1 * qk.z;

            ++dk;
            state_vec = LoadStatePair(smem, dk, dv0);
            const float next_state0 = state_vec.x * decay;
            const float next_state1 = state_vec.y * decay;
            corr0 += next_state0 * qk.y;
            corr1 += next_state1 * qk.y;
            sq0 += next_state0 * qk.w;
            sq1 += next_state1 * qk.w;
        }
    }

    template<int ComputeGroup>
    static __device__ __forceinline__ void ComputeAndPublishPartial(
        SharedStorage& smem, int role_tid, float decay)
    {
        static_assert(0 <= ComputeGroup && ComputeGroup < kComputeGroups);
        const int dv_lane  = role_tid % kCorrDvThreads;
        const int dk_group = role_tid / kCorrDvThreads;
        const int dv_local = dv_lane * kCorrDvPerThread;
        const int dv0      = ComputeGroup * kComputeBlockDv + dv_local;

        if constexpr (kCorrDvPerThread == 1) {
            float corr = 0.0f;
            float sq   = 0.0f;
            AccumulateScalar(smem, dk_group, dv0, decay, corr, sq);
            smem.corr.partial_corr[ComputeGroup][dk_group][dv_local] = corr;
            smem.partial_sq[ComputeGroup][dk_group][dv_local]   = sq;
        }
        else {
            static_assert(kCorrDvPerThread == 2);
            float corr0 = 0.0f;
            float corr1 = 0.0f;
            float sq0   = 0.0f;
            float sq1   = 0.0f;
            AccumulatePair(smem, dk_group, dv0, decay, corr0, corr1, sq0, sq1);
            *reinterpret_cast<float2*>(&smem.corr.partial_corr[ComputeGroup][dk_group][dv_local]) =
                make_float2(corr0, corr1);
            *reinterpret_cast<float2*>(&smem.partial_sq[ComputeGroup][dk_group][dv_local]) =
                make_float2(sq0, sq1);
        }
    }

    template<int ComputeGroup>
    static __device__ __forceinline__ void Finalize(SharedStorage& smem, int role_tid, float decay, float beta)
    {
        static_assert(0 <= ComputeGroup && ComputeGroup < kComputeGroups);
        const int dv_thread = role_tid % kUpdateDvThreads;
        const int dk_worker = role_tid / kUpdateDvThreads;
        const int dv_local  = dv_thread * kUpdateDvPerThread;
        const int dv0       = ComputeGroup * kComputeBlockDv + dv_local;

        float2 delta{};
        if (dk_worker == 0) {
            float2 corr_sum =
                *reinterpret_cast<const float2*>(&smem.corr.partial_corr[ComputeGroup][0][dv_local]);
#pragma unroll
            for (int group = 1; group < kDkSplit; ++group) {
                const float2 corr_part =
                    *reinterpret_cast<const float2*>(&smem.corr.partial_corr[ComputeGroup][group][dv_local]);
                corr_sum.x += corr_part.x;
                corr_sum.y += corr_part.y;
            }
            delta = make_float2((smem.v[dv0] - corr_sum.x) * beta,
                                (smem.v[dv0 + 1] - corr_sum.y) * beta);
            *reinterpret_cast<float2*>(&smem.corr.partial_corr[ComputeGroup][0][dv_local]) = delta;
        }

        // Only Dk worker 0 performs the bitwise-identical reduction above.
        // Publish delta immediately through dead corr scratch so the remaining
        // workers can overlap state updates with worker 0's output reduction.
        cutlass::arch::NamedBarrier::sync(kComputeGroupThreads, kRecurrentBarrierPartial + ComputeGroup);
        if (dk_worker != 0) {
            delta = *reinterpret_cast<const float2*>(&smem.corr.partial_corr[ComputeGroup][0][dv_local]);
        }

        if (dk_worker == 0) {
            float2 sq_sum =
                *reinterpret_cast<const float2*>(&smem.partial_sq[ComputeGroup][0][dv_local]);
            float  kq_sum = smem.partial_kq[0];
#pragma unroll
            for (int group = 1; group < kDkSplit; ++group) {
                const float2 sq_part =
                    *reinterpret_cast<const float2*>(&smem.partial_sq[ComputeGroup][group][dv_local]);
                sq_sum.x += sq_part.x;
                sq_sum.y += sq_part.y;
                kq_sum += smem.partial_kq[group];
            }
            smem.out[dv0]     = __float2bfloat16(sq_sum.x + delta.x * kq_sum);
            smem.out[dv0 + 1] = __float2bfloat16(sq_sum.y + delta.y * kq_sum);
        }

        if (smem.tile_finished == 0) {
            auto* state_tile = reinterpret_cast<StateT*>(
                static_cast<uintptr_t>(smem.tile_state_address));
            constexpr auto qk_layout = QkSmemLayout();
            // Update TV map: role_tid = dk_worker * kUpdateDvThreads + dv_thread,
            // dv_local = 2 * dv_thread. Dv64-equivalent groups assign one Dk
            // worker per warp, so lane l writes Dv {2l, 2l + 1}: a contiguous,
            // non-overlapping 256-B f32 or 128-B bf16 state stripe. Dv32 assigns
            // two Dk workers per warp; each half warp writes one contiguous
            // 128-B f32 or 64-B bf16 stripe to a different Dk row.
#pragma unroll
            for (int local_pair = 0; local_pair < kUpdatePairsPerWorker; ++local_pair) {
                const int global_pair = dk_worker * kUpdatePairsPerWorker + local_pair;
                const int dk_group    = global_pair / kQkPairsPerGroup;
                const int pair        = global_pair % kQkPairsPerGroup;
                const float2 kk = LoadK64(&smem.qk.packed[qk_layout(dk_group, pair)]);
                int          dk = 2 * global_pair;
                UpdateAndStoreStatePair(smem, state_tile, dk, dv0, decay, kk.x, delta);
                ++dk;
                UpdateAndStoreStatePair(smem, state_tile, dk, dv0, decay, kk.y, delta);
            }
        }
    }

    static __device__ __forceinline__ void PreprocessAux(
        SharedStorage& smem, int role_tid, int pipeline_phase)
    {
        static_assert(kQkPairs == 2 * 32);
        static_assert(BlockDv % 2 == 0);

        cute::wait_barrier(smem.aux_ready, pipeline_phase);

        // Raw V uses a linear bfloat162 TV mapping. Pair p starts at bank
        // p % 32. Dv128 makes two conflict-free iterations of the same warp.
        for (int value_pair = role_tid; value_pair < BlockDv / 2; value_pair += 32) {
            const auto* raw_v = reinterpret_cast<const __nv_bfloat162*>(&smem.corr.v_raw[0]);
            const float2 value = __bfloat1622float2(raw_v[value_pair]);
            *reinterpret_cast<float2*>(&smem.v[2 * value_pair]) = value;
        }

        if (role_tid == 0) {
            smem.gate_beta[0] = exp2f(smem.gate_beta[0] * kLog2e);
        }

        // One consumer warp captures all 64 raw Q/K pairs: lane t owns t and
        // t+32. Both bfloat162 loads map to bank t, one conflict-free warp
        // instruction per half, matching the former loader-warp mapping.
        const auto* raw_q = reinterpret_cast<const __nv_bfloat162*>(&smem.qk.raw.q[0]);
        const auto* raw_k = reinterpret_cast<const __nv_bfloat162*>(&smem.qk.raw.k[0]);
        float2      q_lo  = __bfloat1622float2(raw_q[role_tid]);
        float2      q_hi  = __bfloat1622float2(raw_q[role_tid + 32]);
        const float2 k_lo = __bfloat1622float2(raw_k[role_tid]);
        const float2 k_hi = __bfloat1622float2(raw_k[role_tid + 32]);

        q_lo.x *= kHeadScale;
        q_lo.y *= kHeadScale;
        q_hi.x *= kHeadScale;
        q_hi.y *= kHeadScale;
        float kq_lo = k_lo.x * q_lo.x;
        kq_lo       = fmaf(k_lo.y, q_lo.y, kq_lo);
        float kq_hi = k_hi.x * q_hi.x;
        kq_hi       = fmaf(k_hi.y, q_hi.y, kq_hi);

        // QkStorage::packed overlays QkStorage::raw. This warp rendezvous
        // captures every raw pair in registers before packed stores begin.
        __syncwarp();
        constexpr auto qk_layout = QkSmemLayout();
        const int      group_lo  = role_tid / kQkPairsPerGroup;
        const int      pair      = role_tid % kQkPairsPerGroup;
        StoreQk128(&smem.qk.packed[qk_layout(group_lo, pair)], k_lo.x, k_lo.y, q_lo.x, q_lo.y);
        const int group_hi = (role_tid + 32) / kQkPairsPerGroup;
        StoreQk128(&smem.qk.packed[qk_layout(group_hi, pair)], k_hi.x, k_hi.y, q_hi.x, q_hi.y);

#pragma unroll
        for (int offset = kQkPairsPerGroup / 2; offset > 0; offset /= 2) {
            const float other_lo = __shfl_down_sync(0xffffffff, kq_lo, offset, kQkPairsPerGroup);
            const float other_hi = __shfl_down_sync(0xffffffff, kq_hi, offset, kQkPairsPerGroup);
            if (pair < offset) {
                kq_lo += other_lo;
                kq_hi += other_hi;
            }
        }
        if (pair == 0) {
            smem.partial_kq[group_lo] = kq_lo;
            smem.partial_kq[group_hi] = kq_hi;
        }

        // Join preprocessing with the in-flight state transaction. The shared
        // state barrier releases all consumers only after both are complete,
        // eliminating an extra CTA-wide named-barrier round trip.
        __syncwarp();
        if (role_tid == 0) {
            cute::arrive_barrier(smem.state_tma_ready);
        }
    }

    template<int ComputeGroup>
    static __device__ void ComputeTile(SharedStorage& smem, int role_tid)
    {
        static_assert(0 <= ComputeGroup && ComputeGroup < kComputeGroups);
        const float decay = smem.gate_beta[0];
        const float beta  = smem.gate_beta[1];

        ComputeAndPublishPartial<ComputeGroup>(smem, role_tid, decay);

        // Each compute group publishes the same four Dk32 partials before any
        // lane consumes the shared reductions.
        cutlass::arch::NamedBarrier::sync(kComputeGroupThreads, kRecurrentBarrierPartial + ComputeGroup);

        Finalize<ComputeGroup>(smem, role_tid, decay, beta);
    }

    template<int ComputeGroup>
    static __device__ __forceinline__ void RunComputeGroup(SharedStorage& smem, int role_tid, int total_tiles)
    {
        static_assert(0 <= ComputeGroup && ComputeGroup < kComputeGroups);
        int           tile_phase       = 0;
        constexpr int kDvTiles         = kHeadDim / BlockDv;
        const int     total_work_tiles = total_tiles * kDvTiles;
        const int     first_work_tile  = static_cast<int>(blockIdx.x);
        for (int work_tile = first_work_tile; work_tile < total_work_tiles;
             work_tile += static_cast<int>(gridDim.x)) {
            // WG0's first warp preprocesses raw auxiliary inputs while the
            // independent state TMA remains active. Other consumers proceed
            // directly to the joined state/preprocessing barrier.
            if constexpr (ComputeGroup == 0) {
                if (role_tid < 32) {
                    PreprocessAux(smem, role_tid, tile_phase);
                }
            }

            // Acquire state, prepared auxiliary inputs, and tile metadata.
            cute::wait_barrier(smem.state_tma_ready, tile_phase);
            tile_phase ^= 1;
            ComputeTile<ComputeGroup>(smem, role_tid);

            // Rendezvous: every lane finishes computation and state/output
            // writes before lane 0 of its warp may release compute_done.
            __syncwarp();
            if ((role_tid & 31) == 0) {
                // Release this compute warp's updated state and output; the
                // producer warp acquires after every representative arrives.
                cute::arrive_barrier(smem.compute_done);
            }
        }
    }

    static constexpr int kThreads            = kRecurrentThreads;
    static constexpr int kLaunchBoundThreads = kRecurrentLaunchBoundThreads;
    static constexpr int kMinBlocks          = kRecurrentMinBlocksPerSm;

    static __device__ __forceinline__ void Run(const CUtensorMap* q_tma_desc,
                                               const CUtensorMap* k_tma_desc,
                                               const CUtensorMap* v_tma_desc,
                                               __nv_bfloat16* __restrict__ out,
                                               const float* __restrict__ g,
                                               const float* __restrict__ beta,
                                               const bool* __restrict__ finished,
                                               const int64_t* __restrict__ state_ptrs,
                                               const CUtensorMap* __restrict__ state_tma_descs,
                                               int           total_tiles,
                                               int           batch_count,
                                               int           hq,
                                               int           hv,
                                               int64_t       gate_batch_stride,
                                               int64_t       out_batch_stride,
                                               int64_t       out_head_stride,
                                               int           num_head_groups,
                                               int           heads_per_block,
                                               int64_t       state_layer_offset,
                                               int           state_layer,
                                               unsigned char* smem_raw)
    {
        auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

        const int tid    = static_cast<int>(threadIdx.x);
        const int wg_idx = cutlass::canonical_warp_group_idx();
        static_cast<void>(batch_count);

        if (tid == 0) {
            cute::prefetch_tma_descriptor(q_tma_desc);
            cute::prefetch_tma_descriptor(k_tma_desc);
            cute::prefetch_tma_descriptor(v_tma_desc);
            cute::initialize_barrier(smem.aux_tma_ready, 1);
            cute::initialize_barrier(smem.state_tma_ready, 2);
            cute::initialize_barrier(smem.aux_ready, 1);
            cute::initialize_barrier(smem.compute_done, kRecurrentConsumerWarps);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == kRecurrentConsumerWarpGroups) {
            // One complete warp in an incomplete producer WG. Never call
            // warpgroup_reg_* on this path.

            constexpr int kDvTiles         = kHeadDim / BlockDv;
            const int     total_work_tiles = total_tiles * kDvTiles;
            const int     first_work_tile  = static_cast<int>(blockIdx.x);
            const int     role_tid         = tid - kRecurrentConsumerThreads;

            if (first_work_tile >= total_work_tiles) {
                return;
            }

            int aux_phase  = 0;
            int done_phase = 0;
            int work_tile  = first_work_tile;

            // Prologue: no current output is live, so publish tile 0 directly.
            if (role_tid == 0) {
                const IssuedTileMetadata tile = IssueTile(smem,
                                                          work_tile,
                                                          state_ptrs,
                                                          state_tma_descs,
                                                          *q_tma_desc,
                                                          *k_tma_desc,
                                                          *v_tma_desc,
                                                          g,
                                                          beta,
                                                          finished,
                                                          hq,
                                                          hv,
                                                          gate_batch_stride,
                                                          out_batch_stride,
                                                          out_head_stride,
                                                          num_head_groups,
                                                          heads_per_block,
                                                          state_layer_offset,
                                                          state_layer);
                CommitTileMetadata(smem, tile);
                cute::wait_barrier(smem.aux_tma_ready, aux_phase);
                cute::arrive_barrier(smem.aux_ready);
            }
            aux_phase ^= 1;

            for (;;) {
                // All current state stores are issued and every shared lifetime
                // except output and its offset is dead when this wait releases.
                cute::wait_barrier(smem.compute_done, done_phase);
                done_phase ^= 1;

                const int  next_work_tile = work_tile + static_cast<int>(gridDim.x);
                const bool has_next       = next_work_tile < total_work_tiles;

                IssuedTileMetadata next_tile{};
                if (has_next && role_tid == 0) {
                    next_tile = IssueTile(smem,
                                          next_work_tile,
                                          state_ptrs,
                                          state_tma_descs,
                                          *q_tma_desc,
                                          *k_tma_desc,
                                          *v_tma_desc,
                                          g,
                                          beta,
                                          finished,
                                          hq,
                                          hv,
                                          gate_batch_stride,
                                          out_batch_stride,
                                          out_head_stride,
                                          num_head_groups,
                                          heads_per_block,
                                          state_layer_offset,
                                          state_layer);
                }

                // The next transactions are in flight while the whole producer
                // warp copies the current output. IssueTile intentionally leaves
                // current output metadata untouched.
                __syncwarp();
                const int64_t out_offset = smem.tile_out_offset;
                auto*       out_words  = reinterpret_cast<uint32_t*>(out + out_offset);
                const auto* smem_words = reinterpret_cast<const uint32_t*>(&smem.out[0]);
                for (int word = role_tid; word < BlockDv / 2; word += kRecurrentProducerThreads) {
                    out_words[word] = smem_words[word];
                }
                __syncwarp();

                // Epilogue: do not create an unmatched transaction generation.
                if (!has_next) {
                    break;
                }

                if (role_tid == 0) {
                    CommitTileMetadata(smem, next_tile);
                    cute::wait_barrier(smem.aux_tma_ready, aux_phase);
                    cute::arrive_barrier(smem.aux_ready);
                }
                aux_phase ^= 1;
                work_tile = next_work_tile;
            }
            return;
        }
        else if (wg_idx < kRecurrentConsumerWarpGroups) {
            // One Dv compute band per complete consumer WG. Do not call
            // warpgroup_reg_alloc on the incomplete producer WG path above.
            if constexpr (BlockDv == 128) {
                if (wg_idx == 0) {
                    RunComputeGroup<0>(smem, tid, total_tiles);
                }
                else {
                    RunComputeGroup<1>(smem, tid - kComputeGroupThreads, total_tiles);
                }
            }
            else {
                RunComputeGroup<0>(smem, tid, total_tiles);
            }
            return;
        }
    }
};

template<int BlockDv, class StateT>
__global__ __launch_bounds__(Sm90GdrRecurrent<BlockDv, StateT>::kLaunchBoundThreads,
                             Sm90GdrRecurrent<BlockDv, StateT>::kMinBlocks) void Sm90GdrRecurrentKernel(
    const __grid_constant__ CUtensorMap q_tma_desc,
    const __grid_constant__ CUtensorMap k_tma_desc,
    const __grid_constant__ CUtensorMap v_tma_desc,
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__        g,
    const float* __restrict__        beta,
    const bool* __restrict__ finished,
    const int64_t* __restrict__ state_ptrs,
    const CUtensorMap* __restrict__ state_tma_descs,
    int     total_tiles,
    int     batch_count,
    int     hq,
    int     hv,
    int64_t gate_batch_stride,
    int64_t out_batch_stride,
    int64_t out_head_stride,
    int     num_head_groups,
    int     heads_per_block,
    int64_t state_layer_offset,
    int     state_layer)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Sm90GdrRecurrent<BlockDv, StateT>::Run(&q_tma_desc,
                                           &k_tma_desc,
                                           &v_tma_desc,
                                           out,
                                           g,
                                           beta,
                                           finished,
                                           state_ptrs,
                                           state_tma_descs,
                                           total_tiles,
                                           batch_count,
                                           hq,
                                           hv,
                                           gate_batch_stride,
                                           out_batch_stride,
                                           out_head_stride,
                                           num_head_groups,
                                           heads_per_block,
                                           state_layer_offset,
                                           state_layer,
                                           smem_raw);
}

template<int BlockDv, class StateT>
void SetRecurrentGdrSharedMemoryLimit(size_t smem_bytes)
{
    auto kernel = Sm90GdrRecurrentKernel<BlockDv, StateT>;
    TM_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    TM_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

template<int BlockDv, class StateT>
int RecurrentGdrActiveCtasPerSm()
{
    using Kernel = Sm90GdrRecurrent<BlockDv, StateT>;
    auto kernel  = Sm90GdrRecurrentKernel<BlockDv, StateT>;
    SetRecurrentGdrSharedMemoryLimit<BlockDv, StateT>(Kernel::SharedBytes());
    int active_ctas = 0;
    TM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_ctas, kernel, Kernel::kThreads, Kernel::SharedBytes()));
    TM_CHECK(active_ctas > 0) << "SM90 recurrent kernel has zero active CTAs for BlockDv=" << BlockDv;
    return active_ctas;
}

struct RecurrentGdrDispatchCandidate {
    int     block_dv{};
    int64_t memory_bytes_per_tile{};
    int64_t work_tiles{};
    int64_t wave_capacity{};
    int64_t active_sms{};
};

template<int BlockDv, class StateT>
RecurrentGdrDispatchCandidate MakeRecurrentGdrDispatchCandidate(const Problem& problem)
{
    using Kernel = Sm90GdrRecurrent<BlockDv, StateT>;
    const int64_t head_tiles =
        std::max<int64_t>(problem.sequence_num, 0) * std::max<int64_t>(problem.hv, 0);
    const int64_t work_tiles = head_tiles * (Kernel::kHeadDim / BlockDv);
    const int64_t ctas_per_sm = RecurrentGdrActiveCtasPerSm<BlockDv, StateT>();
    const int64_t wave_capacity = std::max<int64_t>(
        1,
        std::min<int64_t>(std::max<int64_t>(problem.sm_count, 1) * ctas_per_sm,
                          Kernel::kRecurrentMaxDescriptorCtas));
    const int64_t grid_blocks = std::min(work_tiles, wave_capacity);
    const int64_t active_sms = std::min<int64_t>(std::max<int64_t>(problem.sm_count, 1), grid_blocks);
    return {BlockDv,
            Kernel::MemoryBytesPerWorkTile(),
            work_tiles,
            wave_capacity,
            active_sms};
}

using RecurrentGdrMemoryCost = uint64_t;

RecurrentGdrMemoryCost EstimateRecurrentGdrMemoryCost(const RecurrentGdrDispatchCandidate& candidate)
{
    // Latency-throughput model in byte-equivalent units:
    // - The current single-slot pipeline has a roughly one-tile boundary
    //   latency: consumers wait for the first TMA fill, and the slot is released
    //   after direct state-store issue plus output copy. With per-SM bandwidth b,
    //   tile_bytes / b. Normalizing by the aggregate active_sms * b turns that
    //   latency into active_sms * tile_bytes. CTAs within each resident wave
    //   remain parallel and do not multiply the per-CTA boundary latency.
    // - Once filled, HBM is shared by the whole grid. Charge the exact traffic
    //   of every work tile, including a partial final wave. This captures the
    //   extra Q/K/control loads from using more Dv tiles per logical head.
    // The common inverse-bandwidth factor cancels when candidates are compared.
    const auto boundary_tiles = static_cast<RecurrentGdrMemoryCost>(candidate.active_sms);
    const auto total_tiles = boundary_tiles + static_cast<RecurrentGdrMemoryCost>(candidate.work_tiles);
    return static_cast<RecurrentGdrMemoryCost>(candidate.memory_bytes_per_tile) * total_tiles;
}

template<class StateT>
int SelectRecurrentGdrBlockDv(const Problem& problem)
{
    const int64_t head_tiles =
        std::max<int64_t>(problem.sequence_num, 0) * std::max<int64_t>(problem.hv, 0);
    if (head_tiles == 0) {
        return 32;
    }

    RecurrentGdrDispatchCandidate candidates[] = {
        MakeRecurrentGdrDispatchCandidate<32, StateT>(problem),
        MakeRecurrentGdrDispatchCandidate<64, StateT>(problem),
        MakeRecurrentGdrDispatchCandidate<128, StateT>(problem),
    };

    // A narrower Dv band exposes more independent CTAs. Keep Dv32 while all of
    // its work fits in one CUDA-reported resident wave; beyond that point Dv64
    // still has enough CTAs to fill the SMs while duplicating less work.
    if (candidates[0].work_tiles <= candidates[0].wave_capacity) {
        return 32;
    }

    // Dv64 and Dv128 use the same Dv64 compute-group shape. Compare packing
    // one versus two such groups into a CTA using the physical memory model
    // above; the crossover follows from SM count, occupancy, state type, and
    // the implementation's calculated traffic.
    const auto dv64_cost  = EstimateRecurrentGdrMemoryCost(candidates[1]);
    const auto dv128_cost = EstimateRecurrentGdrMemoryCost(candidates[2]);
    return dv128_cost <= dv64_cost ? 128 : 64;
}

int SelectRecurrentGdrBlockDv(const Problem& problem, DataType state_dtype)
{
    if (state_dtype == kFloat32) {
        return SelectRecurrentGdrBlockDv<float>(problem);
    }
    return SelectRecurrentGdrBlockDv<__nv_bfloat16>(problem);
}

template<class StateT>
__global__ __launch_bounds__(32, 1) void PrepareGroupedStateDescriptors(
    const __grid_constant__ CUtensorMap state_tma_desc,
    const int64_t* addresses,
    int64_t layer_group_stride,
    int64_t sequence_stride,
    int64_t head_group_stride,
    CUtensorMap* descriptors,
    int sequence_count,
    int num_head_groups)
{
    __shared__ __align__(128) CUtensorMap smem_descriptor;
    const int linear = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int head_group = linear % num_head_groups;
    const int sequence = (linear / num_head_groups) % sequence_count;
    const int layer_group = linear / (sequence_count * num_head_groups);
    const int64_t pointer_index = static_cast<int64_t>(layer_group) * layer_group_stride
                                  + static_cast<int64_t>(sequence) * sequence_stride
                                  + static_cast<int64_t>(head_group) * head_group_stride;
    CopyTmaDescriptor(&smem_descriptor, &state_tma_desc, lane, 32);
    __syncwarp();
    if (lane == 0) {
        auto* state_base = reinterpret_cast<StateT*>(
            static_cast<uintptr_t>(addresses[pointer_index]));
        ReplaceTmaAddress(&smem_descriptor, state_base);
    }
    __syncwarp();
    PublishTmaDescriptor(&descriptors[linear], &smem_descriptor);
    __syncwarp();
    cute::tma_descriptor_fence_acquire(
        reinterpret_cast<cute::TmaDescriptor*>(&descriptors[linear]));
}

template<int BlockDv, class StateT>
void PrepareSm90RecurrentStateTmaDescriptorsTyped(const core::Tensor& state_ptrs,
                                                  core::Tensor& state_tma_descs,
                                                  int layer_groups,
                                                  int layers_per_block,
                                                  int sequence_count,
                                                  int num_head_groups,
                                                  int heads_per_block,
                                                  cudaStream_t stream)
{
    using Kernel = Sm90GdrRecurrent<BlockDv, StateT>;
    const auto state_tma_desc = Kernel::MakeStateTmaDesc(
        reinterpret_cast<StateT*>(state_tma_descs.raw_data()),
        layers_per_block,
        heads_per_block,
        BlockDv);
    const auto* addresses = reinterpret_cast<const int64_t*>(state_ptrs.raw_data());
    auto* descriptors = reinterpret_cast<CUtensorMap*>(state_tma_descs.raw_data());
    const int work = layer_groups * sequence_count * num_head_groups;
    PrepareGroupedStateDescriptors<StateT><<<work, 32, 0, stream>>>(state_tma_desc,
                                                                    addresses,
                                                                    state_ptrs.stride(0),
                                                                    state_ptrs.stride(1),
                                                                    state_ptrs.stride(2),
                                                                    descriptors,
                                                                    sequence_count,
                                                                    num_head_groups);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<int BlockDv, class StateT>
void LaunchSm90GdrRecurrentTyped(const core::Tensor& q,
                                 const core::Tensor& k,
                                 const core::Tensor& v,
                                 const core::Tensor& g,
                                 const core::Tensor& beta,
                                 const core::Tensor& finished,
                                 const core::Tensor& state_ptrs,
                                 const core::Tensor& state_tma_descs,
                                 core::Tensor&       out,
                                 const Problem&      problem,
                                 int64_t             state_layer_offset,
                                 void*               tma_desc_workspace,
                                 cudaStream_t        stream)
{
    const auto* g_ptr = g.data<float>();
    const auto* beta_ptr = beta.data<float>();
    auto*       out_ptr  = reinterpret_cast<__nv_bfloat16*>(out.raw_data());
    using Kernel         = Sm90GdrRecurrent<BlockDv, StateT>;
    const auto* finished_ptr = finished.data<bool>();
    const auto* state_ptrs_ptr = reinterpret_cast<const int64_t*>(state_ptrs.raw_data());
    const int state_layer =
        static_cast<int>(state_layer_offset
                         / (static_cast<int64_t>(problem.heads_per_block) * Kernel::kHeadDim * Kernel::kHeadDim));
    const auto* state_tma_descs_ptr = reinterpret_cast<const CUtensorMap*>(state_tma_descs.raw_data());
    static_cast<void>(tma_desc_workspace);

    constexpr int block_dv         = Kernel::BlockDv;
    const int     total_tiles      = problem.batch * problem.hv;
    constexpr int dv_tiles         = Kernel::kHeadDim / block_dv;
    const int     sm_count         = problem.sm_count;
    const int     active_ctas      = RecurrentGdrActiveCtasPerSm<block_dv, StateT>();
    const int     base_grid_blocks = sm_count * active_ctas;
    const int     total_work_tiles = total_tiles * dv_tiles;
    // Persistent grid: fill SMs up to base_grid (792 @ Dv64/6CTA).
    const int grid_blocks = std::max(
        1,
        std::min({total_work_tiles, base_grid_blocks, Kernel::kRecurrentMaxDescriptorCtas}));
    const size_t smem_bytes = Kernel::SharedBytes();
    const auto   q_tma_desc = Kernel::template MakeQkTmaDesc<__nv_bfloat16>(q);
    const auto   k_tma_desc = Kernel::template MakeQkTmaDesc<__nv_bfloat16>(k);
    const auto   v_tma_desc = Kernel::template MakeValueTmaDesc<__nv_bfloat16>(v, block_dv);
    Sm90GdrRecurrentKernel<block_dv, StateT><<<grid_blocks, Kernel::kThreads, smem_bytes, stream>>>(
        q_tma_desc,
        k_tma_desc,
        v_tma_desc,
        out_ptr,
        g_ptr,
        beta_ptr,
        finished_ptr,
        state_ptrs_ptr,
        state_tma_descs_ptr,
        total_tiles,
        problem.sequence_num,
        problem.hq,
        problem.hv,
        problem.gate_batch_stride,
        out.stride(0),
        out.stride(2),
        problem.num_head_groups,
        problem.heads_per_block,
        state_layer_offset,
        state_layer);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace detail {

void LaunchSm90Recurrent(const core::Tensor& q,
                         const core::Tensor& k,
                         const core::Tensor& v,
                         const core::Tensor& g,
                         const core::Tensor& beta,
                         const core::Tensor& finished,
                         const core::Tensor& state_ptrs,
                         const core::Tensor& state_tma_descs,
                         core::Tensor& out,
                         const Problem& problem,
                         int64_t state_layer_offset,
                         DataType state_dtype,
                         cudaStream_t stream)
{
    const int block_dv = SelectRecurrentGdrBlockDv(problem, state_dtype);
    if (state_dtype == kFloat32) {
        if (block_dv == 32) {
            LaunchSm90GdrRecurrentTyped<32, float>(q,
                                                   k,
                                                   v,
                                                   g,
                                                   beta,
                                                   finished,
                                                   state_ptrs,
                                                   state_tma_descs,
                                                   out,
                                                   problem,
                                                   state_layer_offset,
                                                   nullptr,
                                                   stream);
        }
        else if (block_dv == 64) {
            LaunchSm90GdrRecurrentTyped<64, float>(q,
                                                   k,
                                                   v,
                                                   g,
                                                   beta,
                                                   finished,
                                                   state_ptrs,
                                                   state_tma_descs,
                                                   out,
                                                   problem,
                                                   state_layer_offset,
                                                   nullptr,
                                                   stream);
        }
        else {
            LaunchSm90GdrRecurrentTyped<128, float>(q,
                                                    k,
                                                    v,
                                                    g,
                                                    beta,
                                                    finished,
                                                    state_ptrs,
                                                    state_tma_descs,
                                                    out,
                                                    problem,
                                                    state_layer_offset,
                                                    nullptr,
                                                    stream);
        }
    }
    else if (block_dv == 32) {
        LaunchSm90GdrRecurrentTyped<32, __nv_bfloat16>(q,
                                                       k,
                                                       v,
                                                       g,
                                                       beta,
                                                       finished,
                                                       state_ptrs,
                                                       state_tma_descs,
                                                       out,
                                                       problem,
                                                       state_layer_offset,
                                                       nullptr,
                                                       stream);
    }
    else if (block_dv == 64) {
        LaunchSm90GdrRecurrentTyped<64, __nv_bfloat16>(q,
                                                       k,
                                                       v,
                                                       g,
                                                       beta,
                                                       finished,
                                                       state_ptrs,
                                                       state_tma_descs,
                                                       out,
                                                       problem,
                                                       state_layer_offset,
                                                       nullptr,
                                                       stream);
    }
    else {
        LaunchSm90GdrRecurrentTyped<128, __nv_bfloat16>(q,
                                                        k,
                                                        v,
                                                        g,
                                                        beta,
                                                        finished,
                                                        state_ptrs,
                                                        state_tma_descs,
                                                        out,
                                                        problem,
                                                        state_layer_offset,
                                                        nullptr,
                                                        stream);
    }
}

void PrepareSm90RecurrentStateTmaDescriptors(const core::Tensor& state_ptrs,
                                             core::Tensor& state_tma_descs,
                                             int layer_groups,
                                             int layers_per_block,
                                             const Plan& plan,
                                             cudaStream_t stream)
{
    const int block_dv = SelectRecurrentGdrBlockDv(plan.problem, plan.problem.state_dtype);
    if (plan.problem.state_dtype == kFloat32) {
        if (block_dv == 32) {
            PrepareSm90RecurrentStateTmaDescriptorsTyped<32, float>(
                state_ptrs,
                state_tma_descs,
                layer_groups,
                layers_per_block,
                plan.problem.sequence_num,
                plan.problem.num_head_groups,
                plan.problem.heads_per_block,
                stream);
        }
        else if (block_dv == 64) {
            PrepareSm90RecurrentStateTmaDescriptorsTyped<64, float>(
                state_ptrs,
                state_tma_descs,
                layer_groups,
                layers_per_block,
                plan.problem.sequence_num,
                plan.problem.num_head_groups,
                plan.problem.heads_per_block,
                stream);
        }
        else {
            PrepareSm90RecurrentStateTmaDescriptorsTyped<128, float>(
                state_ptrs,
                state_tma_descs,
                layer_groups,
                layers_per_block,
                plan.problem.sequence_num,
                plan.problem.num_head_groups,
                plan.problem.heads_per_block,
                stream);
        }
    }
    else if (block_dv == 32) {
        PrepareSm90RecurrentStateTmaDescriptorsTyped<32, __nv_bfloat16>(
            state_ptrs,
            state_tma_descs,
            layer_groups,
            layers_per_block,
            plan.problem.sequence_num,
            plan.problem.num_head_groups,
            plan.problem.heads_per_block,
            stream);
    }
    else if (block_dv == 64) {
        PrepareSm90RecurrentStateTmaDescriptorsTyped<64, __nv_bfloat16>(
            state_ptrs,
            state_tma_descs,
            layer_groups,
            layers_per_block,
            plan.problem.sequence_num,
            plan.problem.num_head_groups,
            plan.problem.heads_per_block,
            stream);
    }
    else {
        PrepareSm90RecurrentStateTmaDescriptorsTyped<128, __nv_bfloat16>(
            state_ptrs,
            state_tma_descs,
            layer_groups,
            layers_per_block,
            plan.problem.sequence_num,
            plan.problem.num_head_groups,
            plan.problem.heads_per_block,
            stream);
    }
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
