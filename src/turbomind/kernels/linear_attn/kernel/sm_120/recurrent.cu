#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/linear_attn/kernel/tma_desc.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int      kHeadDim                           = 128;
constexpr float    kHeadScale                         = 0.08838834764831845f;
constexpr int      kRecurrentBlockDv                  = 64;
constexpr int      kRecurrentStages                   = 1;
constexpr int      kRecurrentConsumerThreads          = 256;
constexpr int      kRecurrentProducerThreads          = 128;
constexpr int      kRecurrentStoreThreads             = 32;
constexpr int      kRecurrentLoaderThreads            = kRecurrentProducerThreads - kRecurrentStoreThreads;
constexpr int      kRecurrentThreads                  = kRecurrentConsumerThreads + kRecurrentProducerThreads;
constexpr int      kRecurrentConsumerWarps            = kRecurrentConsumerThreads / 32;
constexpr int      kRecurrentConsumerWarpGroups       = kRecurrentConsumerThreads / 128;
constexpr int      kRecurrentLoaderWarps              = kRecurrentLoaderThreads / 32;
constexpr int      kRecurrentConsumerRegs             = 96;
constexpr int      kRecurrentProducerRegs             = 32;
constexpr int      kRecurrentBaseCtasPerSm            = 2;
constexpr int      kRecurrentMediumCtasPerSm          = 3;
constexpr int      kRecurrentMediumMinWorkPerBaseCta  = 8;
constexpr int      kRecurrentMediumMaxWorkPerBaseCta  = 16;
constexpr int      kRecurrentMaxDescriptorCtas        = 256;
constexpr int      kRecurrentSm120SharedBytes         = 102400;
constexpr int      kRecurrentStaticSharedReserveBytes = 1024;
constexpr int      kRecurrentMaxDynamicSharedBytes    = kRecurrentSm120SharedBytes - kRecurrentStaticSharedReserveBytes;
constexpr float    kLog2e                             = 1.4426950408889634f;
constexpr uint64_t kTmaNoCacheHint                    = 0;
constexpr int      kRecurrentBarrierPartial           = 0;
constexpr int      kRecurrentBarrierStateConvert      = 1;

static_assert(kHeadDim % kRecurrentBlockDv == 0);
static_assert(kRecurrentConsumerThreads % kRecurrentBlockDv == 0);
static_assert(kRecurrentStoreThreads == 32);
static_assert(kRecurrentStoreThreads == kRecurrentBlockDv / 2);
static_assert(kRecurrentLoaderThreads > 0);
static_assert(kRecurrentConsumerThreads % 32 == 0);
static_assert(kRecurrentConsumerThreads % 128 == 0);
static_assert(kRecurrentProducerThreads == 128);
static_assert(kRecurrentThreads == (kRecurrentConsumerWarpGroups + 1) * 128);
static_assert(kRecurrentLoaderThreads % 32 == 0);
static_assert(kRecurrentBarrierStateConvert + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
              < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

template<int BlockDv>
CUTE_HOST_DEVICE constexpr auto RecurrentStateSmemLayout()
{
    static_assert(BlockDv == kRecurrentBlockDv);
    return cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<BlockDv>>,
                        cute::Stride<cute::Int<BlockDv>, cute::_1>>{};
}

template<int BlockDv>
struct __align__(1024) Sm120GdrRecurrentSharedStorage
{
    __align__(1024) float        state[kRecurrentStages][kHeadDim][BlockDv];
    __align__(128) __nv_bfloat16 q_raw[kRecurrentStages][kHeadDim];
    __align__(128) __nv_bfloat16 k_raw[kRecurrentStages][kHeadDim];
    __align__(128) __nv_bfloat16 v_raw[kRecurrentStages][BlockDv];
    __align__(128) float         q[kRecurrentStages][kHeadDim];
    __align__(128) float         k[kRecurrentStages][kHeadDim];
    __align__(128) float         v[kRecurrentStages][BlockDv];
    __align__(128) __nv_bfloat16 out[kRecurrentStages][BlockDv];
    __align__(128) float2        partial_corr[kRecurrentStages][kRecurrentConsumerWarps][BlockDv / 2];
    __align__(128) float2        partial_sq[kRecurrentStages][kRecurrentConsumerWarps][BlockDv / 2];
    __align__(16) float          partial_kq[kRecurrentStages][kRecurrentConsumerWarps];
    __align__(16) float          gate_beta[kRecurrentStages][2];
    __align__(16) int            tile_finished[kRecurrentStages];
    __align__(16) int            tile_batch[kRecurrentStages];
    __align__(16) int            tile_value_head[kRecurrentStages];
    __align__(16) int            tile_dv0[kRecurrentStages];
    __align__(16) int64_t        tile_out_offset[kRecurrentStages];
    __align__(8) cute::uint64_t  state_tma_ready[kRecurrentStages];
    __align__(8) cute::uint64_t  aux_tma_ready[kRecurrentStages];
    __align__(8) cute::uint64_t  tile_ready[kRecurrentStages];
    __align__(8) cute::uint64_t  compute_done[kRecurrentStages];
    __align__(8) cute::uint64_t  stage_free[kRecurrentStages];
};

template<int BlockDv>
constexpr size_t Sm120GdrRecurrentSharedBytes()
{
    static_assert(BlockDv == kRecurrentBlockDv);
    return sizeof(Sm120GdrRecurrentSharedStorage<BlockDv>);
}

static_assert(Sm120GdrRecurrentSharedBytes<kRecurrentBlockDv>() <= kRecurrentMaxDynamicSharedBytes);

template<class T>
constexpr CUtensorMapDataType RecurrentTmaDataType()
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template<>
constexpr CUtensorMapDataType RecurrentTmaDataType<float>()
{
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
}

template<class T>
CUtensorMap MakeRecurrentQkTmaDesc(const core::Tensor& tensor)
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
    const uint32_t box_dims[5] = {64u, 2u, 1u, 1u, 1u};
    return MakeTmaDesc(const_cast<T*>(tensor.data<T>()),
                       RecurrentTmaDataType<T>(),
                       5,
                       global_dims,
                       global_strides,
                       box_dims,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class T>
CUtensorMap MakeRecurrentValueTmaDesc(const core::Tensor& tensor, int block_dv)
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
    const uint32_t box_dims[4] = {static_cast<uint32_t>(block_dv), 1u, 1u, 1u};
    return MakeTmaDesc(const_cast<T*>(tensor.data<T>()),
                       RecurrentTmaDataType<T>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<class StateT>
CUtensorMap MakeRecurrentStateTmaDesc(StateT* ptr, int layers, int hv, int block_dv)
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
        ptr, RecurrentTmaDataType<StateT>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

__device__ __forceinline__ int DecodeHeadTileBatch(int head_tile, int hv)
{
    return head_tile / hv;
}

__device__ __forceinline__ int DecodeHeadTileHead(int head_tile, int hv)
{
    return head_tile % hv;
}

template<class StateT, int BlockDv>
__device__ __forceinline__ constexpr int RecurrentStateTmaBytes()
{
    return kHeadDim * BlockDv * static_cast<int>(sizeof(StateT));
}

template<class StateT, int BlockDv>
__device__ __forceinline__ void RecurrentUnpackStateTma(float (&state_stage)[kHeadDim][BlockDv], int tid, int threads)
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
            cutlass::arch::NamedBarrier::sync(threads, kRecurrentBarrierStateConvert);
        }
        if (tid == 0) {
            const float2 pair = __bfloat1622float2(overlay[0]);
            state_stage[0][0] = pair.x;
            state_stage[0][1] = pair.y;
        }
        cutlass::arch::NamedBarrier::sync(threads, kRecurrentBarrierStateConvert);
    }
}

template<class StateT, int BlockDv>
__device__ __forceinline__ void RecurrentPackStateTma(float (&state_stage)[kHeadDim][BlockDv], int tid, int threads)
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
        __syncwarp();
        for (int begin = 1; begin < kPairs; begin <<= 1) {
            for (int linear = begin + tid; linear < 2 * begin; linear += threads) {
                const int element = 2 * linear;
                const int dk      = element / BlockDv;
                const int dv      = element - dk * BlockDv;
                overlay[linear]   = __float22bfloat162_rn(make_float2(state_stage[dk][dv], state_stage[dk][dv + 1]));
            }
            __syncwarp();
        }
    }
}

template<int BlockDv, class StateT>
__device__ void StageRecurrentTile(Sm120GdrRecurrentSharedStorage<BlockDv>& smem,
                                   int                                      stage,
                                   int                                      head_tile,
                                   int                                      dv_tile,
                                   int                                      loader_tid,
                                   int&                                     aux_tma_phase,
                                   int&                                     state_tma_phase,
                                   int&                                     state_free_phase,
                                   const CUtensorMap* __restrict__ state_tma_descs,
                                   const CUtensorMap& q_tma_desc,
                                   const CUtensorMap& k_tma_desc,
                                   const CUtensorMap& v_tma_desc,
                                   const float* __restrict__ g,
                                   const float* __restrict__ beta,
                                   const bool* __restrict__ finished,
                                   int     hq,
                                   int     hv,
                                   int64_t gate_batch_stride,
                                   int64_t out_batch_stride,
                                   int64_t out_head_stride,
                                   int     num_head_groups,
                                   int     heads_per_block,
                                   int     state_layer)
{
    constexpr int kValueBytes      = BlockDv * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int kQkBytes         = kHeadDim * static_cast<int>(sizeof(__nv_bfloat16));
    const int     batch            = DecodeHeadTileBatch(head_tile, hv);
    const int     value_head       = DecodeHeadTileHead(head_tile, hv);
    const int     dv0              = dv_tile * BlockDv;
    const int     qk_head          = value_head / (hv / hq);
    const int     gate_quad        = value_head / 4;
    const int     gate_lane        = value_head - gate_quad * 4;
    const int     head_group       = value_head / heads_per_block;
    const int     local_head       = value_head % heads_per_block;
    const int     state_desc_index = batch * num_head_groups + head_group;
    const int64_t value_offset =
        static_cast<int64_t>(batch) * out_batch_stride + static_cast<int64_t>(value_head) * out_head_stride + dv0;
    const int gate_offset       = static_cast<int>(static_cast<int64_t>(batch) * gate_batch_stride + gate_quad * 4);
    int       tile_finished_int = (loader_tid & 31) == 0 ? static_cast<int>(finished[batch]) : 0;
    tile_finished_int           = __shfl_sync(0xffffffff, tile_finished_int, 0);
    const bool tile_finished    = tile_finished_int != 0;

    const auto wait_state_slot = [&] {
        cute::wait_barrier(smem.stage_free[stage], state_free_phase);
        state_free_phase ^= 1;
    };

    wait_state_slot();
    if (loader_tid == 0) {
        smem.tile_finished[stage]   = static_cast<int>(tile_finished);
        smem.tile_batch[stage]      = batch;
        smem.tile_value_head[stage] = value_head;
        smem.tile_dv0[stage]        = dv0;
        smem.tile_out_offset[stage] = value_offset;
    }

    if (loader_tid == 0) {
        const int aux_bytes = kValueBytes + 2 * kQkBytes;
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.aux_tma_ready[stage], aux_bytes);
        cute::SM90_TMA_LOAD_4D::copy(
            &v_tma_desc, &smem.aux_tma_ready[stage], kTmaNoCacheHint, &smem.v_raw[stage][0], dv0, value_head, 0, batch);
        cute::SM90_TMA_LOAD_5D::copy(
            &q_tma_desc, &smem.aux_tma_ready[stage], kTmaNoCacheHint, &smem.q_raw[stage][0], 0, 0, qk_head, 0, batch);
        cute::SM90_TMA_LOAD_5D::copy(
            &k_tma_desc, &smem.aux_tma_ready[stage], kTmaNoCacheHint, &smem.k_raw[stage][0], 0, 0, qk_head, 0, batch);
        smem.gate_beta[stage][0] = exp2f(g[gate_offset + gate_lane] * kLog2e);
        smem.gate_beta[stage][1] = beta[gate_offset + gate_lane];
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.state_tma_ready[stage],
                                                                       RecurrentStateTmaBytes<StateT, BlockDv>());
        if (hv >= 32) {
            cute::prefetch_tma_descriptor(
                reinterpret_cast<const cute::TmaDescriptor*>(&state_tma_descs[state_desc_index]));
        }
        cute::SM90_TMA_LOAD_4D::copy(&state_tma_descs[state_desc_index],
                                     &smem.state_tma_ready[stage],
                                     hv >= 32 ? static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST) :
                                                kTmaNoCacheHint,
                                     reinterpret_cast<StateT*>(&smem.state[stage][0][0]),
                                     dv0,
                                     0,
                                     local_head,
                                     state_layer);
    }

    cute::wait_barrier(smem.aux_tma_ready[stage], aux_tma_phase);
    aux_tma_phase ^= 1;

    for (int dv = loader_tid; dv < BlockDv; dv += kRecurrentLoaderThreads) {
        smem.v[stage][dv] = __bfloat162float(smem.v_raw[stage][dv]);
    }
    for (int dk = loader_tid; dk < kHeadDim; dk += kRecurrentLoaderThreads) {
        smem.q[stage][dk] = __bfloat162float(smem.q_raw[stage][dk]) * kHeadScale;
        smem.k[stage][dk] = __bfloat162float(smem.k_raw[stage][dk]);
    }

    cute::wait_barrier(smem.state_tma_ready[stage], state_tma_phase);
    state_tma_phase ^= 1;
    __syncwarp();
    if ((loader_tid & 31) == 0) {
        cute::arrive_barrier(smem.tile_ready[stage]);
    }
}

template<int BlockDv, class StateT>
__device__ void ComputeRecurrentTile(Sm120GdrRecurrentSharedStorage<BlockDv>& smem, int stage, int role_tid)
{
    constexpr int kDvPerThreadPair = 2;
    constexpr int kDvPairsPerWarp  = BlockDv / kDvPerThreadPair;
    constexpr int kDkPerWarp       = kHeadDim / kRecurrentConsumerWarps;
    static_assert(kDvPairsPerWarp == 32);
    static_assert(kHeadDim % kRecurrentConsumerWarps == 0);

    const int   lane     = role_tid & 31;
    const int   warp     = role_tid >> 5;
    const int   dv_pair  = lane;
    const int   dv0      = dv_pair * kDvPerThreadPair;
    const int   dv1      = dv0 + 1;
    const int   dk_begin = warp * kDkPerWarp;
    const int   dk_end   = dk_begin + kDkPerWarp;
    const float decay    = smem.gate_beta[stage][0];
    const float beta     = smem.gate_beta[stage][1];

    float corr0 = 0.0f;
    float corr1 = 0.0f;
    float sq0   = 0.0f;
    float sq1   = 0.0f;
    float kq    = 0.0f;
    for (int dk = dk_begin; dk < dk_end; ++dk) {
        float2 state_vec;
        if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
            const auto* state_ptr = reinterpret_cast<const __nv_bfloat162*>(&smem.state[stage][0][0]);
            state_vec             = __bfloat1622float2(state_ptr[(dk * BlockDv + dv0) / 2]);
        }
        else {
            static_assert(std::is_same_v<StateT, float>);
            const auto* state_ptr = reinterpret_cast<const float2*>(&smem.state[stage][dk][dv0]);
            state_vec             = *state_ptr;
        }
        const float kk     = smem.k[stage][dk];
        const float qq     = smem.q[stage][dk];
        const float state0 = state_vec.x * decay;
        const float state1 = state_vec.y * decay;
        corr0 += state0 * kk;
        corr1 += state1 * kk;
        sq0 += state0 * qq;
        sq1 += state1 * qq;
        if (lane == 0) {
            kq += kk * qq;
        }
    }

    smem.partial_corr[stage][warp][dv_pair] = make_float2(corr0, corr1);
    smem.partial_sq[stage][warp][dv_pair]   = make_float2(sq0, sq1);
    if (lane == 0) {
        smem.partial_kq[stage][warp] = kq;
    }
    cutlass::arch::NamedBarrier::sync(kRecurrentConsumerThreads, kRecurrentBarrierPartial);

    float2 corr_sum = smem.partial_corr[stage][0][dv_pair];
#pragma unroll
    for (int producer_warp = 1; producer_warp < kRecurrentConsumerWarps; ++producer_warp) {
        const float2 corr_part = smem.partial_corr[stage][producer_warp][dv_pair];
        corr_sum.x += corr_part.x;
        corr_sum.y += corr_part.y;
    }
    const float delta0 = (smem.v[stage][dv0] - corr_sum.x) * beta;
    const float delta1 = (smem.v[stage][dv1] - corr_sum.y) * beta;

    if (warp == 0) {
        float2 sq_sum = smem.partial_sq[stage][0][dv_pair];
        float  kq_sum = smem.partial_kq[stage][0];
#pragma unroll
        for (int producer_warp = 1; producer_warp < kRecurrentConsumerWarps; ++producer_warp) {
            const float2 sq_part = smem.partial_sq[stage][producer_warp][dv_pair];
            sq_sum.x += sq_part.x;
            sq_sum.y += sq_part.y;
            kq_sum += smem.partial_kq[stage][producer_warp];
        }
        smem.out[stage][dv0] = __float2bfloat16(sq_sum.x + delta0 * kq_sum);
        smem.out[stage][dv1] = __float2bfloat16(sq_sum.y + delta1 * kq_sum);
    }

    for (int dk = dk_begin; dk < dk_end; ++dk) {
        const float kk = smem.k[stage][dk];
        if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
            auto*  state_ptr                    = reinterpret_cast<__nv_bfloat162*>(&smem.state[stage][0][0]);
            float2 state_vec                    = __bfloat1622float2(state_ptr[(dk * BlockDv + dv0) / 2]);
            state_vec.x                         = state_vec.x * decay + kk * delta0;
            state_vec.y                         = state_vec.y * decay + kk * delta1;
            state_ptr[(dk * BlockDv + dv0) / 2] = __float22bfloat162_rn(state_vec);
        }
        else {
            static_assert(std::is_same_v<StateT, float>);
            auto*  state_ptr = reinterpret_cast<float2*>(&smem.state[stage][dk][dv0]);
            float2 state_vec = *state_ptr;
            state_vec.x      = state_vec.x * decay + kk * delta0;
            state_vec.y      = state_vec.y * decay + kk * delta1;
            *state_ptr       = state_vec;
        }
    }
}

template<int BlockDv, class StateT>
struct Sm120GdrRecurrent {
    using SharedStorage = Sm120GdrRecurrentSharedStorage<BlockDv>;

    static constexpr int    kThreads     = kRecurrentThreads;
    static constexpr int    kMinBlocks   = 2;
    static constexpr size_t kSharedBytes = Sm120GdrRecurrentSharedBytes<BlockDv>();

    static __device__ __forceinline__ void Run(const CUtensorMap& q_tma_desc,
                                               const CUtensorMap& k_tma_desc,
                                               const CUtensorMap& v_tma_desc,
                                               __nv_bfloat16* __restrict__ out,
                                               const float* __restrict__ g,
                                               const float* __restrict__ beta,
                                               const bool* __restrict__ finished,
                                               const CUtensorMap* __restrict__ state_tma_descs,
                                               int            total_tiles,
                                               int            batch_count,
                                               int            hq,
                                               int            hv,
                                               int64_t        gate_batch_stride,
                                               int64_t        out_batch_stride,
                                               int64_t        out_head_stride,
                                               int            num_head_groups,
                                               int            heads_per_block,
                                               int            state_layer,
                                               unsigned char* smem_raw)
    {
        static_assert(BlockDv == kRecurrentBlockDv);
        auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

        const int tid    = static_cast<int>(threadIdx.x);
        const int wg_idx = cutlass::canonical_warp_group_idx();
        static_cast<void>(batch_count);

        if (tid == 0) {
            cute::prefetch_tma_descriptor(&q_tma_desc);
            cute::prefetch_tma_descriptor(&k_tma_desc);
            cute::prefetch_tma_descriptor(&v_tma_desc);
#pragma unroll
            for (int stage = 0; stage < kRecurrentStages; ++stage) {
                cute::initialize_barrier(smem.state_tma_ready[stage], 1);
                cute::initialize_barrier(smem.compute_done[stage], kRecurrentConsumerWarps);
                cute::initialize_barrier(smem.stage_free[stage], 1);
                cute::arrive_barrier(smem.stage_free[stage]);
                cute::initialize_barrier(smem.aux_tma_ready[stage], 1);
                cute::initialize_barrier(smem.tile_ready[stage], kRecurrentLoaderWarps);
            }
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == kRecurrentConsumerWarpGroups) {
            cutlass::arch::warpgroup_reg_dealloc<kRecurrentProducerRegs>();

            constexpr int kDvTiles         = kHeadDim / BlockDv;
            const int     total_work_tiles = total_tiles * kDvTiles;
            const int     first_work_tile  = static_cast<int>(blockIdx.x);
            const int     role_tid         = tid - kRecurrentConsumerThreads;

            if (role_tid < kRecurrentStoreThreads) {
                int done_phase[kRecurrentStages]{};
                int iter = 0;
                for (int work_tile = first_work_tile; work_tile < total_work_tiles;
                     work_tile += static_cast<int>(gridDim.x), ++iter) {
                    const int stage = iter % kRecurrentStages;
                    cute::wait_barrier(smem.compute_done[stage], done_phase[stage]);
                    done_phase[stage] ^= 1;
                    const bool    tile_finished    = smem.tile_finished[stage] != 0;
                    const int     dv0              = smem.tile_dv0[stage];
                    const int64_t out_offset       = smem.tile_out_offset[stage];
                    const int     batch            = smem.tile_batch[stage];
                    const int     value_head       = smem.tile_value_head[stage];
                    const int     head_group       = value_head / heads_per_block;
                    const int     local_head       = value_head % heads_per_block;
                    const int     state_desc_index = batch * num_head_groups + head_group;
                    if constexpr (std::is_same_v<StateT, float>) {
                        RecurrentPackStateTma<StateT, BlockDv>(smem.state[stage], role_tid, kRecurrentStoreThreads);
                    }
                    else {
                        static_assert(std::is_same_v<StateT, __nv_bfloat16>);
                    }
                    __syncwarp();
                    if (role_tid == 0 && !tile_finished) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_4D::copy(&state_tma_descs[state_desc_index],
                                                      reinterpret_cast<StateT*>(&smem.state[stage][0][0]),
                                                      dv0,
                                                      0,
                                                      local_head,
                                                      state_layer);
                        cute::tma_store_arrive();
                    }
                    reinterpret_cast<uint32_t*>(out + out_offset)[role_tid] =
                        reinterpret_cast<const uint32_t*>(&smem.out[stage][0])[role_tid];
                    __syncwarp();
                    if (role_tid == 0) {
                        if (!tile_finished) {
                            cute::tma_store_wait<0>();
                        }
                        cute::arrive_barrier(smem.stage_free[stage]);
                    }
                }
            }
            else if (role_tid >= kRecurrentStoreThreads) {
                const int loader_tid = role_tid - kRecurrentStoreThreads;
                int       state_free_phase[kRecurrentStages]{};
                int       aux_tma_phase[kRecurrentStages]{};
                int       state_tma_phase[kRecurrentStages]{};
                int       iter = 0;
                for (int work_tile = first_work_tile; work_tile < total_work_tiles;
                     work_tile += static_cast<int>(gridDim.x), ++iter) {
                    const int stage     = iter % kRecurrentStages;
                    const int head_tile = work_tile / kDvTiles;
                    const int dv_tile   = work_tile - head_tile * kDvTiles;
                    StageRecurrentTile<BlockDv, StateT>(smem,
                                                        stage,
                                                        head_tile,
                                                        dv_tile,
                                                        loader_tid,
                                                        aux_tma_phase[stage],
                                                        state_tma_phase[stage],
                                                        state_free_phase[stage],
                                                        state_tma_descs,
                                                        q_tma_desc,
                                                        k_tma_desc,
                                                        v_tma_desc,
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
                                                        state_layer);
                }
            }
            return;
        }
        else if (wg_idx < kRecurrentConsumerWarpGroups) {
            cutlass::arch::warpgroup_reg_alloc<kRecurrentConsumerRegs>();

            const int     role_tid = tid;
            int           tile_phase[kRecurrentStages]{};
            constexpr int kDvTiles         = kHeadDim / BlockDv;
            const int     total_work_tiles = total_tiles * kDvTiles;
            const int     first_work_tile  = static_cast<int>(blockIdx.x);
            int           iter             = 0;
            for (int work_tile = first_work_tile; work_tile < total_work_tiles;
                 work_tile += static_cast<int>(gridDim.x), ++iter) {
                const int stage = iter % kRecurrentStages;
                cute::wait_barrier(smem.tile_ready[stage], tile_phase[stage]);
                tile_phase[stage] ^= 1;
                ComputeRecurrentTile<BlockDv, StateT>(smem, stage, role_tid);
                __syncwarp();
                if ((role_tid & 31) == 0) {
                    cute::arrive_barrier(smem.compute_done[stage]);
                }
            }
            return;
        }
    }
};

template<int BlockDv, class StateT>
__global__ __launch_bounds__(
    Sm120GdrRecurrent<BlockDv, StateT>::kThreads,
    Sm120GdrRecurrent<BlockDv,
                      StateT>::kMinBlocks) void Sm120GdrRecurrentKernel(const __grid_constant__ CUtensorMap q_tma_desc,
                                                                        const __grid_constant__ CUtensorMap k_tma_desc,
                                                                        const __grid_constant__ CUtensorMap v_tma_desc,
                                                                        __nv_bfloat16* __restrict__ out,
                                                                        const float* __restrict__ g,
                                                                        const float* __restrict__ beta,
                                                                        const bool* __restrict__ finished,
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
                                                                        int     state_layer)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Sm120GdrRecurrent<BlockDv, StateT>::Run(q_tma_desc,
                                            k_tma_desc,
                                            v_tma_desc,
                                            out,
                                            g,
                                            beta,
                                            finished,
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
                                            state_layer,
                                            smem_raw);
}

template<int BlockDv, class StateT>
void SetRecurrentGdrSharedMemoryLimit()
{
    using Kernel                    = Sm120GdrRecurrent<BlockDv, StateT>;
    auto                     kernel = Sm120GdrRecurrentKernel<BlockDv, StateT>;
    static const cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Kernel::kSharedBytes));
    TM_CUDA_CHECK(status);
}

template<class StateT>
struct Sm120GroupedStateDescPrepare {
    static constexpr int kThreads   = 32;
    static constexpr int kMinBlocks = 1;

    static __device__ __forceinline__ void Run(const CUtensorMap& state_tma_desc,
                                               const int64_t*     addresses,
                                               int64_t            layer_group_stride,
                                               int64_t            sequence_stride,
                                               int64_t            head_group_stride,
                                               CUtensorMap*       descriptors,
                                               int                sequence_count,
                                               int                num_head_groups,
                                               CUtensorMap*       smem_descriptor)
    {
        const int     linear        = static_cast<int>(blockIdx.x);
        const int     lane          = static_cast<int>(threadIdx.x);
        const int     head_group    = linear % num_head_groups;
        const int     sequence      = (linear / num_head_groups) % sequence_count;
        const int     layer_group   = linear / (sequence_count * num_head_groups);
        const int64_t pointer_index = static_cast<int64_t>(layer_group) * layer_group_stride
                                      + static_cast<int64_t>(sequence) * sequence_stride
                                      + static_cast<int64_t>(head_group) * head_group_stride;
        CopyTmaDescriptor(smem_descriptor, &state_tma_desc, lane, kThreads);
        __syncwarp();
        if (lane == 0) {
            auto* state_base = reinterpret_cast<StateT*>(static_cast<uintptr_t>(addresses[pointer_index]));
            ReplaceTmaAddress(smem_descriptor, state_base);
        }
        __syncwarp();
        PublishTmaDescriptor(&descriptors[linear], smem_descriptor);
        __syncwarp();
        cute::tma_descriptor_fence_acquire(reinterpret_cast<cute::TmaDescriptor*>(&descriptors[linear]));
    }
};

template<class StateT>
__global__ __launch_bounds__(
    Sm120GroupedStateDescPrepare<StateT>::kThreads,
    Sm120GroupedStateDescPrepare<StateT>::kMinBlocks) void PrepareGroupedStateDescriptors(const __grid_constant__
                                                                                              CUtensorMap
                                                                                                         state_tma_desc,
                                                                                          const int64_t* addresses,
                                                                                          int64_t layer_group_stride,
                                                                                          int64_t sequence_stride,
                                                                                          int64_t head_group_stride,
                                                                                          CUtensorMap* descriptors,
                                                                                          int          sequence_count,
                                                                                          int          num_head_groups)
{
    __shared__ __align__(128) CUtensorMap smem_descriptor;
    Sm120GroupedStateDescPrepare<StateT>::Run(state_tma_desc,
                                              addresses,
                                              layer_group_stride,
                                              sequence_stride,
                                              head_group_stride,
                                              descriptors,
                                              sequence_count,
                                              num_head_groups,
                                              &smem_descriptor);
}

template<class StateT>
void PrepareSm120RecurrentStateTmaDescriptorsTyped(const core::Tensor& state_ptrs,
                                                   core::Tensor&       state_tma_descs,
                                                   int                 layer_groups,
                                                   int                 layers_per_block,
                                                   int                 sequence_count,
                                                   int                 num_head_groups,
                                                   int                 heads_per_block,
                                                   int                 block_dv,
                                                   cudaStream_t        stream)
{
    using Kernel = Sm120GroupedStateDescPrepare<StateT>;

    const auto state_tma_desc = MakeRecurrentStateTmaDesc(
        reinterpret_cast<StateT*>(state_tma_descs.raw_data()), layers_per_block, heads_per_block, block_dv);
    const auto* addresses   = reinterpret_cast<const int64_t*>(state_ptrs.raw_data());
    auto*       descriptors = reinterpret_cast<CUtensorMap*>(state_tma_descs.raw_data());
    const int   work        = layer_groups * sequence_count * num_head_groups;
    PrepareGroupedStateDescriptors<StateT><<<work, Kernel::kThreads, 0, stream>>>(state_tma_desc,
                                                                                  addresses,
                                                                                  state_ptrs.stride(0),
                                                                                  state_ptrs.stride(1),
                                                                                  state_ptrs.stride(2),
                                                                                  descriptors,
                                                                                  sequence_count,
                                                                                  num_head_groups);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace {

template<class StateT>
void LaunchSm120GdrRecurrentTyped(const core::Tensor& q,
                                  const core::Tensor& k,
                                  const core::Tensor& v,
                                  const core::Tensor& g,
                                  const core::Tensor& beta,
                                  const core::Tensor& finished,
                                  const core::Tensor& state_tma_descs,
                                  core::Tensor&       out,
                                  const Problem&      problem,
                                  int64_t             state_layer_offset,
                                  void*               tma_desc_workspace,
                                  cudaStream_t        stream)
{
    const auto* g_ptr        = g.data<float>();
    const auto* beta_ptr     = beta.data<float>();
    auto*       out_ptr      = reinterpret_cast<__nv_bfloat16*>(out.raw_data());
    const auto* finished_ptr = finished.data<bool>();
    const int   state_layer =
        static_cast<int>(state_layer_offset / (static_cast<int64_t>(problem.heads_per_block) * kHeadDim * kHeadDim));
    const auto* state_tma_descs_ptr = reinterpret_cast<const CUtensorMap*>(state_tma_descs.raw_data());
    static_cast<void>(tma_desc_workspace);

    constexpr int block_dv         = kRecurrentBlockDv;
    const int     total_tiles      = problem.batch * problem.hv;
    constexpr int dv_tiles         = kHeadDim / block_dv;
    const int     sm_count         = problem.sm_count;
    const int     base_grid_blocks = sm_count * kRecurrentBaseCtasPerSm;
    const int     total_work_tiles = total_tiles * dv_tiles;
    // Medium persistent loops benefit from an extra queued CTA wave; smaller loops pay launch-tail overhead,
    // while larger loops already amortize per-CTA setup with the resident two-CTA grid.
    const bool medium_persistent_grid = total_work_tiles > base_grid_blocks * kRecurrentMediumMinWorkPerBaseCta
                                        && total_work_tiles <= base_grid_blocks * kRecurrentMediumMaxWorkPerBaseCta;
    const int target_grid_blocks =
        sm_count * (medium_persistent_grid ? kRecurrentMediumCtasPerSm : kRecurrentBaseCtasPerSm);
    const int grid_blocks =
        std::max(1, std::min(std::min(total_work_tiles, target_grid_blocks), kRecurrentMaxDescriptorCtas));
    using Kernel = Sm120GdrRecurrent<block_dv, StateT>;

    const auto q_tma_desc = MakeRecurrentQkTmaDesc<__nv_bfloat16>(q);
    const auto k_tma_desc = MakeRecurrentQkTmaDesc<__nv_bfloat16>(k);
    const auto v_tma_desc = MakeRecurrentValueTmaDesc<__nv_bfloat16>(v, block_dv);
    SetRecurrentGdrSharedMemoryLimit<block_dv, StateT>();
    Sm120GdrRecurrentKernel<block_dv, StateT>
        <<<grid_blocks, Kernel::kThreads, Kernel::kSharedBytes, stream>>>(q_tma_desc,
                                                                          k_tma_desc,
                                                                          v_tma_desc,
                                                                          out_ptr,
                                                                          g_ptr,
                                                                          beta_ptr,
                                                                          finished_ptr,
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
                                                                          state_layer);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace detail {

void LaunchSm120Recurrent(const core::Tensor& q,
                          const core::Tensor& k,
                          const core::Tensor& v,
                          const core::Tensor& g,
                          const core::Tensor& beta,
                          const core::Tensor& finished,
                          const core::Tensor& state_tma_descs,
                          core::Tensor&       out,
                          const Problem&      problem,
                          int64_t             state_layer_offset,
                          DataType            state_dtype,
                          cudaStream_t        stream)
{
    if (state_dtype == kFloat32) {
        LaunchSm120GdrRecurrentTyped<float>(
            q, k, v, g, beta, finished, state_tma_descs, out, problem, state_layer_offset, nullptr, stream);
    }
    else {
        LaunchSm120GdrRecurrentTyped<__nv_bfloat16>(
            q, k, v, g, beta, finished, state_tma_descs, out, problem, state_layer_offset, nullptr, stream);
    }
}

void PrepareSm120RecurrentStateTmaDescriptors(const core::Tensor& state_ptrs,
                                              core::Tensor&       state_tma_descs,
                                              int                 layer_groups,
                                              int                 layers_per_block,
                                              const Plan&         plan,
                                              cudaStream_t        stream)
{
    if (plan.problem.state_dtype == kFloat32) {
        PrepareSm120RecurrentStateTmaDescriptorsTyped<float>(state_ptrs,
                                                             state_tma_descs,
                                                             layer_groups,
                                                             layers_per_block,
                                                             plan.problem.sequence_num,
                                                             plan.problem.num_head_groups,
                                                             plan.problem.heads_per_block,
                                                             kRecurrentBlockDv,
                                                             stream);
    }
    else {
        PrepareSm120RecurrentStateTmaDescriptorsTyped<__nv_bfloat16>(state_ptrs,
                                                                     state_tma_descs,
                                                                     layer_groups,
                                                                     layers_per_block,
                                                                     plan.problem.sequence_num,
                                                                     plan.problem.num_head_groups,
                                                                     plan.problem.heads_per_block,
                                                                     kRecurrentBlockDv,
                                                                     stream);
    }
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
