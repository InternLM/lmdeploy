// Inspired by https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/underscore.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kChunkSize       = 32;
constexpr int kHeadDim         = 128;
constexpr int kSharedAlign     = 1024;
constexpr int kConsumerRegisters       = 128;
constexpr int kKktSolveRoleThreads     = 128;
constexpr int kKktSolveConsumerWgs     = 1;
constexpr int kTmaDescriptorBytes      = 128;

constexpr int kBlock16       = 16;
// Rows per 16x16 fp32 tile including one padding row. Actual rows stay contiguous;
// the extra row shifts successive tiles away from bank-aligned starts.
constexpr int kBlock16Stride = 17;

constexpr int kKTileSwizzleDim = 64;
enum : int {
    kKTileTmaDim = kKTileSwizzleDim,
};
constexpr int kKTilePlaneElems  = kChunkSize * kKTileSwizzleDim;
constexpr int kA16iElems        = 2 * kBlock16Stride * kBlock16;
constexpr int kA16oElems        = kBlock16Stride * kBlock16;
constexpr int kOutputTileElems  = kChunkSize * kChunkSize;
constexpr uint64_t kTmaNoCacheHint = 0;

enum KktTmaDescIndex : int {
    kKktKDesc = 0,
    kKktBetaDesc,
    kKktResolventDesc,
    kKktTmaDescCount,
};

static_assert(kBlock16 == 16);
static_assert(kHeadDim == 2 * kKTileTmaDim);
static_assert(kChunkSize == 2 * kBlock16);
static_assert(sizeof(CUtensorMap) == kTmaDescriptorBytes);

template<class K>
struct KktMmaTraits;

template<>
struct KktMmaTraits<__nv_bfloat16> {
    using Element = cute::bfloat16_t;
    using Atom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

template<class K>
struct __align__(16) KktSolveScratch {
    __align__(16) float a16i[kA16iElems];
    __align__(16) float neg_l10[kA16oElems];
};

template<class K>
struct __align__(1024) KktSolveSharedStorage {
    using MmaElement = typename KktMmaTraits<K>::Element;

    __align__(1024) MmaElement k_tile[2 * kKTilePlaneElems];
    __align__(1024) MmaElement out_tile[kOutputTileElems];
    __align__(16) KktSolveScratch<K> scratch[kKktSolveConsumerWgs];
    __align__(128) float beta_stage[kChunkSize][4];
    __align__(8) uint64_t k_ready0;
    __align__(8) uint64_t k_ready1;
    __align__(8) uint64_t beta_ready;
};

template<class K, int ConsumerThreads>
constexpr size_t KktSolveSharedBytes()
{
    static_assert(ConsumerThreads >= kChunkSize);
    static_assert(ConsumerThreads % 32 == 0);
    static_assert(alignof(KktSolveSharedStorage<K>) == kSharedAlign);
    return sizeof(KktSolveSharedStorage<K>);
}

__device__ __forceinline__ int KktCeilDiv(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

__device__ __forceinline__ void KktAcquireAndPrefetchTmaDescriptors(const CUtensorMap* desc, int tid)
{
    if (tid < 32) {
        for (int idx = tid; idx < kKktTmaDescCount; idx += 32) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&desc[idx]));
            cute::prefetch_tma_descriptor(&desc[idx]);
        }
    }
}

template<int ConsumerThreads, int BarrierId>
__device__ __forceinline__ void ConsumerWgSync()
{
    static_assert(ConsumerThreads % 32 == 0);
    static_assert(BarrierId >= 0);
    static_assert(BarrierId < kKktSolveConsumerWgs);
    static_assert(BarrierId + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);
    cutlass::arch::NamedBarrier::sync(ConsumerThreads, BarrierId);
}

template<int Tiles>
CUTE_HOST_DEVICE constexpr auto KktBlock16Layout()
{
    return cute::make_layout(cute::make_shape(cute::Int<Tiles>{}, cute::Int<kBlock16Stride>{}, cute::Int<kBlock16>{}),
                             cute::make_stride(cute::Int<kBlock16Stride * kBlock16>{},
                                               cute::Int<kBlock16>{},
                                               cute::Int<1>{}));
}

CUTE_HOST_DEVICE constexpr auto KktKTileLayout()
{
    return cute::composition(cute::Swizzle<3, 3, 3>{},
                             cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Int<kKTileSwizzleDim>>,
                                          cute::Stride<cute::Int<kKTileSwizzleDim>, cute::_1>>{});
}

CUTE_HOST_DEVICE constexpr auto KktOutputTileLayout()
{
    return cute::composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<cute::Shape<cute::_32, cute::_32>,
                                          cute::Stride<cute::_32, cute::_1>>{});
}

CUTE_HOST_DEVICE constexpr auto KktDiagSolveThreadLayout()
{
    return cute::make_layout(cute::Shape<cute::Int<2>, cute::Int<kBlock16>>{},
                             cute::make_stride(cute::Int<kBlock16>{}, cute::Int<1>{}));
}

CUTE_HOST_DEVICE constexpr auto KktOffdiagVec4ThreadLayout()
{
    return cute::make_layout(cute::Shape<cute::Int<kBlock16>, cute::Int<4>>{},
                             cute::make_stride(cute::Int<4>{}, cute::Int<1>{}));
}

template<class Element>
__device__ __forceinline__ void StoreBf16(Element* ptr, float value)
{
    static_assert(sizeof(Element) == sizeof(__nv_bfloat16));
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16_rn(value);
}

template<class K, int ConsumerThreads, int ConsumerRegisters>
__global__ void __launch_bounds__(kKktSolveConsumerWgs * ConsumerThreads, 1)
    Sm120KktSolveKernel(const int32_t* __restrict__ q_offsets,
                   const bool* __restrict__ finished,
                   CUtensorMap* tma_desc_workspace,
                   int total_tokens,
                   int sequence_num,
                   int hq,
                   int hv,
                   int64_t gate_batch_stride,
                   int groups_per_k_head)
{
    static_assert(ConsumerThreads == kKktSolveRoleThreads,
                  "KKT solve MMA and repack layouts currently require 128 consumer threads");
    static_assert(ConsumerThreads >= kChunkSize);
    static_assert(ConsumerThreads % 32 == 0);

    const int tx                = static_cast<int>(threadIdx.x);
    const int qk_head           = static_cast<int>(blockIdx.x);
    int       local_chunk_id    = static_cast<int>(blockIdx.y);
    constexpr int batch_id      = 0;
    const int value_head_base   = qk_head * groups_per_k_head;
    const int wg_idx            = cutlass::canonical_warp_group_idx();
    const int role_tid          = tx % ConsumerThreads;

    int sequence_id = -1;
    for (int b = 0; b < sequence_num; ++b) {
        const int cur_start  = q_offsets[b];
        const int cur_end    = q_offsets[b + 1];
        const int cur_chunks = KktCeilDiv(cur_end - cur_start, kChunkSize);
        if (local_chunk_id < cur_chunks) {
            sequence_id = b;
            break;
        }
        local_chunk_id -= cur_chunks;
    }
    if (sequence_id < 0) {
        return;
    }
    const int token0 = local_chunk_id * kChunkSize;

    extern __shared__ __align__(1024) unsigned char shared_raw[];
    auto& smem = *reinterpret_cast<KktSolveSharedStorage<K>*>(shared_raw);
    uint64_t* k_ready0 = &smem.k_ready0;
    uint64_t* k_ready1 = &smem.k_ready1;
    uint64_t* beta_ready = &smem.beta_ready;

    using MmaElement = typename KktMmaTraits<K>::Element;
    MmaElement* k_tile0 = smem.k_tile;
    MmaElement* k_tile1 = smem.k_tile + kKTilePlaneElems;
    MmaElement* out_tile = smem.out_tile;
    float*      beta_stage = &smem.beta_stage[0][0];
    const auto* gmem_desc = tma_desc_workspace + sequence_id * kKktTmaDescCount;
    KktAcquireAndPrefetchTmaDescriptors(gmem_desc, tx);
    const CUtensorMap* k_desc = &gmem_desc[kKktKDesc];
    const CUtensorMap* beta_desc = &gmem_desc[kKktBetaDesc];
    const CUtensorMap* resolvent_desc = &gmem_desc[kKktResolventDesc];

    if (tx == 0) {
        cute::initialize_barrier(*k_ready0, 1);
        cute::initialize_barrier(*k_ready1, 1);
        cute::initialize_barrier(*beta_ready, 1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (wg_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();

        constexpr int ConsumerWg = 0;
        static_assert(ConsumerWg < kKktSolveConsumerWgs);

        if (ConsumerWg < groups_per_k_head) {
            auto& scratch = smem.scratch[ConsumerWg];
            auto  s_k0    = cute::make_tensor(cute::make_smem_ptr(k_tile0), KktKTileLayout());
            auto  s_k1    = cute::make_tensor(cute::make_smem_ptr(k_tile1), KktKTileLayout());
            auto  s_a16i  = cute::make_tensor(cute::make_smem_ptr(scratch.a16i), KktBlock16Layout<2>());
            auto  s_neg_l10 = cute::make_tensor(cute::make_smem_ptr(scratch.neg_l10), KktBlock16Layout<1>());
            auto  s_out   = cute::make_tensor(cute::make_smem_ptr(out_tile), KktOutputTileLayout());

            float gram_fragment[32];

            using MmaAtom = typename KktMmaTraits<K>::Atom;
            using Mma32 = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                         cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                         cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma32 mma32;
            auto  t_gram_fragment =
                cute::make_tensor(cute::make_rmem_ptr(gram_fragment),
                                  cute::partition_shape_C(mma32, cute::Shape<cute::_32, cute::_32>{}));
            auto  c_a32       = cute::make_identity_tensor(cute::Shape<cute::_32, cute::_32>{});
            auto  t_a32_coord = mma32.get_thread_slice(role_tid).partition_C(c_a32);

            constexpr int k_tma_box_rows = kChunkSize;
            const int     first_beta_quad = value_head_base / 4;
            const int     first_beta_coord = first_beta_quad * 4;
            if (role_tid == 0) {
                cute::set_barrier_transaction_bytes(*k_ready0, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                cute::SM90_TMA_LOAD_5D::copy(
                    k_desc, k_ready0, kTmaNoCacheHint, k_tile0, 0, 0, qk_head, token0, 0);
                cute::set_barrier_transaction_bytes(*k_ready1, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                cute::SM90_TMA_LOAD_5D::copy(
                    k_desc, k_ready1, kTmaNoCacheHint, k_tile1, 0, 1, qk_head, token0, 0);
                cute::set_barrier_transaction_bytes(*beta_ready, k_tma_box_rows * 4 * static_cast<int>(sizeof(float)));
                cute::SM90_TMA_LOAD_3D::copy(
                    beta_desc, beta_ready, kTmaNoCacheHint, beta_stage, first_beta_coord, token0, 0);
            }
            cute::wait_barrier(*k_ready0, 0);

            using namespace cute;
            clear(t_gram_fragment);
            cute::cooperative_gemm(role_tid,
                                   mma32,
                                   s_k0,
                                   s_k0,
                                   t_gram_fragment,
                                   identity{},
                                   identity{},
                                   SM75_U32x4_LDSM_N{},
                                   SM75_U32x4_LDSM_N{});
            ConsumerWgSync<ConsumerThreads, ConsumerWg>();
            cute::wait_barrier(*k_ready1, 0);
            cute::cooperative_gemm(role_tid,
                                   mma32,
                                   s_k1,
                                   s_k1,
                                   t_gram_fragment,
                                   identity{},
                                   identity{},
                                   SM75_U32x4_LDSM_N{},
                                   SM75_U32x4_LDSM_N{});

            int beta_phase        = 0;
            int loaded_beta_quad  = -1;
            int pending_beta_quad = first_beta_quad;
            for (int group = ConsumerWg; group < groups_per_k_head; group += kKktSolveConsumerWgs) {
                const int value_head = value_head_base + group;
                const int beta_quad  = value_head / 4;
                const int beta_coord = beta_quad * 4;
                if (beta_quad != loaded_beta_quad) {
                    if (beta_quad != pending_beta_quad) {
                        if (role_tid == 0) {
                            cute::set_barrier_transaction_bytes(
                                *beta_ready, k_tma_box_rows * 4 * static_cast<int>(sizeof(float)));
                            cute::SM90_TMA_LOAD_3D::copy(
                                beta_desc, beta_ready, kTmaNoCacheHint, beta_stage, beta_coord, token0, 0);
                        }
                        pending_beta_quad = beta_quad;
                    }
                    cute::wait_barrier(*beta_ready, beta_phase);
                    beta_phase ^= 1;
                    loaded_beta_quad = beta_quad;
                }
                const int beta_lane = value_head & 3;
                if (group > 0) {
                    if (role_tid == 0) {
                        cute::tma_store_wait<0>();
                    }
                    ConsumerWgSync<ConsumerThreads, ConsumerWg>();
                }

#pragma unroll
                for (int idx = 0; idx < cute::size(t_gram_fragment); ++idx) {
                    const auto coord = t_a32_coord(idx);
                    const int  row   = cute::get<0>(coord);
                    const int  col   = cute::get<1>(coord);

                    float value = t_gram_fragment(idx) * beta_stage[row * 4 + beta_lane];

                    if (row < col) {
                        value = 0.0f;
                    }
                    else if (row == col) {
                        value = 1.0f;
                    }

                    const int row_block = row / kBlock16;
                    const int col_block = col / kBlock16;
                    if (row_block == col_block) {
                        s_a16i(row_block, row % kBlock16, col % kBlock16) = value;
                    }
                    else if (row_block == 1 && col_block == 0) {
                        s_neg_l10(0, row - kBlock16, col) = -value;
                    }
                }
                ConsumerWgSync<ConsumerThreads, ConsumerWg>();

                if (role_tid < kChunkSize) {
                    const auto solve_coord = KktDiagSolveThreadLayout().get_hier_coord(role_tid);
                    const int  tile_id     = cute::get<0>(solve_coord);
                    const int  col         = cute::get<1>(solve_coord);

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
                    for (int row = 0; row < kBlock16; ++row) {
                        if (col <= row) {
                            s_a16i(tile_id, row, col) = inv_col[row];
                        }
                    }
                }
                ConsumerWgSync<ConsumerThreads, ConsumerWg>();

                // The chunk32 triangular inverse has two 16x16 diagonal blocks. Store inv(L00) and inv(L11)
                // directly, then form the lower-left block as inv(L11) * (-L10) * inv(L00).
                if (role_tid < 64) {
                    const auto copy_coord = KktOffdiagVec4ThreadLayout().get_hier_coord(role_tid);
                    const int  row        = cute::get<0>(copy_coord);
                    const int  col        = cute::get<1>(copy_coord) * 4;

                    const float4 upper = *reinterpret_cast<const float4*>(&s_a16i(0, row, col));
                    const float4 lower = *reinterpret_cast<const float4*>(&s_a16i(1, row, col));
                    StoreBf16(&s_out(row, col + 0), upper.x);
                    StoreBf16(&s_out(row, col + 1), upper.y);
                    StoreBf16(&s_out(row, col + 2), upper.z);
                    StoreBf16(&s_out(row, col + 3), upper.w);
                    StoreBf16(&s_out(kBlock16 + row, kBlock16 + col + 0), lower.x);
                    StoreBf16(&s_out(kBlock16 + row, kBlock16 + col + 1), lower.y);
                    StoreBf16(&s_out(kBlock16 + row, kBlock16 + col + 2), lower.z);
                    StoreBf16(&s_out(kBlock16 + row, kBlock16 + col + 3), lower.w);
                    StoreBf16(&s_out(row, kBlock16 + col + 0), 0.0f);
                    StoreBf16(&s_out(row, kBlock16 + col + 1), 0.0f);
                    StoreBf16(&s_out(row, kBlock16 + col + 2), 0.0f);
                    StoreBf16(&s_out(row, kBlock16 + col + 3), 0.0f);

                    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
                    for (int n = 0; n < kBlock16; ++n) {
                        float mid = 0.0f;
#pragma unroll
                        for (int m = 0; m < kBlock16; ++m) {
                            mid += s_a16i(1, row, m) * s_neg_l10(0, m, n);
                        }
                        const float4 rhs = *reinterpret_cast<const float4*>(&s_a16i(0, n, col));
                        acc.x += mid * rhs.x;
                        acc.y += mid * rhs.y;
                        acc.z += mid * rhs.z;
                        acc.w += mid * rhs.w;
                    }
                    StoreBf16(&s_out(kBlock16 + row, col + 0), acc.x);
                    StoreBf16(&s_out(kBlock16 + row, col + 1), acc.y);
                    StoreBf16(&s_out(kBlock16 + row, col + 2), acc.z);
                    StoreBf16(&s_out(kBlock16 + row, col + 3), acc.w);
                }
                ConsumerWgSync<ConsumerThreads, ConsumerWg>();

                cute::tma_store_fence();
                ConsumerWgSync<ConsumerThreads, ConsumerWg>();
                if (role_tid == 0) {
                    cute::SM90_TMA_STORE_4D::copy(
                        resolvent_desc, out_tile, 0, value_head, token0, 0);
                    cute::tma_store_arrive();
                }
                if (group + kKktSolveConsumerWgs < groups_per_k_head) {
                    ConsumerWgSync<ConsumerThreads, ConsumerWg>();
                }
            }
            static_cast<void>(finished);
            static_cast<void>(total_tokens);
            static_cast<void>(hq);
            static_cast<void>(hv);
            static_cast<void>(gate_batch_stride);
            static_cast<void>(batch_id);
            if (role_tid == 0) {
                cute::tma_store_wait<0>();
            }
        }
    }
}

template<class K, int ConsumerThreads = kKktSolveRoleThreads, int ConsumerRegisters = kConsumerRegisters>
void LaunchKktSolveTyped(const float* g_cumsum_ptr,
                         const core::Tensor& q_offsets,
                         const core::Tensor& finished,
                         void* tma_desc_workspace,
                         const Problem& problem,
                         cudaStream_t stream)
{
    if (problem.total_chunks == 0) {
        return;
    }

    const int groups_per_k_head = problem.hv / problem.hq;
    const dim3 grid(problem.hq, problem.total_chunks, 1);
    const size_t shared_bytes = KktSolveSharedBytes<K, ConsumerThreads>();

    static const cudaError_t smem_attribute_status =
        cudaFuncSetAttribute(Sm120KktSolveKernel<K, ConsumerThreads, ConsumerRegisters>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes));
    TM_CUDA_CHECK(smem_attribute_status);

    const int32_t* offsets_ptr = q_offsets.data<int32_t>();
    const bool*    finished_ptr = finished.data<bool>();
    auto* desc_workspace = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);
    Sm120KktSolveKernel<K, ConsumerThreads, ConsumerRegisters>
        <<<grid, kKktSolveConsumerWgs * ConsumerThreads, shared_bytes, stream>>>(
        offsets_ptr,
        finished_ptr,
        desc_workspace,
        problem.token_num,
        problem.sequence_num,
        problem.hq,
        problem.hv,
        problem.gate_batch_stride,
        groups_per_k_head);
    TM_CUDA_CHECK(cudaGetLastError());
}

void LaunchSm120KktSolveImpl(const core::Tensor&,
                             const core::Tensor&,
                             const core::Tensor& q_offsets,
                             const core::Tensor* g_cumsum,
                             const core::Tensor& finished,
                             core::Tensor&,
                             const Problem& problem,
                             void* tma_desc_workspace,
                             cudaStream_t stream)
{
    const auto* g_cumsum_ptr = g_cumsum->data<float>();
    LaunchKktSolveTyped<__nv_bfloat16>(g_cumsum_ptr, q_offsets, finished, tma_desc_workspace, problem, stream);
}

}  // namespace

namespace detail {

void LaunchSm120KktSolve(const core::Tensor& k,
                         const core::Tensor& beta,
                         const core::Tensor& q_offsets,
                         const core::Tensor* g_cumsum,
                         const core::Tensor& finished,
                         core::Tensor& resolvent,
                         const Problem& problem,
                         void* tma_desc_workspace,
                         cudaStream_t stream)
{
    LaunchSm120KktSolveImpl(
        k, beta, q_offsets, g_cumsum, finished, resolvent, problem, tma_desc_workspace, stream);
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
