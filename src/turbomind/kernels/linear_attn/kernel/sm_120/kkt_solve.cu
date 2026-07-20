// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/tma_desc_prepare.h"

#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class K, int GroupsPerKHead>
struct Sm120KktSolve {
    static constexpr int kChunkSize             = 32;
    static constexpr int kKktHeadDim            = 128;
    static constexpr int kSharedAlign           = 1024;
    static constexpr int kConsumerThreads       = 128;
    static constexpr int kConsumerRegisters     = 96;
    static constexpr int kKktSolveConsumerWgs   = 1;
    static constexpr int kKktTmaDescriptorBytes = 128;

    static constexpr int kBlock16 = 16;
// Rows per 16x16 fp32 tile including one padding row. Actual rows stay contiguous;
// the extra row shifts successive tiles away from bank-aligned starts.
    static constexpr int kBlock16Stride = 17;

    static constexpr int kKTileSwizzleDim = 64;
    enum : int
    {
        kKTileTmaDim = kKTileSwizzleDim,
    };
    static constexpr int      kKTilePlaneElems  = kChunkSize * kKTileSwizzleDim;
    static constexpr int      kA16iElems        = 2 * kBlock16Stride * kBlock16;
    static constexpr int      kA16oElems        = kBlock16Stride * kBlock16;
    static constexpr int      kOutputTileElems  = kChunkSize * kChunkSize;
    static constexpr uint64_t kKktTmaNoCacheHint = 0;

    enum KktTmaDescIndex : int
    {
        kKktKDesc = 0,
        kKktResolventDesc,
        kLocalKktTmaDescCount,
    };

    static_assert(kBlock16 == 16);
    static_assert(kKktHeadDim == 2 * kKTileTmaDim);
    static_assert(kChunkSize == 2 * kBlock16);
    static_assert(sizeof(CUtensorMap) == kKktTmaDescriptorBytes);
    static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT solve only supports bfloat16 inputs");
    static_assert(GroupsPerKHead == 0 || (GroupsPerKHead >= 1 && GroupsPerKHead <= 4));
    static_assert(kConsumerThreads >= kChunkSize);
    static_assert(kConsumerThreads % 32 == 0);

    using MmaElement = cute::bfloat16_t;
    using MmaAtom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;

    struct __align__(16) Scratch
    {
        __align__(16) float a16i[kA16iElems];
        __align__(16) float neg_l10[kA16oElems];
    };

    struct __align__(1024) SharedStorage
    {
        __align__(1024) MmaElement k_tile[2 * kKTilePlaneElems];
        __align__(1024) MmaElement out_tile[kOutputTileElems];
        __align__(16) Scratch       scratch[kKktSolveConsumerWgs];
        __align__(128) float        beta_stage[kChunkSize][4];
        __align__(8) uint64_t       k_ready0;
        __align__(8) uint64_t       k_ready1;
    };

    static constexpr size_t SharedBytes()
    {
        static_assert(alignof(SharedStorage) == kSharedAlign);
        return sizeof(SharedStorage);
    }

    static __device__ __forceinline__ int CeilDiv(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    static __device__ __forceinline__ void AcquireAndPrefetchTmaDescriptors(const CUtensorMap* desc, int tid)
    {
        if (tid < 32) {
            for (int idx = tid; idx < kLocalKktTmaDescCount; idx += 32) {
                cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&desc[idx]));
                cute::prefetch_tma_descriptor(&desc[idx]);
            }
        }
    }

    template<int BarrierId>
    static __device__ __forceinline__ void ConsumerWgSync()
    {
        static_assert(BarrierId == 0);
        asm volatile("bar.sync 0, %0;" : : "r"(kConsumerThreads) : "memory");
    }

    static __device__ __forceinline__ void OffdiagMmaSync()
    {
        constexpr int kMmaThreads = kConsumerThreads / 2;
        static_assert(kConsumerThreads == 2 * kMmaThreads);
        asm volatile("bar.sync 1, %0;" : : "r"(kMmaThreads) : "memory");
    }

    template<int Tiles>
    static CUTE_HOST_DEVICE constexpr auto Block16Layout()
    {
        return cute::make_layout(
            cute::make_shape(cute::Int<Tiles>{}, cute::Int<kBlock16Stride>{}, cute::Int<kBlock16>{}),
            cute::make_stride(cute::Int<kBlock16Stride * kBlock16>{}, cute::Int<kBlock16>{}, cute::Int<1>{}));
    }

    static CUTE_HOST_DEVICE constexpr auto KTileLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Int<kKTileSwizzleDim>>,
                                              cute::Stride<cute::Int<kKTileSwizzleDim>, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto OutputTileLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto DiagSolveThreadLayout()
    {
        return cute::make_layout(cute::Shape<cute::Int<2>, cute::Int<kBlock16>>{},
                                 cute::make_stride(cute::Int<kBlock16>{}, cute::Int<1>{}));
    }

    static CUTE_HOST_DEVICE constexpr auto OffdiagVec4ThreadLayout()
    {
        return cute::make_layout(cute::Shape<cute::Int<kBlock16>, cute::Int<4>>{},
                                 cute::make_stride(cute::Int<4>{}, cute::Int<1>{}));
    }

    template<class Element>
    static __device__ __forceinline__ void StoreBf16(Element* ptr, float value)
    {
        static_assert(sizeof(Element) == sizeof(__nv_bfloat16));
        *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16_rn(value);
    }

    struct Tf32LoadTransform {
        template<class T>
        __device__ __forceinline__ cute::tfloat32_t operator()(T const& value) const
        {
            return cute::tfloat32_t::bitcast(__float_as_uint(static_cast<float>(value)));
        }
    };

    static constexpr int    kThreads     = kKktSolveConsumerWgs * kConsumerThreads;
    static constexpr int    kMinBlocks   = 1;
    static constexpr size_t kSharedBytes = SharedBytes();

    static __device__ __forceinline__ void Run(const int32_t* __restrict__ q_offsets,
                                               const bool* __restrict__ finished,
                                               const float* __restrict__ beta,
                                               CUtensorMap* tma_desc_workspace,
                                               int          token_num,
                                               int          sequence_num,
                                               int          hq,
                                               int          hv,
                                               int64_t      beta_stride,
                                               int64_t      beta_batch_stride,
                                               int          groups_per_k_head,
                                               unsigned char* shared_raw)
    {
        const int     tx              = static_cast<int>(threadIdx.x);
        const int     qk_head         = static_cast<int>(blockIdx.x);
        int           local_chunk_id  = static_cast<int>(blockIdx.y);
        constexpr int batch_id        = 0;
        const int     group_count     = GroupsPerKHead == 0 ? groups_per_k_head : GroupsPerKHead;
        const int     value_head_base = qk_head * group_count;
        const int     wg_idx          = cutlass::canonical_warp_group_idx();
        const int     role_tid        = tx % kConsumerThreads;

        int sequence_id = -1;
        for (int b = 0; b < sequence_num; ++b) {
            const int cur_start  = q_offsets[b];
            const int cur_end    = q_offsets[b + 1];
            const int cur_chunks = CeilDiv(cur_end - cur_start, kChunkSize);
            if (local_chunk_id < cur_chunks) {
                sequence_id = b;
                break;
            }
            local_chunk_id -= cur_chunks;
        }
        if (sequence_id < 0) {
            return;
        }
        const int seq_start       = q_offsets[sequence_id];
        const int seq_end         = q_offsets[sequence_id + 1];
        const int seq_len         = seq_end - seq_start;
        const int token0          = local_chunk_id * kChunkSize;
        const int valid           = min(seq_len - token0, kChunkSize);
        const int physical_batch  = seq_start / token_num;
        const int local_seq_start = seq_start - physical_batch * token_num;

        auto&     smem     = *reinterpret_cast<SharedStorage*>(shared_raw);
        uint64_t* k_ready0 = &smem.k_ready0;
        uint64_t* k_ready1 = &smem.k_ready1;

        MmaElement* k_tile0    = smem.k_tile;
        MmaElement* k_tile1    = smem.k_tile + kKTilePlaneElems;
        MmaElement* out_tile   = smem.out_tile;
        float*      beta_stage = &smem.beta_stage[0][0];
        const auto* gmem_desc = tma_desc_workspace + sequence_id * kLocalKktTmaDescCount;
        AcquireAndPrefetchTmaDescriptors(gmem_desc, tx);
        const CUtensorMap* k_desc          = &gmem_desc[kKktKDesc];
        const CUtensorMap* resolvent_desc  = &gmem_desc[kKktResolventDesc];
        constexpr int      tma_batch       = 0;

        if (tx == 0) {
            cute::initialize_barrier(*k_ready0, 1);
            cute::initialize_barrier(*k_ready1, 1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<kConsumerRegisters>();

            constexpr int ConsumerWg = 0;
            static_assert(ConsumerWg < kKktSolveConsumerWgs);

            if (ConsumerWg < group_count) {
                auto& scratch   = smem.scratch[ConsumerWg];
                auto  s_k0      = cute::make_tensor(cute::make_smem_ptr(k_tile0), KTileLayout());
                auto  s_k1      = cute::make_tensor(cute::make_smem_ptr(k_tile1), KTileLayout());
                auto  s_a16i    = cute::make_tensor(cute::make_smem_ptr(scratch.a16i), Block16Layout<2>());
                auto  s_neg_l10 = cute::make_tensor(cute::make_smem_ptr(scratch.neg_l10), Block16Layout<1>());
                auto  s_out     = cute::make_tensor(cute::make_smem_ptr(out_tile), OutputTileLayout());

                using Mma32   = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                             cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                             cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

                Mma32 mma32;
                float gram_fragment[cute::size(cute::partition_shape_C(mma32, cute::Shape<cute::_32, cute::_32>{}))];
                auto  t_gram_fragment =
                    cute::make_tensor(cute::make_rmem_ptr(gram_fragment),
                                      cute::partition_shape_C(mma32, cute::Shape<cute::_32, cute::_32>{}));
                static_assert(cute::size(t_gram_fragment) == 8);
                auto c_a32       = cute::make_identity_tensor(cute::Shape<cute::_32, cute::_32>{});
                auto t_a32_coord = mma32.get_thread_slice(role_tid).partition_C(c_a32);

                constexpr int k_tma_box_rows = kChunkSize;
                if (role_tid == 0) {
                    cute::set_barrier_transaction_bytes(*k_ready0, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                    cute::SM90_TMA_LOAD_5D::copy(
                        k_desc, k_ready0, kKktTmaNoCacheHint, k_tile0, 0, 0, qk_head, token0, tma_batch);
                    cute::set_barrier_transaction_bytes(*k_ready1, k_tma_box_rows * kKTileTmaDim * sizeof(MmaElement));
                    cute::SM90_TMA_LOAD_5D::copy(
                        k_desc, k_ready1, kKktTmaNoCacheHint, k_tile1, 0, 1, qk_head, token0, tma_batch);
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
                ConsumerWgSync<ConsumerWg>();
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

                if constexpr (GroupsPerKHead != 0) {
                    constexpr int beta_loader_threads = kChunkSize * GroupsPerKHead;
                    if (role_tid < beta_loader_threads) {
                        const int beta_row   = role_tid / GroupsPerKHead;
                        const int beta_group = role_tid - beta_row * GroupsPerKHead;
                        const int beta_head  = value_head_base + beta_group;
                        float     beta_value = 0.0f;
                        if (beta_row < valid) {
                            const int64_t beta_offset =
                                static_cast<int64_t>(physical_batch) * beta_batch_stride
                                + static_cast<int64_t>(local_seq_start + token0 + beta_row) * beta_stride + beta_head;
                            beta_value = beta[beta_offset];
                        }
                        beta_stage[beta_row * 4 + (beta_head & 3)] = beta_value;
                    }
                    ConsumerWgSync<ConsumerWg>();
                }

                int loaded_beta_quad = -1;
#pragma unroll
                for (int group = ConsumerWg; group < group_count; group += kKktSolveConsumerWgs) {
                    const int value_head = value_head_base + group;
                    if constexpr (GroupsPerKHead == 0) {
                        const int beta_quad = value_head / 4;
                        if (beta_quad != loaded_beta_quad) {
                            const int beta_row  = role_tid / 4;
                            const int beta_lane = role_tid % 4;
                            const int beta_head = beta_quad * 4 + beta_lane;
                            float     beta_value = 0.0f;
                            if (beta_row < valid && beta_head < hv) {
                                const int64_t beta_offset =
                                    static_cast<int64_t>(physical_batch) * beta_batch_stride
                                    + static_cast<int64_t>(local_seq_start + token0 + beta_row) * beta_stride
                                    + beta_head;
                                beta_value = beta[beta_offset];
                            }
                            beta_stage[beta_row * 4 + beta_lane] = beta_value;
                            ConsumerWgSync<ConsumerWg>();
                            loaded_beta_quad = beta_quad;
                        }
                    }
                    const int beta_lane = value_head & 3;
                    if (group > 0) {
                        if (role_tid == 0) {
                            cute::tma_store_wait<0>();
                        }
                        ConsumerWgSync<ConsumerWg>();
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
                    ConsumerWgSync<ConsumerWg>();

                    if (role_tid < kChunkSize) {
                        const auto solve_coord = DiagSolveThreadLayout().get_hier_coord(role_tid);
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
                    ConsumerWgSync<ConsumerWg>();

                    // The chunk32 triangular inverse has two 16x16 diagonal blocks. Store inv(L00) and inv(L11)
                    // directly, then form the lower-left block as inv(L11) * (-L10) * inv(L00).
                    if (role_tid < 64) {
                        const auto copy_coord = OffdiagVec4ThreadLayout().get_hier_coord(role_tid);
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
                    }

                    if (role_tid < 64) {
                        using namespace cute;
                        using MmaAtom = SM80_16x8x8_F32TF32TF32F32_TN;
                        using Mma16   = TiledMMA<MMA_Atom<MmaAtom>,
                                                   Layout<Shape<_1, _2, _1>>,
                                                   Tile<Underscore, Int<kBlock16>, Underscore>>;

                        Mma16 mma16;
                        auto  identity = make_identity_tensor(Shape<_16, _16>{});
                        auto  thr_mma  = mma16.get_thread_slice(role_tid);
                        auto  tAcA     = thr_mma.partition_A(identity);
                        auto  tBcB     = thr_mma.partition_B(identity);

                        tfloat32_t a_fragment[size(partition_shape_A(mma16, Shape<_16, _16>{}))];
                        tfloat32_t b_fragment[size(partition_shape_B(mma16, Shape<_16, _16>{}))];
                        float      block_fragment[4];
                        auto tA = make_tensor(make_rmem_ptr(a_fragment), partition_shape_A(mma16, Shape<_16, _16>{}));
                        auto tB = make_tensor(make_rmem_ptr(b_fragment), partition_shape_B(mma16, Shape<_16, _16>{}));
                        auto tC = make_tensor(make_rmem_ptr(block_fragment), partition_shape_C(mma16, Shape<_16, _16>{}));

                        clear(tC);
#pragma unroll
                        for (int i = 0; i < size(tA); ++i) {
                            const auto coord = tAcA(i);
                            tA(i) = Tf32LoadTransform{}(s_a16i(1, get<0>(coord), get<1>(coord)));
                        }
#pragma unroll
                        for (int i = 0; i < size(tB); ++i) {
                            const auto coord = tBcB(i);
                            tB(i) = Tf32LoadTransform{}(s_neg_l10(0, get<1>(coord), get<0>(coord)));
                        }
#pragma unroll
                        for (int k_block = 0; k_block < size<2>(tA); ++k_block) {
                            cute::gemm(mma16, tA(_, _, k_block), tB(_, _, k_block), tC);
                        }

                        OffdiagMmaSync();
                        const int lane       = role_tid & 31;
                        const int warp       = role_tid >> 5;
                        const int pair_col   = lane & 3;
                        const int pair_row   = lane >> 2;
#pragma unroll
                        for (int row_select = 0; row_select < 2; ++row_select) {
                            const int row  = pair_row + 8 * row_select;
                            const int col  = 8 * warp + 2 * pair_col;
                            const int frag = 2 * row_select;
                            s_neg_l10(0, row, col + 0) = block_fragment[frag + 0];
                            s_neg_l10(0, row, col + 1) = block_fragment[frag + 1];
                        }
                        OffdiagMmaSync();

                        clear(tC);
#pragma unroll
                        for (int i = 0; i < size(tA); ++i) {
                            const auto coord = tAcA(i);
                            tA(i) = Tf32LoadTransform{}(s_neg_l10(0, get<0>(coord), get<1>(coord)));
                        }
#pragma unroll
                        for (int i = 0; i < size(tB); ++i) {
                            const auto coord = tBcB(i);
                            tB(i) = Tf32LoadTransform{}(s_a16i(0, get<1>(coord), get<0>(coord)));
                        }
#pragma unroll
                        for (int k_block = 0; k_block < size<2>(tA); ++k_block) {
                            cute::gemm(mma16, tA(_, _, k_block), tB(_, _, k_block), tC);
                        }

#pragma unroll
                        for (int row_select = 0; row_select < 2; ++row_select) {
                            const int row  = pair_row + 8 * row_select;
                            const int col  = 8 * warp + 2 * pair_col;
                            const int frag = 2 * row_select;
                            StoreBf16(&s_out(kBlock16 + row, col + 0), block_fragment[frag + 0]);
                            StoreBf16(&s_out(kBlock16 + row, col + 1), block_fragment[frag + 1]);
                        }
                    }
                    ConsumerWgSync<ConsumerWg>();

                    cute::tma_store_fence();
                    ConsumerWgSync<ConsumerWg>();
                    if (role_tid == 0) {
                        cute::SM90_TMA_STORE_4D::copy(resolvent_desc, out_tile, 0, value_head, token0, tma_batch);
                        cute::tma_store_arrive();
                    }
                    if (group + kKktSolveConsumerWgs < group_count) {
                        ConsumerWgSync<ConsumerWg>();
                    }
                }
                static_cast<void>(finished);
                static_cast<void>(hq);
                static_cast<void>(batch_id);
                if (role_tid == 0) {
                    cute::tma_store_wait<0>();
                }
            }
        }
    }
};

template<class K, int GroupsPerKHead>
__global__ void __launch_bounds__(Sm120KktSolve<K, GroupsPerKHead>::kThreads,
                                  Sm120KktSolve<K, GroupsPerKHead>::kMinBlocks)
    Sm120KktSolveKernel(const int32_t* __restrict__ q_offsets,
                        const bool* __restrict__ finished,
                        const float* __restrict__ beta,
                        CUtensorMap* tma_desc_workspace,
                        int          token_num,
                        int          sequence_num,
                        int          hq,
                        int          hv,
                        int64_t      beta_stride,
                        int64_t      beta_batch_stride,
                        int          groups_per_k_head)
{
    extern __shared__ __align__(1024) unsigned char shared_raw[];
    Sm120KktSolve<K, GroupsPerKHead>::Run(
        q_offsets,
        finished,
        beta,
        tma_desc_workspace,
        token_num,
        sequence_num,
        hq,
        hv,
        beta_stride,
        beta_batch_stride,
        groups_per_k_head,
        shared_raw);
}

template<class K, int GroupsPerKHead>
void LaunchKktSolveTyped(const float*        beta_ptr,
                         const core::Tensor& q_offsets,
                         const core::Tensor& finished,
                         void*               tma_desc_workspace,
                         const Problem&      problem,
                         cudaStream_t        stream)
{
    if (problem.total_chunks == 0) {
        return;
    }

    using Kernel = Sm120KktSolve<K, GroupsPerKHead>;

    const dim3 grid(problem.hq, problem.total_chunks, 1);

    static const cudaError_t smem_attribute_status =
        cudaFuncSetAttribute(Sm120KktSolveKernel<K, GroupsPerKHead>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Kernel::kSharedBytes));
    TM_CUDA_CHECK(smem_attribute_status);
    static const cudaError_t carveout_attribute_status =
        cudaFuncSetAttribute(Sm120KktSolveKernel<K, GroupsPerKHead>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
    TM_CUDA_CHECK(carveout_attribute_status);

    const int32_t* offsets_ptr    = q_offsets.data<int32_t>();
    const bool*    finished_ptr   = finished.data<bool>();
    auto*          desc_workspace = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);
    Sm120KktSolveKernel<K, GroupsPerKHead>
        <<<grid, Kernel::kThreads, Kernel::kSharedBytes, stream>>>(offsets_ptr,
                                                                  finished_ptr,
                                                                  beta_ptr,
                                                                  desc_workspace,
                                                                  problem.token_num,
                                                                  problem.sequence_num,
                                                                  problem.hq,
                                                                  problem.hv,
                                                                  problem.beta_stride,
                                                                  problem.beta_batch_stride,
                                                                  problem.hv / problem.hq);
    TM_CUDA_CHECK(cudaGetLastError());
}

void LaunchSm120KktSolveImpl(const core::Tensor&,
                             const core::Tensor& beta,
                             const core::Tensor& q_offsets,
                             const core::Tensor* g_cumsum,
                             const core::Tensor& finished,
                             core::Tensor&,
                             const Problem& problem,
                             void*          tma_desc_workspace,
                             cudaStream_t   stream)
{
    const auto* beta_ptr = beta.data<float>();
    static_cast<void>(g_cumsum);
    const int groups_per_k_head = problem.hv / problem.hq;
#define TM_LAUNCH_SM120_KKT(GROUPS)                                                                                  \
    LaunchKktSolveTyped<__nv_bfloat16, GROUPS>(                                                                       \
        beta_ptr, q_offsets, finished, tma_desc_workspace, problem, stream)
    switch (groups_per_k_head) {
        case 1: TM_LAUNCH_SM120_KKT(1); break;
        case 2: TM_LAUNCH_SM120_KKT(2); break;
        case 3: TM_LAUNCH_SM120_KKT(3); break;
        case 4: TM_LAUNCH_SM120_KKT(4); break;
        default: TM_LAUNCH_SM120_KKT(0); break;
    }
#undef TM_LAUNCH_SM120_KKT
}

}  // namespace

namespace detail {

void LaunchSm120KktSolve(const core::Tensor& k,
                         const core::Tensor& beta,
                         const core::Tensor& q_offsets,
                         const core::Tensor* g_cumsum,
                         const core::Tensor& finished,
                         core::Tensor&       resolvent,
                         const Problem&      problem,
                         void*               tma_desc_workspace,
                         cudaStream_t        stream)
{
    LaunchSm120KktSolveImpl(
        k, beta, q_offsets, g_cumsum, finished, resolvent, problem, tma_desc_workspace, stream);
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
