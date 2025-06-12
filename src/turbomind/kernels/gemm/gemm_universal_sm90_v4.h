#pragma once

#include <numeric>
#include <utility>

#include <cuda_fp8.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/smem.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/iterator_sm90.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

namespace GMMA = cute::SM90::GMMA;

inline __device__ cute::GmmaDescriptor make_smem_desc(void* smem_ptr, int layout_type)
{
    auto uint_ptr = cast_smem_ptr_to_uint(smem_ptr);

    cute::GmmaDescriptor desc{};
    desc.bitfield.start_address_       = uint_ptr >> 4;
    desc.bitfield.layout_type_         = layout_type;
    desc.bitfield.leading_byte_offset_ = 0;
    desc.bitfield.stride_byte_offset_  = 1024 >> 4;
    desc.bitfield.base_offset_         = 0;

    return desc;
}

template<int Stages, int Step>
struct SmemDescIterV2 {
    union {
        uint32_t u32_[2];
        uint64_t u64_;
    };

    uint32_t base_;

    __device__ SmemDescIterV2(uint64_t desc): u64_{desc}, base_{u32_[0]} {}

    __device__ void Advance(int stage)
    {
        u32_[0] += Step;
        if (stage == Stages - 1) {
            u32_[0] = base_;
        }
    }

    __device__ void Reset(int stage)
    {
        u32_[0] = base_ + stage * Step;
    }

    __device__ SmemDescIterV2& operator+=(int offset)
    {
        u32_[0] += offset;
        return *this;
    }

    __device__ SmemDescIterV2& operator-=(int offset)
    {
        u32_[0] -= offset;
        return *this;
    }

    __device__ operator uint64_t()
    {
        return u64_;
    }
};

template<class MMA_Atom, size_t... Is>
inline __device__ void
wgmma_impl(uint64_t desc_a, uint64_t desc_b, float* frag_C, bool clear, std::index_sequence<Is...>)
{
    return MMA_Atom::fma(desc_a, desc_b, frag_C[Is]..., clear ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One);
}

template<class MMA_Atom, int N>
inline __device__ void wgmma(uint64_t desc_a, uint64_t desc_b, float (&frag_C)[N], bool clear)
{
    return wgmma_impl<MMA_Atom>(desc_a, desc_b, frag_C, clear, std::make_index_sequence<N>{});
}

inline __device__ void warpgroup_fence_operand(float& reg)
{
    asm volatile("" : "+f"(reg)::"memory");
}

template<int M, int N, int K>
inline __device__ void warpgroup_fence_operand(float (&x)[M][N][K])
{
    PRAGMA_UNROLL
    for (int m = 0; m < M; ++m) {
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
                warpgroup_fence_operand(x[m][n][k]);
            }
        }
    }
}

template<int N, int K>
inline __device__ void warpgroup_fence_operand(float (&x)[N][K])
{
    PRAGMA_UNROLL
    for (int n = 0; n < N; ++n) {
        PRAGMA_UNROLL
        for (int k = 0; k < K; ++k) {
            warpgroup_fence_operand(x[n][k]);
        }
    }
}

template<class Func, size_t... Is>
__device__ void for_(std::index_sequence<Is...>, Func func)
{
    return (func(constant<Is>{}), ...);
}

namespace arch {

template<int M_, int N_, Order order>
struct Cluster {
    static constexpr int M = M_;
    static constexpr int N = N_;

    static constexpr int C = mk2cs<order>(M, N).x;
    static constexpr int S = mk2cs<order>(M, N).y;

    static constexpr int size = M * N;

    static constexpr uint16_t kMaskC = (1 << C) - 1;
    static constexpr uint16_t kMaskS = ((1 << size) - 1) / kMaskC;

    __device__ static ushort2 mask_cs(int cta_id)
    {
        const auto [c, s] = cta_cs(cta_id);
        return make_ushort2(kMaskS << c, kMaskC << s * C);
    }

    __device__ static ushort2 mask_mn(int cta_id)
    {
        auto [c, s] = mask_cs(cta_id);
        return order == kColMajor ? ushort2{c, s} : ushort2{s, c};
    }

    __device__ static int2 cta_cs(int cta_id)
    {
        return {C > 1 ? cta_id % C : 0, S > 1 ? cta_id / C : 0};
    }

    __device__ static int2 cta_mn(int cta_id)
    {
        return cs2mk<order>(cta_cs(cta_id));
    }

    int2    cta_mn_;
    ushort2 mask_mn_;

    __device__ explicit Cluster(int cta_id): cta_mn_(cta_mn(cta_id)), mask_mn_(mask_mn(cta_id)) {}

    __device__ int cta_m()
    {
        return cta_mn_.x;
    }

    __device__ int cta_n()
    {
        return cta_mn_.y;
    }

    __device__ uint16_t mask_m()
    {
        return mask_mn_.x;
    }

    __device__ uint16_t mask_n()
    {
        return mask_mn_.y;
    }
};

}  // namespace arch

struct GemmUniversalSm90_v3 {

    static constexpr bool kDebug = false;

    using Arch = Sm90;

    // using MMA_Atom = GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
    using MMA_Atom = GMMA::MMA_64x192x32_F32E4M3E4M3_SS_TN<>;
    static constexpr typename cute::MMA_Traits<MMA_Atom>::Shape_MNK MMA_Shape{};

    static constexpr int MMA_ATOM_M = cute::get<0>(MMA_Shape);
    static constexpr int MMA_ATOM_N = cute::get<1>(MMA_Shape);
    static constexpr int MMA_ATOM_K = cute::get<2>(MMA_Shape);

    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 192;
    static constexpr int TILE_K = 128;

    static constexpr int WG_M = 2;
    static constexpr int WG_N = 1;

    static constexpr int WG_TILE_M = TILE_M / WG_M;
    static constexpr int WG_TILE_N = TILE_N / WG_N;

    static constexpr int kSchedWarpGroups = 1;

    static constexpr int WARPGORUPS = WG_M * WG_N;

    static constexpr int MMA_ITER_M = WG_TILE_M / MMA_ATOM_M;
    static constexpr int MMA_ITER_N = WG_TILE_N / MMA_ATOM_N;
    static constexpr int MMA_ITER_K = TILE_K / MMA_ATOM_K;

    static constexpr int kMulticastA = 1;
    static constexpr int kMulticastB = 2;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = 4;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = TILE_K;

    static constexpr int WARPGROUP_SIZE = 128;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGORUPS + 1);

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tc = nv_bfloat16;

    using Tu = float;
    using Tv = float;

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;

    using Scheduler = TileScheduler<kRowMajor, Cluster, true, true, false>;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int MAX_K = 32768;

    static constexpr int TILE_M_U = cdiv(TILE_M, 1);
    static constexpr int CTA_K_U  = cdiv(TILE_K, 128);

    static constexpr int kTmaTxBytes =
        sizeof(Ta) * (TILE_M * TILE_K) + sizeof(Tb) * (TILE_N * TILE_K) + sizeof(Tu) * TILE_M_U * CTA_K_U;

    // ! Smem addr must be SBO aligned for TMA load/store
    struct SharedStorage {
        struct Source {
            __align__(1024) Array<Ta, Stages * TILE_M * TILE_K> A;
            __align__(1024) Array<Tb, Stages * TILE_N * TILE_K> B;
            __align__(1024) Tu U[Stages][TILE_M_U * CTA_K_U];
            // __align__(1024) Tv V[2][WARPGORUPS][cdiv(MAX_K, 128)];
            __align__(1024) Tv V[Stages][2 * cdiv(MAX_K, 128)];
        };
        Source source;
        __align__(1024) Array<Tc, TILE_M * TILE_N> C;
        __align__(128) uint64_t producer_bar[Stages];
        __align__(128) uint64_t consumer_bar[Stages];
        __align__(128) CUtensorMap tma_desc_C[WARPGORUPS];
    };

    static constexpr int kSmemSize = sizeof(SharedStorage);

    static constexpr int kSwizzleC = 2 * std::gcd(WG_TILE_N, 128 / sizeof(Tc));

    using LayoutC = std::conditional_t<kSwizzleC >= 32,
                                       SmemLayoutV2<WG_TILE_M, WG_TILE_N, -1, kSwizzleC / sizeof(Tc)>,
                                       SmemLayoutV2<WG_TILE_M, WG_TILE_N>>;

    __device__ void operator()(const CUtensorMap& tm_a,
                               const CUtensorMap& tm_b,
                               const CUtensorMap& tm_c,
                               const CUtensorMap& tm_u,
                               const CUtensorMap& tm_v,
                               const MatrixParam& param_A,
                               const MatrixParam& param_B,
                               const MatrixParam& param_U,
                               const MatrixParam& param_V,
                               const MatrixParam& param_C,
                               uint2              box_V,
                               Scheduler          sched,
                               CUtensorMap*       tensormap_buf,
                               char*              smem_buf)
    {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], 1);
                ConsumerBar::init(&consumer_bar[s], WARPGORUPS * kClusterSize * 4);
            }
            cutlass::arch::fence_view_async_shared();
            if constexpr (kClusterSize > 1) {
                cutlass::arch::fence_barrier_init();
            }
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        const int wg_idx = cutlass::canonical_warp_group_idx();

        if (wg_idx == WARPGORUPS) {
            cutlass::arch::warpgroup_reg_dealloc<40>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            // if (threadIdx.x == WARPGORUPS * WARPGROUP_SIZE) {
            if (threadIdx.x % WARPGROUP_SIZE / WARP_SIZE == 0) {

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto  smem_A = storage.source.A.data() + mc_offset_m * TILE_K;
                auto  smem_B = storage.source.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.source.U;
                auto& smem_V = storage.source.V;

                sched.grid_init();

                cutlass::PipelineState<Stages> write_state{0, 1, 0};
                cutlass::PipelineState<Stages> v_state{0, 1, 0};

                while (sched.next()) {
                    if (cute::elect_one_sync()) {
                        auto [valid_cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                        if (!cluster_tile_p) {
                            // OOB tile caused by swizzle pattern
                            continue;
                        }

                        const auto tile_offset              = sched.tile_offset();
                        const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                        const int offset_k = iter_k_beg * TILE_K;

                        const uint16_t mask_A = cluster.mask_m();
                        const uint16_t mask_B = cluster.mask_n();

                        const int offset_m = tile_offset.x * TILE_M;
                        const int offset_n = tile_offset.y * TILE_N;

                        int k_iter = iter_k_end - iter_k_beg;

                        GmemIteratorSm90<kMulticastA> gmem_A{&tm_a, {offset_k, offset_m + mc_offset_m}, {TILE_K, 0}};
                        GmemIteratorSm90<kMulticastB> gmem_B{&tm_b, {offset_k, offset_n + mc_offset_n}, {TILE_K, 0}};
                        GmemIteratorSm90<kMulticastA> gmem_U{&tm_u, {offset_m + mc_offset_m, offset_k / 128}, {0, 1}};
                        GmemIteratorSm90<1>           gmem_V(&tm_v, {0, offset_n / 128}, {0, 0});

                        {
                            int pipe = write_state.index();
                            ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                            const int v_bytes = sizeof(Tv) * box_V.x * box_V.y;
                            ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes + v_bytes);
                            gmem_A.Step(&producer_bar[pipe], &smem_A[pipe * TILE_M * TILE_K], mask_A);
                            gmem_B.Step(&producer_bar[pipe], &smem_B[pipe * TILE_N * TILE_K], mask_B);
                            gmem_U.Step(&producer_bar[pipe], &smem_U[pipe][0] + mc_offset_m, mask_A);
                            gmem_V.Step(&producer_bar[pipe], &smem_V[v_state.index()], 0);
                            ++write_state;
                            --k_iter;
                        }

                        for (; k_iter > 0; --k_iter) {
                            int pipe = write_state.index();
                            ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                            ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                            gmem_A.Step(&producer_bar[pipe], &smem_A[pipe * TILE_M * TILE_K], mask_A);
                            gmem_B.Step(&producer_bar[pipe], &smem_B[pipe * TILE_N * TILE_K], mask_B);
                            gmem_U.Step(&producer_bar[pipe], &smem_U[pipe][0] + mc_offset_m, mask_A);
                            ++write_state;
                        }

                        ++v_state;
                    }
                }
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<232>();

            sched.grid_init(kSchedWarpGroups);

            auto& smem_A = storage.source.A;
            auto& smem_B = storage.source.B;
            auto& smem_U = storage.source.U;

            const int wg_idx_m = WG_M > 1 ? wg_idx % WG_M : 0;
            const int wg_idx_n = WG_N > 1 ? wg_idx / WG_M : 0;

            auto smem_desc_A = make_smem_desc(&smem_A[wg_idx_m * WG_TILE_M * TILE_K], 1);
            auto smem_desc_B = make_smem_desc(&smem_B[wg_idx_n * WG_TILE_N * TILE_K], 1);

            SmemDescIterV2<Stages, ((sizeof(Ta) * TILE_M * TILE_K) >> 4)> smem_iter_A{smem_desc_A};
            SmemDescIterV2<Stages, ((sizeof(Tb) * TILE_N * TILE_K) >> 4)> smem_iter_B{smem_desc_B};

            constexpr int kStepMA = (sizeof(Ta) * MMA_ATOM_M * TILE_K) >> 4;
            constexpr int kStepNB = (sizeof(Tb) * MMA_ATOM_N * TILE_K) >> 4;
            constexpr int kStepKA = (sizeof(Ta) * MMA_ATOM_K) >> 4;
            constexpr int kStepKB = (sizeof(Tb) * MMA_ATOM_K) >> 4;

            cutlass::arch::NamedBarrier wg_barrier(WARPGROUP_SIZE, wg_idx + 2);  // 2,3

            auto epi_barrier = [&](int phase) {  // 0, 1
                return EmptyBarrier{};
                // return cutlass::arch::NamedBarrier(WARPGORUPS * WARPGROUP_SIZE, wg_idx ^ phase);
            };

            if (wg_idx == 1) {
                epi_barrier(1).arrive_unaligned();
            }

            cutlass::PipelineState<Stages> pipe_state{};
            cutlass::PipelineState<Stages> v_state{};

            while (sched.next(kSchedWarpGroups)) {
                auto [cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                if (!cluster_tile_p) {
                    // OOB tile caused by swizzle pattern
                    continue;
                }

                MMA_Atom::CRegisters frag_C[MMA_ITER_N];
                MMA_Atom::CRegisters accum_C[MMA_ITER_M][MMA_ITER_N]{};

                const auto tile_offset              = sched.tile_offset();
                const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                // const auto [M, N, K, L] = sched.gemm_shape();

                const int offset_m = tile_offset.x * TILE_M;
                const int offset_n = tile_offset.y * TILE_N;

                const int wg_offset_n = offset_n + wg_idx_n * WG_TILE_N;

                int k_iter = iter_k_end - iter_k_beg;

                const int warp_id = threadIdx.x / WARP_SIZE;
                const int lane_id = threadIdx.x % WARP_SIZE;

                auto consumer_arrive = [&] {
                    __syncwarp();
                    if constexpr (kClusterSize > 1) {
                        ConsumerBar::arrive(&consumer_bar[pipe_state.index()], lane_id, lane_id < kClusterSize);
                    }
                    else {
                        if (lane_id == 0) {
                            ConsumerBar::arrive(&consumer_bar[pipe_state.index()]);
                        }
                    }
                    __syncwarp();
                };

                if constexpr (kClusterSize > 1) {
                    if (!cta_tile_p) {  // other CTAs in the cluster are still alive
                        for (; k_iter > 0; --k_iter) {
                            ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                            consumer_arrive();
                            ++pipe_state;
                        }
                        epi_barrier(0).arrive_and_wait_unaligned();
                        epi_barrier(1).arrive_unaligned();
                        continue;
                    }
                }

                // auto Copy = [k = cdiv(K, 128)](Tv* dst, const Tv* src) {
                //     PRAGMA_NO_UNROLL
                //     for (int i = threadIdx.x % WARPGROUP_SIZE; i < k; i += WARPGROUP_SIZE) {
                //         dst[i] = __ldg(&src[i]);
                //     }
                // };
                // auto gmem_V = (const Tv*)param_V.ptr + (wg_offset_n / 128) * param_V.stride + (offset_k / 128);
                // Copy(storage.source.V[0][wg_idx], gmem_V);

                uint32_t pred_V{};
                int      iter_V{};

                constexpr int OUTER_N = std::gcd(MMA_ATOM_N, 128);
                if constexpr (OUTER_N != 128) {

                    static_assert(MMA_ATOM_N <= 128 + OUTER_N, "MMA inst is crossing more than 2 scale blocks");

                    constexpr uint32_t mask = (1UL << (WG_TILE_N / OUTER_N)) - 1;

                    int phase = 128 - wg_offset_n % 128;
                    pred_V    = (mask << (phase / OUTER_N)) & mask;

                    // if (pred_V && wg_offset_n / 128 + 1 < cdiv(N, 128)) {
                    //     Copy(storage.source.V[1][wg_idx], gmem_V + param_V.stride);
                    // }
                    // if constexpr (WG_N > 1) {
                    //     constexpr int tiles = MMA_ATOM_N / OUTER_N;
                    //     pred_V              = (pred_V >> (wg_idx_n * tiles)) & ((1 << tiles) - 1);
                    // }
                }

                __syncwarp();

                float scale_V[2];
                // auto  Load_V = [&] {
                //     scale_V[0] = storage.source.V[0][wg_idx][iter_V];
                //     if (pred_V) {
                //         scale_V[1] = storage.source.V[1][wg_idx][iter_V];
                //     }
                //     ++iter_V;
                // };
                auto Load_V = [&] {
                    // scale_V[0] = scale_V[1] = 1;
                    scale_V[0] = storage.source.V[v_state.index()][iter_V];
                    if (pred_V) {
                        scale_V[1] = storage.source.V[v_state.index()][box_V.x + iter_V];
                    }
                    ++iter_V;
                };

                float     scale_U[MMA_ITER_M][2];
                const int offset_U = wg_idx_m * WG_TILE_M + warp_id % 4 * 16 + lane_id / 4;
                auto      Load_U   = [&] {
                    for (int m = 0; m < MMA_ITER_M; ++m) {
                        scale_U[m][0] = smem_U[pipe_state.index()][offset_U + m * MMA_ATOM_M];
                        scale_U[m][1] = smem_U[pipe_state.index()][offset_U + m * MMA_ATOM_M + 8];
                    }
                };

                auto scale_accum = [&](int m) {  // cta_n = mma_iter_n * wg_n * mma_atom_n
                    float scales[2][2];
                    scales[0][0] = scale_U[m][0] * scale_V[0];
                    scales[1][0] = scale_U[m][1] * scale_V[0];
                    scales[0][1] = scale_U[m][0] * scale_V[1];
                    scales[1][1] = scale_U[m][1] * scale_V[1];
                    cute::warpgroup_wait<0>();
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {
                        PRAGMA_UNROLL
                        for (int c0 = 0; c0 < MMA_ATOM_N; c0 += OUTER_N) {
                            bool pred = (pred_V & (1U << (c0 / OUTER_N)));
                            PRAGMA_UNROLL
                            for (int cc = 0; cc < OUTER_N; cc += 8) {
                                int c = c0 + cc;
                                // clang-format off
                                accum_C[m][n][c / 2 + 0] += (pred ? scales[0][1] : scales[0][0]) * frag_C[n][c / 2 + 0];
                                accum_C[m][n][c / 2 + 1] += (pred ? scales[0][1] : scales[0][0]) * frag_C[n][c / 2 + 1];
                                accum_C[m][n][c / 2 + 2] += (pred ? scales[1][1] : scales[1][0]) * frag_C[n][c / 2 + 2];
                                accum_C[m][n][c / 2 + 3] += (pred ? scales[1][1] : scales[1][0]) * frag_C[n][c / 2 + 3];
                                // clang-format on
                            }
                        }
                    }
                };

                auto gmma = [&](int m) {
                    PRAGMA_UNROLL
                    for (int k = 0; k < MMA_ITER_K; ++k) {
                        PRAGMA_UNROLL
                        for (int n = 0; n < MMA_ITER_N; ++n) {
                            wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[n], k == 0);
                            smem_iter_B += kStepNB;
                        }
                        smem_iter_B -= MMA_ITER_N * kStepNB;
                        smem_iter_A += kStepKA;
                        smem_iter_B += kStepKB;
                    }
                    smem_iter_A -= MMA_ITER_K * kStepKA;
                    smem_iter_B -= MMA_ITER_K * kStepKB;
                    smem_iter_A += kStepMA;
                    cute::warpgroup_commit_batch();
                };

                static_assert(MMA_ITER_N == 1);

                wg_barrier.sync();

                ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                Load_V();
                Load_U();
                smem_iter_A.Reset(pipe_state.index());
                smem_iter_B.Reset(pipe_state.index());
                cute::warpgroup_arrive();
                gmma(0);
                scale_accum(0);
                consumer_arrive();
                ++pipe_state;
                --k_iter;

                Load_V();
                ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                Load_U();
                smem_iter_A.Reset(pipe_state.index());
                smem_iter_B.Reset(pipe_state.index());

                for (; k_iter > 1; --k_iter) {
                    cute::warpgroup_arrive();
                    gmma(0);
                    scale_accum(0);
                    consumer_arrive();
                    ++pipe_state;
                    Load_V();
                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                    Load_U();
                    smem_iter_A.Reset(pipe_state.index());
                    smem_iter_B.Reset(pipe_state.index());
                }

                cute::warpgroup_arrive();
                gmma(0);
                scale_accum(0);
                consumer_arrive();
                ++pipe_state;
                ++v_state;

                const int wg_lane = threadIdx.x % WARPGROUP_SIZE;

                if (wg_lane < LayoutC::C1) {
                    cute::tma_store_wait<0>();
                }

                epi_barrier(0).arrive_and_wait_unaligned();

                wg_barrier.sync();

                // void* Cdesc{};
                // if (threadIdx.x % WARPGROUP_SIZE / WARP_SIZE == 0) {
                //     Cdesc = update_tma_desc(tm_c, tensormap_buf, &storage.tma_desc_C[wg_idx], wg_idx, param_C.ptr,
                //     M);
                // }

                Tc* smem_C = &storage.C[wg_idx_m * WG_TILE_M * TILE_N + wg_idx_n * WG_TILE_N];

                // epilogue
                PRAGMA_UNROLL
                for (int m = 0; m < MMA_ITER_M; ++m) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {

                        constexpr int N       = LayoutC::C0;
                        constexpr int SW_bits = log2(kSwizzleC / 16);

                        static_assert(!SW_bits || MMA_ATOM_N % LayoutC::C0 == 0);

                        const int m0 = m * MMA_ATOM_M;
                        const int n0 = n * MMA_ATOM_N;

                        PRAGMA_UNROLL
                        for (int i = 0; i < MMA_ATOM_N; i += 16) {
                            __align__(16) Array<Tc, 8> tvec = cast<Tc>(*(Array<float, 8>*)&accum_C[m][n][i / 2]);

                            int mm = m0 + warp_id % 4 * 16 + (lane_id & 8);
                            int nn = n0 + i / N * N;

                            int addr = ((nn / N) * WG_TILE_M * N) + (mm * N) + (nn % N);

                            int s = lane_id % 8;
                            int c = (lane_id & 16) / 2 + i % N;

                            addr += Swizzle<SW_bits, 3, 3>::apply(s * N + c);

                            auto& uvec = (Array<uint32_t, 4>&)tvec;
                            cute::SM90_U32x4_STSM_N::copy(uvec[0],  //
                                                          uvec[1],
                                                          uvec[2],
                                                          uvec[3],
                                                          (cutlass::uint128_t&)smem_C[addr]);
                        }
                    }
                }

                cute::tma_store_fence();  // visibility: smem -> async proxy

                wg_barrier.sync();

                if (wg_lane < LayoutC::C1) {
                    const int tma_n = wg_lane * LayoutC::C0;
                    // cute::tma_descriptor_fence_acquire((cute::TmaDescriptor*)Cdesc);
                    cute::SM90_TMA_STORE::copy(&tm_c,
                                               &smem_C[wg_lane * WG_TILE_M * LayoutC::C0],
                                               offset_n + wg_idx_n * WG_TILE_N + tma_n,
                                               offset_m + wg_idx_m * WG_TILE_M);
                    cute::tma_store_arrive();
                }

                epi_barrier(1).arrive_unaligned();

            }  // scheduler loop

            if (threadIdx.x % WARPGROUP_SIZE < LayoutC::C1) {
                cute::tma_store_wait<0>();
            }

            if (wg_idx == 0) {
                epi_barrier(0).arrive_and_wait_unaligned();
            }
        }

        if constexpr (kClusterSize > 1) {
            cute::cluster_arrive();
            cute::cluster_wait();
        }

    }  // operator()

    struct EmptyBarrier {
        __device__      EmptyBarrier(...) {}
        __device__ void arrive_and_wait_unaligned() {}
        __device__ void arrive_unaligned() {}
    };

    __device__ void* update_tma_desc(const CUtensorMap& param_desc,
                                     CUtensorMap*       gmem_desc,
                                     CUtensorMap*       smem_desc,
                                     int                index,
                                     void*              global_addr,
                                     int                dim)
    {
        uint32_t uint_ptr = cast_smem_ptr_to_uint(smem_desc);

        const int lane_id = threadIdx.x % WARP_SIZE;

        if (lane_id < sizeof(CUtensorMap) / sizeof(uint2)) {
            ((uint2*)smem_desc)[lane_id] = ((uint2*)&param_desc)[lane_id];
        }

        __syncwarp();

        if (lane_id == 0) {
            // clang-format off
            asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" ::"r"(uint_ptr), "l"(global_addr));
            asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(uint_ptr), "r"(dim));
            // clang-format on
        }

        __syncwarp();

        constexpr int kNumPerCta = 4;

        auto gmem_ptr = (void*)&gmem_desc[blockIdx.x * kNumPerCta + index];

        // clang-format off
        asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(gmem_ptr), "r"(uint_ptr));
        // clang-format on

        return gmem_ptr;
    }
};

}  // namespace turbomind::gemm
