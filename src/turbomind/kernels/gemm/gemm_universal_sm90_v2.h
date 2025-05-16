#pragma once

#include <numeric>
#include <utility>

#include <cuda_fp8.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
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
#include "src/turbomind/kernels/gemm/iterator_sm90.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/types.h"

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

template<class Arch_>
struct GemmUniversalSm90_v2 {

    static constexpr bool kDebug = false;

    // using MMA_Atom = GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
    using MMA_Atom = GMMA::MMA_64x96x32_F32E4M3E4M3_SS_TN<>;
    static constexpr typename cute::MMA_Traits<MMA_Atom>::Shape_MNK MMA_Shape{};

    static constexpr int MMA_ATOM_M = cute::get<0>(MMA_Shape);
    static constexpr int MMA_ATOM_N = cute::get<1>(MMA_Shape);
    static constexpr int MMA_ATOM_K = cute::get<2>(MMA_Shape);

    // static constexpr int kWorkGroupM = 2;
    // static constexpr int kWorkGroupN = 1;

    // static constexpr int CTA_M = 128;
    // static constexpr int CTA_N = MMA_ATOM_N * kWorkGroupN;
    // static constexpr int CTA_K = 128;

    static constexpr int WARPGORUPS = 2;  // kWorkGroupM * kWorkGroupN;

    // static constexpr int MMA_M = MMA_ATOM_M * kWorkGroupM;
    // static constexpr int MMA_N = MMA_ATOM_N * kWorkGroupN;
    // static constexpr int MMA_K = MMA_ATOM_K;

    // static constexpr int WG_M = CTA_M / kWorkGroupM;

    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = MMA_ATOM_N;
    static constexpr int TILE_K = 128;

    static constexpr int MMA_ITER_M = TILE_M / MMA_ATOM_M;
    static constexpr int MMA_ITER_N = TILE_N / MMA_ATOM_N;
    static constexpr int MMA_ITER_K = TILE_K / MMA_ATOM_K;

    static constexpr int kMulticastA = 1;
    static constexpr int kMulticastB = 4;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = 6;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = TILE_K;

    static constexpr int WARPGROUP_SIZE = 128;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGORUPS + 1);

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tc = nv_bfloat16;

    using Tu = float;
    using Tv = float;

    using Arch      = Arch_;
    using Scheduler = TileScheduler<kRowMajor, kMulticastB, kMulticastA, true, true>;

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
            __align__(1024) Tu U[Stages][round_up(TILE_M_U * CTA_K_U, 32)];
            __align__(1024) Tv V[2][WARPGORUPS][cdiv(MAX_K, 128)];
        };
        Source source;
        __align__(1024) Array<Tc, TILE_M * TILE_N> C;
        __align__(128) uint64_t producer_bar[Stages];
        __align__(128) uint64_t consumer_bar[Stages];
        int pipe_count;
        int alive_count;
    };

    static constexpr int kSmemSize = sizeof(SharedStorage);

    static constexpr int kSwizzleC = 2 * std::gcd(TILE_N, 128 / sizeof(Tc));

    using LayoutC = std::conditional_t<kSwizzleC >= 32,
                                       SmemLayoutV2<TILE_M, TILE_N, -1, kSwizzleC / sizeof(Tc)>,
                                       SmemLayoutV2<TILE_M, TILE_N>>;

    __device__ void operator()(const CUtensorMap& tm_a,
                               const CUtensorMap& tm_b,
                               const CUtensorMap& tm_c,
                               const CUtensorMap& tm_u,
                               const CUtensorMap& tm_v,
                               const void*        U_,
                               int                ldU,
                               const void*        V_,
                               int                ldV,
                               Scheduler          sched,
                               char*              smem_buf)
    {
        // sched.grid_init();

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], 1);
                ConsumerBar::init(&consumer_bar[s], kClusterSize * 4);
            }
            cutlass::arch::fence_view_async_shared();
            if constexpr (kClusterSize > 1) {
                cutlass::arch::fence_barrier_init();
            }
            storage.pipe_count  = 0;
            storage.alive_count = 0;
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        const int warpgroup_id = cutlass::canonical_warp_group_idx();

        if (warpgroup_id == WARPGORUPS) {
            cutlass::arch::warpgroup_reg_dealloc<40>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            if (threadIdx.x == WARPGORUPS * WARPGROUP_SIZE) {

                const int cta_id = cute::block_id_in_cluster().x;

                // NOTE: ternary operator is used to convince the compiler the result is constant in the trivial case
                const int cta_id_m = kMulticastB > 1 ? cta_id % kMulticastB : 0;
                const int cta_id_n = kMulticastA > 1 ? cta_id / kMulticastB : 0;

                const int mc_offset_m = cta_id_n * (TILE_M / kMulticastA);
                const int mc_offset_n = cta_id_m * (TILE_N / kMulticastB);

                auto  smem_A = storage.source.A.data() + mc_offset_m * TILE_K;
                auto  smem_B = storage.source.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.source.U;

                sched.grid_init();

                cutlass::PipelineState<Stages> write_state{0, 1, 0};

                if (kDebug) {
                    printf("producer %d,0: BASE %d\n", blockIdx.x, sched.cluster_idx_);
                }

                while (sched.next()) {
                    auto [valid_cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                    if (!cluster_tile_p) {
                        // OOB tile caused by swizzle pattern
                        continue;
                    }

                    const auto tile_offset              = sched.tile_offset();
                    const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                    const int offset_k = 0 * TILE_K;

                    constexpr uint16_t kBaseMaskB = (1 << kMulticastB) - 1;
                    constexpr uint16_t kBaseMaskA = ((1 << kClusterSize) - 1) / kBaseMaskB;

                    const uint16_t mask_B = kBaseMaskB << (cta_id_n * kMulticastB);
                    const uint16_t mask_A = kBaseMaskA << (cta_id_m * 1);

                    const int offset_m = tile_offset.x * TILE_M;
                    const int offset_n = tile_offset.y * TILE_N;

                    if (kDebug) {
                        printf("producer %d,0: TILE %d %d\n", blockIdx.x, offset_m, offset_n);
                    }

                    int k_iter = iter_k_end - iter_k_beg;

                    GmemIteratorSm90<kMulticastA> gmem_A{&tm_a, {offset_k, offset_m + mc_offset_m}, {TILE_K, 0}};
                    GmemIteratorSm90<kMulticastB> gmem_B{&tm_b, {offset_k, offset_n + mc_offset_n}, {TILE_K, 0}};

                    // column-major
                    GmemIteratorSm90<false> gmem_U{&tm_u, {offset_m, offset_k / 128}, {0, 1}};

                    for (; k_iter > 0; --k_iter) {
                        int pipe = write_state.index();

                        if (kDebug) {
                            printf("producer %d,0: pipe=%d\n", blockIdx.x, pipe);
                        }

                        ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());

                        if (kDebug) {
                            printf("producer %d,0: ok\n", blockIdx.x);
                        }

                        ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);

                        gmem_A.Load(&producer_bar[pipe], &smem_A[pipe * TILE_M * TILE_K], mask_A);
                        gmem_B.Load(&producer_bar[pipe], &smem_B[pipe * TILE_N * TILE_K], mask_B);

                        gmem_U.Load(&producer_bar[pipe], &smem_U[pipe][0], 0);

                        ++write_state;
                    }
                }

                if (kDebug) {
                    printf("producer %d,0: EXIT\n", blockIdx.x);
                }
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<232>();

            sched.grid_init(WARPGORUPS);

            auto& smem_A = storage.source.A;
            auto& smem_B = storage.source.B;
            auto& smem_U = storage.source.U;

            // const int warp_group_id_m = kWorkGroupM > 1 ? warpgroup_id % kWorkGroupM : 0;
            // const int warp_group_id_n = kWorkGroupN > 1 ? warpgroup_id / kWorkGroupM : 0;

            // auto smem_desc_A = make_smem_desc(&smem_A[0][warp_group_id_m * MMA_ATOM_M * CTA_K], 1);
            // auto smem_desc_B = make_smem_desc(&smem_B[0][warp_group_id_n * MMA_ATOM_N * CTA_K], 1);

            auto smem_desc_A = make_smem_desc(&smem_A, 1);
            auto smem_desc_B = make_smem_desc(&smem_B, 1);

            SmemDescIterV2<Stages, ((sizeof(Ta) * TILE_M * TILE_K) >> 4)> smem_iter_A{smem_desc_A};
            SmemDescIterV2<Stages, ((sizeof(Tb) * TILE_N * TILE_K) >> 4)> smem_iter_B{smem_desc_B};

            constexpr int kStepMA = (sizeof(Ta) * MMA_ATOM_M * TILE_K) >> 4;
            constexpr int kStepNB = (sizeof(Tb) * MMA_ATOM_N * TILE_K) >> 4;
            constexpr int kStepKA = (sizeof(Ta) * MMA_ATOM_K) >> 4;
            constexpr int kStepKB = (sizeof(Tb) * MMA_ATOM_K) >> 4;

            cutlass::PipelineState<Stages> pipe_acquire{};
            cutlass::PipelineState<Stages> pipe_release{};

            auto math_barrier_sync = [&](int phase, int alive = 1) {
                constexpr int base    = (int)cutlass::arch::ReservedNamedBarriers::FirstUserBarrier;
                constexpr int threads = WARPGORUPS * WARPGROUP_SIZE;
                int           res;
                asm volatile("{\n"
                             "  .reg.pred p;\n"
                             "  setp.ne.b32 p, %3, 0;\n"
                             "  barrier.cta.red.or.pred p, %1, %2, p;\n"
                             "  selp.s32 %0, 1, 0, p;\n"
                             "}\n"
                             : "=r"(res)
                             : "r"(base + warpgroup_id ^ phase), "r"(threads), "r"(alive));
                return res;
            };

            cutlass::arch::NamedBarrier wg_barrier(WARPGROUP_SIZE, warpgroup_id + 2);  // 2,3

            sched.next(warpgroup_id);

            if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                printf("consumer %d,%d: BASE %d\n", blockIdx.x, warpgroup_id, sched.cluster_idx_);
            }

            if (warpgroup_id == 1) {
                math_barrier_sync(1);
            }

            while (sched.next(WARPGORUPS)) {
                auto [cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                if (!cluster_tile_p) {
                    // OOB tile caused by swizzle pattern
                    continue;
                }

                MMA_Atom::CRegisters frag_C[MMA_ITER_M][MMA_ITER_N];
                MMA_Atom::CRegisters accum_C[MMA_ITER_M][MMA_ITER_N]{};  /// TODO: check the z-fill is eliminated

                const auto tile_offset              = sched.tile_offset();
                const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                const auto [M, N, K, L] = sched.gemm_shape();

                const int offset_m = tile_offset.x * TILE_M;
                const int offset_n = tile_offset.y * TILE_N;
                const int offset_k = 0;

                int k_iter = iter_k_end - iter_k_beg;

                const int warp_id = threadIdx.x / WARP_SIZE;
                const int lane_id = threadIdx.x % WARP_SIZE;

                const int wg_lane = threadIdx.x % WARPGROUP_SIZE;

                auto consumer_arrive = [&] {
                    __syncwarp();
                    if constexpr (kClusterSize > 1) {
                        ConsumerBar::arrive(&consumer_bar[pipe_release.index()], lane_id, lane_id < kClusterSize);
                    }
                    else {
                        if (lane_id == 0) {
                            ConsumerBar::arrive(&consumer_bar[pipe_release.index()]);
                        }
                    }
                };

                if constexpr (kClusterSize > 1) {
                    if (!cta_tile_p) {  // other CTAs in the cluster are still alive
                        math_barrier_sync(0);
                        const int distance = storage.pipe_count - pipe_acquire.count();
                        pipe_acquire.advance(distance);
                        pipe_release.advance(distance);
                        smem_iter_A.Reset(pipe_acquire.index());
                        smem_iter_B.Reset(pipe_acquire.index());
                        wg_barrier.sync();
                        for (; k_iter > 0; --k_iter) {
                            ProducerBar::wait(&producer_bar[pipe_acquire.index()], pipe_acquire.phase());
                            consumer_arrive();
                            smem_iter_A.Advance(pipe_acquire.index());
                            smem_iter_B.Advance(pipe_acquire.index());
                            ++pipe_acquire;
                            ++pipe_release;
                        }
                        if (wg_lane == 0) {
                            storage.pipe_count = pipe_acquire.count();
                        }
                        math_barrier_sync(1);
                        continue;
                    }
                }

                auto Copy = [k = cdiv(K, 128)](Tv* dst, const Tv* src) {
                    for (int i = threadIdx.x % WARPGROUP_SIZE; i < k; i += WARPGROUP_SIZE) {
                        dst[i] = __ldg(&src[i]);
                    }
                };
                auto gmem_V = (const Tv*)V_ + (offset_n / 128) * ldV + (offset_k / 128);
                Copy(storage.source.V[0][warpgroup_id], gmem_V);

                uint32_t pred_V{};

                constexpr int OUTER_N = std::gcd(MMA_ATOM_N, 128);
                if constexpr (OUTER_N != 128) {

                    static_assert(MMA_ATOM_N <= 128 + OUTER_N, "MMA inst is crossing more than 2 scale blocks");

                    constexpr uint32_t mask = (1UL << (TILE_M / OUTER_N)) - 1;

                    int phase = 128 - offset_n % 128;
                    pred_V    = (mask << (phase / OUTER_N)) & mask;

                    if (pred_V && offset_n / 128 + 1 < cdiv(N, 128)) {
                        Copy(storage.source.V[1][warpgroup_id], gmem_V + ldV);
                    }

                    // if constexpr (kWorkGroupN > 1) {
                    //     constexpr int tiles = MMA_ATOM_N / OUTER_N;
                    //     pred_V              = (pred_V >> (warp_group_id_n * tiles)) & ((1 << tiles) - 1);
                    // }
                }

                // barrier for reading `storage.pipe_count` and writing `smem_V`
                // wg_barrier.sync();

                // math_barrier(0).sync();

                int iter_V = 0;

                // auto tile_gemm = [&] {
                //     if constexpr (MMA_ITER_M == 1) {
                //         PRAGMA_UNROLL
                //         for (int k = 0; k < MMA_ITER_K; ++k) {
                //             PRAGMA_UNROLL
                //             for (int m = 0; m < MMA_ITER_M; ++m) {
                //                 PRAGMA_UNROLL
                //                 for (int n = 0; n < MMA_ITER_N; ++n) {
                //                     wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[m][n], k == 0);
                //                     smem_iter_B += kStepNB;
                //                 }
                //                 smem_iter_B -= MMA_ITER_N * kStepNB;
                //                 smem_iter_A += kStepMA;
                //             }
                //             smem_iter_A += kStepKA - MMA_ITER_M * kStepMA;
                //             smem_iter_B += kStepKB;
                //         }
                //         smem_iter_A -= MMA_ITER_K * kStepKA;
                //         smem_iter_B -= MMA_ITER_K * kStepKB;
                //         cute::warpgroup_commit_batch();
                //     }
                //     else {
                //         PRAGMA_UNROLL
                //         for (int m = 0; m < MMA_ITER_M; ++m) {
                //             PRAGMA_UNROLL
                //             for (int k = 0; k < MMA_ITER_K; ++k) {
                //                 PRAGMA_UNROLL
                //                 for (int n = 0; n < MMA_ITER_N; ++n) {
                //                     wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[m][n], k == 0);
                //                     smem_iter_B += kStepNB;
                //                 }
                //                 smem_iter_B -= MMA_ITER_N * kStepNB;
                //                 smem_iter_A += kStepKA;
                //                 smem_iter_B += kStepKB;
                //             }
                //             cute::warpgroup_commit_batch();
                //             smem_iter_A -= MMA_ITER_K * kStepKA;
                //             smem_iter_B -= MMA_ITER_K * kStepKB;
                //             smem_iter_A += kStepMA;
                //         }
                //         smem_iter_A -= MMA_ITER_M * kStepMA;
                //     }

                //     smem_iter_A.Advance(pipe_acquire.index());
                //     smem_iter_B.Advance(pipe_acquire.index());
                // };

                auto gmma = [&](int m) {
                    // warpgroup_fence_operand(frag_C[m]);
                    PRAGMA_UNROLL
                    for (int k = 0; k < MMA_ITER_K; ++k) {
                        PRAGMA_UNROLL
                        for (int n = 0; n < MMA_ITER_N; ++n) {
                            wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[m][n], k == 0);
                            smem_iter_B += kStepNB;
                        }
                        smem_iter_B -= MMA_ITER_N * kStepNB;
                        smem_iter_A += kStepKA;
                        smem_iter_B += kStepKB;
                    }
                    cute::warpgroup_commit_batch();
                    // warpgroup_fence_operand(frag_C[m]);
                    smem_iter_A -= MMA_ITER_K * kStepKA;
                    smem_iter_B -= MMA_ITER_K * kStepKB;
                    smem_iter_A += kStepMA;
                    if (m == MMA_ITER_M - 1) {
                        smem_iter_A -= MMA_ITER_M * kStepMA;
                        smem_iter_A.Advance(pipe_acquire.index());
                        smem_iter_B.Advance(pipe_acquire.index());
                    }
                };

                static_assert(MMA_ITER_N == 1);

                float scale_V[2];

                auto Load_V = [&] {
                    scale_V[0] = storage.source.V[0][warpgroup_id][iter_V];
                    if (pred_V) {
                        scale_V[1] = storage.source.V[1][warpgroup_id][iter_V];
                    }
                    ++iter_V;
                };

                float scale_U[MMA_ITER_M][2];

                const int offset_U = warp_id % 4 * 16 + lane_id / 4;

                auto Load_U = [&] {
                    for (int m = 0; m < MMA_ITER_M; ++m) {
                        scale_U[m][0] = smem_U[pipe_acquire.index()][offset_U + m * MMA_ATOM_M];
                        scale_U[m][1] = smem_U[pipe_acquire.index()][offset_U + m * MMA_ATOM_M + 8];
                    }
                };

                auto scale_accum = [&](auto m) {  // cta_n = mma_iter_n * wg_n * mma_atom_n
                    float scales[2][2];

                    scales[0][0] = scale_U[m][0] * scale_V[0];
                    scales[1][0] = scale_U[m][1] * scale_V[0];
                    scales[0][1] = scale_U[m][0] * scale_V[1];
                    scales[1][1] = scale_U[m][1] * scale_V[1];

                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {
                        PRAGMA_UNROLL
                        for (int c0 = 0; c0 < MMA_ATOM_N; c0 += OUTER_N) {
                            bool pred = (pred_V & (1U << (c0 / OUTER_N)));
                            PRAGMA_UNROLL
                            for (int cc = 0; cc < OUTER_N; cc += 8) {
                                int c = c0 + cc;
                                // clang-format off
                                    accum_C[m][n][c / 2 + 0] += (pred ? scales[0][1] : scales[0][0]) * frag_C[m][n][c / 2 + 0];
                                    accum_C[m][n][c / 2 + 1] += (pred ? scales[0][1] : scales[0][0]) * frag_C[m][n][c / 2 + 1];
                                    accum_C[m][n][c / 2 + 2] += (pred ? scales[1][1] : scales[1][0]) * frag_C[m][n][c / 2 + 2];
                                    accum_C[m][n][c / 2 + 3] += (pred ? scales[1][1] : scales[1][0]) * frag_C[m][n][c / 2 + 3];
                                // clang-format on
                            }
                        }
                    }
                    // });
                };

                math_barrier_sync(0);

                const int distance = storage.pipe_count - pipe_acquire.count();
                pipe_acquire.advance(distance);
                pipe_release.advance(distance);
                smem_iter_A.Reset(pipe_acquire.index());
                smem_iter_B.Reset(pipe_acquire.index());

                wg_barrier.sync();

                k_iter = 16;

                Load_V();
                ProducerBar::wait(&producer_bar[pipe_acquire.index()], pipe_acquire.phase());
                Load_U();
                cute::warpgroup_arrive();
                warpgroup_fence_operand(frag_C[0]);
                gmma(0);
                warpgroup_fence_operand(frag_C[1]);
                gmma(1);
                cute::warpgroup_wait<1>();
                warpgroup_fence_operand(frag_C[0]);
                scale_accum(0);
                ++pipe_acquire;
                --k_iter;

                PRAGMA_UNROLL
                for (; k_iter > 0; --k_iter) {

                    ProducerBar::wait(&producer_bar[pipe_acquire.index()], pipe_acquire.phase());

                    cute::warpgroup_arrive();
                    warpgroup_fence_operand(frag_C[0]);
                    gmma(0);

                    cute::warpgroup_wait<1>();
                    warpgroup_fence_operand(frag_C[1]);
                    scale_accum(1);

                    Load_V();
                    Load_U();

                    consumer_arrive();

                    cute::warpgroup_arrive();
                    warpgroup_fence_operand(frag_C[1]);
                    gmma(1);

                    cute::warpgroup_wait<1>();
                    warpgroup_fence_operand(frag_C[0]);
                    scale_accum(0);

                    ++pipe_acquire;
                    ++pipe_release;
                }

                // cute::warpgroup_wait<1>();
                // warpgroup_fence_operand(frag_C[0]);
                // scale_accum(0);
                cute::warpgroup_wait<0>();
                warpgroup_fence_operand(frag_C[1]);
                scale_accum(1);
                consumer_arrive();
                ++pipe_release;

#if 0
                Load_V();
                ProducerBar::wait(&producer_bar[pipe_acquire.index()], pipe_acquire.phase());
                Load_U();
                cute::warpgroup_arrive();
                warpgroup_fence_operand(frag_C[0]);
                gmma(0);
                warpgroup_fence_operand(frag_C[1]);
                gmma(1);
                ++pipe_acquire;
                --k_iter;

                for (; k_iter > 0; --k_iter) {
                    cute::warpgroup_wait<1>();
                    warpgroup_fence_operand(frag_C[0]);
                    scale_accum(0);

                    cute::warpgroup_wait<0>();
                    warpgroup_fence_operand(frag_C[1]);
                    scale_accum(1);

                    consumer_arrive();

                    Load_V();
                    ProducerBar::wait(&producer_bar[pipe_acquire.index()], pipe_acquire.phase());
                    Load_U();

                    cute::warpgroup_arrive();

                    warpgroup_fence_operand(frag_C[0]);
                    gmma(0);

                    warpgroup_fence_operand(frag_C[1]);
                    gmma(1);

                    ++pipe_acquire;
                    ++pipe_release;
                }

                cute::warpgroup_wait<1>();
                warpgroup_fence_operand(frag_C[0]);
                scale_accum(0);
                cute::warpgroup_wait<0>();
                warpgroup_fence_operand(frag_C[1]);
                scale_accum(1);
                consumer_arrive();
                ++pipe_release;
#endif

                if (wg_lane == 0) {
                    storage.pipe_count = pipe_acquire.count();
                }

                if (wg_lane == 0 && kDebug) {
                    printf("consumer %d,%d: post, count=%d\n", blockIdx.x, warpgroup_id, pipe_acquire.count());
                }

                // math_barrier(1).arrive_and_wait_unaligned();
                math_barrier_sync(1);

                if (wg_lane == 0 && kDebug) {
                    printf("consumer %d,%d: epi\n", blockIdx.x, warpgroup_id);
                }

                const int wg_thread_id = threadIdx.x % WARPGROUP_SIZE;

                if (wg_thread_id < LayoutC::C1) {
                    cute::tma_store_wait<0>();
                }

                wg_barrier.sync();

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

                            int addr = ((nn / N) * TILE_M * N) + (mm * N) + (nn % N);

                            int s = lane_id % 8;
                            int c = (lane_id & 16) / 2 + i % N;

                            addr += Swizzle<SW_bits, 3, 3>::apply(s * N + c);

                            auto& uvec = (Array<uint32_t, 4>&)tvec;
                            cute::SM90_U32x4_STSM_N::copy(
                                uvec[0], uvec[1], uvec[2], uvec[3], (cutlass::uint128_t&)storage.C[addr]);
                        }
                    }
                }

                cute::tma_store_fence();  // visibility: smem -> async proxy

                wg_barrier.sync();

                if (wg_thread_id < LayoutC::C1) {
                    const int tma_n = wg_thread_id * LayoutC::C0;
                    cute::SM90_TMA_STORE::copy(
                        &tm_c, &storage.C[wg_thread_id * TILE_M * LayoutC::C0], offset_n + tma_n, offset_m);
                    cute::tma_store_arrive();
                }

            }  // scheduler loop

            if (warpgroup_id == 0) {
                math_barrier_sync(0, 0);
                while (math_barrier_sync(1, 0)) {
                    math_barrier_sync(0, 0);
                }
            }
            else {
                while (math_barrier_sync(0, 0)) {
                    math_barrier_sync(1, 0);
                }
            }

#if 0
            if (warpgroup_id == 0) {
                if (threadIdx.x % WARPGROUP_SIZE == 0) {  // [1 -> 0]
                    --storage.alive_count;
                }
                math_barrier(0).arrive_and_wait_unaligned();
                math_barrier(1).arrive_and_wait_unaligned();  // A
                // int alive_count = storage.alive_count;
                // if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                //     printf("consumer %d, alive=%d: tail\n", warpgroup_id, alive_count);
                // }
                // while (alive_count) {
                //     alive_count = storage.alive_count;
                //     math_barrier(0).arrive_and_wait_unaligned();
                //     if (!alive_count) {
                //         break;
                //     }
                //     math_barrier(1).arrive_and_wait_unaligned();
                // }

                if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                    int alive_count = storage.alive_count;
                    printf("consumer %d,%d, alive=%d: tail\n", blockIdx.x, warpgroup_id, alive_count);
                }
                while (storage.alive_count) {
                    math_barrier(0).arrive_and_wait_unaligned();
                    math_barrier(1).arrive_and_wait_unaligned();
                }
            }
            else {
                int alive_count = storage.alive_count - 1;
                wg_barrier.sync();
                if (threadIdx.x % WARPGROUP_SIZE == 0) {  // [1 -> 0]
                    --storage.alive_count;
                }
                if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                    printf("consumer %d,%d, alive=%d: tail\n", blockIdx.x, warpgroup_id, alive_count);
                }
                math_barrier(0).arrive_and_wait_unaligned();  // A
                while (alive_count) {
                    if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                        printf("consumer %d,%d, alive=%d: wait 1\n", blockIdx.x, warpgroup_id, alive_count);
                    }
                    math_barrier(1).arrive_and_wait_unaligned();
                    alive_count = storage.alive_count;
                    if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                        printf("consumer %d,%d, alive=%d: wait 0\n", blockIdx.x, warpgroup_id, alive_count);
                    }
                    math_barrier(0).arrive_and_wait_unaligned();
                }
            }
#endif

            // if (alive_count) {
            //     while (true) {
            //         math_barrier(1).arrive_and_wait_unaligned();
            //         if (storage.alive_count == 0) {
            //             break;
            //         }
            //         math_barrier(0).arrive_and_wait_unaligned();
            //     }
            // }

            // while (true) {
            //     math_barrier(0).arrive_and_wait_unaligned();
            //     if (storage.alive_count == 0) {
            //         break;
            //     }
            //     math_barrier(1).arrive_and_wait_unaligned();
            // }

            // if (warpgroup_id == 0) {
            //     if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
            //         printf("consumer %d: tail 1\n", warpgroup_id);
            //     }
            //     math_barrier(1).arrive_and_wait_unaligned();
            // }

            if (threadIdx.x % WARPGROUP_SIZE < LayoutC::C1) {
                cute::tma_store_wait<0>();
            }

            if (threadIdx.x % WARPGROUP_SIZE == 0 && kDebug) {
                printf("consumer %d,%d: EXIT\n", blockIdx.x, warpgroup_id);
            }
        }

        if constexpr (kClusterSize > 1) {
            cute::cluster_arrive();
            cute::cluster_wait();
        }

    }  // operator()
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1) gemm_kernel_sm90(const __grid_constant__ CUtensorMap tm_a,
                                                                        const __grid_constant__ CUtensorMap tm_b,
                                                                        const __grid_constant__ CUtensorMap tm_c,
                                                                        const __grid_constant__ CUtensorMap tm_u,
                                                                        const __grid_constant__ CUtensorMap tm_v,
                                                                        const void*                         U_,
                                                                        int                                 ldU,
                                                                        const void*                         V_,
                                                                        int                                 ldV,
                                                                        typename Kernel::Scheduler          sched)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(tm_a, tm_b, tm_c, tm_u, tm_v, U_, ldU, V_, ldV, sched, smem_buf);
    }
#endif
}

}  // namespace turbomind::gemm