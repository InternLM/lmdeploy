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

template<class Func, size_t... Is>
__device__ void for_(std::index_sequence<Is...>, Func func)
{
    return (func(constant<Is>{}), ...);
}

template<class Arch_>
struct GemmUniversalSm90_v2 {

    // using MMA_Atom = GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
    using MMA_Atom = GMMA::MMA_64x192x32_F32E4M3E4M3_SS_TN<>;
    static constexpr typename cute::MMA_Traits<MMA_Atom>::Shape_MNK MMA_Shape{};

    static constexpr int MMA_ATOM_M = cute::get<0>(MMA_Shape);
    static constexpr int MMA_ATOM_N = cute::get<1>(MMA_Shape);
    static constexpr int MMA_ATOM_K = cute::get<2>(MMA_Shape);

    static constexpr int kWorkGroupM = 2;
    static constexpr int kWorkGroupN = 1;

    static constexpr int CTA_M = 128;
    static constexpr int CTA_N = MMA_ATOM_N * kWorkGroupN;
    static constexpr int CTA_K = 128;

    static constexpr int WARPGORUPS = kWorkGroupM * kWorkGroupN;

    static constexpr int MMA_M = MMA_ATOM_M * kWorkGroupM;
    static constexpr int MMA_N = MMA_ATOM_N * kWorkGroupN;
    static constexpr int MMA_K = MMA_ATOM_K;

    static constexpr int MMA_ITER_M = CTA_M / MMA_M;
    static constexpr int MMA_ITER_N = CTA_N / MMA_N;
    static constexpr int MMA_ITER_K = CTA_K / MMA_K;

    static constexpr int kMulticastA = 1;
    static constexpr int kMulticastB = 2;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = 4;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = CTA_K;

    static constexpr int WARPGROUP_SIZE = 128;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGORUPS + 1);

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tc = nv_bfloat16;

    using Tu = float;
    using Tv = float;

    using Arch      = Arch_;
    using Scheduler = TileScheduler<kRowMajor, kMulticastB, kMulticastA>;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int CTA_M_U = cdiv(CTA_M, 1);
    static constexpr int CTA_K_U = cdiv(CTA_K, 128);

    static constexpr int CTA_K_V = cdiv(CTA_K, 128);
    static constexpr int CTA_N_V = cdiv(CTA_N, 128);

    static constexpr int kTmaTxBytes =
        sizeof(Ta) * (CTA_M * CTA_K) + sizeof(Tb) * (CTA_K * CTA_N) + sizeof(Tu) * CTA_M_U * CTA_K_U;

    // ! Smem addr must be SBO aligned for TMA load/store
    struct SharedStorage {
        struct Source {
            __align__(1024) Array<Ta, Stages * CTA_M * CTA_K> A;
            __align__(1024) Array<Tb, Stages * CTA_K * CTA_N> B;
            __align__(128) Tu U[Stages][round_up(CTA_M_U * CTA_K_U, 32)];
            __align__(128) Tv V[Stages][round_up(CTA_N_V * CTA_K_V, 32)];  // (k1,n256)
        };
        Source source;
        __align__(1024) Array<Tc, CTA_M * CTA_N> C;
        __align__(128) uint64_t producer_bar[Stages];
        __align__(128) uint64_t consumer_bar[Stages];
    };

    static constexpr int kSmemSize = sizeof(SharedStorage);

    static constexpr int kSwizzleC = 16;

    using LayoutC = std::conditional_t<kSwizzleC >= 32,
                                       SmemLayoutV2<CTA_M, CTA_N, -1, kSwizzleC / sizeof(Tc)>,
                                       SmemLayoutV2<CTA_M, CTA_N>>;

    static_assert(LayoutC::S1 == 1);

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
        sched.grid_init();

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], 1);
                ConsumerBar::init(&consumer_bar[s], kClusterSize * WARPGORUPS * 4);
            }
            cutlass::arch::fence_view_async_shared();
            if constexpr (kClusterSize > 1) {
                cutlass::arch::fence_barrier_init();
            }
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        const int warpgroup_id = cutlass::canonical_warp_group_idx();

        if (warpgroup_id == WARPGORUPS) {
            cutlass::arch::warpgroup_reg_dealloc<32>();

            static_assert(CTA_M % kMulticastA == 0);
            static_assert(CTA_N % kMulticastB == 0);

            const int cta_id = cute::block_id_in_cluster().x;

            const int mc_offset_m = kMulticastA > 1 ? cta_id * (CTA_M / kMulticastA) : 0;
            const int mc_offset_n = kMulticastB > 1 ? cta_id * (CTA_N / kMulticastB) : 0;

            auto  smem_A = storage.source.A.data() + mc_offset_m * CTA_K;
            auto  smem_B = storage.source.B.data() + mc_offset_n * CTA_K;
            auto& smem_U = storage.source.U;
            auto& smem_V = storage.source.V;

            if (threadIdx.x == WARPGORUPS * WARPGROUP_SIZE) {
                cutlass::PipelineState<Stages> write_state{0, 1, 0};
                while (sched.next()) {
                    auto [valid_cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                    if (!cluster_tile_p) {
                        // OOB tile caused by swizzle pattern
                        continue;
                    }

                    const auto tile_offset              = sched.tile_offset();
                    const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                    const int offset_m = tile_offset.x * CTA_M;
                    const int offset_n = tile_offset.y * CTA_N;
                    const int offset_k = 0 * CTA_K;

                    int k_iter = iter_k_end - iter_k_beg;

                    GmemIteratorSm90<kMulticastA> gmem_A{&tm_a, {offset_k, offset_m + mc_offset_m}, {CTA_K, 0}};
                    GmemIteratorSm90<kMulticastB> gmem_B{&tm_b, {offset_k, offset_n + mc_offset_n}, {CTA_K, 0}};

                    // column-major
                    GmemIteratorSm90<false> gmem_U{&tm_u, {offset_m, offset_k / 128}, {0, 1}};

                    for (; k_iter > 0; --k_iter) {
                        int pipe = write_state.index();
                        ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                        ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                        gmem_A.Load(&producer_bar[pipe], &smem_A[pipe * CTA_M * CTA_K]);
                        gmem_B.Load(&producer_bar[pipe], &smem_B[pipe * CTA_N * CTA_K]);
                        gmem_U.Load(&producer_bar[pipe], &smem_U[pipe][0]);
                        ++write_state;
                    }
                }
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<232>();

            auto& smem_A = storage.source.A;
            auto& smem_B = storage.source.B;
            auto& smem_U = storage.source.U;

            const int warp_group_id_m = kWorkGroupM > 1 ? warpgroup_id % kWorkGroupM : 0;
            const int warp_group_id_n = kWorkGroupN > 1 ? warpgroup_id / kWorkGroupM : 0;

            auto smem_desc_A = make_smem_desc(&smem_A[warp_group_id_m * MMA_ATOM_M * CTA_K], 1);
            auto smem_desc_B = make_smem_desc(&smem_B[warp_group_id_n * MMA_ATOM_N * CTA_K], 1);

            SmemDescIterV2<Stages, ((sizeof(Ta) * CTA_M * CTA_K) >> 4)> smem_iter_A{smem_desc_A};
            SmemDescIterV2<Stages, ((sizeof(Tb) * CTA_N * CTA_K) >> 4)> smem_iter_B{smem_desc_B};

            constexpr int kStepMA = (sizeof(Ta) * MMA_M * CTA_K) >> 4;
            constexpr int kStepNB = (sizeof(Tb) * MMA_N * CTA_K) >> 4;
            constexpr int kStepKA = (sizeof(Ta) * MMA_K) >> 4;
            constexpr int kStepKB = (sizeof(Tb) * MMA_K) >> 4;

            cutlass::PipelineState<Stages> pipe_state{};

            while (sched.next()) {
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

                const int offset_m = tile_offset.x * CTA_M;
                const int offset_n = tile_offset.y * CTA_N;
                const int offset_k = 0;

                const int wg_offset_n = offset_n + warp_group_id_n * MMA_ATOM_N;

                auto gmem_V = (const Tv*)V_ + (wg_offset_n / 128) * ldV + (offset_k / 128);
                auto step_V = 1;

                int k_iter = iter_k_end - iter_k_beg;

                auto tile_gemm = [&] {
                    PRAGMA_UNROLL
                    for (int k = 0; k < MMA_ITER_K; ++k) {
                        PRAGMA_UNROLL
                        for (int m = 0; m < MMA_ITER_M; ++m) {
                            PRAGMA_UNROLL
                            for (int n = 0; n < MMA_ITER_N; ++n) {
                                wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[m][n], k == 0);
                                smem_iter_B += kStepNB;
                            }
                            smem_iter_B -= MMA_ITER_N * kStepNB;
                            smem_iter_A += kStepMA;
                        }
                        smem_iter_A += kStepKA - MMA_ITER_M * kStepMA;
                        smem_iter_B += kStepKB;
                    }
                    smem_iter_A -= MMA_ITER_K * kStepKA;
                    smem_iter_B -= MMA_ITER_K * kStepKB;
                    cute::warpgroup_commit_batch();

                    // PRAGMA_UNROLL
                    // for (int m = 0; m < MMA_ITER_M; ++m) {
                    //     PRAGMA_UNROLL
                    //     for (int k = 0; k < MMA_ITER_K; ++k) {
                    //         PRAGMA_UNROLL
                    //         for (int n = 0; n < MMA_ITER_N; ++n) {
                    //             wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[m][n], k == 0);
                    //             smem_iter_B += kStepNB;
                    //         }
                    //         smem_iter_B -= MMA_ITER_N * kStepNB;
                    //         smem_iter_A += kStepKA;
                    //         smem_iter_B += kStepKB;
                    //     }
                    //     cute::warpgroup_commit_batch();
                    //     smem_iter_A -= MMA_ITER_K * kStepKA;
                    //     smem_iter_B -= MMA_ITER_K * kStepKB;
                    //     smem_iter_A += kStepMA;
                    // }
                    // smem_iter_A -= MMA_ITER_M * kStepMA;

                    smem_iter_A.Advance(pipe_state.index());
                    smem_iter_B.Advance(pipe_state.index());
                };

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
                };

                if constexpr (kClusterSize > 1) {
                    if (!cta_tile_p) {
                        // other CTAs in the cluster are still alive
                        for (; k_iter > 0; --k_iter) {
                            ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                            consumer_arrive();
                            smem_iter_A.Advance(pipe_state.index());
                            smem_iter_B.Advance(pipe_state.index());
                            ++pipe_state;
                        }
                        continue;
                    }
                }

                static_assert(MMA_ITER_N == 1);

                bool has_V_1 = wg_offset_n + 128 < M;

                uint32_t pred_V{};

                constexpr int OUTER_N = std::gcd(MMA_ATOM_N, 128);
                if constexpr (OUTER_N != 128) {

                    static_assert(MMA_ATOM_N <= 128 + OUTER_N, "Single MMA op is covering more than 2 scale blocks");

                    // int next_V = MMA_ATOM_N;
                    // PRAGMA_UNROLL
                    // for (int c = 0; c < MMA_ATOM_N; c += OUTER_N) {
                    //     if (c && (wg_offset_n + c) % 128 == 0) {
                    //         next_V = c;
                    //     }
                    //     if (c >= next_V) {
                    //         pred_V |= 1U << c / OUTER_N;
                    //     }
                    // }

                    constexpr uint32_t mask = (1UL << (MMA_ATOM_N / OUTER_N)) - 1;

                    int phase = 128 - wg_offset_n % 128;
                    pred_V    = (mask << (phase / OUTER_N)) & mask;

                    // pred_V = (1 << (phase / OUTER_N)) & mask;
                }

                float scale_V[2];

                auto Load_V = [&] {
                    scale_V[0] = __ldg(gmem_V);
                    if (pred_V && has_V_1) {
                        scale_V[1] = __ldg(gmem_V + ldV);
                    }
                    gmem_V += step_V;
                };

                float scale_U[MMA_ITER_M][2];

                const int offset_U = warp_group_id_m * MMA_ATOM_M + warp_id % 4 * 16 + lane_id / 4;

                auto Load_U = [&] {
                    for (int m = 0; m < MMA_ITER_M; ++m) {
                        scale_U[m][0] = smem_U[pipe_state.index()][offset_U + m * MMA_M];
                        scale_U[m][1] = smem_U[pipe_state.index()][offset_U + m * MMA_M + 8];
                    }
                };

                auto scale_accum = [&]() {  // cta_n = mma_iter_n * wg_n * mma_atom_n
                    for_(std::make_index_sequence<MMA_ITER_M>{}, [&](auto m) {
                        float scales[2][2];

                        scales[0][0] = scale_U[m][0] * scale_V[0];
                        scales[1][0] = scale_U[m][1] * scale_V[0];
                        scales[0][1] = scale_U[m][0] * scale_V[1];
                        scales[1][1] = scale_U[m][1] * scale_V[1];

                        cute::warpgroup_wait<MMA_ITER_M - 1 - m>();

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
                    });

                };

                Load_V();
                ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                Load_U();
                cute::warpgroup_arrive();
                warpgroup_fence_operand(frag_C);
                tile_gemm();
                warpgroup_fence_operand(frag_C);
                scale_accum();
                consumer_arrive();
                ++pipe_state;
                --k_iter;

                for (; k_iter > 0; --k_iter) {
                    Load_V();
                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                    Load_U();
                    cute::warpgroup_arrive();
                    warpgroup_fence_operand(frag_C);
                    tile_gemm();
                    warpgroup_fence_operand(frag_C);
                    scale_accum();
                    consumer_arrive();
                    ++pipe_state;
                }

                if (threadIdx.x < LayoutC::C1) {
                    cute::tma_store_wait<0>();
                }

                cutlass::arch::NamedBarrier(WARPGORUPS * WARPGROUP_SIZE).sync();

                // epilogue
                PRAGMA_UNROLL
                for (int m = 0; m < MMA_ITER_M; ++m) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {

                        const int m0 = m * MMA_M + warp_group_id_m * MMA_ATOM_M;
                        const int n0 = n * MMA_N + warp_group_id_n * MMA_ATOM_N;
#if 1
                        PRAGMA_UNROLL
                        for (int i = 0; i < MMA_ATOM_N; i += 16) {
                            __align__(16) Array<Tc, 8> tvec = cast<Tc>(*(Array<float, 8>*)&accum_C[m][n][i / 2]);

                            constexpr int N       = LayoutC::C0;
                            constexpr int SW_bits = log2(kSwizzleC / 16);

                            int mm = m0 + warp_id % 4 * 16 + (lane_id & 8);
                            int nn = n0 + i / N * N;

                            int s = lane_id % 8;
                            int c = (lane_id & 16) / 2 + i % N;

                            int addr = (nn / N * CTA_M * N) + (mm * N) + Swizzle<SW_bits, 3, 3>::apply(s * N + c);

                            auto& uvec = (Array<uint32_t, 4>&)tvec;
                            cute::SM90_U32x4_STSM_N::copy(
                                uvec[0], uvec[1], uvec[2], uvec[3], (cutlass::uint128_t&)storage.C[addr]);
                        }
#else
                        PRAGMA_UNROLL
                        for (int i = 0; i < MMA_ATOM_N; i += 8) {
                            __align__(16) Array<Tc, 4> tvec = cast<Tc>(*(Array<float, 4>*)&accum_C[m][n][i / 2]);

                            constexpr int N       = LayoutC::C0;
                            constexpr int SW_bits = log2(kSwizzleC / 16);

                            int mm = m0 + warp_id % 4 * 16 + (lane_id & 8);
                            int nn = n0 + i / N * N;

                            int s = lane_id % 8;
                            int c = i % N;

                            int addr = (nn / N * CTA_M * N) + (mm * N) + Swizzle<SW_bits, 3, 3>::apply(s * N + c);

                            auto& uvec = (Array<uint32_t, 2>&)tvec;
                            cute::SM90_U32x2_STSM_N::copy(uvec[0], uvec[1], (cutlass::uint128_t&)storage.C[addr]);
                        }
#endif
                    }
                }

                cute::tma_store_fence();  // visibility: smem -> async proxy
                cutlass::arch::NamedBarrier(WARPGORUPS * WARPGROUP_SIZE).sync();

                if (threadIdx.x < LayoutC::C1) {
                    const int tma_n = threadIdx.x * LayoutC::C0;
                    cute::SM90_TMA_STORE::copy(&tm_c, &storage.C[CTA_M * tma_n], offset_n + tma_n, offset_m);
                    cute::tma_store_arrive();
                }

            }  // scheduler loop

            if (threadIdx.x < LayoutC::C1) {
                cute::tma_store_wait<0>();
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