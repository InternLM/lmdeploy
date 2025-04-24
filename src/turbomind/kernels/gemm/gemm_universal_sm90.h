#pragma once

#include <utility>

#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/iterator_sm90.h"

namespace turbomind::gemm {

inline __device__ uint64_t make_smem_desc(void* smem_ptr, int layout_type)
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

template<class MMA_Atom, size_t... Is>
inline __device__ void wgmma_impl(uint64_t desc_a, uint64_t desc_b, float* frag_C, std::index_sequence<Is...>)
{
    return MMA_Atom::fma(desc_a, desc_b, frag_C[Is]...);
}

template<class MMA_Atom, int N>
inline __device__ void wgmma(uint64_t desc_a, uint64_t desc_b, float (&frag_C)[N])
{
    return wgmma_impl<MMA_Atom>(desc_a, desc_b, frag_C, std::make_index_sequence<N>{});
}

template<class Arch_, class Scheduler_>
struct GemmUniversalSm90 {

    static constexpr int CTA_M = 128;
    static constexpr int CTA_N = 128;
    static constexpr int CTA_K = 64;

    static constexpr int MMA_M = 64;
    static constexpr int MMA_N = 128;
    static constexpr int MMA_K = 16;

    static constexpr int MMA_ITER_M = CTA_M / MMA_M;
    static constexpr int MMA_ITER_N = CTA_N / MMA_N;
    static constexpr int MMA_ITER_K = CTA_K / MMA_K;

    static constexpr int Stages = 2;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = CTA_K;

    static constexpr int CTA_SIZE = 128;

    using MMA_Atom =
        cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>;

    using Ta = nv_bfloat16;
    using Tb = nv_bfloat16;
    using Tc = nv_bfloat16;

    using Arch      = Arch_;
    using Scheduler = Scheduler_;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int kTmaTxBytes = sizeof(Ta) * (CTA_M * CTA_K) + sizeof(Tb) * (CTA_K * CTA_N);

    struct SharedStorage {
        __align__(16) Array<Ta, Stages * CTA_M * CTA_K> A;
        __align__(16) Array<Tb, Stages * CTA_K * CTA_N> B;
        // __align__(16) Array<Tc, CTA_M * CTA_N> C;
        uint64_t producer_bar[Stages];
        uint64_t consumer_bar[Stages];
    };

    __device__ void operator()(const CUtensorMap& tm_a,
                               const CUtensorMap& tm_b,
                               const CUtensorMap& tm_c,
                               Tc*                C,
                               int                ldC,
                               Scheduler          sched,
                               char*              smem_buf)
    {
        if (!sched.init()) {
            return;
        }

        const auto [M, N, K, L] = sched.gemm_shape();

        const auto tile_offset              = sched.tile_offset();
        const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();
        int k_iter                          = iter_k_end - iter_k_beg;

        const int offset_m = tile_offset.x * CTA_M;
        const int offset_n = tile_offset.y * CTA_N;
        const int offset_k = 0 * CTA_K;

        // if (threadIdx.x == 0) {
        //     printf("(%d %d %d) (%d %d) (%d %d %d)\n",
        //            (int)blockIdx.x,
        //            (int)blockIdx.y,
        //            (int)blockIdx.z,
        //            tile_offset.x,
        //            tile_offset.y,
        //            offset_m,
        //            offset_n,
        //            offset_k);
        // }

        if (offset_m >= M || offset_n >= N || offset_k >= K) {  // empty tile
            return;
        }

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        GmemIteratorSm90<false> gmem_A{&tm_a, {offset_k, offset_m}, {CTA_K, 0}};
        GmemIteratorSm90<false> gmem_B{&tm_b, {offset_k, offset_n}, {CTA_K, 0}};

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], 1);
                ConsumerBar::init(&consumer_bar[s], CTA_SIZE);
            }
            cutlass::arch::fence_view_async_shared();
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::arrive_and_expect_tx(&producer_bar[s], kTmaTxBytes);
                gmem_A.Load(&producer_bar[s], &storage.A[s * CTA_M * CTA_K]);
                gmem_B.Load(&producer_bar[s], &storage.B[s * CTA_N * CTA_K]);
            }
        }

        k_iter -= Stages;

        cutlass::PipelineState<Stages> write_state{};
        cutlass::PipelineState<Stages> read_state{};

        MMA_Atom::CRegisters frag_C[MMA_ITER_M][MMA_ITER_N]{};  // zero fill

        auto smem_desc_A = make_smem_desc(&storage.A, 1);
        auto smem_desc_B = make_smem_desc(&storage.B, 1);

        while (k_iter > -Stages) {
            if (1) {
                int pipe = read_state.index();
                ProducerBar::wait(&producer_bar[pipe], read_state.phase());
                cute::warpgroup_arrive();  // wgmma.fence.sync.aligned

                PRAGMA_UNROLL
                for (int m = 0; m < MMA_ITER_M; ++m) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {
                        PRAGMA_UNROLL
                        for (int k = 0; k < MMA_ITER_K; ++k) {
                            // clang-format off
                            auto smem_A = smem_desc_A + pipe * ((sizeof(Ta) * CTA_M * CTA_K) >> 4) + (sizeof(Ta) * ((m * MMA_M) * CTA_K + k * MMA_K) >> 4);
                            auto smem_B = smem_desc_B + pipe * ((sizeof(Tb) * CTA_N * CTA_K) >> 4) + (sizeof(Tb) * ((n * MMA_N) * CTA_K + k * MMA_K) >> 4);
                            // clang-format of
                            wgmma<MMA_Atom>(smem_A, smem_B, frag_C[m][n]);
                        }
                    }
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                ConsumerBar::arrive(&consumer_bar[pipe]);
                ++read_state;
            }

            if (threadIdx.x == 0) {
                int pipe = write_state.index();
                ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                gmem_A.Load(&producer_bar[pipe], &storage.A[pipe * CTA_M * CTA_K]);
                gmem_B.Load(&producer_bar[pipe], &storage.B[pipe * CTA_N * CTA_K]);
                ++write_state;
            }

            --k_iter;
        }

        // epilogue
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < MMA_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < MMA_ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int i = 0; i < MMA_N; i += 8) {
                    int mm = offset_m + m * MMA_M + lane_id / 4 + warp_id * 16;
                    int nn = offset_n + n * MMA_N + (lane_id & 3) * 2 + i;
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        PRAGMA_UNROLL
                        for (int c = 0; c < 2; ++c) {
                            C[(nn + c) * ldC + mm + s * 8] = (Tc)frag_C[m][n][i / 2 + s * 2 + c];
                        }
                    }
                }
            }
        }

        // end
    }
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void gemm_kernel_sm90(const __grid_constant__ CUtensorMap tm_a,
                                 const __grid_constant__ CUtensorMap tm_b,
                                 const __grid_constant__ CUtensorMap tm_c,
                                 typename Kernel::Tc*                C,
                                 int                                 ldC,
                                 typename Kernel::Scheduler          sched)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(tm_a, tm_b, tm_c, C, ldC, sched, smem_buf);
    }
#endif
}

}  // namespace turbomind::gemm