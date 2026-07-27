#pragma once

/*
 * SM90 dense BF16 GEMM — CuTe cooperative mainloop (PipelineTmaAsync + TiledMma).
 *
 * Operand swap (API vs GMMA):
 *   LlamaLinear / KernelImpl: A = activations (M_batch, K), B = weights (K, N_out)
 *     → host TMA: act as (M,K) row-major; weight transposed to (N,K).
 *   GMMA TileShape = (OUT, BATCH, K) = (TILE_N, TILE_M, TILE_K)
 *     → GMMA-A SMEM = weight  (OUT × K), K-major SoT (SmemLayoutAtomA)
 *     → GMMA-B SMEM = act     (BATCH × K), K-major SoT (SmemLayoutAtomB)
 *   Epilogue maps GMMA C (OUT, BATCH) → problem C (M_batch, N_out) = (BATCH, OUT).
 *
 * Layout contract (Task 1 SoT):
 *   SmemLayout{A,B} = tile_to_shape(SmemLayoutAtom*, Shape<MN,K,Stages>, Step<_1,_2,_3>)
 *   Host CUtensorMap uses CU_TENSOR_MAP_SWIZZLE_128B (KernelImplSm90Bf16) matching
 *   Layout_K_SW128_Atom for TILE_K=64 BF16. Multicast boxes are multiples of the
 *   8-row SW128 atom so linear mc_offset*TILE_K stitching matches the composed SoT.
 *
 * Layout gate — producer-store TV vs consumer GMMA TV (same ComposedLayout SoT from
 * gmma_bf16_sm90.h / GmmaBF16Traits::{SmemLayoutAtomA,SmemLayoutAtomB}):
 *   Weight (GMMA-A): SmemLayoutA = tile_to_shape(SmemLayoutAtomA, (TILE_N,TILE_K,Stages),
 *     Step<_1,_2,_3>). Producer TMA (host CUtensorMap SW128) writes the multicast box
 *     into stage pipe of that ComposedLayout; consumer TiledMma DescriptorIterator
 *     (partition_A → make_fragment_A) reads the same SoT — no alternate swizzle.
 *   Act (GMMA-B): SmemLayoutB = tile_to_shape(SmemLayoutAtomB, (TILE_M,TILE_K,Stages),
 *     Step<_1,_2,_3>). Dense: TMA store TV ↔ GMMA load TV share SmemLayoutAtomB.
 *   Indexed-A gather (grouped): full producer WG (128 threads) cooperative cp.async
 *     through SmemLayoutAtomB / SmemLayoutB_2D (`&sB(m,k)`); GMEM addressing only
 *     (idxs[m0+offset_m+m] → token row). All 128 threads are Pipeline Producers so
 *     producer_acquire runs lockstep (no mid-stage gather_bar). Dynamic TileScheduler
 *     folded onto warp0 (next → acquire → release); producers_bar unused here.
 *     Indexed load as iterator_sm80 (idxs→src_data_vec_, +=TILE_K); TMA before gather.
 *
 * Barrier contract (PipelineTmaAsync full = ClusterTransactionBarrier):
 *   Dense / blocked-A grouped: expect_tx(weight+act bytes); both operands TMA-arrive
 *     via complete_tx (act + weight). Grouped/flat blocked: TMA maps and scheduler
 *     offsets prepared by prepare_tma_descs_sm90_bf16; GEMM indexes by group_idx
 *     (no fence_acquire — that is only for in-kernel tensormap replace).
 *   Indexed-A (v3 / PTX Example-2 .noinc contract): expect_tx(weight bytes only).
 *     Full-barrier init arrive count = 1 (leader arrive_and_expect_tx) + N_noinc
 *     (one cpasync_barrier_arrive_noinc per gather thread). PTX: with .noinc, init
 *     MUST account for those arrive-ons. Per stage: acquire → gather → arrive_noinc
 *     → weight TMA .with(*bar). No producer cp_async_wait / no software act
 *     complete_tx. Indexed A does not TMA-multicast; each CTA gathers full TILE_M.
 *     Weight TMA multicast OK.
 *
 * Epilogue (CUTLASS Sm90TmaWarpSpecialized-style):
 *   EpilogueTile = (_128,_32) for STSM R2S into SW128 SMEM. Host TMA box is
 *   (_64,_32) — bf16 SW128 make_2d_tma_desc requires innermost 64. Device issues
 *   2× SM90_TMA_STORE per epi tile along OUT. Warp0 after fence + EpilogueBarrier.
 */

#include <numeric>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/sm90_bf16_traits.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

namespace detail {

// CuTe TMA issue from host CUtensorMap (KernelImpl): Copy_Atom + .with(*bar[, mcast]).
// Descriptor stays host-built; in-kernel path matches PIPELINING.md / TMA.md contract.
// (copy_traits_sm90_tma.hpp arrives via cute/tensor.hpp → copy_atom.hpp; do not
// include it before tensor.hpp — that breaks CuTe's include order.)
template<int Multicast, int BoxMN, int BoxK, class Element>
__device__ void tma_load_with_barrier(const cute::TmaDescriptor* desc,
                                      uint64_t*                  bar,
                                      Element*                   smem,
                                      int                        crd0,
                                      int                        crd1,
                                      uint16_t                   mcast_mask,
                                      uint64_t cache_hint = (uint64_t)cute::TMA::CacheHintSm90::EVICT_NORMAL)
{
    constexpr int kNumBits = BoxMN * BoxK * (int)cute::sizeof_bits_v<Element>;
    constexpr int kNumVals = BoxMN * BoxK;

    // Dummy Aux — descriptor is swapped in via .with(desc, *bar[, mask]).
    using Aux = cute::
        AuxTmaParams<cute::Stride<cute::_1, cute::_1>, cute::Layout<cute::Shape<cute::_1>>, cute::Swizzle<0, 4, 3>>;

    auto g = cute::make_tensor(cute::make_inttuple_iter(crd0, crd1), cute::Layout<cute::Int<kNumVals>>{});
    auto s = cute::make_tensor(cute::make_smem_ptr(smem), cute::Layout<cute::Int<kNumVals>>{});

    if constexpr (Multicast > 1) {
        using Traits = cute::Copy_Traits<cute::SM90_TMA_LOAD_MULTICAST, cute::Int<kNumBits>, Aux>;
        using Atom   = cute::Copy_Atom<Traits, Element>;
        Atom tma{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(tma.with(desc, *bar, mcast_mask, (cute::TMA::CacheHintSm90)cache_hint), g, s);
    }
    else {
        using Traits = cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::Int<kNumBits>, Aux>;
        using Atom   = cute::Copy_Atom<Traits, Element>;
        Atom tma{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(tma.with(desc, *bar, 0, (cute::TMA::CacheHintSm90)cache_hint), g, s);
        (void)mcast_mask;
    }
}

// STSM atom aliases, tiered by per-WG accumulator vals/thread: big path must match
// pre-templatize (U32x4 / U16x8) exactly; small (4 vals) covers the narrow per-WG
// GMMA-N of 1x2 tiles with TILE_M = 16. (TILE_M = 8 on 1x2 would need 2-val atoms,
// but its per-WG GMMA-N of 4 is already below the GMMA atom minimum of 8.)
template<int kStsmVals>
struct EpiStsmAtoms {
    static_assert(kStsmVals >= 8);
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x8_STSM_T;
};
template<>
struct EpiStsmAtoms<4> {
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x2_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x4_STSM_T;
};

__device__ __forceinline__ float silu_mul(float g, float u)
{
    return fdividef(g, 1.f + expf(-g)) * u;
}

// Device TMA map helpers for MoE prepare kernel (copy → replace addr/dim1 → publish).
__device__ __forceinline__ void copy_tma_desc(CUtensorMap* dst, const CUtensorMap* src, int lane)
{
    constexpr int kWords = (int)(sizeof(CUtensorMap) / sizeof(uint2));
    if (lane < kWords) {
        ((uint2*)dst)[lane] = ((const uint2*)src)[lane];
    }
}

__device__ __forceinline__ void replace_tma_addr_dim1(CUtensorMap* desc, void* global_addr, int dim1)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(desc);
    // clang-format off
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" ::"r"(uint_ptr), "l"(global_addr));
    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(uint_ptr), "r"(dim1));
    // clang-format on
}

__device__ __forceinline__ void publish_tma_desc(CUtensorMap* gmem_desc, CUtensorMap* smem_desc)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(smem_desc);
    // clang-format off
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(gmem_desc), "r"(uint_ptr));
    // clang-format on
}

template<int N>
__device__ __forceinline__ void rebase_publish_tma_descs(CUtensorMap*                 gmem_out,
                                                         CUtensorMap*                 smem_desc,
                                                         Array<const CUtensorMap*, N> templates,
                                                         Array<void*, N>              global_addrs,
                                                         Array<int, N>                dims,
                                                         int                          stride_desc_idx,
                                                         uint64_t                     stride_bytes,
                                                         int                          lane)
{
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        copy_tma_desc(&smem_desc[i], templates[i], lane);
    }
    __syncwarp();
    if (lane == 0) {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            replace_tma_addr_dim1(&smem_desc[i], global_addrs[i], dims[i]);
        }
        replace_tma_global_stride(&smem_desc[stride_desc_idx], stride_bytes);
    }
    __syncwarp();
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        publish_tma_desc(&gmem_out[i], &smem_desc[i]);
    }
    __syncwarp();
}

}  // namespace detail

// Rebase grouped TMA templates and materialize scheduler offsets in workspace.
// Indexed: [B, C]. Blocked: [A, B, C]. Flat input is one blocked group.
template<Striding kStridingA>
__global__ void __launch_bounds__(32, 1) prepare_tma_descs_sm90_bf16(const __grid_constant__ CUtensorMap tm_a,
                                                                     const __grid_constant__ CUtensorMap tm_b,
                                                                     const __grid_constant__ CUtensorMap tm_c,
                                                                     MatrixParam                         param_A,
                                                                     MatrixParam                         param_B,
                                                                     MatrixParam                         param_C,
                                                                     CUtensorMap*                        out,
                                                                     int*                                offsets,
                                                                     int                                 M_total,
                                                                     int                                 N)
{
    constexpr int kNumAB = (kStridingA == Striding::kBlocked) ? 2 : 1;
    constexpr int kNum   = kNumAB + 1;

    __shared__ __align__(128) CUtensorMap smem_desc[kNum];

    const int g    = (int)blockIdx.x;
    const int lane = (int)threadIdx.x & 31;

    using Ta = nv_bfloat16;
    using Tb = nv_bfloat16;
    using Tc = nv_bfloat16;

    const int m0 = param_A.offsets ? __ldg(param_A.offsets + g) : 0;
    const int m1 = param_A.offsets ? __ldg(param_A.offsets + g + 1) : M_total;
    const int M  = m1 - m0;

    CUtensorMap* gmem_out = out + g * kNum;

    if (lane == 0) {
        offsets[g] = m0;
        if (g + 1 == gridDim.x) {
            offsets[g + 1] = m1;
        }
    }

    if constexpr (kStridingA == Striding::kBlocked) {
        Array<const CUtensorMap*, 3> templates;
        templates[0] = &tm_a;
        templates[1] = &tm_b;
        templates[2] = &tm_c;
        Array<void*, 3> addrs;
        const auto      b = resolve<Tb, Striding::kBlocked>(param_B, g);
        addrs[0]          = resolve<Ta, Striding::kBlocked>(param_A, g).ptr.ptr;
        addrs[1]          = b.ptr.ptr;
        addrs[2]          = resolve<Tc, Striding::kBlocked>(param_C, g).ptr.ptr;
        Array<int, 3> dims;
        dims[0] = M;
        dims[1] = N;
        dims[2] = M;
        detail::rebase_publish_tma_descs<3>(
            gmem_out, smem_desc, templates, addrs, dims, 1, (uint64_t)b.ptr.stride * sizeof(Tb), lane);
    }
    else {
        // Indexed-A: gather activations; prepare weight B + output C only.
        Array<const CUtensorMap*, 2> templates;
        templates[0] = &tm_b;
        templates[1] = &tm_c;
        Array<void*, 2> addrs;
        const auto      b = resolve<Tb, Striding::kBlocked>(param_B, g);
        addrs[0]          = b.ptr.ptr;
        addrs[1]          = resolve<Tc, Striding::kBlocked>(param_C, g).ptr.ptr;
        Array<int, 2> dims;
        dims[0] = N;
        dims[1] = M;
        detail::rebase_publish_tma_descs<2>(
            gmem_out, smem_desc, templates, addrs, dims, 0, (uint64_t)b.ptr.stride * sizeof(Tb), lane);
    }
}

template<Order    raster_order,
         int      multicast_a,
         int      multicast_b,
         bool     is_grouped_gemm_,
         Striding kStridingA_     = (is_grouped_gemm_ ? Striding::kIndexed : Striding::kFlat),
         class Tile_              = Sm90Bf16Tile_128x128_2x1,
         bool kSupportsFusedSilu_ = false,
         int  kL2HintW_           = 0>
struct GemmUniversalSm90_Bf16 {

    static constexpr bool kDebug = false;

    static constexpr Order kRasterOrder = raster_order;

    // L2 eviction policy for mainloop weight loads. Instantiation axis (desc policy_b):
    // 0 = EVICT_NORMAL, 1 = EVICT_FIRST (weight panel is streamed once when the problem
    // has a single M-tile). Variants co-exist in the catalog; the tuner picks.
    static constexpr int      kL2HintW = kL2HintW_;
    static constexpr uint64_t kWeightL2Policy =
        kL2HintW ? (uint64_t)cute::TMA::CacheHintSm90::EVICT_FIRST : (uint64_t)cute::TMA::CacheHintSm90::EVICT_NORMAL;

    using Arch = Sm90;
    using Tile = Tile_;

    static constexpr bool kSupportsFusedSilu = kSupportsFusedSilu_;

    // Problem CTA tile: M = batch (act rows), N = out (weight cols), K
    static constexpr int TILE_M = Tile::TILE_M;
    static constexpr int TILE_N = Tile::TILE_N;
    static constexpr int TILE_K = Tile::TILE_K;
    static_assert(TILE_N % 128 == 0);
    static_assert(TILE_M >= 8 && TILE_M % 8 == 0);
    static_assert(TILE_K == 64);  // host TMA still SW128 / K-atom for this step

    using AtomLayoutMNK = typename Tile::AtomLayoutMNK;
    // Fused SiLU: kAtomM == 1 (1x2/1x1) covers the full OUT extent per WG, so the
    // epilogue's C-ownership R2S/TMA mapping applies directly. kAtomM == 2 (2x1)
    // splits the [g64|u64] blocks across WGs (WG0 gate, WG1 up); the epilogue pairs
    // them through an f32 smem staging buffer (run_epilogue), staged per whole CTA
    // tile: kEpiBatch == TILE_M and the silu output exactly fills one epi tile.
    static_assert(!kSupportsFusedSilu || cute::size<0>(AtomLayoutMNK{}) == 1
                  || (cute::size<0>(AtomLayoutMNK{}) * 128 == TILE_N && TILE_M <= 32));

    // Traits: OUT=N_out=TILE_N, BATCH=M_batch=TILE_M; AtomLayout from Tile_
    using Traits   = GmmaBF16Traits<TILE_N, TILE_M, TILE_K, AtomLayoutMNK>;
    using TiledMma = typename Traits::TiledMma;

    static constexpr int WARPGORUPS = cute::size(AtomLayoutMNK{});  // math WGs (cooperative)

    static constexpr int kMulticastA = multicast_a;  // act along TILE_M
    static constexpr int kMulticastB = multicast_b;  // weight along TILE_N

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = Tile::Stages;

    static constexpr int WARPGROUP_SIZE = 128;
    static constexpr int kMathGroupSize = WARPGROUP_SIZE * WARPGORUPS;
    static constexpr int CTA_SIZE       = WARPGROUP_SIZE * (WARPGORUPS + 1);

    static constexpr int K_PIPE_MMAS = 1;

    using Ta = nv_bfloat16;  // API A = activations
    using Tb = nv_bfloat16;  // API B = weights
    using Tc = nv_bfloat16;

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;

    static constexpr auto is_grouped_gemm = is_grouped_gemm_;

    static constexpr Striding kStridingA = kStridingA_;
    static constexpr Striding kStridingB = is_grouped_gemm_ ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC = is_grouped_gemm_ ? Striding::kBlocked : Striding::kFlat;

    static constexpr bool kIndexedGather = (kStridingA_ == Striding::kIndexed);

    // setmaxnreg: each WG ≤ 256, multiples of 8. Budgets from Tile (TMA vs indexed).
    static constexpr int kProducerRegs = kIndexedGather ? Tile::kProducerRegsIndexed : Tile::kProducerRegsTma;
    static constexpr int kMathRegs     = kIndexedGather ? Tile::kMathRegsIndexed : Tile::kMathRegsTma;
    static_assert(kProducerRegs >= 24 && kProducerRegs % 8 == 0);
    static_assert(kMathRegs >= 24 && kMathRegs % 8 == 0 && kMathRegs <= 256);
    static_assert(WARPGORUPS == 1 || WARPGORUPS == 2);
    static_assert(WARPGORUPS != 2 || kProducerRegs + 2 * kMathRegs <= 504);
    static_assert(WARPGORUPS != 1 || kProducerRegs + kMathRegs <= 512);

    using Scheduler = TileScheduler<raster_order, Cluster, true, true, TILE_M, TILE_N, Stages, is_grouped_gemm>;

    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using PipelineState    = typename MainloopPipeline::PipelineState;
    using PipelineStorage  = typename MainloopPipeline::SharedStorage;

    // SMEM SoT: weight → GMMA-A (OUT,K,PIPE); act → GMMA-B (BATCH,K,PIPE). K-major Step<_1,_2,_3>.
    using SmemLayoutA =
        decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomA{},
                                     cute::make_shape(cute::Int<TILE_N>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB =
        decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomB{},
                                     cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    // Per-stage 2D view for indexed gather store TV (same atom as SmemLayoutB / GMMA-B).
    using SmemLayoutB_2D = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomB{},
                                                        cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_K>{}),
                                                        cute::Step<cute::_1, cute::_2>{}));

    static constexpr int kTmaTxBytesWeight = (int)sizeof(Tb) * (TILE_N * TILE_K);
    static constexpr int kTmaTxBytesAct    = (int)sizeof(Ta) * (TILE_M * TILE_K);
    // Dense / blocked-A: expect_tx(weight+act); both via TMA complete_tx.
    // Indexed-A: expect_tx(weight only); gather gated by cpasync_barrier_arrive_noinc.
    static constexpr int kTmaTxBytes =
        (kStridingA_ == Striding::kIndexed) ? kTmaTxBytesWeight : (kTmaTxBytesWeight + kTmaTxBytesAct);

    // Grouped: per-expert maps in workspace (indexed [B,C]; blocked [A,B,C]).
    static constexpr int kTmaDescNumAB = is_grouped_gemm_ ? (kStridingA_ == Striding::kBlocked ? 2 : 1) : 0;
    static constexpr int kTmaDescNumC  = is_grouped_gemm_ ? 1 : 0;
    static constexpr int kTmaDescNum   = (kTmaDescNumAB + kTmaDescNumC) > 0 ? (kTmaDescNumAB + kTmaDescNumC) : 1;

    // Epilogue tile must match TiledMma::tile_size ownership (MMA.md / CUTLASS
    // Sm90 TMA epi): kEpiOut = 64 * (WGs along OUT). AtomLayout<_2,_1,_1> → (128, ≤32);
    // <_1,_2,_1> / <_1,_1,_1> → (64, TILE_M). Epi SMEM must cover full tile_size N
    // or ThrN=1 STSM writes OOB past kEpiBatch=32.
    static constexpr int kAtomM          = Traits::kAtomM;
    static constexpr int kAtomN          = Traits::kAtomN;
    static constexpr int kEpiOut         = 64 * kAtomM;
    static constexpr int kEpiBatch       = (kAtomN == 2) ? TILE_M : ((TILE_M <= 32) ? TILE_M : 32);
    static constexpr int kTmaOut         = 64;  // TMA box along OUT (64 bf16 = SW128B)
    static constexpr int kSwizzleC       = 128;
    static constexpr int kFusedSiluBlock = 64;
    static constexpr int kFragmentSize   = (kEpiOut * kEpiBatch) / kMathGroupSize;
    static_assert(TILE_N % kEpiOut == 0);
    static_assert(TILE_M % kEpiBatch == 0);
    static_assert(kEpiOut % kTmaOut == 0);
    static_assert(kFragmentSize >= 1);
    using EpilogueTile = cute::Shape<cute::Int<kEpiOut>, cute::Int<kEpiBatch>>;

    using SmemLayoutAtomD = decltype(
        gmma_ss_smem_selector<cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::Int<kEpiOut>, cute::Int<kEpiBatch>>());
    using SmemLayoutD =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiOut>{}, cute::Int<kEpiBatch>{}, cute::_1{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));

    // Retile atom (CUTLASS builder always uses STSM_N for C_atom); R2S is STSM_T (M-major).
    // Atom size = per-WG accumulator vals/thread: 2x1 tiles split GMMA-M, so per-WG
    // GMMA-N = TILE_M (TILE_M/2 vals); 1x2 splits GMMA-N too (TILE_M/4 vals). Matches the
    // old kCValsPerThread>=8 criterion on every pre-existing tile; TILE_M=8 2x1 and
    // 16x256_1x2 get U32x2 / U16x4.
    static constexpr int kCValsPerThread = (kAtomN == 2) ? TILE_M / 4 : TILE_M / 2;
    static_assert(kCValsPerThread >= 4, "STSM needs at least U32x2 / 4 bf16 vals per thread");
    using CopyAtomC = typename detail::EpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyAtomC;
    using CopyOpR2S = typename detail::EpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyOpR2S;

    struct LayoutC {
        static constexpr int S0 = kEpiBatch;
        static constexpr int C0 = kTmaOut;  // TMA box, not full epi OUT
        static constexpr int C1 = 1;
    };

    struct SharedStorage {
        cute::array_aligned<typename Traits::ElementA, cute::cosize_v<SmemLayoutA>> A;
        cute::array_aligned<typename Traits::ElementB, cute::cosize_v<SmemLayoutB>> B;
        // Shared epilogue D (CUTLASS Sm90TmaWarpSpecialized — one buffer for all math WGs).
        cute::array_aligned<Tc, cute::cosize_v<SmemLayoutD>, 128> D;
        // f32 gate/up staging for 2x1 fused-SiLU (the [g64|u64] blocks are split
        // across the two math WGs, so silu pairs are exchanged through smem).
        static constexpr int kSiluStageElems = (kSupportsFusedSilu && kAtomM == 2) ? TILE_N * TILE_M : 1;
        cute::array_aligned<float, kSiluStageElems, 128> silu_stage;
        PipelineStorage                                  pipeline;
        typename Scheduler::Storage                      sched;
        int                                              gather_alive;
        int                                              gather_k_iters;
        int                                              gather_m0;
        int                                              gather_M_group;
        int                                              gather_offset_m;
    };

    static constexpr int kSmemSize = (int)sizeof(SharedStorage);

    using ClusterShape = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;

    // Host: launch TMA-map/offset preparation and return the prepared offset table.
    static int* PrepareTmaDescs(const CUtensorMap& tm_a,
                                const CUtensorMap& tm_b,
                                const CUtensorMap& tm_c,
                                const MatrixParam& param_A,
                                const MatrixParam& param_B,
                                const MatrixParam& param_C,
                                CUtensorMap*       out,
                                int                num_groups,
                                int                M,
                                int                N,
                                cudaStream_t       stream)
    {
        if constexpr (!is_grouped_gemm_) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_tma_descs_sm90_bf16<kStridingA_>
            <<<num_groups, 32, 0, stream>>>(tm_a, tm_b, tm_c, param_A, param_B, param_C, out, offsets, M, N);
        return offsets;
    }

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
                               bool               fuse_silu,
                               Scheduler          sched,
                               CUtensorMap*       tensormap_buf,
                               char*              smem_buf)
    {
        (void)tm_u;
        (void)tm_v;
        (void)param_U;
        (void)param_V;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int wg_idx = cutlass::canonical_warp_group_idx();

        if (threadIdx.x == 0) {
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGORUPS * 4 + 1));
        }

        typename MainloopPipeline::Params pp;
        pp.transaction_bytes = (uint32_t)kTmaTxBytes;
        pp.num_consumers     = (uint32_t)kMathGroupSize;
        // Indexed: PTX Example 2 — init must include arrive-ons from each
        // cpasync_barrier_arrive_noinc (1 per gather thread) plus leader
        // arrive_and_expect_tx. Dense/blocked: TMA-only, count = 1.
        pp.num_producers     = (kStridingA == Striding::kIndexed) ? (1 + WARPGROUP_SIZE) : 1;
        pp.initializing_warp = 0;

        if (wg_idx == WARPGORUPS) {
            const int warp_id_in_wg = (threadIdx.x / WARP_SIZE) % 4;
            if constexpr (kStridingA == Striding::kIndexed) {
                // All 128 gather threads are Producers (lockstep acquire + noinc).
                pp.role      = MainloopPipeline::ThreadCategory::Producer;
                pp.is_leader = (warp_id_in_wg == 0) && (threadIdx.x % WARP_SIZE == 0);
            }
            else {
                pp.role      = (warp_id_in_wg == 0) ? MainloopPipeline::ThreadCategory::Producer :
                                                      MainloopPipeline::ThreadCategory::NonParticipant;
                pp.is_leader = (warp_id_in_wg == 0) && (threadIdx.x % WARP_SIZE == 0);
            }
        }
        else {
            pp.role      = MainloopPipeline::ThreadCategory::Consumer;
            pp.is_leader = 0;
        }

        MainloopPipeline pipeline(storage.pipeline, pp, ClusterShape{});

        if (threadIdx.x == 0) {
            cutlass::arch::fence_view_async_shared();
        }
        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        if (wg_idx == WARPGORUPS) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            cutlass::arch::NamedBarrier producers_bar(WARP_SIZE * 2, 7);

            const int  warp_id    = cutlass::canonical_warp_idx_sync();
            const int  warp_in_wg = warp_id % 4;
            const bool cta_0      = cute::block_id_in_cluster().x == 0;

            if constexpr (kStridingA == Striding::kIndexed) {
                // Full producer WG gather. Scheduler folded onto warp0.
                // Full-barrier init = 1 + 128 (expect_tx + one noinc per gather thr).
                cutlass::arch::NamedBarrier gather_bar(
                    /*num_threads=*/WARPGROUP_SIZE, cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto* smem_act    = storage.B.data();
                auto* smem_weight = storage.A.data() + mc_offset_n * TILE_K;

                PipelineState                     write_state = cutlass::make_producer_start_state<MainloopPipeline>();
                typename Scheduler::ConsumerState sched_state = sched.init_consumer(storage.sched);
                typename Scheduler::ProducerState prod_state  = sched.init_producer(storage.sched);
                int                               lane_predicate = 0;
                const int                         lane_id        = threadIdx.x % WARP_SIZE;
                const int                         prod_tid       = threadIdx.x - WARPGORUPS * WARPGROUP_SIZE;

                if (warp_in_wg == 0) {
                    lane_predicate = cute::elect_one_sync();
                }

                const Ta*  act_gmem = (const Ta*)param_A.ptr;
                const int  ldA      = param_A.stride;
                const int* idxs     = param_A.idxs;
                const int  K        = sched.gemm_shape().z;

                constexpr int kVec = 8;
                constexpr int nvec = TILE_M * (TILE_K / kVec);

                typename Scheduler::Tile* tile;

                while (true) {
                    const CUtensorMap* Bdesc   = &tm_b;
                    uint16_t           mask_B  = 0;
                    int                coord_n = 0;
                    int                k_iters = 0;

                    if (warp_in_wg == 0 && cta_0) {
                        (void)prod_state.next();
                    }

                    if (warp_in_wg == 0) {
                        const bool alive = sched_state.acquire(tile);
                        int        m0 = 0, M_group = 0, offset_m = 0;

                        if (alive && tile->is_valid_cluster) {
                            if constexpr (is_grouped_gemm) {
                                const int g = tile->group_idx;
                                Bdesc       = &tensormap_buf[g * kTmaDescNum];
                            }

                            mask_B   = cluster.mask_n();
                            coord_n  = tile->offset_n + mc_offset_n;
                            offset_m = tile->offset_m;
                            m0       = [&] {
                                if constexpr (is_grouped_gemm) {
                                    return tile->m0;
                                }
                                return 0;
                            }();
                            M_group = [&] {
                                if constexpr (is_grouped_gemm) {
                                    return tile->m1 - tile->m0;
                                }
                                return sched.gemm_shape().x;
                            }();
                            k_iters = sched.k_iters_;
                        }

                        if (lane_id == 0) {
                            storage.gather_alive    = alive ? 1 : 0;
                            storage.gather_k_iters  = k_iters;
                            storage.gather_m0       = m0;
                            storage.gather_M_group  = M_group;
                            storage.gather_offset_m = offset_m;
                        }
                        __syncwarp();
                    }

                    // Tile header only.
                    gather_bar.arrive_and_wait();

                    if (storage.gather_alive == 0) {
                        break;
                    }

                    k_iters            = storage.gather_k_iters;
                    const int m0       = storage.gather_m0;
                    const int M_group  = storage.gather_M_group;
                    const int offset_m = storage.gather_offset_m;
                    int       coord_k  = 0;

                    // Indexed load pattern (iterator_sm80): per-tile idxs → src_data_vec_,
                    // then Prefetch advances each base by TILE_K (no idxs reload).
                    // TILE_M=8: nvec=64 < 128 → idle producers must skip ZFILL (pred=false
                    // still zeros dst). TILE_M>=16: exact pre-templatize tight loop.
                    if constexpr (TILE_M == 8) {
                        constexpr int kSlots = (nvec + WARPGROUP_SIZE - 1) / WARPGROUP_SIZE;
                        static_assert(kSlots >= 1);
                        const Ta* src_data_vec_[kSlots];
                        int       m_own[kSlots];
                        int       kk_own[kSlots];
                        bool      pred_row[kSlots];
                        bool      in_rng_slot[kSlots];
                        PRAGMA_UNROLL
                        for (int t = 0; t < kSlots; ++t) {
                            const int  i      = prod_tid + t * WARPGROUP_SIZE;
                            const bool in_rng = i < nvec;
                            const int  m      = in_rng ? (i / (TILE_K / kVec)) : 0;
                            const int  kk     = in_rng ? ((i % (TILE_K / kVec)) * kVec) : 0;
                            const int  packed = m0 + offset_m + m;
                            const bool row_ok = in_rng && (offset_m + m) < M_group;
                            const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                            m_own[t]          = m;
                            kk_own[t]         = kk;
                            pred_row[t]       = row_ok;
                            in_rng_slot[t]    = in_rng;
                            src_data_vec_[t]  = act_gmem + (int64_t)token * ldA + kk;
                        }

                        for (; k_iters > 0; --k_iters) {
                            pipeline.producer_acquire(write_state);
                            auto*     bar  = pipeline.producer_get_barrier(write_state);
                            const int pipe = write_state.index();

                            if (warp_in_wg == 0 && lane_predicate) {
                                detail::tma_load_with_barrier<kMulticastB, TILE_N / kMulticastB, TILE_K>(
                                    Bdesc,
                                    bar,
                                    smem_weight + pipe * TILE_N * TILE_K,
                                    coord_k,
                                    coord_n,
                                    mask_B,
                                    kWeightL2Policy);
                            }

                            {
                                cute::Tensor sB = cute::make_tensor(
                                    cute::make_smem_ptr(smem_act + pipe * TILE_M * TILE_K), SmemLayoutB_2D{});

                                PRAGMA_UNROLL
                                for (int t = 0; t < kSlots; ++t) {
                                    if constexpr ((nvec % WARPGROUP_SIZE) != 0) {
                                        if (!in_rng_slot[t]) {
                                            continue;
                                        }
                                    }
                                    const bool pred = pred_row[t] && (coord_k + kk_own[t]) < K;
                                    auto*      dst  = &sB(m_own[t], kk_own[t]);
                                    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(
                                        *reinterpret_cast<const uint4*>(src_data_vec_[t]),
                                        *reinterpret_cast<uint4*>(dst),
                                        pred);
                                    src_data_vec_[t] += TILE_K;
                                }
                                cutlass::arch::cpasync_barrier_arrive_noinc(bar);
                            }

                            ++write_state;
                            coord_k += TILE_K;
                        }
                    }
                    else {
                        // Identical to 7f8e860 gather (TILE_M=128 → kSlots=8).
                        constexpr int kSlots = nvec / WARPGROUP_SIZE;
                        static_assert(nvec % WARPGROUP_SIZE == 0);
                        const Ta* src_data_vec_[kSlots];
                        int       m_own[kSlots];
                        int       kk_own[kSlots];
                        bool      pred_row[kSlots];
                        PRAGMA_UNROLL
                        for (int t = 0; t < kSlots; ++t) {
                            const int  i      = prod_tid + t * WARPGROUP_SIZE;
                            const int  m      = i / (TILE_K / kVec);
                            const int  kk     = (i % (TILE_K / kVec)) * kVec;
                            const int  packed = m0 + offset_m + m;
                            const bool row_ok = (offset_m + m) < M_group;
                            const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                            m_own[t]          = m;
                            kk_own[t]         = kk;
                            pred_row[t]       = row_ok;
                            src_data_vec_[t]  = act_gmem + (int64_t)token * ldA + kk;
                        }

                        // acquire → weight TMA → gather → arrive_noinc. No producer wait.
                        for (; k_iters > 0; --k_iters) {
                            pipeline.producer_acquire(write_state);
                            auto*     bar  = pipeline.producer_get_barrier(write_state);
                            const int pipe = write_state.index();

                            if (warp_in_wg == 0 && lane_predicate) {
                                detail::tma_load_with_barrier<kMulticastB, TILE_N / kMulticastB, TILE_K>(
                                    Bdesc,
                                    bar,
                                    smem_weight + pipe * TILE_N * TILE_K,
                                    coord_k,
                                    coord_n,
                                    mask_B,
                                    kWeightL2Policy);
                            }

                            {
                                cute::Tensor sB = cute::make_tensor(
                                    cute::make_smem_ptr(smem_act + pipe * TILE_M * TILE_K), SmemLayoutB_2D{});

                                PRAGMA_UNROLL
                                for (int t = 0; t < kSlots; ++t) {
                                    const bool pred = pred_row[t] && (coord_k + kk_own[t]) < K;
                                    auto*      dst  = &sB(m_own[t], kk_own[t]);
                                    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(
                                        *reinterpret_cast<const uint4*>(src_data_vec_[t]),
                                        *reinterpret_cast<uint4*>(dst),
                                        pred);
                                    src_data_vec_[t] += TILE_K;
                                }
                                cutlass::arch::cpasync_barrier_arrive_noinc(bar);
                            }

                            ++write_state;
                            coord_k += TILE_K;
                        }
                    }

                    if (warp_in_wg == 0) {
                        sched_state.release();
                    }
                }

                if (warp_in_wg == 0) {
                    sched_state.release();
                    if (cta_0) {
                        sched.tail(prod_state);
                    }
                    if (lane_predicate) {
                        pipeline.producer_tail(write_state);
                    }
                }
            }
            else if (warp_in_wg == 0) {
                Cluster cluster(cute::block_id_in_cluster().x);

                // API A = act → GMMA-B SMEM; API B = weight → GMMA-A SMEM
                const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto* smem_act    = storage.B.data();
                auto* smem_weight = storage.A.data() + mc_offset_n * TILE_K;

                PipelineState write_state = cutlass::make_producer_start_state<MainloopPipeline>();

                auto sched_state = sched.init_consumer(storage.sched);

                int lane_predicate = cute::elect_one_sync();

                typename Scheduler::Tile* tile;

                while (sched_state.acquire(tile)) {

                    if (tile->is_valid_cluster) {

                        const CUtensorMap* Adesc = &tm_a;
                        const CUtensorMap* Bdesc = &tm_b;

                        if constexpr (is_grouped_gemm) {
                            // Descs published by prepare_moe_tma_descs on this stream;
                            // fence_acquire only needed after in-kernel tensormap replace.
                            const int          g     = tile->group_idx;
                            CUtensorMap* const descs = tensormap_buf + g * kTmaDescNum;
                            if constexpr (kStridingA == Striding::kBlocked) {
                                Adesc = &descs[0];
                                Bdesc = &descs[1];
                            }
                            else {
                                Bdesc = &descs[0];
                            }
                        }

                        const uint16_t mask_B = cluster.mask_n();  // weight multicast

                        const int offset_m = tile->offset_m;
                        const int offset_n = tile->offset_n;

                        int k_iter = sched.k_iters_;

                        int       coord_k = 0;
                        const int coord_m = offset_m + mc_offset_m;
                        const int coord_n = offset_n + mc_offset_n;

                        if (lane_predicate) {
                            // Dense Flat-A or grouped Blocked-A: both operands TMA into SoT.
                            const uint16_t mask_A = cluster.mask_m();
                            for (; k_iter > 0; --k_iter) {
                                pipeline.producer_acquire(write_state);
                                auto*     bar  = pipeline.producer_get_barrier(write_state);
                                const int pipe = write_state.index();

                                detail::tma_load_with_barrier<kMulticastA, TILE_M / kMulticastA, TILE_K>(
                                    Adesc,
                                    bar,
                                    smem_act + mc_offset_m * TILE_K + pipe * TILE_M * TILE_K,
                                    coord_k,
                                    coord_m,
                                    mask_A);
                                detail::tma_load_with_barrier<kMulticastB, TILE_N / kMulticastB, TILE_K>(
                                    Bdesc,
                                    bar,
                                    smem_weight + pipe * TILE_N * TILE_K,
                                    coord_k,
                                    coord_n,
                                    mask_B,
                                    kWeightL2Policy);

                                coord_k += TILE_K;
                                ++write_state;
                            }
                        }
                    }

                    if constexpr (Scheduler::is_dynamic) {
                        if (cta_0) {
                            producers_bar.arrive_unaligned();
                        }
                    }

                    sched_state.release();
                }

                sched_state.release();

                if (lane_predicate) {
                    pipeline.producer_tail(write_state);
                }
            }
            else if (warp_in_wg == 1 && cta_0) {
                auto state = sched.init_producer(storage.sched);
                while (state.next()) {
                    if constexpr (Scheduler::is_dynamic) {
                        producers_bar.arrive_and_wait_unaligned();
                    }
                }
                sched.tail(state);
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<kMathRegs>();

            // mma_tid in [0, kMathGroupSize) — math WGs share one TiledMma TV (cooperative).
            const int mma_tid = threadIdx.x;

            cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), SmemLayoutA{});
            cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});

            TiledMma tiled_mma;
            auto     thr_mma = tiled_mma.get_thread_slice(mma_tid);

            cute::Tensor tCsA = thr_mma.partition_A(sA);
            cute::Tensor tCsB = thr_mma.partition_B(sB);
            cute::Tensor tCrA = thr_mma.make_fragment_A(tCsA);
            cute::Tensor tCrB = thr_mma.make_fragment_B(tCsB);

            PipelineState pipe_state{};
            PipelineState pipe_release = pipe_state;

            auto sched_state = sched.init_consumer(storage.sched);

            typename Scheduler::Tile* tile;
            sched_state.acquire(tile);

            // CUTLASS Sm90TmaWarpSpecialized R2S: as_position_independent +
            // make_tiled_copy_C_atom → STSM_T; warp0 TMA after fence + EpilogueBarrier.
            cute::Tensor sD = cute::as_position_independent_swizzle_tensor(
                cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));

            CopyAtomC tiled_copy_atom_c{};
            auto      tiled_copy_C_atom = cute::make_tiled_copy_C_atom(tiled_copy_atom_c, tiled_mma);
            auto      tiled_r2s =
                cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, cutlass::bfloat16_t>{}, tiled_copy_C_atom);
            auto thr_r2s = tiled_r2s.get_slice(mma_tid);

            cute::Tensor tRS_sD        = thr_r2s.partition_D(sD);  // (R2S,R2S_M,R2S_N,PIPE)
            auto         tRS_rD_layout = cute::make_layout(cute::take<0, 3>(cute::shape(thr_r2s.partition_S(sD))));

            const bool issue_tma_store = (mma_tid / WARP_SIZE) == 0;

            auto epi_synchronize = [&] {
                cutlass::arch::NamedBarrier::sync(kMathGroupSize,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            };

            while (tile->alive) {

                if (tile->is_valid_cta) {
                    cute::Tensor accum =
                        cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    cute::clear(accum);

                    int k_iter = sched.k_iters_;

                    {
                        auto token = pipeline.consumer_try_wait(pipe_state);
                        pipeline.consumer_wait(pipe_state, token);
                        const int read = pipe_state.index();
                        cute::warpgroup_fence_operand(accum);
                        cute::warpgroup_arrive();
                        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                        CUTE_UNROLL
                        for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                            cute::gemm(tiled_mma,
                                       tCrA(cute::_, cute::_, k_block, read),
                                       tCrB(cute::_, cute::_, k_block, read),
                                       accum);
                            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                        }
                        cute::warpgroup_commit_batch();
                        ++pipe_state;
                        --k_iter;
                    }

                    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;

                    PRAGMA_NO_UNROLL
                    for (; k_iter > 0; --k_iter) {
                        auto token = pipeline.consumer_try_wait(pipe_state);
                        pipeline.consumer_wait(pipe_state, token);
                        const int read = pipe_state.index();
                        cute::warpgroup_fence_operand(accum);
                        cute::warpgroup_arrive();
                        cute::gemm(tiled_mma,
                                   tCrA(cute::_, cute::_, cute::_, read),
                                   tCrB(cute::_, cute::_, cute::_, read),
                                   accum);
                        cute::warpgroup_commit_batch();
                        cute::warpgroup_wait<K_PIPE_MMAS>();
                        cute::warpgroup_fence_operand(accum);
                        pipeline.consumer_release(pipe_release);
                        ++pipe_state;
                        ++pipe_release;
                    }

                    cute::warpgroup_wait<0>();
                    pipeline.consumer_release(pipe_release);
                    ++pipe_release;

                    if (issue_tma_store) {
                        cute::tma_store_wait<0>();
                    }
                    epi_synchronize();

                    const int offset_m = tile->offset_m;
                    const int offset_n = tile->offset_n;

                    const void* Cdesc = &tm_c;
                    if constexpr (is_grouped_gemm) {
                        Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kTmaDescNumAB;
                    }

                    cute::Tensor tRS_rAcc = thr_r2s.retile_S(accum);  // ((R2S,R2S_V),MMA_M,MMA_N)
                    cute::Tensor tRS_rD   = cute::make_tensor<cutlass::bfloat16_t>(tRS_rD_layout);

                    cute::Tensor tRS_rAcc_frg = cute::recast<cutlass::Array<float, kFragmentSize>>(tRS_rAcc);
                    cute::Tensor tRS_rD_frg = cute::recast<cutlass::Array<cutlass::bfloat16_t, kFragmentSize>>(tRS_rD);

                    const int     mma_tile_m  = cute::size<0>(typename Traits::TileShape{}) / cute::size<1>(tRS_rAcc);
                    const int     mma_tile_n  = cute::size<1>(typename Traits::TileShape{}) / cute::size<2>(tRS_rAcc);
                    constexpr int epi_tile_m  = kEpiOut;
                    constexpr int epi_tile_n  = kEpiBatch;
                    constexpr int epi_n_count = TILE_M / kEpiBatch;
                    constexpr int tma_m_count = kEpiOut / kTmaOut;
                    (void)epi_tile_m;
                    (void)mma_tile_m;

                    auto run_epilogue = [&](auto fused_silu) {
                        constexpr bool kFuseSilu = decltype(fused_silu)::value;
                        static_assert(!kFuseSilu || kSupportsFusedSilu);
                        // 2x1 silu: the [g64|u64] blocks are split across the two math WGs
                        // (WG0 gate, WG1 up), so pairs are exchanged through the f32
                        // silu_stage buffer and the silu output (TILE_N/2) fills exactly
                        // one epi tile. 1xN silu pairs gate/up in-register instead.
                        constexpr bool kStagedSilu = kFuseSilu && kAtomM == 2;
                        static_assert(!kStagedSilu || (TILE_M == kEpiBatch && TILE_N / 2 == kEpiOut));

                        float* stage = storage.silu_stage.data();
                        (void)stage;
                        // R2S-element (OUT, BATCH) coords within the epi tile; the staged
                        // fill uses them to locate each output's gate/up in silu_stage.
                        // Source-side partition: fragment element j holds the value for
                        // coord tRS_cEpi(j) (same coords as retile_S(accum)).
                        cute::Tensor cEpi     = cute::make_identity_tensor(EpilogueTile{});
                        cute::Tensor tRS_cEpi = thr_r2s.partition_S(cEpi);
                        (void)tRS_cEpi;

                        if constexpr (kStagedSilu) {
                            cute::Tensor cD =
                                cute::make_identity_tensor(cute::make_shape(cute::Int<TILE_N>{}, cute::Int<TILE_M>{}));
                            cute::Tensor tCcD = thr_mma.partition_C(cD);
                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(accum); ++i) {
                                stage[cute::get<0>(tCcD(i)) + cute::get<1>(tCcD(i)) * TILE_N] = accum(i);
                            }
                            epi_synchronize();
                        }

                        constexpr int kStoreOut   = kFuseSilu ? (TILE_N / 2) : TILE_N;
                        constexpr int epi_m_count = kStoreOut / kEpiOut;
                        static_assert(kStoreOut % kEpiOut == 0);

                        CUTE_UNROLL
                        for (int epi_n = 0; epi_n < epi_n_count; ++epi_n) {
                            CUTE_UNROLL
                            for (int epi_m = 0; epi_m < epi_m_count; ++epi_m) {
                                // OUT strips of kEpiOut; in-register gate/up occupy adjacent
                                // strips (staged silu reads pairs from silu_stage instead).
                                const int mma_m = (kFuseSilu && !kStagedSilu) ? 2 * epi_m : epi_m;
                                const int mma_n = (epi_n * epi_tile_n) / mma_tile_n;
                                (void)mma_m;
                                (void)mma_n;

                                const int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
                                const int r2s_v        = epi_n_in_mma * cute::size(tRS_rD_frg);
                                CUTE_UNROLL
                                for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                    cutlass::Array<cutlass::bfloat16_t, kFragmentSize> dst;
                                    if constexpr (kStagedSilu) {
                                        // tRS_cEpi has exactly kFragmentSize coords per thread
                                        // (whole epi tile); staged tiles have epi_n_count ==
                                        // epi_m_count == 1 (assert above).
                                        CUTE_UNROLL
                                        for (int j = 0; j < kFragmentSize; ++j) {
                                            const auto coord = tRS_cEpi(epi_v * kFragmentSize + j);
                                            const int  o     = cute::get<0>(coord);
                                            const int  n     = cute::get<1>(coord);
                                            const int  g_off = (o / kFusedSiluBlock) * (2 * kFusedSiluBlock)
                                                              + o % kFusedSiluBlock + n * TILE_N;
                                            dst[j] = cutlass::bfloat16_t(
                                                detail::silu_mul(stage[g_off], stage[g_off + kFusedSiluBlock]));
                                        }
                                    }
                                    else if constexpr (kFuseSilu) {
                                        // Block-pack [g64|u64]: up is the next MMA_M strip.
                                        auto gate = tRS_rAcc_frg(cute::_, mma_m, mma_n)(r2s_v + epi_v);
                                        auto up   = tRS_rAcc_frg(cute::_, mma_m + 1, mma_n)(r2s_v + epi_v);
                                        CUTE_UNROLL
                                        for (int j = 0; j < kFragmentSize; ++j) {
                                            dst[j] = cutlass::bfloat16_t(detail::silu_mul(gate[j], up[j]));
                                        }
                                    }
                                    else {
                                        auto src = tRS_rAcc_frg(cute::_, mma_m, mma_n)(r2s_v + epi_v);
                                        CUTE_UNROLL
                                        for (int j = 0; j < kFragmentSize; ++j) {
                                            dst[j] = cutlass::bfloat16_t(src[j]);
                                        }
                                    }
                                    tRS_rD_frg(epi_v) = dst;
                                }

                                // Single D buffer: drain prior TMA before R2S (convert above
                                // overlaps the in-flight store). Do not wait right after arrive.
                                if (!(epi_n == 0 && epi_m == 0)) {
                                    if (issue_tma_store) {
                                        cute::tma_store_wait<0>();
                                    }
                                    epi_synchronize();
                                }

                                cute::copy(tiled_r2s, tRS_rD, tRS_sD(cute::_, cute::_, cute::_, 0));

                                cutlass::arch::fence_view_async_shared();
                                epi_synchronize();

                                if (issue_tma_store) {
                                    constexpr int kAtomElems = kTmaOut * kEpiBatch;
                                    CUTE_UNROLL
                                    for (int tma_m = 0; tma_m < tma_m_count; ++tma_m) {
                                        const int out_off = epi_m * kEpiOut + tma_m * kTmaOut;
                                        const int store_n = kFuseSilu ? (offset_n / 2 + out_off) : (offset_n + out_off);
                                        cute::SM90_TMA_STORE::copy(Cdesc,
                                                                   storage.D.data() + tma_m * kAtomElems,
                                                                   store_n,
                                                                   offset_m + epi_n * kEpiBatch);
                                        cute::tma_store_arrive();
                                    }
                                }
                                epi_synchronize();
                            }
                        }
                    };

                    if constexpr (kSupportsFusedSilu) {
                        if (fuse_silu) {
                            run_epilogue(std::true_type{});
                        }
                        else {
                            run_epilogue(std::false_type{});
                        }
                    }
                    else {
                        run_epilogue(std::false_type{});
                    }
                }
                else if (tile->is_valid_cluster) {
                    int k_iter = sched.k_iters_;
                    for (; k_iter > 0; --k_iter) {
                        auto token = pipeline.consumer_try_wait(pipe_state);
                        pipeline.consumer_wait(pipe_state, token);
                        pipeline.consumer_release(pipe_state);
                        ++pipe_state;
                    }
                    pipe_release = pipe_state;
                }

                sched_state.release();
                sched_state.acquire(tile);
            }

            sched_state.release();

            if (issue_tma_store) {
                cute::tma_store_wait<0>();
            }
        }
    }
};

}  // namespace turbomind::gemm
