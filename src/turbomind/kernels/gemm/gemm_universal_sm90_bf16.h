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
 *   Indexed-A gather (grouped): full producer WG (128 threads) cooperatively issues
 *     cp.async through a CuTe TiledCopy whose destination is SmemLayoutB. The small-M
 *     fallback keeps the same affine (M,K) coordinates and skips nonexistent slots.
 *     Warp 0 advances the dynamic scheduler and broadcasts the resolved group pointer,
 *     tile offsets, and bounds through shared storage before all producer warps load.
 *   Fused WG_1x2 consumer: CuTe's native A-fragment RestM stride assigns WG0/WG1
 *     alternating OUT64 segments. The fused descriptor view halves that affine
 *     stride and rebases WG1, assigning each WG a contiguous [gate64|up64] range
 *     without changing the TMA/GMMA swizzle or accumulator ownership.
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
 *   The tile is (kEpiM, 64*kAtomM) in public (M,N) order. kEpiM defaults to the
 *   largest power-of-two divisor of the per-WG M extent up to 32, with explicit
 *   tile overrides for large kernels. Each elected math-warp leader owns one TMA
 *   store slice. Independent WGs synchronize locally; only cross-WG SiLU exchange
 *   uses the full math-group barrier. Two pipeline buffers are selected whenever
 *   there are multiple epilogue passes and the combined mainloop/epilogue SMEM fits.
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
#include "src/turbomind/kernels/core/sync.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"
#include "src/turbomind/kernels/gemm/gmma_issue.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
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
// GMMA-N of WG_2x1 tiles with TILE_M = 16. (TILE_M = 8 on WG_2x1 would need 2-val atoms,
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
    const int M_desc = M > 0 ? M : 1;

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
        dims[0] = M_desc;
        dims[1] = N;
        dims[2] = M_desc;
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
        dims[1] = M_desc;
        detail::rebase_publish_tma_descs<2>(
            gmem_out, smem_desc, templates, addrs, dims, 0, (uint64_t)b.ptr.stride * sizeof(Tb), lane);
    }
}

template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB, int L2HintW, int MmaN, bool SeparateMmaAtoms, int EpiM, int EpiStages_>
struct GemmUniversalSm90_Bf16 {

    static constexpr bool kDebug = false;

    static constexpr Order kRasterOrder = Raster;

    // L2 eviction policy for mainloop weight loads. Instantiation axis (desc policy_b):
    // 0 = EVICT_NORMAL, 1 = EVICT_FIRST (weight panel is streamed once when the problem
    // has a single M-tile). Variants co-exist in the catalog; the tuner picks.
    static constexpr int      kL2HintW = L2HintW;
    static constexpr uint64_t kWeightL2Policy =
        kL2HintW ? (uint64_t)cute::TMA::CacheHintSm90::EVICT_FIRST : (uint64_t)cute::TMA::CacheHintSm90::EVICT_NORMAL;

    using Arch = Sm90;
    using Tile = typename Config_::Tile;
    using Groups = typename Config_::Groups;
    using RegisterConfig = typename Config_::RegisterConfig;

    static constexpr bool kSupportsFusedSilu = Silu;

    // Problem CTA tile: M = batch (act rows), N = out (weight cols), K
    static constexpr int TILE_M = Tile::M;
    static constexpr int TILE_N = Tile::N;
    static constexpr int TILE_K = Tile::K;
    static_assert(TILE_N % 128 == 0);
    static_assert(TILE_M >= 8 && TILE_M % 8 == 0);
    static_assert(TILE_K == 64);  // host TMA still SW128 / K-atom for this step

    using WGLayout = cute::Layout<cute::Shape<cute::Int<Groups::M>, cute::Int<Groups::N>>>;
    using AtomLayoutMNK = GmmaAtomLayoutMNK<WGLayout>;

    // Traits: OUT=N_out=TILE_N, BATCH=M_batch=TILE_M; AtomLayout reverses WGLayout MN.
    using Traits   = GmmaBF16Traits<TILE_N, TILE_M, TILE_K, AtomLayoutMNK, MmaN>;
    using TiledMma = typename Traits::TiledMma;
    using MmaIssue = detail::GmmaIssue<Traits::kMmaNSlices, SeparateMmaAtoms, 2>;

    static constexpr int WARPGROUPS = cute::size(AtomLayoutMNK{});  // math WGs (cooperative)

    static constexpr int kMulticastA = MulticastA;  // act along TILE_M
    static constexpr int kMulticastB = MulticastB;  // weight along TILE_N

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = Stages_;

    static constexpr int WARPGROUP_SIZE = 128;
    static constexpr int kMathGroupSize = WARPGROUP_SIZE * WARPGROUPS;
    static constexpr int CTA_SIZE       = WARPGROUP_SIZE * (WARPGROUPS + 1);

    static constexpr int kEpilogueBarrierId = 1;
    static constexpr int kProducerBarrierId = 8;
    static_assert(kEpilogueBarrierId + WARPGROUPS <= kProducerBarrierId);

    static constexpr int K_PIPE_MMAS = 1;

    using Ta = nv_bfloat16;  // API A = activations
    using Tb = nv_bfloat16;  // API B = weights
    using Tc = nv_bfloat16;

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;

    static constexpr bool is_grouped_gemm = Mode != Striding::kFlat;

    static constexpr Striding kStridingA = Mode;
    static constexpr Striding kStridingB = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;

    static constexpr bool kIndexedGather = (Mode == Striding::kIndexed);

    // setmaxnreg: each WG ≤ 256, multiples of 8. The configuration supplies the active producer/math budgets.
    static constexpr int kProducerRegs = RegisterConfig::Producer;
    static constexpr int kMathRegs = RegisterConfig::Math;
    static_assert(kProducerRegs >= 24 && kProducerRegs % 8 == 0);
    static_assert(kMathRegs >= 24 && kMathRegs % 8 == 0 && kMathRegs <= 256);
    static_assert(WARPGROUPS == 1 || WARPGROUPS == 2);
    static_assert(WARPGROUPS != 2 || kProducerRegs + 2 * kMathRegs <= 504);
    static_assert(WARPGROUPS != 1 || kProducerRegs + kMathRegs <= 512);

    using Scheduler = TileScheduler<Raster, Cluster, true, true, TILE_M, TILE_N, Stages, is_grouped_gemm>;

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

    static constexpr int kGatherVec      = 16 / (int)sizeof(Ta);
    static constexpr int kGatherThreadsK = TILE_K / kGatherVec;
    static constexpr int kGatherThreadsM = WARPGROUP_SIZE / kGatherThreadsK;
    using GatherCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>, Ta>;
    using GatherTiledCopy = decltype(cute::make_tiled_copy(GatherCopyAtom{},
                                                           cute::Layout<cute::Shape<cute::Int<kGatherThreadsM>, cute::Int<kGatherThreadsK>>,
                                                                        cute::Stride<cute::Int<kGatherThreadsK>, cute::_1>>{},
                                                           cute::Layout<cute::Shape<cute::_1, cute::Int<kGatherVec>>>{}));
    static_assert(kGatherVec * (int)sizeof(Ta) == 16);
    static_assert(kGatherThreadsM * kGatherThreadsK == WARPGROUP_SIZE);
    static_assert(cute::size(GatherTiledCopy{}) == WARPGROUP_SIZE);

    static constexpr int kTmaCountM = cute::ceil_div(TILE_M / kMulticastA, 256);
    static constexpr int kTmaBoxM   = TILE_M / (kMulticastA * kTmaCountM);
    static_assert(TILE_M % (kMulticastA * kTmaCountM) == 0);
    static_assert(kTmaBoxM <= 256);

    static constexpr int kTmaTxBytesWeight = (int)sizeof(Tb) * (TILE_N * TILE_K);
    static constexpr int kTmaTxBytesAct    = (int)sizeof(Ta) * (TILE_M * TILE_K);
    // Dense / blocked-A: expect_tx(weight+act); both via TMA complete_tx.
    // Indexed-A: expect_tx(weight only); gather gated by cpasync_barrier_arrive_noinc.
    static constexpr int kTmaTxBytes =
        (Mode == Striding::kIndexed) ? kTmaTxBytesWeight : (kTmaTxBytesWeight + kTmaTxBytesAct);

    // Grouped: per-expert maps in workspace (indexed [B,C]; blocked [A,B,C]).
    static constexpr int kTmaDescNumAB = is_grouped_gemm ? (Mode == Striding::kBlocked ? 2 : 1) : 0;
    static constexpr int kTmaDescNumC  = is_grouped_gemm ? 1 : 0;
    static constexpr int kTmaDescNum   = (kTmaDescNumAB + kTmaDescNumC) > 0 ? (kTmaDescNumAB + kTmaDescNumC) : 1;

    // The epilogue tile follows public GEMM (M,N) order. Its N extent covers
    // 64 columns per output-split WG; its M extent must cover the STSM-owned
    // tile or an M-unsplit STSM can write beyond the shared-memory buffer.
    static constexpr int kAtomM          = Traits::kAtomM;
    static constexpr int kAtomN          = Traits::kAtomN;
    static constexpr int kRestM                    = TILE_N / (64 * kAtomM);
    static constexpr bool kNeedsCrossWgSiluExchange = kSupportsFusedSilu && kRestM % 2 != 0 && kAtomM == 2 && WARPGROUPS == 2;
    static constexpr bool kUsesContiguousFusedMapping = kSupportsFusedSilu && kAtomM == 2 && !kNeedsCrossWgSiluExchange;
    static constexpr bool kSplitEpiM      = kAtomN == 2;
    static constexpr int kWgM            = TILE_M / kAtomN;
    static constexpr int kEpiN           = 64 * kAtomM;
    static constexpr int kWgMLowBit      = kWgM & -kWgM;
    static constexpr int kEpiMDefault    = kWgMLowBit < 32 ? kWgMLowBit : 32;
    static constexpr int kEpiM           = EpiM ? EpiM : kEpiMDefault;
    static constexpr int kEpiPlanes      = kSplitEpiM ? kAtomN : 1;
    static constexpr int kTmaStoreN      = 64;
    static constexpr int kTmaStoreM      = kEpiM <= 256 ? kEpiM : 64;
    static constexpr int kTmaStoreCountM = kEpiM / kTmaStoreM;
    static constexpr int kSwizzleC       = 128;
    static constexpr int kEpiThreads     = kSplitEpiM ? WARPGROUP_SIZE : kMathGroupSize;
    static constexpr int kFragmentSize   = (kEpiM * kEpiN) / kEpiThreads;
    static constexpr int kEpiStripsM     = kWgM / kEpiM;
    static constexpr int kEpiStripsN     = TILE_N / kEpiN;
    static constexpr int kEpiPasses      = kEpiStripsM * kEpiStripsN;
    static_assert(TILE_N % kEpiN == 0);
    static_assert(!kSupportsFusedSilu || kRestM % 2 == 0 || kNeedsCrossWgSiluExchange);
    static_assert(TILE_M % kEpiM == 0);
    static_assert(kWgM % kEpiM == 0);
    static_assert(kEpiM % kTmaStoreM == 0);
    static_assert(kTmaStoreM <= 256);
    static_assert(kEpiN % kTmaStoreN == 0);
    static_assert(kFragmentSize >= 1);
    static_assert(!kNeedsCrossWgSiluExchange || (kAtomM == 2 && kAtomN == 1 && kEpiPlanes == 1 && kTmaStoreCountM == 1));
    using CrossWgSiluLayout = cute::Layout<cute::Shape<cute::Int<kEpiM>, cute::Int<kTmaStoreN>>, cute::Stride<cute::Int<kTmaStoreN>, cute::_1>>;

    using SmemLayoutAtomD = decltype(
        gmma_ss_smem_selector<cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::Int<kEpiN>, cute::Int<kEpiM>>());
    using SmemLayoutDPlane =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::_1{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    static constexpr int kEpiStageElems = cute::cosize_v<SmemLayoutDPlane> * kEpiPlanes;
    static_assert(!kNeedsCrossWgSiluExchange || cute::cosize_v<CrossWgSiluLayout> * (int)sizeof(float) == kEpiStageElems * (int)sizeof(Tc));

    // Retile atom (CUTLASS builder always uses STSM_N for C_atom); R2S is STSM_T (M-major).
    // Atom size = per-WG accumulator vals/thread: WG_1x2 tiles split GMMA-M, so per-WG
    // GMMA-N = TILE_M (TILE_M/2 vals); WG_2x1 splits GMMA-N too (TILE_M/4 vals). Matches the
    // old kCValsPerThread>=8 criterion on every pre-existing tile; TILE_M=8 WG_1x2 and
    // 16x256_2x1 get U32x2 / U16x4.
    static constexpr int kCValsPerThread = (kAtomN == 2) ? TILE_M / 4 : TILE_M / 2;
    static_assert(kCValsPerThread >= 4, "STSM needs at least U32x2 / 4 bf16 vals per thread");
    using CopyAtomC = typename detail::EpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyAtomC;
    using CopyOpR2S = typename detail::EpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyOpR2S;

    struct LayoutC {
        static constexpr int S0 = kEpiM;
        static constexpr int C0 = kTmaStoreN;
        static constexpr int C1 = 1;
    };

    template<int EpiPipeStages>
    struct SharedStorageT {
        cute::array_aligned<typename Traits::ElementA, cute::cosize_v<SmemLayoutA>> A;
        cute::array_aligned<typename Traits::ElementB, cute::cosize_v<SmemLayoutB>> B;
        cute::array_aligned<Tc, kEpiStageElems * EpiPipeStages, 1024> D;
        PipelineStorage                                                  pipeline;
        typename Scheduler::Storage                                      sched;
        StridedPtr                                                       gather_A;
        const int*                                                       gather_idxs;
        int                                                              gather_alive;
        int                                                              gather_k_iters;
        int                                                              gather_M_group;
        int                                                              gather_offset_m;
        volatile int                                                     gather_group_idx;
        volatile int                                                     gather_offset_n;
    };

    static constexpr int kSmemCapacity = 228 << 10;
    static constexpr int GetEpiPipeStages()
    {
        if constexpr (EpiStages_) {
            return EpiStages_;
        }
        else if constexpr (kEpiPasses >= 2 && sizeof(SharedStorageT<2>) <= kSmemCapacity) {
            return 2;
        }
        else {
            return 1;
        }
    }
    static constexpr int kEpiPipeStages = GetEpiPipeStages();
    static_assert(1 <= kEpiPipeStages && kEpiPipeStages <= kEpiPasses);
    static constexpr int kEpiSmemSlices = kEpiPlanes * kEpiPipeStages;
    using SmemLayoutD = decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::Int<kEpiSmemSlices>{}),
                                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    using EpiStageLayout = cute::Layout<cute::Shape<cute::Int<kEpiPlanes>, cute::Int<kEpiPipeStages>>, cute::Stride<cute::_1, cute::Int<kEpiPlanes>>>;
    static_assert(cute::cosize_v<SmemLayoutD> == kEpiStageElems * kEpiPipeStages);

    using SharedStorage = SharedStorageT<kEpiPipeStages>;
    static constexpr int kSmemSize = (int)sizeof(SharedStorage);
    static_assert(kSmemSize <= kSmemCapacity);

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
        if constexpr (!is_grouped_gemm) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_tma_descs_sm90_bf16<Mode>
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
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGROUPS * 4 + 1));
        }

        typename MainloopPipeline::Params pp;
        pp.transaction_bytes = (uint32_t)kTmaTxBytes;
        pp.num_consumers     = (uint32_t)kMathGroupSize;
        // Indexed: PTX Example 2 — init must include arrive-ons from each
        // cpasync_barrier_arrive_noinc (1 per gather thread) plus leader
        // arrive_and_expect_tx. Dense/blocked: TMA-only, count = 1.
        pp.num_producers     = (kStridingA == Striding::kIndexed) ? (1 + WARPGROUP_SIZE) : 1;
        pp.initializing_warp = 0;

        if (wg_idx == WARPGROUPS) {
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

        if (wg_idx == WARPGROUPS) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            const int  warp_id    = cutlass::canonical_warp_idx_sync();
            const int  warp_in_wg = warp_id % 4;
            const bool cta_0      = cute::block_id_in_cluster().x == 0;

            if constexpr (kStridingA == Striding::kIndexed) {
                Cluster cluster(cute::block_id_in_cluster().x);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);
                auto* smem_act    = storage.B.data();
                auto* smem_weight = storage.A.data() + mc_offset_n * TILE_K;
                PipelineState                     write_state = cutlass::make_producer_start_state<MainloopPipeline>();
                typename Scheduler::ConsumerState sched_state = sched.init_consumer(storage.sched);
                typename Scheduler::ProducerState prod_state  = sched.init_producer(storage.sched);
                int                               elected        = 0;
                const int                         lane_id        = threadIdx.x % WARP_SIZE;
                const int                         prod_tid       = threadIdx.x - WARPGROUPS * WARPGROUP_SIZE;
                if (warp_in_wg == 0) {
                    elected = cute::elect_one_sync();
                }
                const int K = sched.gemm_shape().z;
                constexpr int kGatherVectors = TILE_M * kGatherThreadsK;
                constexpr int kGatherSlots   = (kGatherVectors + WARPGROUP_SIZE - 1) / WARPGROUP_SIZE;
                typename Scheduler::Tile* tile;
                while (true) {
                    if (warp_in_wg == 0) {
                        if (cta_0) {
                            (void)prod_state.next();
                        }
                        const bool alive = sched_state.acquire(tile);
                        if (lane_id == 0) {
                            MatrixData a{{param_A.ptr, param_A.stride}, param_A.idxs};
                            storage.gather_alive     = alive ? 1 : 0;
                            storage.gather_k_iters   = 0;
                            storage.gather_M_group   = 0;
                            storage.gather_offset_m  = 0;
                            storage.gather_group_idx = 0;
                            storage.gather_offset_n  = 0;
                            if (alive && tile->is_valid_cluster) {
                                a = resolve<Ta, kStridingA>(param_A, tile->group_idx);
                                storage.gather_k_iters   = sched.k_iters_;
                                storage.gather_M_group   = is_grouped_gemm ? tile->m1 - tile->m0 : sched.gemm_shape().x;
                                storage.gather_offset_m  = tile->offset_m;
                                storage.gather_group_idx = tile->group_idx;
                                storage.gather_offset_n  = tile->offset_n;
                            }
                            storage.gather_A    = a.ptr;
                            storage.gather_idxs = a.idxs;
                        }
                        __syncwarp();
                    }
                    named_barrier_arrive_and_wait(WARPGROUP_SIZE, kProducerBarrierId);
                    if (storage.gather_alive == 0) {
                        break;
                    }
                    const int tile_k_iters = storage.gather_k_iters;
                    const int group_idx    = storage.gather_group_idx;
                    const int packed_m0    = storage.gather_offset_m;
                    const int row_count    = storage.gather_M_group - storage.gather_offset_m;
                    const Ta* act_gmem     = static_cast<const Ta*>(storage.gather_A.ptr);
                    const int ldA          = storage.gather_A.stride;
                    const int* idxs        = storage.gather_idxs;
                    const CUtensorMap* Bdesc = &tm_b;
                    if constexpr (is_grouped_gemm) {
                        Bdesc = tensormap_buf + group_idx * kTmaDescNum;
                    }
                    const int coord_n = storage.gather_offset_n + mc_offset_n;
                    const uint16_t mask_B = cluster.mask_n();

                    const Ta* gather_src[kGatherSlots];
                    typename Traits::ElementB* gather_dst[kGatherSlots];
                    int  gather_tile_k[kGatherSlots];
                    bool gather_pred[kGatherSlots];
                    bool gather_slot_valid[kGatherSlots];
                    if constexpr (TILE_M >= kGatherThreadsM && TILE_M % kGatherThreadsM == 0) {
                        static_assert(kGatherVectors % WARPGROUP_SIZE == 0);
                        auto gather_thr = GatherTiledCopy{}.get_slice(prod_tid);
                        auto smem_act_tensor = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
                        auto gather_smem = gather_thr.partition_D(smem_act_tensor);
                        auto identity = cute::make_identity_tensor(cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_K>>{});
                        auto gather_coord = gather_thr.partition_D(identity);
                        static_assert(cute::size<0>(gather_coord) == kGatherVec);
                        static_assert(cute::size<1>(gather_coord) == kGatherSlots);
                        static_assert(cute::size<2>(gather_coord) == 1);
                        PRAGMA_UNROLL
                        for (int slot = 0; slot < kGatherSlots; ++slot) {
                            const auto tile_coord = gather_coord(0, slot, 0);
                            const int tile_m      = cute::get<0>(tile_coord);
                            const int tile_k      = cute::get<1>(tile_coord);
                            const int packed_row  = packed_m0 + tile_m;
                            const bool row_valid  = tile_m < row_count;
                            const int source_row  = (idxs && row_valid) ? __ldg(idxs + packed_row) : packed_row;
                            gather_src[slot]        = act_gmem + (int64_t)source_row * ldA + tile_k;
                            gather_dst[slot]        = &gather_smem(0, slot, 0, 0);
                            gather_tile_k[slot]     = tile_k;
                            gather_pred[slot]       = row_valid;
                            gather_slot_valid[slot] = true;
                        }
                    }
                    else {
                        auto smem_act_tensor = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB_2D{});
                        PRAGMA_UNROLL
                        for (int slot = 0; slot < kGatherSlots; ++slot) {
                            const int vector_idx = prod_tid + slot * WARPGROUP_SIZE;
                            const bool slot_valid = vector_idx < kGatherVectors;
                            const int tile_m = slot_valid ? vector_idx / kGatherThreadsK : 0;
                            const int tile_k = slot_valid ? (vector_idx % kGatherThreadsK) * kGatherVec : 0;
                            const int packed_row = packed_m0 + tile_m;
                            const bool row_valid = slot_valid && tile_m < row_count;
                            const int source_row = (idxs && row_valid) ? __ldg(idxs + packed_row) : packed_row;
                            gather_src[slot]        = act_gmem + (int64_t)source_row * ldA + tile_k;
                            gather_dst[slot]        = &smem_act_tensor(tile_m, tile_k);
                            gather_tile_k[slot]     = tile_k;
                            gather_pred[slot]       = row_valid;
                            gather_slot_valid[slot] = slot_valid;
                        }
                    }

                    for (int k_tile = 0; k_tile < tile_k_iters; ++k_tile) {
                        pipeline.producer_acquire(write_state);
                        auto* bar = pipeline.producer_get_barrier(write_state);
                        const int stage = write_state.index();
                        if (warp_in_wg == 0 && elected) {
                            detail::tma_load_with_barrier<kMulticastB, TILE_N / kMulticastB, TILE_K>(Bdesc, bar, smem_weight + stage * TILE_N * TILE_K, k_tile * TILE_K, coord_n, mask_B, kWeightL2Policy);
                        }
                        PRAGMA_UNROLL
                        for (int slot = 0; slot < kGatherSlots; ++slot) {
                            if constexpr (kGatherVectors % WARPGROUP_SIZE) {
                                if (!gather_slot_valid[slot]) {
                                    continue;
                                }
                            }
                            auto* dst = gather_dst[slot] + stage * TILE_M * TILE_K;
                            const bool pred = gather_pred[slot] && k_tile * TILE_K + gather_tile_k[slot] < K;
                            cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(*reinterpret_cast<const uint4*>(gather_src[slot]), *reinterpret_cast<uint4*>(dst), pred);
                            gather_src[slot] += TILE_K;
                        }
                        cutlass::arch::cpasync_barrier_arrive_noinc(bar);
                        ++write_state;
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
                    if (elected) {
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

                                CUTE_UNROLL
                                for (int tma_m = 0; tma_m < kTmaCountM; ++tma_m) {
                                    detail::tma_load_with_barrier<kMulticastA, kTmaBoxM, TILE_K>(
                                        Adesc,
                                        bar,
                                        smem_act + (mc_offset_m + tma_m * kTmaBoxM) * TILE_K + pipe * TILE_M * TILE_K,
                                        coord_k,
                                        coord_m + tma_m * kTmaBoxM,
                                        mask_A);
                                }
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
                            named_barrier_arrive_unaligned(WARP_SIZE * 2, kProducerBarrierId);
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
                        named_barrier_arrive_and_wait_unaligned(WARP_SIZE * 2, kProducerBarrierId);
                    }
                }
                sched.tail(state);
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<kMathRegs>();

            // mma_tid in [0, kMathGroupSize) — math WGs share one TiledMma TV (cooperative).
            const int mma_tid   = threadIdx.x;
            const int local_tid = mma_tid % WARPGROUP_SIZE;

            cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), SmemLayoutA{});
            cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});

            TiledMma tiled_mma;
            auto     thr_mma = tiled_mma.get_thread_slice(mma_tid);

            cute::Tensor tCsA = thr_mma.partition_A(sA);
            cute::Tensor tCsB = thr_mma.partition_B(sB);
            cute::Tensor tCrA = thr_mma.make_fragment_A(tCsA);
            cute::Tensor tCrB = thr_mma.make_fragment_B(tCsB);

            CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::Int<kRestM>{});
            // Native segment = rest_m * kAtomM + wg_m. Fused segment = wg_m * kRestM + rest_m.
            const int wg_m = cutlass::canonical_warp_group_idx() % kAtomM;
            auto a_segment_stride = cute::stride<1>(tCrA.layout()) / cute::Int<kAtomM>{};
            auto contiguous_a_layout = cute::make_layout(cute::shape(tCrA), cute::replace<1>(cute::stride(tCrA), a_segment_stride));
            auto tCrA_contiguous = cute::make_tensor(tCrA.data() + wg_m * (kRestM - 1) * a_segment_stride, contiguous_a_layout);

            PipelineState pipe_state{};
            PipelineState pipe_release = pipe_state;

            auto sched_state = sched.init_consumer(storage.sched);

            typename Scheduler::Tile* tile;
            sched_state.acquire(tile);

            // CUTLASS Sm90TmaWarpSpecialized R2S: as_position_independent +
            // make_tiled_copy_C_atom → STSM_T; warp0 TMA after fence + EpilogueBarrier.
            cute::Tensor sD = cute::as_position_independent_swizzle_tensor(
                cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));

            CopyAtomC copy_atom_c{};
            using EpiTiledMma = std::conditional_t<kSplitEpiM, typename Traits::WgTiledMma, TiledMma>;
            EpiTiledMma epi_tiled_mma;
            auto tiled_copy_C_atom = cute::make_tiled_copy_C_atom(copy_atom_c, epi_tiled_mma);
            auto tiled_r2s = cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, cutlass::bfloat16_t>{}, tiled_copy_C_atom);
            auto thr_r2s = tiled_r2s.get_slice(kSplitEpiM ? local_tid : mma_tid);
            auto         tRS_rD_layout = cute::make_layout(cute::take<0, 3>(cute::shape(thr_r2s.partition_S(sD))));
            const int  tma_store_warp   = mma_tid / WARP_SIZE;
            const bool tma_store_leader = cute::elect_one_sync();
            int        epi_store_count  = 0;

            while (tile->alive) {

                if (tile->is_valid_cta) {
                    cute::Tensor accum =
                        cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    cute::clear(accum);

                    auto run_mainloop = [&](auto const& tCrA_) {
                        if constexpr (Traits::kMmaNSlices > 1 || SeparateMmaAtoms) {
                            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                            cute::warpgroup_fence_operand(accum);
                            const int k_iters = sched.k_iters_;
                            for (int k_tile = 0; k_tile < k_iters; ++k_tile) {
                                auto token = pipeline.consumer_try_wait(pipe_state);
                                pipeline.consumer_wait(pipe_state, token);
                                const int read = pipe_state.index();
                                ++pipe_state;
                                CUTE_UNROLL
                                for (int k_block = 0; k_block < cute::size<2>(tCrA_); ++k_block) {
                                    MmaIssue::run(tiled_mma, tCrA_(cute::_, cute::_, k_block, read), tCrB(cute::_, cute::_, k_block, read), accum);
                                    if (k_block == 1 && k_tile > 0) {
                                        pipeline.consumer_release(pipe_release);
                                        ++pipe_release;
                                    }
                                }
                                cute::warpgroup_fence_operand(accum);
                            }
                            cute::warpgroup_wait<0>();
                            cute::warpgroup_fence_operand(accum);
                            pipeline.consumer_release(pipe_release);
                            ++pipe_release;
                        }
                        else {
                            int k_iter = sched.k_iters_;
                            {
                                auto token = pipeline.consumer_try_wait(pipe_state);
                                pipeline.consumer_wait(pipe_state, token);
                                const int read = pipe_state.index();
                                cute::warpgroup_fence_operand(accum);
                                cute::warpgroup_arrive();
                                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                                CUTE_UNROLL
                                for (int k_block = 0; k_block < cute::size<2>(tCrA_); ++k_block) {
                                    cute::gemm(tiled_mma, tCrA_(cute::_, cute::_, k_block, read), tCrB(cute::_, cute::_, k_block, read), accum);
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
                                cute::gemm(tiled_mma, tCrA_(cute::_, cute::_, cute::_, read), tCrB(cute::_, cute::_, cute::_, read), accum);
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
                        }
                    };

                    if constexpr (kUsesContiguousFusedMapping) {
                        if (fuse_silu) {
                            run_mainloop(tCrA_contiguous);
                        }
                        else {
                            run_mainloop(tCrA);
                        }
                    }
                    else {
                        run_mainloop(tCrA);
                    }

                    const void* Cdesc = &tm_c;
                    if constexpr (is_grouped_gemm) {
                        Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kTmaDescNumAB;
                    }

                    cute::Tensor tRS_rAcc = thr_r2s.retile_S(accum);  // ((R2S,R2S_V),MMA_M,MMA_N)
                    cute::Tensor tRS_rD   = cute::make_tensor<cutlass::bfloat16_t>(tRS_rD_layout);

                    cute::Tensor tRS_rAcc_frg = cute::recast<cutlass::Array<float, kFragmentSize>>(tRS_rAcc);
                    cute::Tensor tRS_rD_frg = cute::recast<cutlass::Array<cutlass::bfloat16_t, kFragmentSize>>(tRS_rD);

                    constexpr int kMmaTileN = cute::size<0>(typename Traits::TileShape{}) / cute::size<1>(decltype(tRS_rAcc){});
                    constexpr int kMmaTileM = (kSplitEpiM ? kWgM : cute::size<1>(typename Traits::TileShape{})) / cute::size<2>(decltype(tRS_rAcc){});
                    constexpr int kTmaStoreCountN = kEpiN / kTmaStoreN;
                    (void)kMmaTileN;

                    auto run_epilogue = [&](auto fused_silu) {
                        constexpr bool kFuseSilu = decltype(fused_silu)::value;
                        static_assert(!kFuseSilu || kSupportsFusedSilu);
                        constexpr bool kCrossWgSilu = kFuseSilu && kNeedsCrossWgSiluExchange;
                        constexpr int kStoreN    = kFuseSilu ? TILE_N / 2 : TILE_N;
                        constexpr int kEpiCountN = kCrossWgSilu ? 1 : kStoreN / kEpiN;
                        static_assert(kCrossWgSilu || kStoreN % kEpiN == 0);
                        static_assert(!kCrossWgSilu || (kAtomM == 2 && kAtomN == 1 && kEpiN == 2 * kTmaStoreN && kEpiPlanes == 1));

                        auto epi_synchronize = [&] {
                            if constexpr (kCrossWgSilu) {
                                named_barrier_arrive_and_wait(kMathGroupSize, kEpilogueBarrierId);
                            }
                            else {
                                constexpr int kWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
                                const int barrier_id = kEpilogueBarrierId + tma_store_warp / kWarpsPerWg;
                                named_barrier_arrive_and_wait(WARPGROUP_SIZE, barrier_id);
                            }
                        };

                        auto epi_pass_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiStripsM>{}, cute::Int<kEpiCountN>{}), cute::make_stride(cute::Int<kEpiCountN>{}, cute::_1{}));
                        auto store_offset_m_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiPlanes>{}, cute::Int<kEpiStripsM>{}, cute::Int<kTmaStoreCountM>{}), cute::make_stride(cute::Int<kWgM>{}, cute::Int<kEpiM>{}, cute::Int<kTmaStoreM>{}));
                        auto cEpiNM     = cute::make_identity_tensor(cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}));
                        auto tRS_cEpiNM = thr_r2s.partition_S(cEpiNM);
                        auto r2s_coord_layout = cute::make_layout(cute::make_shape(cute::Int<kFragmentSize>{}, cute::size(tRS_rD_frg)));
                        auto r2s_value_layout = cute::make_layout(cute::make_shape(cute::size(tRS_rD_frg), cute::Int<kMmaTileM / kEpiM>{}));
                        constexpr int kTmaStoreWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
                        const int store_wg = tma_store_warp / kTmaStoreWarpsPerWg;

                        CUTE_UNROLL
                        for (int epi_m = 0; epi_m < kEpiStripsM; ++epi_m) {
                            CUTE_UNROLL
                            for (int epi_n = 0; epi_n < kEpiCountN; ++epi_n) {
                                const int epi_pass = epi_pass_layout(epi_m, epi_n);
                                const int epi_stage = epi_store_count % kEpiPipeStages;
                                const int mma_n = kFuseSilu && !kCrossWgSilu ? 2 * epi_n : epi_n;
                                const int mma_m = (epi_m * kEpiM) / kMmaTileM;
                                const int epi_m_in_mma = epi_m % (kMmaTileM / kEpiM);
                                const int r2s_v = r2s_value_layout(0, epi_m_in_mma);
                                const int epi_smem_slice = EpiStageLayout{}(kSplitEpiM ? store_wg : 0, epi_stage);
                                auto sD_epi = sD(cute::_, cute::_, epi_smem_slice);
                                auto tRS_sD = thr_r2s.partition_D(sD_epi);

                                if constexpr (kCrossWgSilu) {
                                    if (tma_store_leader) {
                                        cute::tma_store_wait<kEpiPipeStages - 1>();
                                    }
                                    epi_synchronize();
                                    auto silu_exchange = cute::make_tensor(reinterpret_cast<float*>(storage.D.data() + epi_smem_slice * cute::cosize_v<SmemLayoutDPlane>), CrossWgSiluLayout{});
                                    if (store_wg == 1) {
                                        CUTE_UNROLL
                                        for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                            auto up = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                            CUTE_UNROLL
                                            for (int j = 0; j < kFragmentSize; ++j) {
                                                const auto coord_nm = tRS_cEpiNM(r2s_coord_layout(j, epi_v));
                                                const int n = cute::get<0>(coord_nm) - kTmaStoreN;
                                                const int m = cute::get<1>(coord_nm);
                                                silu_exchange(m, n) = up[j];
                                            }
                                        }
                                    }
                                    epi_synchronize();
                                }

                                if constexpr (!kCrossWgSilu) {
                                    CUTE_UNROLL
                                    for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                        cutlass::Array<cutlass::bfloat16_t, kFragmentSize> dst;
                                        if constexpr (kFuseSilu) {
                                            auto gate = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                            auto up = tRS_rAcc_frg(cute::_, mma_n + 1, mma_m)(r2s_v + epi_v);
                                            CUTE_UNROLL
                                            for (int j = 0; j < kFragmentSize; ++j) {
                                                dst[j] = cutlass::bfloat16_t(detail::silu_mul(gate[j], up[j]));
                                            }
                                        }
                                        else {
                                            auto src = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                            CUTE_UNROLL
                                            for (int j = 0; j < kFragmentSize; ++j) {
                                                dst[j] = cutlass::bfloat16_t(src[j]);
                                            }
                                        }
                                        tRS_rD_frg(epi_v) = dst;
                                    }
                                }
                                else if (store_wg == 0) {
                                    auto silu_exchange = cute::make_tensor(reinterpret_cast<float*>(storage.D.data() + epi_smem_slice * cute::cosize_v<SmemLayoutDPlane>), CrossWgSiluLayout{});
                                    CUTE_UNROLL
                                    for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                        cutlass::Array<cutlass::bfloat16_t, kFragmentSize> dst;
                                        auto gate = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                        CUTE_UNROLL
                                        for (int j = 0; j < kFragmentSize; ++j) {
                                            const auto coord_nm = tRS_cEpiNM(r2s_coord_layout(j, epi_v));
                                            const int n = cute::get<0>(coord_nm);
                                            const int m = cute::get<1>(coord_nm);
                                            dst[j] = cutlass::bfloat16_t(detail::silu_mul(gate[j], silu_exchange(m, n)));
                                        }
                                        tRS_rD_frg(epi_v) = dst;
                                    }
                                }

                                if constexpr (!kCrossWgSilu) {
                                    if (tma_store_leader) {
                                        cute::tma_store_wait<kEpiPipeStages - 1>();
                                    }
                                    epi_synchronize();
                                    cute::copy(tiled_r2s, tRS_rD, tRS_sD);
                                }
                                else {
                                    // D aliases the float exchange tile. All WG0 threads must finish reading
                                    // the up fragment before any warp overwrites that storage with BF16 STSM.
                                    epi_synchronize();
                                    if (store_wg == 0) {
                                        cute::copy(tiled_r2s, tRS_rD, tRS_sD);
                                    }
                                }
                                cutlass::arch::fence_view_async_shared();
                                epi_synchronize();

                                if (tma_store_leader) {
                                    if constexpr (kCrossWgSilu) {
                                        constexpr int kTmaStoreWarps = kMathGroupSize / WARP_SIZE;
                                        if (tma_store_warp == epi_pass % kTmaStoreWarps) {
                                            const int store_m = tile->offset_m + store_offset_m_layout(0, epi_m, 0);
                                            const int store_n = tile->offset_n / 2;
                                            auto sD_tma = cute::local_tile(sD(cute::_, cute::_, EpiStageLayout{}(0, epi_stage)), cute::make_shape(cute::Int<kTmaStoreN>{}, cute::Int<kTmaStoreM>{}), cute::make_coord(0, 0));
                                            cute::SM90_TMA_STORE::copy(Cdesc, cute::raw_pointer_cast(sD_tma.data()), store_n, store_m);
                                        }
                                    }
                                    else {
                                        constexpr int kStoreTmaCountN = kTmaStoreCountN;
                                        auto store_offset_n_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiCountN>{}, cute::Int<kStoreTmaCountN>{}), cute::make_stride(cute::Int<kEpiN>{}, cute::Int<kTmaStoreN>{}));
                                        static_assert(kEpiPlanes * kStoreTmaCountN == WARPGROUPS);
                                        static_assert(kTmaStoreCountM <= kTmaStoreWarpsPerWg);
                                        auto store_tile_layout = cute::make_layout(cute::make_shape(cute::Int<kStoreTmaCountN>{}, cute::Int<kEpiPlanes>{}));
                                        const int warp_in_wg = tma_store_warp % kTmaStoreWarpsPerWg;
                                        const int first_warp = (epi_pass * kTmaStoreCountM) % kTmaStoreWarpsPerWg;
                                        const int tma_m = (warp_in_wg + kTmaStoreWarpsPerWg - first_warp) % kTmaStoreWarpsPerWg;
                                        if (tma_m < kTmaStoreCountM) {
                                            const auto store_tile_coord = store_tile_layout.get_flat_coord(store_wg);
                                            const int tma_n = cute::get<0>(store_tile_coord);
                                            const int epi_plane = cute::get<1>(store_tile_coord);
                                            const int store_m = tile->offset_m + store_offset_m_layout(epi_plane, epi_m, tma_m);
                                            const int store_n = (kFuseSilu ? tile->offset_n / 2 : tile->offset_n) + store_offset_n_layout(epi_n, tma_n);
                                            auto sD_tma = cute::local_tile(sD(cute::_, cute::_, EpiStageLayout{}(epi_plane, epi_stage)), cute::make_shape(cute::Int<kTmaStoreN>{}, cute::Int<kTmaStoreM>{}), cute::make_coord(tma_n, tma_m));
                                            cute::SM90_TMA_STORE::copy(Cdesc, cute::raw_pointer_cast(sD_tma.data()), store_n, store_m);
                                        }
                                    }
                                    cute::tma_store_arrive();
                                }
                                ++epi_store_count;
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

            if (tma_store_leader) {
                cute::tma_store_wait<0>();
            }
        }
    }
};

}  // namespace turbomind::gemm
