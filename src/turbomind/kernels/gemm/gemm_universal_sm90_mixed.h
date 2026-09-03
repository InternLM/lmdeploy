// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

/*
 * SM90 BF16 x prepacked mixed-precision GEMM.
 *
 * Public problem:
 *   A: BF16 activation, row-major [BATCH, K]
 *   B: packed U4, MXFP4, NVFP4, or E4M3 weight, col-major logical [K, OUT]
 *   V: format-specific K-group metadata
 *   C: BF16 output, row-major [BATCH, OUT]
 *
 * Hardware problem:
 *   WGMMA operand A = dequantized weight RS fragment [OUT, K64]
 *   WGMMA operand B = activation descriptor [BATCH, K64]
 *
 * The persistent weight layout is [K/16][OUT/64][RS fragment]. TMA projects
 * the CTA's K16 and OUT64 fragment tile into shared memory. A fused WG_1x2
 * consumer assigns `wg_m * RestM + rest_m`, so each math WG receives a
 * contiguous [gate64|up64] range. Other consumers retain CuTe's native
 * `rest_m * AtomM + wg_m` assignment.
 */

#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/gmma_issue.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/sm90_mixed_dequant.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_traits.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace detail {

template<int Multicast, int BoxMN, int BoxK, class Element>
__device__ void
mixed_tma_load_with_barrier(const cute::TmaDescriptor* desc,
                            uint64_t*                  bar,
                            Element*                   smem,
                            int                        crd0,
                            int                        crd1,
                            uint16_t                   mcast_mask)
{
    constexpr int kNumBits = BoxMN * BoxK * (int)cute::sizeof_bits_v<Element>;
    constexpr int kNumVals = BoxMN * BoxK;

    using Aux = cute::
        AuxTmaParams<cute::Stride<cute::_1, cute::_1>, cute::Layout<cute::Shape<cute::_1>>, cute::Swizzle<0, 4, 3>>;
    auto g = cute::make_tensor(cute::make_inttuple_iter(crd0, crd1), cute::Layout<cute::Int<kNumVals>>{});
    auto s = cute::make_tensor(cute::make_smem_ptr(smem), cute::Layout<cute::Int<kNumVals>>{});

    if constexpr (Multicast > 1) {
        using Traits = cute::Copy_Traits<cute::SM90_TMA_LOAD_MULTICAST, cute::Int<kNumBits>, Aux>;
        using Atom   = cute::Copy_Atom<Traits, Element>;
        Atom tma{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(tma.with(desc, *bar, mcast_mask, cute::TMA::CacheHintSm90::EVICT_NORMAL), g, s);
    }
    else {
        using Traits = cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::Int<kNumBits>, Aux>;
        using Atom   = cute::Copy_Atom<Traits, Element>;
        Atom tma{Traits{cute::TmaDescriptor{}, Aux{}}};
        cute::copy(tma.with(desc, *bar, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL), g, s);
        (void)mcast_mask;
    }
}

// One word contains eight U4 values in operand-A fragment order.  LOP3
// materializes four BF16x2 pairs biased by +128; metadata stores
// {scale, effective_zero=zero+128}.
__device__ __forceinline__ void u4_unpack_dequant(uint32_t     packed,
                                                  uint32_t     scale_lo_pair,
                                                  uint32_t     zero_lo_pair,
                                                  uint32_t     scale_hi_pair,
                                                  uint32_t     zero_hi_pair,
                                                  nv_bfloat16* out)
{
    constexpr uint32_t kBf16x2_128 = 0x43004300u;
    constexpr uint32_t kNibbleMask = 0x000f000fu;
    constexpr uint32_t kLut        = (0xf0 & 0xcc) | 0xaa;

    auto* h = reinterpret_cast<uint32_t*>(out);
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(h[0])
                 : "r"(packed), "n"(kNibbleMask), "n"(kBf16x2_128), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(h[1])
                 : "r"(packed >> 4), "n"(kNibbleMask), "n"(kBf16x2_128), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(h[2])
                 : "r"(packed >> 8), "n"(kNibbleMask), "n"(kBf16x2_128), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(h[3])
                 : "r"(packed >> 12), "n"(kNibbleMask), "n"(kBf16x2_128), "n"(kLut));

    auto& scale_lo = reinterpret_cast<const nv_bfloat162&>(scale_lo_pair);
    auto& zero_lo  = reinterpret_cast<const nv_bfloat162&>(zero_lo_pair);
    auto& scale_hi = reinterpret_cast<const nv_bfloat162&>(scale_hi_pair);
    auto& zero_hi  = reinterpret_cast<const nv_bfloat162&>(zero_hi_pair);
    auto* h2       = reinterpret_cast<nv_bfloat162*>(out);

    h2[0] = __hmul2(__hsub2(h2[0], zero_lo), scale_lo);
    h2[1] = __hmul2(__hsub2(h2[1], zero_hi), scale_hi);
    h2[2] = __hmul2(__hsub2(h2[2], zero_lo), scale_lo);
    h2[3] = __hmul2(__hsub2(h2[3], zero_hi), scale_hi);
}

// Decode two direct and two nibble-rotated E4M3 pair planes into the BF16 register fragment consumed by one
// m64n*k16 RS WGMMA operand-A lane, then apply the B128 weight scale.
__device__ __forceinline__ void fp8_e4m3_unpack_dequant(const uint32_t* packed, uint32_t scale_pair, nv_bfloat16* out)
{
    constexpr uint32_t kDirectPairMask = 0x87f087f0u;
    constexpr uint32_t kShiftLeftMask  = 0x080f080fu;
    constexpr uint32_t kShiftRightMask = 0x70007000u;
    constexpr uint32_t kNormalize      = (127u - 7u + 127u) << 7u;
    constexpr uint32_t kNormalizePair  = kNormalize | (kNormalize << 16);
    static_assert((kDirectPairMask | kShiftLeftMask | kShiftRightMask) == 0xffffffffu);

    auto* h = reinterpret_cast<uint32_t*>(out);
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
        const uint32_t x = packed[i];
        h[i]             = x & kDirectPairMask;
        h[i + 2]         = ((x & kShiftLeftMask) << 4) | ((x & kShiftRightMask) >> 4);
    }

    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h[i]) : "r"(h[i]), "r"(kNormalizePair));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h[i]) : "r"(h[i]), "r"(scale_pair));
    }
}

template<>
struct Sm90MixedDequant<Sm90U4Format> {
    static constexpr int kWordsPerThreadKBlock = 1;

    struct SharedStorage {};

    __device__ static void init(SharedStorage&) {}

    __device__ static int packed_lane(int local_tid)
    {
        return local_tid;
    }

    template<int RestM>
    struct Registers {
        uint32_t scale_lo_pair[RestM]{};
        uint32_t zero_lo_pair[RestM]{};
        uint32_t scale_hi_pair[RestM]{};
        uint32_t zero_hi_pair[RestM]{};
    };

    template<int RestM, int AtomM, int TileOut>
    __device__ static void load(Registers<RestM>& regs, const uint8_t* smem, int segment_base, int segment_stride, int local_tid)
    {
        static_assert(TileOut == RestM * AtomM * 64);
        const auto* q = reinterpret_cast<const uint32_t*>(smem);

        CUTE_UNROLL
        for (int rest_m = 0; rest_m < RestM; ++rest_m) {
            const int      segment     = segment_base + rest_m * segment_stride;
            const int      pair        = local_tid / 4;
            const uint2    lo_hi       = reinterpret_cast<const uint2*>(q + segment * 64)[pair];
            const uint32_t lo          = lo_hi.x;
            const uint32_t hi          = lo_hi.y;
            const uint32_t s_lo        = lo & 0xffffu;
            const uint32_t z_lo        = lo >> 16;
            const uint32_t s_hi        = hi & 0xffffu;
            const uint32_t z_hi        = hi >> 16;
            regs.scale_lo_pair[rest_m] = s_lo | (s_lo << 16);
            regs.zero_lo_pair[rest_m]  = z_lo | (z_lo << 16);
            regs.scale_hi_pair[rest_m] = s_hi | (s_hi << 16);
            regs.zero_hi_pair[rest_m]  = z_hi | (z_hi << 16);
        }
    }

    template<int RestM>
    __device__ static void
    dequant(const uint32_t*      packed,
            const Registers<RestM>& regs,
            int                  rest_m,
            int /*kb*/,
            const SharedStorage&,
            nv_bfloat16* out)
    {
        u4_unpack_dequant(packed[0],
                          regs.scale_lo_pair[rest_m],
                          regs.zero_lo_pair[rest_m],
                          regs.scale_hi_pair[rest_m],
                          regs.zero_hi_pair[rest_m],
                          out);
    }
};

template<>
struct Sm90MixedDequant<Sm90MxFp4Format> {
    static constexpr int kWordsPerThreadKBlock = 1;
    static constexpr int kScaleGroups          = kSm90MixedTileK / Sm90MxFp4Format::kGroupSize;
    static_assert(kScaleGroups == 2);

    struct SharedStorage {};

    __device__ static void init(SharedStorage&) {}

    __device__ static int packed_lane(int local_tid)
    {
        return local_tid;
    }

    template<int RestM>
    struct Registers {
        uint16_t exponent_pair[kScaleGroups][RestM]{};
    };

    template<int RestM, int AtomM, int TileOut>
    __device__ static void load(Registers<RestM>& regs, const uint8_t* q, int segment_base, int segment_stride, int local_tid)
    {
        static_assert(TileOut == RestM * AtomM * 64);
        const int pair = local_tid / 4;

        CUTE_UNROLL
        for (int group = 0; group < kScaleGroups; ++group) {
            CUTE_UNROLL
            for (int rest_m = 0; rest_m < RestM; ++rest_m) {
                const int   segment  = segment_base + rest_m * segment_stride;
                const auto* fragment = q + group * TileOut + segment * kSm90MixedFragmentN;
                regs.exponent_pair[group][rest_m] = reinterpret_cast<const uint16_t*>(fragment)[pair];
            }
        }
    }

    template<int RestM>
    __device__ static void
    dequant(const uint32_t*      packed,
            const Registers<RestM>& regs,
            int                  rest_m,
            int                  kb,
            const SharedStorage&,
            nv_bfloat16* out)
    {
        const int group = kb / (Sm90MxFp4Format::kGroupSize / 16);
        e2m1_prmt_unpack_scaled(packed[0], regs.exponent_pair[group][rest_m], out);
    }
};

template<>
struct Sm90MixedDequant<Sm90Fp8E4M3Format> {
    static constexpr int kWordsPerThreadKBlock = 2;

    struct SharedStorage {};

    __device__ static void init(SharedStorage&) {}

    __device__ static int packed_lane(int local_tid)
    {
        return local_tid;
    }

    template<int RestM>
    struct Registers {
        uint32_t scale_pair[RestM]{};
    };

    template<int RestM, int AtomM, int TileOut>
    __device__ static void load(Registers<RestM>& regs, const uint8_t* smem, int segment_base, int segment_stride, int /*local_tid*/)
    {
        static_assert(TileOut == RestM * AtomM * 64);
        const auto* scale = reinterpret_cast<const uint16_t*>(smem);
        CUTE_UNROLL
        for (int rest_m = 0; rest_m < RestM; ++rest_m) {
            const int      segment  = segment_base + rest_m * segment_stride;
            const uint16_t value    = scale[segment * Sm90Fp8E4M3Format::kQparamValuesFragment];
            regs.scale_pair[rest_m] = uint32_t(value) | (uint32_t(value) << 16);
        }
    }

    template<int RestM>
    __device__ static void
    dequant(const uint32_t*      packed,
            const Registers<RestM>& regs,
            int                  rest_m,
            int /*kb*/,
            const SharedStorage&,
            nv_bfloat16* out)
    {
        fp8_e4m3_unpack_dequant(packed, regs.scale_pair[rest_m], out);
    }
};

__device__ __forceinline__ float mixed_silu_mul(float gate, float up)
{
    return fdividef(gate, 1.f + expf(-gate)) * up;
}

template<int kStsmVals>
struct MixedEpiStsmAtoms {
    static_assert(kStsmVals >= 8);
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x8_STSM_T;
};

template<>
struct MixedEpiStsmAtoms<4> {
    using CopyAtomC = cute::Copy_Atom<cute::SM90_U32x2_STSM_N, cutlass::half_t>;
    using CopyOpR2S = cute::SM90_U16x4_STSM_T;
};

// Packed weights and qparams use StridedPtr tables for grouped GEMM.  Routing
// offsets describe activation rows and must never be applied to these
// nonlinear packed byte streams.
__device__ __forceinline__ StridedPtr resolve_mixed_group_ptr(const MatrixParam& param, int group_idx)
{
    StridedPtr ptr{param.ptr, param.stride};
    if (ptr.stride == 0) {
        reinterpret_cast<uint4&>(ptr) = __ldg(reinterpret_cast<const uint4*>(param.ptr) + group_idx);
    }
    return ptr;
}

__device__ __forceinline__ void copy_mixed_tma_desc(CUtensorMap* dst, const CUtensorMap* src, int lane)
{
    constexpr int kWords = (int)(sizeof(CUtensorMap) / sizeof(uint2));
    if (lane < kWords) {
        reinterpret_cast<uint2*>(dst)[lane] = reinterpret_cast<const uint2*>(src)[lane];
    }
}

__device__ __forceinline__ void
replace_mixed_tma_addr_dim1_stride(CUtensorMap* desc, void* global_addr, int dim1, uint64_t stride_bytes)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(desc);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :
                 : "r"(uint_ptr), "l"(global_addr));
    if (dim1 >= 0) {
        asm volatile(
            "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" : : "r"(uint_ptr), "r"(dim1));
    }
    if (stride_bytes) {
        replace_tma_global_stride(desc, stride_bytes);
    }
}

__device__ __forceinline__ void publish_mixed_tma_desc(CUtensorMap* gmem_desc, CUtensorMap* smem_desc)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(smem_desc);
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned "
                 "[%0], [%1], 128;"
                 :
                 : "l"(gmem_desc), "r"(uint_ptr));
}

template<int N>
__device__ __forceinline__ void rebase_publish_mixed_tma_descs(CUtensorMap*                 gmem_out,
                                                               CUtensorMap*                 smem_desc,
                                                               Array<const CUtensorMap*, N> templates,
                                                               Array<void*, N>              global_addrs,
                                                               Array<int, N>                dims,
                                                               Array<uint64_t, N>           strides,
                                                               int                          lane)
{
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        copy_mixed_tma_desc(&smem_desc[i], templates[i], lane);
    }
    __syncwarp();
    if (lane == 0) {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            replace_mixed_tma_addr_dim1_stride(&smem_desc[i], global_addrs[i], dims[i], strides[i]);
        }
    }
    __syncwarp();
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        publish_mixed_tma_desc(&gmem_out[i], &smem_desc[i]);
    }
    __syncwarp();
}

template<class Tile, class = void>
struct MixedMmaN {
    static constexpr int value = 0;
};

template<class Tile>
struct MixedMmaN<Tile, std::void_t<decltype(Tile::kMmaN)>> {
    static constexpr int value = Tile::kMmaN;
};

template<class Tile, class = void>
struct MixedSeparateMmaAtoms {
    static constexpr bool value = false;
};

template<class Tile>
struct MixedSeparateMmaAtoms<Tile, std::void_t<decltype(Tile::kSeparateMmaAtoms)>> {
    static constexpr bool value = Tile::kSeparateMmaAtoms;
};

template<class Tile, class = void>
struct MixedEpiM {
    static constexpr int value = 0;
};

template<class Tile>
struct MixedEpiM<Tile, std::void_t<decltype(Tile::kEpiM)>> {
    static constexpr int value = Tile::kEpiM;
};

template<class Tile, class = void>
struct MixedEpiPipeStages {
    static constexpr int value = 0;
};

template<class Tile>
struct MixedEpiPipeStages<Tile, std::void_t<decltype(Tile::kEpiPipeStages)>> {
    static constexpr int value = Tile::kEpiPipeStages;
};

}  // namespace detail

// Grouped descriptor preparation. Every mixed grouped instantiation publishes
// [A, packed B, V, C]. Indexed kernels normally gather A, but blocked/flat
// descriptors dispatched through the same instantiation use A's affine map.
template<Striding kStridingA>
__global__ void __launch_bounds__(32, 1) prepare_tma_descs_sm90_mixed(const __grid_constant__ CUtensorMap tm_a,
                                                                      const __grid_constant__ CUtensorMap tm_b,
                                                                      const __grid_constant__ CUtensorMap tm_v,
                                                                      const __grid_constant__ CUtensorMap tm_c,
                                                                      MatrixParam                         param_A,
                                                                      MatrixParam                         param_B,
                                                                      MatrixParam                         param_V,
                                                                      MatrixParam                         param_C,
                                                                      CUtensorMap*                        out,
                                                                      int*                                offsets,
                                                                      int                                 M_total)
{
    static_assert(kStridingA == Striding::kBlocked || kStridingA == Striding::kIndexed);
    constexpr int kNum = 4;
    __shared__ __align__(128) CUtensorMap smem_desc[kNum];

    const int g      = (int)blockIdx.x;
    const int lane   = (int)threadIdx.x & 31;
    const int m0     = param_A.offsets ? __ldg(param_A.offsets + g) : 0;
    const int m1     = param_A.offsets ? __ldg(param_A.offsets + g + 1) : M_total;
    const int M      = m1 - m0;
    const int M_desc = M > 0 ? M : 1;

    if (lane == 0) {
        offsets[g] = m0;
        if (g + 1 == gridDim.x) {
            offsets[g + 1] = m1;
        }
    }

    const auto a = resolve<nv_bfloat16, kStridingA>(param_A, g);
    const auto b = detail::resolve_mixed_group_ptr(param_B, g);
    const auto v = detail::resolve_mixed_group_ptr(param_V, g);
    const auto c = resolve<nv_bfloat16, Striding::kBlocked>(param_C, g);

    Array<const CUtensorMap*, 4> templates;
    templates[0] = &tm_a;
    templates[1] = &tm_b;
    templates[2] = &tm_v;
    templates[3] = &tm_c;
    Array<void*, 4> addrs;
    addrs[0] = a.ptr.ptr;
    addrs[1] = b.ptr;
    addrs[2] = v.ptr;
    addrs[3] = c.ptr.ptr;
    Array<int, 4> dims;
    dims[0] = M_desc;
    dims[1] = -1;
    dims[2] = -1;
    dims[3] = M_desc;
    Array<uint64_t, 4> strides;
    strides[0] = (uint64_t)a.ptr.stride * sizeof(nv_bfloat16);
    strides[1] = 0;
    strides[2] = 0;
    strides[3] = (uint64_t)c.ptr.stride * sizeof(nv_bfloat16);
    detail::rebase_publish_mixed_tma_descs<4>(out + g * kNum, smem_desc, templates, addrs, dims, strides, lane);
}

template<Order    raster_order,
         int      multicast_a,
         int      multicast_b,
         bool     is_grouped_gemm_,
         Striding kStridingA_,
         class Tile_,
         bool kSupportsFusedSilu_ = false,
         class Format_            = Sm90U4Format>
struct GemmUniversalSm90Mixed {
    using Arch    = Sm90;
    using Tile    = Tile_;
    using Format  = Format_;
    using Dequant = detail::Sm90MixedDequant<Format>;

    static constexpr Order kRasterOrder       = raster_order;
    static constexpr bool  is_grouped_gemm    = is_grouped_gemm_;
    static constexpr bool  kSupportsFusedSilu = kSupportsFusedSilu_;

    static constexpr int kMulticastA = multicast_a;
    static constexpr int kMulticastB = multicast_b;
    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr Striding kStridingA     = kStridingA_;
    static constexpr Striding kStridingB     = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC     = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
    static constexpr bool     kIndexedGather = kStridingA == Striding::kIndexed;

    static_assert(is_grouped_gemm == (kStridingA != Striding::kFlat));
    static_assert(kMulticastA == 1 || kMulticastA == 2);
    static_assert(kMulticastB == 1 || kMulticastB == 2);
    static_assert(kClusterSize <= 2);
    // Public scheduler axes: M=BATCH, N=OUT.
    static constexpr int TILE_M     = Tile::TILE_BATCH;
    static constexpr int TILE_N     = Tile::TILE_OUT;
    static constexpr int TILE_K     = kSm90MixedTileK;
    static constexpr int kGroupSize = Format::kGroupSize;

    using Ta = nv_bfloat16;
    using Tb = typename Format::WeightType;
    using Tv = typename Format::QparamType;
    using Tc = nv_bfloat16;

    static_assert(TILE_K % kGroupSize == 0 || kGroupSize % TILE_K == 0);
    static constexpr int kKTilesPerQGroup = kGroupSize >= TILE_K ? kGroupSize / TILE_K : 1;
    static constexpr int kQGroupsPerStage = kGroupSize < TILE_K ? TILE_K / kGroupSize : 1;
    static_assert(kKTilesPerQGroup * TILE_K == kQGroupsPerStage * kGroupSize);

    using WGLayout      = typename Tile::WGLayout;
    using Traits        = GmmaMixedTraits<TILE_N, TILE_M, Tile::Stages, WGLayout, detail::MixedMmaN<Tile>::value>;
    using AtomLayoutMNK = typename Traits::AtomLayoutMNK;
    using TiledMma      = typename Traits::TiledMma;
    using MmaIssue      = detail::GmmaIssue<Traits::kMmaNSlices, detail::MixedSeparateMmaAtoms<Tile>::value, 2>;

    static constexpr int WARPGROUP_SIZE = 128;
    static constexpr int WARPGROUPS     = Traits::kMathWarpgroups;
    static constexpr int kMathThreads   = Traits::kMathThreads;
    static constexpr int CTA_SIZE       = Traits::kCtaThreads;
    static constexpr int Stages         = Tile::Stages;

    // Named-barrier IDs belong to this kernel. Gather and TMA producers are
    // mutually exclusive, so their producer coordination can share ID 8.
    static constexpr int kEpilogueBarrierId = 1;
    static constexpr int kProducerBarrierId = 8;
    static_assert(kEpilogueBarrierId + WARPGROUPS <= kProducerBarrierId);

    static constexpr int kProducerRegsTma     = Tile::kProducerRegsTma;
    static constexpr int kMathRegsTma         = Tile::kMathRegsTma;
    static constexpr int kProducerRegsIndexed = Tile::kProducerRegsIndexed;
    static constexpr int kMathRegsIndexed     = Tile::kMathRegsIndexed;
    static_assert(kProducerRegsTma >= 24 && kProducerRegsTma % 8 == 0);
    static_assert(kMathRegsTma >= 24 && kMathRegsTma % 8 == 0 && kMathRegsTma <= 256);
    static_assert(kProducerRegsIndexed >= 24 && kProducerRegsIndexed % 8 == 0);
    static_assert(kMathRegsIndexed >= 24 && kMathRegsIndexed % 8 == 0 && kMathRegsIndexed <= 256);
    static_assert(WARPGROUPS != 2 || kProducerRegsTma + 2 * kMathRegsTma <= 504);
    static_assert(WARPGROUPS != 1 || kProducerRegsTma + kMathRegsTma <= 512);
    static_assert(WARPGROUPS != 3 || kProducerRegsTma + 3 * kMathRegsTma <= 512);
    static_assert(WARPGROUPS != 2 || kProducerRegsIndexed + 2 * kMathRegsIndexed <= 504);
    static_assert(WARPGROUPS != 1 || kProducerRegsIndexed + kMathRegsIndexed <= 512);
    static_assert(WARPGROUPS != 3 || kProducerRegsIndexed + 3 * kMathRegsIndexed <= 512);
    static constexpr int  kRestM                    = TILE_N / (64 * Traits::kAtomM);
    static constexpr bool kNeedsCrossWgSiluExchange = kSupportsFusedSilu && kRestM % 2 != 0 && Traits::kAtomM == 2 && WARPGROUPS == 2;
    static_assert(!kSupportsFusedSilu || kRestM % 2 == 0 || kNeedsCrossWgSiluExchange);

    using Cluster   = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;
    using Scheduler = TileScheduler<raster_order, Cluster, true, true, TILE_M, TILE_N, Stages, is_grouped_gemm>;

    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using MainloopState    = typename MainloopPipeline::PipelineState;
    using MainloopStorage  = typename MainloopPipeline::SharedStorage;

    using SmemLayoutB    = typename Traits::SmemLayoutB;
    using SmemLayoutB_2D = typename Traits::SmemLayoutB_2D;

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

    static constexpr int kOutputFragments      = TILE_N / kSm90MixedFragmentN;
    static constexpr int kPackedWordsFragment  = kSm90MixedFragmentN * TILE_K * Format::kWeightBits / 32;
    static constexpr int kPackedWordsStage     = kOutputFragments * kPackedWordsFragment;
    static constexpr int kPackedBytesStage     = kPackedWordsStage * (int)sizeof(uint32_t);
    static constexpr int kQparamFragmentBytes  = Format::kQparamValuesFragment * (int)sizeof(Tv);
    static constexpr int kQparamGroupBytes     = kOutputFragments * kQparamFragmentBytes;
    static constexpr int kQparamDataBytesStage = kQGroupsPerStage * kQparamGroupBytes;
    static constexpr int kQparamBytesStage     = (kQparamDataBytesStage + 127) / 128 * 128;
    static constexpr int kInputBytesStage      = TILE_M * TILE_K * (int)sizeof(Ta);
    static constexpr int kTmaCountM            = cute::ceil_div(TILE_M / kMulticastA, 256);
    static constexpr int kTmaBoxM              = TILE_M / (kMulticastA * kTmaCountM);
    static_assert(TILE_N % kSm90MixedFragmentN == 0);
    static_assert(TILE_M % kMulticastA == 0);
    static_assert(TILE_M % (kMulticastA * kTmaCountM) == 0);
    static_assert(kTmaBoxM <= 256);
    static_assert(Format::kQparamFragmentN == kSm90MixedFragmentN);
    static_assert(Format::kQparamValuesTile == 2 * Format::kQparamValuesFragment);
    static_assert(kPackedWordsStage == Traits::kPackedWordsStage * Format::kWeightBits / 4);
    static_assert(kPackedWordsStage == TILE_N * TILE_K / (32 / Format::kWeightBits));
    static_assert(kQparamGroupBytes % 16 == 0);

    // Persistent weights are [K16 fragment, OUT64 fragment, RS payload].
    // The tensor-map view is (payload, OUT64 fragment, K16 fragment).
    static constexpr int kPackedInnerWords  = 128 * Dequant::kWordsPerThreadKBlock;
    static constexpr int kKFragmentsPerTile = TILE_K / kSm90MixedFragmentK;
    static_assert(kPackedWordsFragment == kPackedInnerWords * kKFragmentsPerTile);
    static_assert(kOutputFragments % kMulticastB == 0);
    using PackedCtaTile = cute::Shape<cute::Int<kPackedInnerWords>,
                                      cute::Int<kOutputFragments>,
                                      cute::Int<kKFragmentsPerTile>>;
    using PackedSmemLayout = decltype(cute::make_layout(PackedCtaTile{}));

    static auto MakeTmaPacked(void* ptr, int n, int k)
    {
        const int k_fragments   = k / kSm90MixedFragmentK;
        const int out_fragments = n / kSm90MixedFragmentN;
        auto layout = cute::make_layout(
            cute::make_shape(cute::Int<kPackedInnerWords>{}, out_fragments, k_fragments),
            cute::make_stride(cute::_1{},
                              cute::Int<kPackedInnerWords>{},
                              (int64_t)kPackedInnerWords * out_fragments));
        auto gmem    = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint32_t*>(ptr)), layout);
        using CopyOp = std::conditional_t<kMulticastB == 1, cute::SM90_TMA_LOAD, cute::SM90_TMA_LOAD_MULTICAST>;
        return cute::make_tma_copy(
            CopyOp{}, gmem, PackedSmemLayout{}, PackedCtaTile{}, cute::Int<kMulticastB>{});
    }

    using TmaPacked = decltype(MakeTmaPacked(nullptr, TILE_N, TILE_K));

    // Persistent qparams are [K/group, OUT64 fragment, consumer payload].
    // The tensor-map view is (payload, OUT64 fragment, K/group).
    using QparamCtaTile = cute::Shape<cute::Int<Format::kQparamValuesFragment>,
                                      cute::Int<kOutputFragments>,
                                      cute::Int<kQGroupsPerStage>>;
    using QparamSmemLayout = decltype(cute::make_layout(QparamCtaTile{}));
    using TmaQparamElement = std::conditional_t<sizeof(Tv) == 2, uint16_t, Tv>;
    static_assert(sizeof(TmaQparamElement) == sizeof(Tv));

    static auto MakeTmaQparam(void* ptr, int n, int k)
    {
        const int out_fragments = n / kSm90MixedFragmentN;
        const int q_groups      = k / kGroupSize;
        auto layout = cute::make_layout(
            cute::make_shape(cute::Int<Format::kQparamValuesFragment>{}, out_fragments, q_groups),
            cute::make_stride(cute::_1{},
                              cute::Int<Format::kQparamValuesFragment>{},
                              (int64_t)Format::kQparamValuesFragment * out_fragments));
        auto gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<TmaQparamElement*>(ptr)), layout);
        using CopyOp = std::conditional_t<kMulticastB == 1, cute::SM90_TMA_LOAD, cute::SM90_TMA_LOAD_MULTICAST>;
        return cute::make_tma_copy(CopyOp{}, gmem, QparamSmemLayout{}, QparamCtaTile{}, cute::Int<kMulticastB>{});
    }

    using TmaQparam = decltype(MakeTmaQparam(nullptr, TILE_N, TILE_K));

    static constexpr int kTmaDescNumA = is_grouped_gemm ? 1 : 0;
    static constexpr int kTmaDescNumB = is_grouped_gemm ? 1 : 0;
    static constexpr int kTmaDescNumV = is_grouped_gemm ? 1 : 0;
    static constexpr int kTmaDescNumC = is_grouped_gemm ? 1 : 0;
    static constexpr int kTmaDescNum =
        is_grouped_gemm ? kTmaDescNumA + kTmaDescNumB + kTmaDescNumV + kTmaDescNumC : 1;

    static constexpr int kAtomM          = Traits::kAtomM;
    static constexpr int kAtomN          = Traits::kAtomN;
    static constexpr bool kSplitEpiM      = kAtomN == 2;
    static constexpr int kWgM            = TILE_M / kAtomN;
    static constexpr int kEpiN           = 64 * kAtomM;
    static constexpr int kWgMLowBit      = kWgM & -kWgM;
    static constexpr int kEpiMDefault    = kWgMLowBit < 32 ? kWgMLowBit : 32;
    static constexpr int kEpiM           = detail::MixedEpiM<Tile>::value ? detail::MixedEpiM<Tile>::value : kEpiMDefault;
    static constexpr int kEpiPlanes      = kSplitEpiM ? kAtomN : 1;
    static constexpr int kTmaStoreN      = 64;
    static constexpr int kTmaStoreM      = kEpiM <= 256 ? kEpiM : 64;
    static constexpr int kTmaStoreCountM = kEpiM / kTmaStoreM;
    static constexpr int kSwizzleC       = 128;
    static constexpr int kEpiThreads     = kSplitEpiM ? WARPGROUP_SIZE : kMathThreads;
    static constexpr int kFragmentSize   = (kEpiM * kEpiN) / kEpiThreads;
    static constexpr int kEpiStripsM     = kWgM / kEpiM;
    static constexpr int kEpiStripsN     = TILE_N / kEpiN;
    static constexpr int kEpiPasses      = kEpiStripsM * kEpiStripsN;
    static_assert(TILE_M % kEpiM == 0);
    static_assert(TILE_N % kEpiN == 0);
    static_assert(kWgM % kEpiM == 0);
    static_assert(kEpiM % kTmaStoreM == 0);
    static_assert(kTmaStoreM <= 256);
    static_assert(kEpiN % kTmaStoreN == 0);
    static_assert(kFragmentSize >= 1);
    static_assert(!kNeedsCrossWgSiluExchange || (kAtomM == 2 && kAtomN == 1 && kEpiPlanes == 1 && kTmaStoreCountM == 1));

    // Public tile geometry follows GEMM's conventional (M,N) order. WGMMA C and
    // the STSM destination are (N,M), so that physical permutation stays here.
    using CrossWgSiluLayout = cute::Layout<cute::Shape<cute::Int<kEpiM>, cute::Int<kTmaStoreN>>, cute::Stride<cute::Int<kTmaStoreN>, cute::_1>>;
    using SmemLayoutAtomD = decltype(
        gmma_ss_smem_selector<cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::Int<kEpiN>, cute::Int<kEpiM>>());
    using SmemLayoutDPlane =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::_1{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    static constexpr int kEpiStageElems = cute::cosize_v<SmemLayoutDPlane> * kEpiPlanes;
    static_assert(!kNeedsCrossWgSiluExchange || cute::cosize_v<CrossWgSiluLayout> * (int)sizeof(float) == kEpiStageElems * (int)sizeof(Tc));

    static constexpr int kCValsPerThread = kAtomN == 2 ? TILE_M / 4 : TILE_M / 2;
    static_assert(kCValsPerThread >= 4);
    using CopyAtomC = typename detail::MixedEpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyAtomC;
    using CopyOpR2S = typename detail::MixedEpiStsmAtoms<(kCValsPerThread >= 8) ? 8 : 4>::CopyOpR2S;

    struct LayoutC {
        static constexpr int S0 = kEpiM;
        static constexpr int C0 = kTmaStoreN;
        static constexpr int C1 = 1;
    };

    template<int EpiPipeStages>
    struct SharedStorageT: Dequant::SharedStorage {
        cute::array_aligned<uint32_t, kPackedWordsStage * Stages, 128>              A;
        cute::array_aligned<typename Traits::ElementB, cute::cosize_v<SmemLayoutB>> B;
        // Each pipeline stage owns one independently committed epilogue pass.
        cute::array_aligned<Tc, kEpiStageElems * EpiPipeStages, 1024> D;
        cute::array_aligned<uint8_t, kQparamBytesStage * Stages, 128> Q;
        MainloopStorage                                                pipeline;
        typename Scheduler::Storage                                    sched;
        StridedPtr                                                     gather_A;
        const int*                                                     gather_idxs;
        int                                                            gather_alive;
        int                                                            gather_k_iters;
        int                                                            gather_M_group;
        int                                                            gather_offset_m;
        volatile int                                                   gather_group_idx;
        volatile int                                                   gather_out_fragment;
    };

    static constexpr int kSmemCapacity = 228 << 10;

    static constexpr int GetEpiPipeStages()
    {
        constexpr int requested = detail::MixedEpiPipeStages<Tile>::value;
        if constexpr (requested) {
            return requested;
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
    using SmemLayoutD =
        decltype(cute::tile_to_shape(SmemLayoutAtomD{},
                                     cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}, cute::Int<kEpiSmemSlices>{}),
                                     cute::Step<cute::_2, cute::_1, cute::_3>{}));
    using EpiStageLayout = cute::Layout<cute::Shape<cute::Int<kEpiPlanes>, cute::Int<kEpiPipeStages>>, cute::Stride<cute::_1, cute::Int<kEpiPlanes>>>;
    static_assert(cute::cosize_v<SmemLayoutD> == kEpiStageElems * kEpiPipeStages);

    using SharedStorage = SharedStorageT<kEpiPipeStages>;
    static constexpr int kSmemSize = (int)sizeof(SharedStorage);
    static_assert(kSmemSize <= kSmemCapacity);

    using ClusterShape = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;

    static int* PrepareTmaDescs(const CUtensorMap& tm_a,
                                const CUtensorMap& tm_b,
                                const CUtensorMap& tm_v,
                                const CUtensorMap& tm_c,
                                const MatrixParam& param_A,
                                const MatrixParam& param_B,
                                const MatrixParam& param_V,
                                const MatrixParam& param_C,
                                CUtensorMap*       out,
                                int                num_groups,
                                int                M,
                                cudaStream_t       stream)
    {
        if constexpr (!is_grouped_gemm) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_tma_descs_sm90_mixed<kStridingA>
            <<<num_groups, 32, 0, stream>>>(tm_a, tm_b, tm_v, tm_c, param_A, param_B, param_V, param_C, out, offsets, M);
        return offsets;
    }

    __device__ void operator()(const CUtensorMap& tm_a,
                               const TmaPacked&   tm_b,
                               const TmaQparam&   tm_v,
                               const CUtensorMap& tm_c,
                               const MatrixParam& param_A,
                               const MatrixParam& param_G,
                               const MatrixParam& param_C,
                               bool               fuse_silu,
                               Scheduler          sched,
                               CUtensorMap*       tensormap_buf,
                               char*              smem_buf)
    {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int wg_idx     = cutlass::canonical_warp_group_idx();
        const int warp_in_wg = (cutlass::canonical_warp_idx_sync() % 4);
        const int lane       = (int)threadIdx.x % WARP_SIZE;

        Dequant::init(storage);

        if (threadIdx.x == 0) {
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGROUPS * 4 + 1));
        }

        typename MainloopPipeline::Params main_params;
        main_params.transaction_bytes = (uint32_t)((kIndexedGather ? kPackedBytesStage : kPackedBytesStage + kInputBytesStage) + kQparamDataBytesStage);
        main_params.num_consumers     = (uint32_t)kMathThreads;
        main_params.num_producers     = kIndexedGather ? (1 + WARPGROUP_SIZE) : 1;
        main_params.initializing_warp = 0;

        if (wg_idx == WARPGROUPS) {
            if constexpr (kIndexedGather) {
                main_params.role      = MainloopPipeline::ThreadCategory::Producer;
                main_params.is_leader = warp_in_wg == 0 && lane == 0;
            }
            else {
                main_params.role      = warp_in_wg == 0 ? MainloopPipeline::ThreadCategory::Producer :
                                                          MainloopPipeline::ThreadCategory::NonParticipant;
                main_params.is_leader = warp_in_wg == 0 && lane == 0;
            }
        }
        else {
            main_params.role      = MainloopPipeline::ThreadCategory::Consumer;
            main_params.is_leader = 0;
        }

        MainloopPipeline pipeline(storage.pipeline, main_params, ClusterShape{});
        if (threadIdx.x == 0) {
            cutlass::arch::fence_view_async_shared();
        }
        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();
        if (wg_idx == WARPGROUPS) {
            run_producer(tm_a, tm_b, tm_v, param_A, sched, tensormap_buf, storage, pipeline);
        }
        else {
            run_consumer(tm_c, param_G, fuse_silu, sched, tensormap_buf, storage, pipeline);
        }
    }

private:
    __device__ __noinline__ static void issue_packed_copy(const TmaPacked&          tma,
                                             const cute::TmaDescriptor* desc,
                                             int                       out_fragment,
                                             int                       k_iters,
                                             int                       out_fragments,
                                             int                       k_tile,
                                             int                       issuer,
                                             uint16_t                  mcast_mask,
                                             uint64_t*                 bar,
                                             uint32_t*                 smem)
    {
        auto gmem = tma.get_tma_tensor(cute::make_shape(cute::Int<kPackedInnerWords>{},
                                                        out_fragments,
                                                        k_iters * kKFragmentsPerTile));
        auto gtiles        = cute::flat_divide(gmem, PackedCtaTile{});
        auto cta_tma       = tma.get_slice(issuer);
        auto src_partition = cta_tma.partition_S(gtiles);
        auto src           = cute::group_modes<1, cute::rank(src_partition)>(src_partition);
        auto stile         = cute::make_tensor(cute::make_smem_ptr(smem), PackedSmemLayout{});
        auto dst_partition = cta_tma.partition_D(stile);
        auto dst           = cute::group_modes<1, cute::rank(dst_partition)>(dst_partition);
        const int tile     = out_fragment / kOutputFragments + (out_fragments / kOutputFragments) * k_tile;
        cute::copy(tma.with(desc, *bar, mcast_mask, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                   src(cute::_, tile),
                   dst(cute::_, 0));
    }

    __device__ static void run_producer(const CUtensorMap& tm_a,
                                        const TmaPacked&   tm_b,
                                        const TmaQparam&   tm_v,
                                        const MatrixParam& param_A,
                                        Scheduler          sched,
                                        CUtensorMap*       tensormap_buf,
                                        SharedStorage&     storage,
                                        MainloopPipeline&  pipeline)
    {
        if constexpr (kIndexedGather) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegsIndexed>();
            run_producer_gather(tm_b, tm_v, param_A, sched, tensormap_buf, storage, pipeline);
        }
        else {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegsTma>();
            run_producer_tma(tm_a, tm_b, tm_v, sched, tensormap_buf, storage, pipeline);
        }
    }

    __device__ __forceinline__ static void run_producer_gather(const TmaPacked&   tm_b,
                                                             const TmaQparam&   tm_v,
                                                             const MatrixParam& param_A,
                                                             Scheduler          sched,
                                                             CUtensorMap*       tensormap_buf,
                                                             SharedStorage&     storage,
                                                             MainloopPipeline&  pipeline)
    {
        static_assert(kIndexedGather);
        const int                   warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const int                   lane_id    = (int)threadIdx.x % WARP_SIZE;
        const int                   prod_tid   = (int)threadIdx.x - WARPGROUPS * WARPGROUP_SIZE;
        const bool                  cta_0      = cute::block_id_in_cluster().x == 0;
        Cluster                     cluster(cute::block_id_in_cluster().x);

        static_assert(TILE_N % kMulticastB == 0);

        const uint16_t mask_B = cluster.mask_n();

        MainloopState write_state = cutlass::make_producer_start_state<MainloopPipeline>();
        auto          sched_state = sched.init_consumer(storage.sched);
        auto          prod_state  = sched.init_producer(storage.sched);
        const int     elected     = warp_in_wg == 0 ? cute::elect_one_sync() : 0;

        const int k_iters       = sched.k_iters_;
        const int out_fragments = sched.gemm_shape().y / kSm90MixedFragmentN;

        auto packed_gmem          = tm_b.get_tma_tensor(cute::make_shape(cute::Int<kPackedInnerWords>{}, out_fragments, k_iters * kKFragmentsPerTile));
        auto packed_gtiles        = cute::flat_divide(packed_gmem, PackedCtaTile{});
        auto packed_cta_tma       = tm_b.get_slice(cluster.cta_m());
        auto packed_src_partition = packed_cta_tma.partition_S(packed_gtiles);
        auto packed_src           = cute::group_modes<1, cute::rank(packed_src_partition)>(packed_src_partition);

        const int q_tile_count      = k_iters / kKTilesPerQGroup;
        auto      qparam_gmem       = tm_v.get_tma_tensor(cute::make_shape(cute::Int<Format::kQparamValuesFragment>{}, out_fragments, q_tile_count * kQGroupsPerStage));
        auto      qparam_gtiles     = cute::flat_divide(qparam_gmem, QparamCtaTile{});
        auto      qparam_cta_tma    = tm_v.get_slice(cluster.cta_m());
        auto      qparam_src_part   = qparam_cta_tma.partition_S(qparam_gtiles);
        auto      qparam_src        = cute::group_modes<1, cute::rank(qparam_src_part)>(qparam_src_part);

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
                    storage.gather_alive        = alive ? 1 : 0;
                    storage.gather_k_iters      = 0;
                    storage.gather_M_group      = 0;
                    storage.gather_offset_m     = 0;
                    storage.gather_group_idx    = 0;
                    storage.gather_out_fragment = 0;
                    if (alive && tile->is_valid_cluster) {
                        a = resolve<Ta, kStridingA>(param_A, tile->group_idx);
                        // A partial output cluster still runs the invalid CTA so multicast and pipeline arrivals
                        // stay balanced. Tensor TMA zero-fills packed-weight and qparam tiles.
                        storage.gather_k_iters      = sched.k_iters_;
                        storage.gather_M_group      = tile->m1 - tile->m0;
                        storage.gather_offset_m     = tile->offset_m;
                        storage.gather_group_idx    = tile->group_idx;
                        storage.gather_out_fragment = tile->offset_n / kSm90MixedFragmentN;
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

            const int tile_k_iters  = storage.gather_k_iters;
            const int out_fragment  = storage.gather_out_fragment;
            const int group_idx     = storage.gather_group_idx;
            const int packed_m0     = storage.gather_offset_m;
            const int row_count     = storage.gather_M_group - storage.gather_offset_m;
            const Ta*  act_gmem     = static_cast<const Ta*>(storage.gather_A.ptr);
            const int  ldA          = storage.gather_A.stride;
            const int* idxs         = storage.gather_idxs;

            const cute::TmaDescriptor* Bdesc = tm_b.get_tma_descriptor();
            const cute::TmaDescriptor* Vdesc = tm_v.get_tma_descriptor();
            if constexpr (is_grouped_gemm) {
                Bdesc = tensormap_buf + group_idx * kTmaDescNum + kTmaDescNumA;
                Vdesc = Bdesc + kTmaDescNumB;
            }

            const Ta* gather_src[kGatherSlots];
            typename Traits::ElementB* gather_dst[kGatherSlots];
            bool      gather_pred[kGatherSlots];
            bool      gather_slot_valid[kGatherSlots];
            if constexpr (TILE_M >= kGatherThreadsM && TILE_M % kGatherThreadsM == 0) {
                static_assert(kGatherVectors % WARPGROUP_SIZE == 0);
                auto gather_thr = GatherTiledCopy{}.get_slice(prod_tid);
                auto smem_act = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
                auto tBsB = gather_thr.partition_D(smem_act);
                auto identity = cute::make_identity_tensor(cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_K>>{});
                auto tBcB = gather_thr.partition_D(identity);
                static_assert(cute::size<0>(tBcB) == kGatherVec);
                static_assert(cute::size<1>(tBcB) == kGatherSlots);
                static_assert(cute::size<2>(tBcB) == 1);
                PRAGMA_UNROLL
                for (int slot = 0; slot < kGatherSlots; ++slot) {
                    const auto tile_coord = tBcB(0, slot, 0);
                    const int  tile_m     = cute::get<0>(tile_coord);
                    const int  tile_k     = cute::get<1>(tile_coord);
                    const int  packed_row = packed_m0 + tile_m;
                    const bool row_valid  = tile_m < row_count;
                    const int  source_row = (idxs && row_valid) ? __ldg(idxs + packed_row) : packed_row;
                    gather_src[slot]        = act_gmem + (int64_t)source_row * ldA + tile_k;
                    gather_dst[slot]        = &tBsB(0, slot, 0, 0);
                    gather_pred[slot]       = row_valid;
                    gather_slot_valid[slot] = true;
                }
            }
            else {
                auto smem_act = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB_2D{});
                PRAGMA_UNROLL
                for (int slot = 0; slot < kGatherSlots; ++slot) {
                    const int  vector_idx = prod_tid + slot * WARPGROUP_SIZE;
                    const bool slot_valid = vector_idx < kGatherVectors;
                    const int  tile_m     = slot_valid ? vector_idx / kGatherThreadsK : 0;
                    const int  tile_k     = slot_valid ? (vector_idx % kGatherThreadsK) * kGatherVec : 0;
                    const int  packed_row = packed_m0 + tile_m;
                    const bool row_valid  = slot_valid && tile_m < row_count;
                    const int  source_row = (idxs && row_valid) ? __ldg(idxs + packed_row) : packed_row;
                    gather_src[slot]        = act_gmem + (int64_t)source_row * ldA + tile_k;
                    gather_dst[slot]        = &smem_act(tile_m, tile_k);
                    gather_pred[slot]       = row_valid;
                    gather_slot_valid[slot] = slot_valid;
                }
            }

            for (int k_tile = 0; k_tile < tile_k_iters; ++k_tile) {
                pipeline.producer_acquire(write_state);
                auto*     bar   = pipeline.producer_get_barrier(write_state);
                const int stage = write_state.index();

                if (warp_in_wg == 0 && elected) {
                    auto qparam_stile = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<TmaQparamElement*>(storage.Q.data() + stage * kQparamBytesStage)), QparamSmemLayout{});
                    auto qparam_dst_part = qparam_cta_tma.partition_D(qparam_stile);
                    auto qparam_dst = cute::group_modes<1, cute::rank(qparam_dst_part)>(qparam_dst_part);
                    const int qparam_tile = out_fragment / kOutputFragments + (out_fragments / kOutputFragments) * (k_tile / kKTilesPerQGroup);
                    cute::copy(tm_v.with(Vdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_LAST), qparam_src(cute::_, qparam_tile), qparam_dst(cute::_, 0));
                }

                if (warp_in_wg == 0 && elected) {
                    auto packed_stile = cute::make_tensor(cute::make_smem_ptr(storage.A.data() + stage * kPackedWordsStage), PackedSmemLayout{});
                    auto packed_dst_part = packed_cta_tma.partition_D(packed_stile);
                    auto packed_dst = cute::group_modes<1, cute::rank(packed_dst_part)>(packed_dst_part);
                    const int packed_tile = out_fragment / kOutputFragments + (out_fragments / kOutputFragments) * k_tile;
                    cute::copy(tm_b.with(Bdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_NORMAL), packed_src(cute::_, packed_tile), packed_dst(cute::_, 0));
                }

                PRAGMA_UNROLL
                for (int slot = 0; slot < kGatherSlots; ++slot) {
                    if constexpr (kGatherVectors % WARPGROUP_SIZE) {
                        if (!gather_slot_valid[slot]) {
                            continue;
                        }
                    }
                    auto* dst = gather_dst[slot] + stage * TILE_M * TILE_K;
                    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(*reinterpret_cast<const uint4*>(gather_src[slot]), *reinterpret_cast<uint4*>(dst), gather_pred[slot]);
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

    __device__ __forceinline__ static void run_producer_tma(const CUtensorMap& tm_a,
                                                          const TmaPacked&   tm_b,
                                                          const TmaQparam&   tm_v,
                                                          Scheduler          sched,
                                                          CUtensorMap*       tensormap_buf,
                                                          SharedStorage&     storage,
                                                          MainloopPipeline&  pipeline)
    {
        const int  warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const bool cta_0      = cute::block_id_in_cluster().x == 0;
        Cluster    cluster(cute::block_id_in_cluster().x);

        static_assert(TILE_M % kMulticastA == 0);
        static_assert(TILE_N % kMulticastB == 0);

        const int      mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
        const uint16_t mask_A      = cluster.mask_m();
        const uint16_t mask_B      = cluster.mask_n();

        if (warp_in_wg == 0) {
            MainloopState write_state = cutlass::make_producer_start_state<MainloopPipeline>();
            auto          sched_state = sched.init_consumer(storage.sched);
            const int     elected     = cute::elect_one_sync();
            const int     k_iters       = sched.k_iters_;
            const int     out_fragments = sched.gemm_shape().y / kSm90MixedFragmentN;
            const int     q_tile_count  = k_iters / kKTilesPerQGroup;
            auto          qparam_gmem   = tm_v.get_tma_tensor(cute::make_shape(cute::Int<Format::kQparamValuesFragment>{}, out_fragments, q_tile_count * kQGroupsPerStage));
            auto          qparam_gtiles = cute::flat_divide(qparam_gmem, QparamCtaTile{});
            auto          qparam_cta_tma = tm_v.get_slice(cluster.cta_m());
            auto          qparam_src_part = qparam_cta_tma.partition_S(qparam_gtiles);
            auto          qparam_src = cute::group_modes<1, cute::rank(qparam_src_part)>(qparam_src_part);

            typename Scheduler::Tile* tile;
            while (sched_state.acquire(tile)) {
                if (tile->is_valid_cluster && elected) {
                    const CUtensorMap* Adesc = &tm_a;
                    const cute::TmaDescriptor* Bdesc = tm_b.get_tma_descriptor();
                    const cute::TmaDescriptor* Vdesc = tm_v.get_tma_descriptor();

                    if constexpr (is_grouped_gemm) {
                        CUtensorMap* descs = tensormap_buf + tile->group_idx * kTmaDescNum;
                        Adesc              = &descs[0];
                        Bdesc              = &descs[kTmaDescNumA];
                        Vdesc              = &descs[kTmaDescNumA + kTmaDescNumB];
                    }

                    const int out_fragment = tile->offset_n / kSm90MixedFragmentN;

                    for (int q_tile = 0; q_tile < q_tile_count; ++q_tile) {
                        const int qparam_tile = out_fragment / kOutputFragments + (out_fragments / kOutputFragments) * q_tile;
                        CUTE_UNROLL
                        for (int in_group = 0; in_group < kKTilesPerQGroup; ++in_group) {
                            const int k_tile = q_tile * kKTilesPerQGroup + in_group;
                            pipeline.producer_acquire(write_state);
                            auto*     bar   = pipeline.producer_get_barrier(write_state);
                            const int stage = write_state.index();
                            auto qparam_stile = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<TmaQparamElement*>(storage.Q.data() + stage * kQparamBytesStage)), QparamSmemLayout{});
                            auto qparam_dst_part = qparam_cta_tma.partition_D(qparam_stile);
                            auto qparam_dst = cute::group_modes<1, cute::rank(qparam_dst_part)>(qparam_dst_part);
                            cute::copy(tm_v.with(Vdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_LAST), qparam_src(cute::_, qparam_tile), qparam_dst(cute::_, 0));
                            issue_packed_copy(tm_b,
                                              Bdesc,
                                              out_fragment,
                                              k_iters,
                                              out_fragments,
                                              k_tile,
                                              cluster.cta_m(),
                                              mask_B,
                                              bar,
                                              storage.A.data() + stage * kPackedWordsStage);
                            CUTE_UNROLL
                            for (int tma_m = 0; tma_m < kTmaCountM; ++tma_m) {
                                detail::mixed_tma_load_with_barrier<kMulticastA, kTmaBoxM, TILE_K>(Adesc, bar, storage.B.data() + stage * TILE_M * TILE_K + (mc_offset_m + tma_m * kTmaBoxM) * TILE_K, k_tile * TILE_K, tile->offset_m + mc_offset_m + tma_m * kTmaBoxM, mask_A);
                            }
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
            if (elected) {
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

    __device__ static void run_consumer(const CUtensorMap& tm_c,
                                        const MatrixParam& param_G,
                                        bool               fuse_silu,
                                        Scheduler          sched,
                                        CUtensorMap*       tensormap_buf,
                                        SharedStorage&     storage,
                                        MainloopPipeline&  pipeline)
    {
        if constexpr (kIndexedGather) {
            cutlass::arch::warpgroup_reg_alloc<kMathRegsIndexed>();
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<kMathRegsTma>();
        }

        const int mma_tid   = (int)threadIdx.x;
        const int wg_idx    = cutlass::canonical_warp_group_idx();
        const int local_tid = mma_tid % WARPGROUP_SIZE;

        TiledMma tiled_mma;
        auto     thr_mma = tiled_mma.get_thread_slice(mma_tid);

        auto dummy_sA =
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<typename Traits::ElementA*>(storage.A.data())),
                              typename Traits::DummySmemLayoutA{});
        auto dummy_tCsA = thr_mma.partition_A(dummy_sA);
        auto tCrA       = thr_mma.make_fragment_A(dummy_tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}));

        auto sB = cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB{});
        auto warp_group_thread_layout = cute::make_layout(cute::Int<WARPGROUPS>{}, cute::Int<WARPGROUP_SIZE>{});
        auto wg_mma = tiled_mma.get_slice(warp_group_thread_layout(wg_idx));
        auto tCsB   = wg_mma.partition_B(sB);
        auto tCrB   = wg_mma.make_fragment_B(tCsB);

        static_assert(cute::rank(tCrA) == 3);
        static_assert(cute::size<0>(tCrA) == 8);
        static_assert(cute::size<2>(tCrA) == Traits::kKBlocksPerStage);
        static_assert(cute::size<1>(tCrA) == kRestM);
        static_assert(kRestM >= 1 && kRestM <= 4);

        auto sD = cute::as_position_independent_swizzle_tensor(cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));
        CopyAtomC copy_atom_c{};
        using EpiTiledMma = std::conditional_t<kSplitEpiM, typename Traits::WgTiledMma, TiledMma>;
        EpiTiledMma epi_tiled_mma;
        auto tiled_copy_C_atom = cute::make_tiled_copy_C_atom(copy_atom_c, epi_tiled_mma);
        auto tiled_r2s = cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, cutlass::bfloat16_t>{}, tiled_copy_C_atom);
        auto thr_r2s   = tiled_r2s.get_slice(kSplitEpiM ? local_tid : mma_tid);
        auto tRS_rD_layout = cute::make_layout(cute::take<0, 3>(cute::shape(thr_r2s.partition_S(sD))));

        const int  tma_store_warp   = mma_tid / WARP_SIZE;
        const bool tma_store_leader = cute::elect_one_sync();
        int        epi_store_count  = 0;

        MainloopState pipe_state{};
        MainloopState pipe_release{};

        auto                      sched_state = sched.init_consumer(storage.sched);
        typename Scheduler::Tile* tile;
        sched_state.acquire(tile);

        while (tile->alive) {
            if (tile->is_valid_cta) {
                auto accum = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                cute::clear(accum);

                float output_scale = 1.f;
                if constexpr (Format::kHasGlobalScale) {
                    int group_idx = 0;
                    if constexpr (is_grouped_gemm) {
                        group_idx = tile->group_idx;
                    }
                    const auto g = detail::resolve_mixed_group_ptr(param_G, group_idx);
                    output_scale = __ldg(static_cast<const float*>(g.ptr));
                }

                typename Dequant::template Registers<kRestM> qregs;

                const int  wg_m                  = wg_idx % kAtomM;
                const bool contiguous_fused_wg   = fuse_silu && kAtomM == 2;
                const int  packed_segment_base   = contiguous_fused_wg ? wg_m * kRestM : wg_m;
                const int  packed_segment_stride = contiguous_fused_wg ? 1 : kAtomM;

                auto load_qparams = [&](int stage) {
                    auto*     q     = storage.Q.data() + stage * kQparamBytesStage;
                    Dequant::template load<kRestM, kAtomM, TILE_N>(qregs, q, packed_segment_base, packed_segment_stride, local_tid);
                };

                auto load_k_block = [&](int kb, int stage) {
                    constexpr int kWordsPerThread  = Dequant::kWordsPerThreadKBlock;
                    constexpr int kWordsPerKBlock  = WARPGROUP_SIZE * kWordsPerThread;
                    constexpr int kWordsPerKSlice  = kOutputFragments * kWordsPerKBlock;
                    CUTE_UNROLL
                    for (int rest_m = 0; rest_m < kRestM; ++rest_m) {
                        const int       segment = packed_segment_base + rest_m * packed_segment_stride;
                        const uint32_t* base = storage.A.data() + stage * kPackedWordsStage + kb * kWordsPerKSlice
                                               + segment * kWordsPerKBlock;
                        const uint32_t* packed = base + Dequant::packed_lane(local_tid) * kWordsPerThread;
                        auto            frag   = tCrA(cute::_, rest_m, kb);
                        Dequant::template dequant<kRestM>(
                            packed,
                            qregs,
                            rest_m,
                            kb,
                            storage,
                            reinterpret_cast<nv_bfloat16*>(frag.data()));
                    }
                };

                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                cute::warpgroup_fence_operand(accum);

                const int k_iters = sched.k_iters_;
                for (int k_tile = 0; k_tile < k_iters; ++k_tile) {
                    auto token = pipeline.consumer_try_wait(pipe_state);
                    pipeline.consumer_wait(pipe_state, token);
                    const int stage = pipe_state.index();
                    ++pipe_state;
                    if constexpr (kKTilesPerQGroup == 1) {
                        load_qparams(stage);
                    }
                    else if (k_tile % kKTilesPerQGroup == 0) {
                        load_qparams(stage);
                    }

                    load_k_block(0, stage);
                    CUTE_UNROLL
                    for (int kb = 0; kb < Traits::kKBlocksPerStage; ++kb) {
                        if (kb + 1 < Traits::kKBlocksPerStage) {
                            load_k_block(kb + 1, stage);
                        }

                        MmaIssue::run(tiled_mma, tCrA(cute::_, cute::_, kb), tCrB(cute::_, cute::_, kb, stage), accum);

                        if (kb == 1 && k_tile > 0) {
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

                const void* Cdesc = &tm_c;
                if constexpr (is_grouped_gemm) {
                    Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kTmaDescNumA + kTmaDescNumB
                            + kTmaDescNumV;
                }
                run_epilogue(Cdesc, tile, fuse_silu, output_scale, thr_mma, thr_r2s, tiled_r2s, tRS_rD_layout, accum, storage, tma_store_warp, tma_store_leader, epi_store_count);
            }
            else if (tile->is_valid_cluster) {
                const int k_iters = sched.k_iters_;
                for (int k_tile = 0; k_tile < k_iters; ++k_tile) {
                    pipeline.consumer_wait(pipe_state);
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

    template<class ThrMma, class ThrR2S, class TiledR2S, class LayoutRD, class Accum>
    __device__ static void run_epilogue(const void*               Cdesc,
                                        typename Scheduler::Tile* tile,
                                        bool                      fuse_silu,
                                        float                     output_scale,
                                        ThrMma&                   thr_mma,
                                        ThrR2S&                   thr_r2s,
                                        TiledR2S&                 tiled_r2s,
                                        const LayoutRD&           tRS_rD_layout,
                                        Accum&                    accum,
                                        SharedStorage&            storage,
                                        int                       tma_store_warp,
                                        bool                      tma_store_leader,
                                        int&                      epi_store_count)
    {
        auto tRS_rAcc     = thr_r2s.retile_S(accum);
        auto tRS_rD       = cute::make_tensor<cutlass::bfloat16_t>(tRS_rD_layout);
        auto tRS_rAcc_frg = cute::recast<cutlass::Array<float, kFragmentSize>>(tRS_rAcc);
        auto tRS_rD_frg   = cute::recast<cutlass::Array<cutlass::bfloat16_t, kFragmentSize>>(tRS_rD);

        constexpr int kMmaTileN = cute::size<0>(typename Traits::TileShape{}) / cute::size<1>(decltype(tRS_rAcc){});
        constexpr int kMmaTileM = (kSplitEpiM ? kWgM : cute::size<1>(typename Traits::TileShape{})) / cute::size<2>(decltype(tRS_rAcc){});
        constexpr int kTmaStoreCountN = kEpiN / kTmaStoreN;
        (void)kMmaTileN;

        auto emit = [&](auto fused) {
            constexpr bool kFuse = decltype(fused)::value;
            static_assert(!kFuse || kSupportsFusedSilu);
            constexpr bool kCrossWgSilu = kFuse && kNeedsCrossWgSiluExchange;
            constexpr int kStoreN     = kFuse ? TILE_N / 2 : TILE_N;
            constexpr int kEpiCountN  = kCrossWgSilu ? 1 : kStoreN / kEpiN;
            static_assert(kCrossWgSilu || kStoreN % kEpiN == 0);
            static_assert(!kCrossWgSilu || (kAtomM == 2 && kAtomN == 1 && kEpiN == 2 * kTmaStoreN && kEpiPlanes == 1));

            auto epi_synchronize = [&] {
                if constexpr (kCrossWgSilu) {
                    named_barrier_arrive_and_wait(kMathThreads, kEpilogueBarrierId);
                }
                else {
                    constexpr int kWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
                    const int barrier_id = kEpilogueBarrierId + tma_store_warp / kWarpsPerWg;
                    named_barrier_arrive_and_wait(WARPGROUP_SIZE, barrier_id);
                }
            };

            auto epi_pass_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiStripsM>{}, cute::Int<kEpiCountN>{}), cute::make_stride(cute::Int<kEpiCountN>{}, cute::_1{}));
            auto store_offset_m_layout = cute::make_layout(cute::make_shape(cute::Int<kEpiPlanes>{}, cute::Int<kEpiStripsM>{}, cute::Int<kTmaStoreCountM>{}), cute::make_stride(cute::Int<kWgM>{}, cute::Int<kEpiM>{}, cute::Int<kTmaStoreM>{}));
            auto sD = cute::as_position_independent_swizzle_tensor(cute::make_tensor(cute::make_smem_ptr(storage.D.data()), SmemLayoutD{}));
            // The copy atom follows WGMMA C's (N,M) mode order. Keep that
            // permutation local and expose conventional (M,N) coordinates below.
            auto cEpiNM     = cute::make_identity_tensor(cute::make_shape(cute::Int<kEpiN>{}, cute::Int<kEpiM>{}));
            auto tRS_cEpiNM = thr_r2s.partition_S(cEpiNM);
            auto r2s_coord_layout = cute::make_layout(cute::make_shape(cute::Int<kFragmentSize>{}, cute::size(tRS_rD_frg)));
            auto r2s_value_layout = cute::make_layout(cute::make_shape(cute::size(tRS_rD_frg), cute::Int<kMmaTileM / kEpiM>{}));
            constexpr int kTmaStoreWarpsPerWg = WARPGROUP_SIZE / WARP_SIZE;
            const int     store_wg             = tma_store_warp / kTmaStoreWarpsPerWg;

            CUTE_UNROLL
            for (int epi_m = 0; epi_m < kEpiStripsM; ++epi_m) {
                CUTE_UNROLL
                for (int epi_n = 0; epi_n < kEpiCountN; ++epi_n) {
                    const int epi_pass  = epi_pass_layout(epi_m, epi_n);
                    const int epi_stage = epi_store_count % kEpiPipeStages;
                    const int mma_n         = kFuse && !kCrossWgSilu ? 2 * epi_n : epi_n;
                    const int mma_m         = (epi_m * kEpiM) / kMmaTileM;
                    const int epi_m_in_mma  = epi_m % (kMmaTileM / kEpiM);
                    const int r2s_v         = r2s_value_layout(0, epi_m_in_mma);
                    const int epi_smem_slice = EpiStageLayout{}(kSplitEpiM ? store_wg : 0, epi_stage);
                    auto sD_epi = sD(cute::_, cute::_, epi_smem_slice);
                    auto tRS_sD = thr_r2s.partition_D(sD_epi);

                    if constexpr (kCrossWgSilu) {
                        if (tma_store_leader) {
                            cute::tma_store_wait<kEpiPipeStages - 1>();
                        }
                        epi_synchronize();

                        auto silu_exchange = cute::make_tensor(
                            reinterpret_cast<float*>(storage.D.data() + epi_smem_slice * cute::cosize_v<SmemLayoutDPlane>),
                            CrossWgSiluLayout{});
                        if (store_wg == 1) {
                            CUTE_UNROLL
                            for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                                auto up = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                CUTE_UNROLL
                                for (int j = 0; j < kFragmentSize; ++j) {
                                    const auto coord_nm = tRS_cEpiNM(r2s_coord_layout(j, epi_v));
                                    const int  n        = cute::get<0>(coord_nm) - kTmaStoreN;
                                    const int  m        = cute::get<1>(coord_nm);
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
                            if constexpr (kFuse) {
                                auto gate = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                auto up   = tRS_rAcc_frg(cute::_, mma_n + 1, mma_m)(r2s_v + epi_v);
                                CUTE_UNROLL
                                for (int j = 0; j < kFragmentSize; ++j) {
                                    if constexpr (Format::kHasGlobalScale) {
                                        dst[j] = cutlass::bfloat16_t(
                                            detail::mixed_silu_mul(gate[j] * output_scale, up[j] * output_scale));
                                    }
                                    else {
                                        dst[j] = cutlass::bfloat16_t(detail::mixed_silu_mul(gate[j], up[j]));
                                    }
                                }
                            }
                            else {
                                auto src = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                                CUTE_UNROLL
                                for (int j = 0; j < kFragmentSize; ++j) {
                                    if constexpr (Format::kHasGlobalScale) {
                                        dst[j] = cutlass::bfloat16_t(src[j] * output_scale);
                                    }
                                    else {
                                        dst[j] = cutlass::bfloat16_t(src[j]);
                                    }
                                }
                            }
                            tRS_rD_frg(epi_v) = dst;
                        }
                    }
                    else if (store_wg == 0) {
                        auto silu_exchange = cute::make_tensor(
                            reinterpret_cast<float*>(storage.D.data() + epi_smem_slice * cute::cosize_v<SmemLayoutDPlane>),
                            CrossWgSiluLayout{});
                        CUTE_UNROLL
                        for (int epi_v = 0; epi_v < cute::size(tRS_rD_frg); ++epi_v) {
                            cutlass::Array<cutlass::bfloat16_t, kFragmentSize> dst;
                            auto gate = tRS_rAcc_frg(cute::_, mma_n, mma_m)(r2s_v + epi_v);
                            CUTE_UNROLL
                            for (int j = 0; j < kFragmentSize; ++j) {
                                const auto coord_nm = tRS_cEpiNM(r2s_coord_layout(j, epi_v));
                                const int  n        = cute::get<0>(coord_nm);
                                const int  m        = cute::get<1>(coord_nm);
                                const float up      = silu_exchange(m, n);
                                if constexpr (Format::kHasGlobalScale) {
                                    dst[j] = cutlass::bfloat16_t(
                                        detail::mixed_silu_mul(gate[j] * output_scale, up * output_scale));
                                }
                                else {
                                    dst[j] = cutlass::bfloat16_t(detail::mixed_silu_mul(gate[j], up));
                                }
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
                            constexpr int kTmaStoreWarps = kMathThreads / WARP_SIZE;
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
                            const int warp_in_wg     = tma_store_warp % kTmaStoreWarpsPerWg;
                            const int first_warp     = (epi_pass * kTmaStoreCountM) % kTmaStoreWarpsPerWg;
                            const int tma_m          = (warp_in_wg + kTmaStoreWarpsPerWg - first_warp) % kTmaStoreWarpsPerWg;
                            if (tma_m < kTmaStoreCountM) {
                                const auto store_tile_coord = store_tile_layout.get_flat_coord(store_wg);
                                const int  tma_n            = cute::get<0>(store_tile_coord);
                                const int  epi_plane        = cute::get<1>(store_tile_coord);
                                const int store_m   = tile->offset_m + store_offset_m_layout(epi_plane, epi_m, tma_m);
                                const int store_n   = (kFuse ? tile->offset_n / 2 : tile->offset_n) + store_offset_n_layout(epi_n, tma_n);
                                auto sD_tma = cute::local_tile(sD(cute::_, cute::_, EpiStageLayout{}(epi_plane, epi_stage)), cute::make_shape(cute::Int<kTmaStoreN>{}, cute::Int<kTmaStoreM>{}), cute::make_coord(tma_n, tma_m));
                                cute::SM90_TMA_STORE::copy(Cdesc, cute::raw_pointer_cast(sD_tma.data()), store_n, store_m);
                            }
                        }
                    }
                    if (tma_store_leader) {
                        cute::tma_store_arrive();
                    }
                    ++epi_store_count;
                }
            }
        };

        if constexpr (kSupportsFusedSilu) {
            if (fuse_silu) {
                emit(std::true_type{});
            }
            else {
                emit(std::false_type{});
            }
        }
        else {
            emit(std::false_type{});
        }
    }
};

}  // namespace turbomind::gemm
