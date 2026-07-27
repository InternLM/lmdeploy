#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {
namespace detail {

// Device TMA map helpers for FP8 MoE prepare (copy → replace addr/dim → publish).
__device__ __forceinline__ void copy_tma_desc_fp8(CUtensorMap* dst, const CUtensorMap* src, int lane)
{
    constexpr int kWords = (int)(sizeof(CUtensorMap) / sizeof(uint2));
    if (lane < kWords) {
        ((uint2*)dst)[lane] = ((const uint2*)src)[lane];
    }
}

__device__ __forceinline__ void replace_tma_addr_dim(CUtensorMap* desc, void* global_addr, int dim_idx, int dim)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(desc);
    // clang-format off
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" ::"r"(uint_ptr), "l"(global_addr));
    if (dim_idx == 0) {
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"r"(uint_ptr), "r"(dim));
    }
    else {
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(uint_ptr), "r"(dim));
    }
    // clang-format on
}

__device__ __forceinline__ void publish_tma_desc_fp8(CUtensorMap* gmem_desc, CUtensorMap* smem_desc)
{
    uint32_t uint_ptr = cast_smem_ptr_to_uint(smem_desc);
    // clang-format off
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(gmem_desc), "r"(uint_ptr));
    // clang-format on
}

template<int N>
__device__ __forceinline__ void rebase_publish_tma_descs_fp8(CUtensorMap*                 gmem_out,
                                                             CUtensorMap*                 smem_desc,
                                                             Array<const CUtensorMap*, N> templates,
                                                             Array<void*, N>              global_addrs,
                                                             Array<int, N>                dims,
                                                             Array<int, N>                dim_idxs,
                                                             int                          stride_desc_idx,
                                                             uint64_t                     stride_bytes,
                                                             int                          lane)
{
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        copy_tma_desc_fp8(&smem_desc[i], templates[i], lane);
    }
    __syncwarp();
    if (lane == 0) {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            replace_tma_addr_dim(&smem_desc[i], global_addrs[i], dim_idxs[i], dims[i]);
        }
        replace_tma_global_stride(&smem_desc[stride_desc_idx], stride_bytes);
    }
    __syncwarp();
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        publish_tma_desc_fp8(&gmem_out[i], &smem_desc[i]);
    }
    __syncwarp();
}

}  // namespace detail

// Per-group MoE TMA prepare for FP8 GMMA.
// Blocked-A kernels accept Blocked or Flat input; Indexed-A kernels accept Indexed, Blocked, or Flat input.
// Blocked-A: workspace [A, B, U, C]. Indexed-A: gather A/U in GEMM; workspace [B, C].
// Weight B uses either a direct Flat pointer or a StridedPtr table selected by MatrixLayout.ld == 0.
// Output C is bf16 (non-fused) or fp8_e4m3 (fused SiLU + post-quant).
template<int kAlignmentU, Striding kStridingA>
__global__ void __launch_bounds__(32, 1) prepare_moe_tma_descs_sm90_fp8(const __grid_constant__ CUtensorMap tm_a,
                                                                        const __grid_constant__ CUtensorMap tm_b,
                                                                        const __grid_constant__ CUtensorMap tm_u,
                                                                        const __grid_constant__ CUtensorMap tm_c,
                                                                        MatrixParam                         param_A,
                                                                        MatrixParam                         param_B,
                                                                        MatrixParam                         param_U,
                                                                        MatrixParam                         param_C,
                                                                        bool                                fuse_silu,
                                                                        CUtensorMap*                        out,
                                                                        int*                                offsets,
                                                                        int                                 M_total,
                                                                        int                                 N)
{
    const int g    = (int)blockIdx.x;
    const int lane = (int)threadIdx.x & 31;

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tu = float;

    const int m0 = param_A.offsets ? __ldg(param_A.offsets + g) : 0;
    const int m1 = param_A.offsets ? __ldg(param_A.offsets + g + 1) : M_total;
    const int M  = m1 - m0;

    if (lane == 0) {
        offsets[g] = m0;
        if (g + 1 == gridDim.x) {
            offsets[g + 1] = m1;
        }
    }

    if constexpr (kStridingA == Striding::kBlocked) {
        constexpr int kNum = 4;

        __shared__ __align__(128) CUtensorMap smem_desc[kNum];

        const int beg_u = m0 / kAlignmentU * kAlignmentU;
        const int end_u = round_up(m1, kAlignmentU);

        CUtensorMap* gmem_out = out + g * kNum;

        Array<const CUtensorMap*, kNum> templates;
        templates[0] = &tm_a;
        templates[1] = &tm_b;
        templates[2] = &tm_u;
        templates[3] = &tm_c;

        Array<void*, kNum> addrs;
        addrs[0]     = resolve<Ta, Striding::kBlocked>(param_A, g).ptr.ptr;
        const auto b = resolve<Tb, Striding::kBlocked>(param_B, g);
        addrs[1]     = b.ptr.ptr;
        addrs[2]     = (Tu*)param_U.ptr + beg_u;
        addrs[3]     = fuse_silu ? resolve<__nv_fp8_e4m3, Striding::kBlocked>(param_C, g).ptr.ptr :
                                   resolve<nv_bfloat16, Striding::kBlocked>(param_C, g).ptr.ptr;

        Array<int, kNum> dims;
        dims[0] = M;
        dims[1] = N;
        dims[2] = end_u - beg_u;
        dims[3] = M;

        Array<int, kNum> dim_idxs;
        dim_idxs[0] = 1;  // A
        dim_idxs[1] = 1;  // B
        dim_idxs[2] = 0;  // U (scale extent along dim0)
        dim_idxs[3] = 1;  // C

        detail::rebase_publish_tma_descs_fp8<kNum>(
            gmem_out, smem_desc, templates, addrs, dims, dim_idxs, 1, (uint64_t)b.ptr.stride * sizeof(Tb), lane);
    }
    else {
        // Indexed-A: gather activations + U scales; prepare weight B + output C only.
        constexpr int kNum = 2;

        __shared__ __align__(128) CUtensorMap smem_desc[kNum];

        CUtensorMap* gmem_out = out + g * kNum;

        Array<const CUtensorMap*, kNum> templates;
        templates[0] = &tm_b;
        templates[1] = &tm_c;

        Array<void*, kNum> addrs;
        const auto         b = resolve<Tb, Striding::kBlocked>(param_B, g);
        addrs[0]             = b.ptr.ptr;
        addrs[1]             = fuse_silu ? resolve<__nv_fp8_e4m3, Striding::kBlocked>(param_C, g).ptr.ptr :
                                           resolve<nv_bfloat16, Striding::kBlocked>(param_C, g).ptr.ptr;

        Array<int, kNum> dims;
        dims[0] = N;
        dims[1] = M;

        Array<int, kNum> dim_idxs;
        dim_idxs[0] = 1;  // B
        dim_idxs[1] = 1;  // C

        (void)tm_a;
        (void)tm_u;
        (void)param_U;
        detail::rebase_publish_tma_descs_fp8<kNum>(
            gmem_out, smem_desc, templates, addrs, dims, dim_idxs, 0, (uint64_t)b.ptr.stride * sizeof(Tb), lane);
    }
}

}  // namespace turbomind::gemm
