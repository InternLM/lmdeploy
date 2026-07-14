

#pragma once

#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/smem.h"

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

}  // namespace turbomind::gemm
