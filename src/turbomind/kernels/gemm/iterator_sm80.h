// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <cassert>
#include <type_traits>

namespace turbomind::gemm {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<int S, int C, bool AlignedS, bool AlignedC>
struct Predicate {

    static constexpr int kSizeC = AlignedC ? 1 : C;

    static_assert(S * kSizeC <= 32);

    uint32_t pred_{};

    static constexpr std::true_type active()
    {
        return {};
    }

    __device__ int operator()(int s, int c) const
    {
        return (pred_ & (1 << (s * kSizeC + c))) != 0;
    }

    __device__ void set(int s, int c)
    {
        pred_ |= (1 << (s * kSizeC + c));
    }

    __device__ void clear()
    {
        pred_ = 0;
    }
};

template<int S, int C>
struct Predicate<S, C, true, true> {

    static constexpr std::false_type active()
    {
        return {};
    }

    __device__ constexpr std::integral_constant<int, 1> operator()(int, int) const
    {
        return {};
    }

    __device__ void set(int, int) {}

    __device__ void clear()
    {
        // pred_ = 0;
    }
};

template<class T, class Map, class SmemLayout, Pack kPack, Order kOrder, bool AlignedC, bool AlignedS>
struct GmemIteratorSm80 {

    using ThreadMap = Map;

    using AccessType = Array<T, Map::kAccessC>;
    using Pointer    = get_pointer_type<T>;

    static constexpr int ITER_S = Map::kIterS;
    static constexpr int ITER_C = Map::kIterC;

    const char* src_data_;

    int src_offset_;
    int dst_offset_;

    int offset_c_;
    int offset_s_;

    int src_step_c_;
    int src_step_s_;

    // int stride_s_;

    Predicate<Map::kIterS, Map::kIterC, (AlignedC && Map::kAlignedC), (AlignedS && Map::kAlignedS)> pred_;

    bool g_mask{true};

    SmemAccessor<T, SmemLayout> smem_data_;

    static constexpr int2 kMK0     = cs2mk<kOrder>(SmemLayout::C0, SmemLayout::S0);
    static constexpr int  kPeriodC = ceil_div(SmemLayout::C0, Map::kDeltaC);
    static constexpr int  kPeriodS = ceil_div(SmemLayout::S0, Map::kDeltaS);

    int phases_[kPeriodS][kPeriodC];

    __device__ static constexpr int2 pack(int2 mk)
    {
        return Packing_v2<kPack, kOrder>::apply(mk);
    }

    __device__ static constexpr int2 to_cs(int2 mk)
    {
        return mk2cs<kOrder>(mk.x, mk.y);
    }

    __device__ GmemIteratorSm80(): smem_data_{Pointer{nullptr}} {};

    __device__ GmemIteratorSm80(Pointer data, int stride_s, int2 offset, int2 extent): smem_data_{Pointer{(T*)nullptr}}
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        data   = data + cs2idx(to_cs(pack(offset)), stride_s);
        extent = to_cs(pack(extent));

        int2 offsets    = Map::get_offset(warp_id, lane_id);
        int  src_offset = offsets.x + offsets.y * stride_s;

        offset_c_ = offsets.x;
        offset_s_ = offsets.y;

        auto src_ptr = reinterpret_cast<const char*>((T*)data);

        if constexpr (pred_.active()) {
            PRAGMA_UNROLL
            for (int s = 0; s < Map::kIterS; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < Map::kIterC; ++c) {
                    int ss = offset_s_ + s * Map::kDeltaS;
                    int cc = offset_c_ + c * Map::kDeltaC;
                    if (ss < extent.y && cc < extent.x) {
                        pred_.set(s, c);
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < kPeriodS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < kPeriodC; ++c) {
                phases_[s][c] = SmemLayout::apply(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
            }
        }

        src_offset_ = src_offset * bitsof<T> / bitsof<char>;
        src_step_c_ = Map::kDeltaC * bitsof<T> / bitsof<char>;
        src_step_s_ = Map::kDeltaS * stride_s * bitsof<T> / bitsof<char>;

        // initialize for the first tile
        src_data_ = src_ptr + src_offset_;
    }

    __device__ constexpr int _src_step_k() const
    {
        return cs2mk<kOrder>(src_step_c_ * Map::kIterC * Map::kWarpC, src_step_s_ * Map::kIterS * Map::kWarpS).y;
    }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int pred_s = offset_s_ + s * Map::kDeltaS < Map::kDimS;
                const int pred_c = offset_c_ + c * Map::kDeltaC < Map::kDimC;
                auto      ptr    = &smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
                if ((Map::kAlignedC && Map::kAlignedS) || (pred_s && pred_c)) {
                    Store(ptr, Array<T, Map::kAccessC>{});
                }
            }
        }
    }

    __device__ void Prefetch(int begin, int count, bool tile_mask)
    {
        PRAGMA_UNROLL
        for (int s = begin; s < begin + count && s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                // auto dst = &smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);

                const int i0  = SmemLayout::apply(  //
                    s / kPeriodS * kPeriodS * Map::kDeltaS,
                    c / kPeriodC * kPeriodC * Map::kDeltaC);
                const int i1  = phases_[s % kPeriodS][c % kPeriodC];
                auto      dst = &smem_data_.ptr_[i0 + i1];

                CpAsync(std::true_type{}, dst, src_data_ + src_step_c_ * c, tile_mask && g_mask && pred_(s, c));
            }
            src_data_ += src_step_s_;
            if (s == Map::kIterS - 1) {
                src_data_ -= src_step_s_ * Map::kIterS;
                src_data_ += _src_step_k();
            }
        }
    }

    __device__ void Prefetch(bool tile_mask)
    {
        Prefetch(0, Map::kIterS, tile_mask);
    }

    __device__ void Advance()
    {
        if (!g_mask) {
            src_data_ -= _src_step_k();
        }
    }

    __device__ void CpAsync(std::true_type, T* dst, const char* __restrict__ src, bool mask)
    {
#if TURBOMIND_ARCH_SM80
        constexpr int size = sizeof(AccessType);
        static_assert(size <= 16);
        auto ptr = cast_smem_ptr_to_uint(dst);
        // clang-format off
        if constexpr (size == 16) {
            asm volatile("{\n"
                        "  .reg .pred p;\n"
                        "  setp.ne.b32 p, %0, 0;\n"
                        "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                        "}\n" ::"r"((int)mask),
                        "r"(ptr),
                        "l"(src),
                        "n"(size));
        } else {
            asm volatile("{\n"
                        "  .reg .pred p;\n"
                        "  setp.ne.b32 p, %0, 0;\n"
                        "  @p cp.async.ca.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                        "}\n" ::"r"((int)mask),
                        "r"(ptr),
                        "l"(src),
                        "n"(size));
        }
        // clang-format on
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    __device__ void CpAsync(std::false_type, T* dst, const char* __restrict__ src, bool)
    {
#if TURBOMIND_ARCH_SM80
        auto          ptr  = cast_smem_ptr_to_uint(dst);
        constexpr int size = sizeof(AccessType);
        if constexpr (size == 16) {
            asm volatile(
                "cp.async.cg.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
        }
        else {
            asm volatile(
                "cp.async.ca.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
        }
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }
};

struct IteratorSm80 {
    template<class T, class Map, class SmemLayout, Pack kPack, Order kOrder, bool AlignedC, bool AlignedS>
    using Type = GmemIteratorSm80<T, Map, SmemLayout, kPack, kOrder, AlignedC, AlignedS>;
};

}  // namespace turbomind::gemm
