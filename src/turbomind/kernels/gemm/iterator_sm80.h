// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/predicate.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <cassert>
#include <type_traits>

namespace turbomind::gemm {

template<class T, class Map, class SmemLayout, Pack kPack, Order kOrder, bool AlignedC, bool AlignedS, class Policy_>
struct GmemIteratorSm80 {

    using ThreadMap = Map;

    using AccessType = Array<T, Map::kAccessC>;
    using Pointer    = get_pointer_type<T>;

    using Policy = Policy_;

    static constexpr int ITER_S = Map::kIterS;
    static constexpr int ITER_C = Map::kIterC;

    const char* src_data_;

    int src_offset_;
    int dst_offset_;

    int offset_c_;
    int offset_s_;

    int src_step_c_;
    int src_step_s_;

    int src_step_k_;

    Predicate<Map::kIterS, Map::kIterC, (AlignedC && Map::kAlignedC), (AlignedS && Map::kAlignedS)> pred_;

    bool g_mask{true};

    SmemAccessor<T, SmemLayout> smem_data_;

    static constexpr int2 kMK0     = cs2mk<kOrder>(SmemLayout::C0, SmemLayout::S0);
    static constexpr int  kPeriodC = ceil_div(SmemLayout::C0, Map::kDeltaC);
    static constexpr int  kPeriodS = ceil_div(SmemLayout::S0, Map::kDeltaS);

    int phases_[kPeriodS][kPeriodC];

    uint64_t cache_policy_{};

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

        if constexpr (pred_.is_active) {
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

        src_step_c_ = bitsof<T> * Map::kDeltaC / bitsof<char>;
        src_step_s_ = bitsof<T> * Map::kDeltaS * stride_s / bitsof<char>;

        src_step_k_ = bitsof<T> * cs2mk<kOrder>(Map::kDimC, Map::kDimS * stride_s).y / bitsof<char>;

        // initialize for the first tile
        src_data_ = src_ptr + src_offset_;

#if TURBOMIND_ARCH_SM80
        if constexpr (Policy::kEvictPolicy != EvictPolicy::kEvictNormal) {
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;\n" : "=l"(cache_policy_) :);
        }
#endif
    }

    __device__ constexpr int _src_step_k() const
    {
        return src_step_k_;
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

        constexpr int prefetch_size = std::min(256, size * Map::kWarpThreadC);

        auto ptr = cast_smem_ptr_to_uint(dst);

        static constexpr auto cache_op = GetCacheOp<Policy::kCacheOp, size>::value;

        if constexpr (Policy::kEvictPolicy != EvictPolicy::kEvictNormal) {
            CP_ASYNC<cache_op, size, prefetch_size>::apply(ptr, src, cache_policy_, mask);
        }
        else {
            CP_ASYNC<cache_op, size, prefetch_size>::apply(ptr, src, mask);
        }
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }
};

template<class Policy>
struct IteratorSm80 {
    template<class T, class Map, class SmemLayout, Pack kPack, Order kOrder, bool AlignedC, bool AlignedS>
    using Type = GmemIteratorSm80<T, Map, SmemLayout, kPack, kOrder, AlignedC, AlignedS, Policy>;
};

}  // namespace turbomind::gemm
