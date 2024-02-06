#pragma once

#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class SmemLayout, int Idx>
struct Sm80GmemIterator: BaseGmemIterator<T, Map, SmemLayout> {

    using Base = BaseGmemIterator<T, Map, SmemLayout>;

    using typename Base::AccessType;

    using Base::Base;
    using Base::kElementSize;
    using Base::src_offset_;
    using Base::dst_offset_;
    using Base::offset_c_;
    using Base::offset_s_;
    using Base::smem_;

    template<bool is_residue, class TileIter>
    __device__ void Prefetch(const TileIter& tile_iter, int s_begin, int s_count, int max_s, int offset)
    {
        auto src_data = tile_iter.OffsetData<Idx>(src_offset_);

        PRAGMA_UNROLL
        for (int s = s_begin; s < s_begin + s_count; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                auto dst = SmemLayout::apply(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
                auto src = &src_data[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC];
                if constexpr (is_residue) {
                    CpAsync(dst + offset, (const T*)src, offset_s_ + s * Map::kDeltaS < max_s);
                }
                else {
                    CpAsync(dst + offset, (const T*)src);
                }
            }
        }
    }

    template<bool is_residue, class TileIter>
    __device__ void Prefetch(const TileIter& tile_iter, int max_s, int offset)
    {
        Prefetch<is_residue>(tile_iter, 0, Map::kIterS, max_s, offset);
    }

    __device__ void CpAsync(int dst, const T* __restrict__ src, bool mask)
    {
#if TURBOMIND_ARCH_SM80
        constexpr int cp_size = sizeof(AccessType);
        uint32_t      ptr     = sizeof(T) * dst;
        // clang-format off
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(ptr),
                     "l"(src),
                     "n"(cp_size));
        // clang-format on
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    __device__ void CpAsync(int dst, const T* __restrict__ src)
    {
#if TURBOMIND_ARCH_SM80
        constexpr int cp_size = sizeof(AccessType);
        uint32_t      ptr     = sizeof(T) * dst;
        asm volatile(
            "cp.async.cg.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(cp_size));
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }
};

}  // namespace turbomind