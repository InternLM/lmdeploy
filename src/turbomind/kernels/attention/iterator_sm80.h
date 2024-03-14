#pragma once

#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <type_traits>

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
    using Base::offset_c_;
    using Base::offset_s_;
    using Base::smem_;

    using Base::partial_c_;
    using Base::pred_c_;

    template<class PartialS, class TileIter>
    __device__ void
    Prefetch(PartialS partial_s, const TileIter& tile_iter, int s_begin, int s_count, int max_s, int pipe_iter)
    {
        // `src_data` may be `SubBytePtr`
        auto src_data = tile_iter.OffsetPtr<Idx>(src_offset_);

        SmemAccessor<T, SmemLayout> dst_data{smem_};

        PRAGMA_UNROLL
        for (int s = s_begin; s < s_begin + s_count && s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                auto dst = cast_smem_ptr_to_uint(&dst_data(offset_s_ + s * Map::kDeltaS,  //
                                                           offset_c_ + c * Map::kDeltaC,
                                                           pipe_iter * SmemLayout::kSize));
                auto src = &src_data[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC];

                if constexpr (partial_c_) {
                    CpAsync(std::true_type{}, dst, (const T*)src, pred_c_);
                }
                else {
                    CpAsync(partial_s, dst, (const T*)src, offset_s_ + s * Map::kDeltaS < max_s);
                }
            }
        }
    }

    template<class Partial, class TileIter>
    __device__ void Prefetch(Partial partial, const TileIter& tile_iter, int max_s, int pipe_iter)
    {
        Prefetch(partial, tile_iter, 0, Map::kIterS, max_s, pipe_iter);
    }

    __device__ void CpAsync(std::true_type, int ptr, const T* __restrict__ src, bool mask)
    {
#if TURBOMIND_ARCH_SM80
        constexpr int size = sizeof(AccessType);
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

    __device__ void CpAsync(std::false_type, int ptr, const T* __restrict__ src, bool)
    {
#if TURBOMIND_ARCH_SM80
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

}  // namespace turbomind
