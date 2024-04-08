// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include <cassert>

namespace turbomind::gemm {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class SmemLayout, int Idx>
struct GmemIteratorSm80 {

    using AccessType = Array<T, Map::kAccessC>;
    using Pointer    = get_pointer_type<T>;

    Pointer smem_;

    int src_offset_;
    int offset_c_;
    int offset_s_;

    int stride_s_;

    __device__ GmemIteratorSm80(int stride_s): stride_s_{stride_s}
    {
        int  warp_id = threadIdx.x / WARP_SIZE;
        int  lane_id = threadIdx.x % WARP_SIZE;
        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * stride_s_;
        offset_c_    = offsets.x;
        offset_s_    = offsets.y;
    }

    __device__ void SetSmem(Pointer smem)
    {
        smem_ = smem;
    }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        SmemAccessor<T, SmemLayout> data{smem_};
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&data(offset_s_ + s * Map::kDeltaS,  //
                            offset_c_ + c * Map::kDeltaC,
                            pipe_iter * SmemLayout::kSize),
                      Array<T, Map::kAccessC>{});
            }
        }
    }

    template<class Iter>
    __device__ void Prefetch(const Iter& iter, int begin, int count, int pipe_iter)
    {
        auto src_data = iter.OffsetPtr<Idx>(src_offset_);

        SmemAccessor<T, SmemLayout> dst_data{smem_};

        PRAGMA_UNROLL
        for (int s = begin; s < begin + count && s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                auto       dst = cast_smem_ptr_to_uint(&dst_data(offset_s_ + s * Map::kDeltaS,  //
                                                           offset_c_ + c * Map::kDeltaC,
                                                           pipe_iter * SmemLayout::kSize));
                auto       src = &src_data[s * Map::kDeltaS * stride_s_ + c * Map::kDeltaC];
                // AccessType vec;
                // Load(vec, src_data);
                // if (Idx == 0) {
                //     for (int i = 0; i < vec.size(); ++i) {
                //         printf("%d %d %d %d %f\n", (threadIdx.x % WARP_SIZE), i, offset_s_, offset_c_, (float)vec[i]);
                //     }
                // }
                CpAsync(std::false_type{}, dst, (const T*)src, true);
            }
        }
    }

    template<class Iter>
    __device__ void Prefetch(const Iter& iter, int pipe_iter)
    {
        Prefetch(iter, 0, Map::kIterS, pipe_iter);
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

}  // namespace turbomind::gemm