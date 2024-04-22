// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include <cassert>

namespace turbomind::gemm {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class SmemLayout, int Idx, int G_CTA = 1, bool Disable = false>
struct GmemIteratorSm80 {

    using AccessType = Array<T, Map::kAccessC>;
    using Pointer    = get_pointer_type<T>;

    const char* src_data_;
    const char* src_ptr_;
    // int         src_delta_;

    int src_offset_;
    int dst_offset_;

    int offset_c_;
    int offset_s_;

    // int iter_c_{};
    // int iter_s_{};

    int src_step_c_;
    int src_step_s_;
    // int src_step_k_;

    int stride_s_;
    int stride_k_;
    // int smem_offset_ = 0;

    int  g_counter_{};
    bool g_mask_{true};

    SmemAccessor<T, SmemLayout> smem_data_;

    __device__ GmemIteratorSm80(Pointer data, int stride_s, int stride_k): smem_data_{Pointer{nullptr}}
    {
        if constexpr (Disable) {
            return;
        }

        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * stride_s;

        offset_c_ = offsets.x;
        offset_s_ = offsets.y;

        src_ptr_ = reinterpret_cast<const char*>((T*)data);

        stride_s_   = stride_s * bitsof<T> / bitsof<char>;
        stride_k_   = stride_k * bitsof<T> / bitsof<char>;
        src_offset_ = src_offset_ * bitsof<T> / bitsof<char>;

        src_step_c_ = Map::kDeltaC * bitsof<T> / bitsof<char>;
        src_step_s_ = Map::kDeltaS * stride_s_;  // - Map::kIterC * Map::kDeltaC * bitsof<T> / bitsof<char>;
        // src_step_k_ = stride_k_ - Map::kIterS * Map::kDeltaS * stride_s_;

        // initialize for the first tile
        src_data_ = src_ptr_ + src_offset_;
        src_ptr_ += stride_k_;
    }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        if constexpr (Disable) {
            return;
        }
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC),
                      Array<T, Map::kAccessC>{});
            }
        }
    }

    __device__ void Prefetch(int begin, int count, bool tile_mask)
    {
        if constexpr (Disable) {
            return;
        }
        // if constexpr (SmemLayout::kIsTrivial) {
        //     if (begin == 0) {
        //         smem_data_.ptr_ += dst_offset_;
        //     }
        // }

        // if (G_CTA > 1 && begin == 0 && threadIdx.x == 0 && blockIdx.x == 0 && Idx == 2) {
        //     printf("[prefetch] counter=%d, g_mask=%d, t_mask=%d, src_data=%p\n", g_counter_, (int)g_mask_,
        //     (int)tile_mask, src_data_);
        // }

        // __syncthreads();

        // if (!g_mask_) {
        //     return;
        // }

        PRAGMA_UNROLL
        for (int s = begin; s < begin + count && s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                auto dst = &smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
                // if constexpr (SmemLayout::kIsTrivial) {
                //     dst = smem_data_.ptr_;
                //     smem_data_.ptr_ += Map::kDeltaC;
                // }

                CpAsync(std::true_type{}, dst, src_data_ + src_step_c_ * c, g_mask_ && tile_mask);

                // if (g_mask_ && tile_mask) {
                //     AccessType tmp;
                //     Load(tmp, (T*)src_data_);
                //     Store(dst, tmp);
                //     if constexpr (Idx == 2) {
                //         printf("fuck: %f %f\n", (float)tmp[0].x, (float)tmp[0].y);
                //     }
                // }

                // if constexpr (Idx != 2) {
                // src_data_ += src_step_c_;
                // }
            }
            // if constexpr (SmemLayout::kIsTrivial) {
            //     smem_data_.ptr_ += Map::kDeltaS * SmemLayout::C - Map::kIterC * Map::kDeltaC;
            // }
            if (g_mask_) {
                src_data_ += src_step_s_;
            }
        }

        // PRAGMA_UNROLL
        // for (int s = 0; s < Map::kIterS; ++s) {
        //     PRAGMA_UNROLL
        //     for (int c = 0; c < Map::kIterC; ++c) {
        //         auto dst = &smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
        //         if (g_mask_ && tile_mask) {
        //             auto tmp = *(AccessType*)dst;
        //             if constexpr (Idx == 2) {
        //                 printf("%f %f\n", (float)tmp[0], (float)tmp[1]);
        //             }
        //         }
        //     }
        // }
    }

    __device__ void Prefetch(bool tile_mask)
    {
        Prefetch(0, Map::kIterS, tile_mask);
    }

    __device__ void Advance()
    {
        if constexpr (Disable) {
            return;
        }
        // src_ptr_ += stride_k_;
        // src_data_ = src_ptr_ + src_offset_;

        // if (G_CTA > 1 && threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("[advance]  counter=%d, g_mask=%d\n", g_counter_, (int)g_mask_);
        // }

        if (g_mask_) {
            src_data_ = src_ptr_ + src_offset_;
            src_ptr_ += stride_k_;
        }

        ++g_counter_;
        g_mask_ = g_counter_ % G_CTA == 0;
        // ++g_counter_;
        // if constexpr (Idx == 2) {
        //     if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //         printf("%d\n", g_counter_);
        //     }
        // }
        // if (g_counter_ % G_CTA == 0 && G_CTA == 1) {
        //     src_ptr_ += stride_k_;
        //     // if constexpr (Idx == 2) {
        //     //     printf("%p\n", src_ptr_);
        //     // }
        // }
    }

#if 0
    // generates almost the same binary as batch version above
    __device__ void Prefetch(bool tile_mask)
    {
        auto dst = &smem_data_(offset_s_ + iter_s_ * Map::kDeltaS, offset_c_ + iter_c_ * Map::kDeltaC);
        CpAsync(std::true_type{}, dst, src_data_, tile_mask);

        src_data_ += src_step_c_;
        ++iter_c_;

        if (iter_c_ < Map::kIterC) {
            return;
        }

        iter_c_ = 0;
        src_data_ += src_step_s_;
        ++iter_s_;

        if (iter_s_ < Map::kIterS) {
            return;
        }

        iter_s_ = 0;

        // advancing to next stage in a parallel data path is faster
        // src_data_ += src_step_k_;
    }
#endif

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

}  // namespace turbomind::gemm