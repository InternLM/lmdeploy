// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

namespace turbomind {

enum class CacheOp
{
    kDefault,  // use global when possible
    kAlways,
    kGlobal,
};

template<CacheOp cache_op, int size>
struct GetCacheOp {
    static constexpr auto value = cache_op;
};

template<>
struct GetCacheOp<CacheOp::kDefault, 16> {
    static constexpr auto value = CacheOp::kGlobal;
};

template<int size>
struct GetCacheOp<CacheOp::kDefault, size> {
    static constexpr auto value = CacheOp::kAlways;
};

enum class EvictPolicy
{
    kEvictNormal,
    kEvictFirst,
    kEvictLast,
};

namespace cache_policy {

struct Default {
    static constexpr auto kCacheOp     = CacheOp::kDefault;
    static constexpr auto kEvictPolicy = EvictPolicy::kEvictNormal;
};

struct Stream {
    static constexpr auto kCacheOp     = CacheOp::kDefault;
    static constexpr auto kEvictPolicy = EvictPolicy::kEvictFirst;
};

struct Reuse {
    static constexpr auto kCacheOp     = CacheOp::kAlways;
    static constexpr auto kEvictPolicy = EvictPolicy::kEvictNormal;
};

};  // namespace cache_policy

template<CacheOp, int size, int prefetch_size>
struct CP_ASYNC {
};

template<int prefetch_size>
struct CP_ASYNC<CacheOp::kGlobal, 16, prefetch_size> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global [%1], [%2], 16;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global.L2::cache_hint [%1], [%2], 16, %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "l"(cache_policy));
    }
    // clang-format on
};

template<>
struct CP_ASYNC<CacheOp::kGlobal, 16, 64> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global" L2_CACHEHINT(64) " [%1], [%2], 16;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global.L2::cache_hint" L2_CACHEHINT(64) " [%1], [%2], 16, %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "l"(cache_policy));
    }
    // clang-format on
};

template<>
struct CP_ASYNC<CacheOp::kGlobal, 16, 128> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], 16;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global.L2::cache_hint" L2_CACHEHINT(128) " [%1], [%2], 16, %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "l"(cache_policy));
    }
    // clang-format on
};

template<>
struct CP_ASYNC<CacheOp::kGlobal, 16, 256> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global" L2_CACHEHINT(256) " [%1], [%2], 16;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global.L2::cache_hint" L2_CACHEHINT(256) " [%1], [%2], 16, %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "l"(cache_policy));
    }
    // clang-format on
};

template<int size, int prefetch_size>
struct CP_ASYNC<CacheOp::kAlways, size, prefetch_size> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global.L2::cache_hint [%1], [%2], %3, %4;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size), "l"(cache_policy));
    }
    // clang-format on
};

template<int size>
struct CP_ASYNC<CacheOp::kAlways, size, 64> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global" L2_CACHEHINT(64) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global.L2::cache_hint" L2_CACHEHINT(64) " [%1], [%2], %3, %4;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size), "l"(cache_policy));
    }
    // clang-format on
};

template<int size>
struct CP_ASYNC<CacheOp::kAlways, size, 128> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global.L2::cache_hint" L2_CACHEHINT(128) " [%1], [%2], %3, %4;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size), "l"(cache_policy));
    }
    // clang-format on
};

template<int size>
struct CP_ASYNC<CacheOp::kAlways, size, 256> {
    // clang-format off
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global" L2_CACHEHINT(256) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size));
    }
    __device__ static void apply(int smem_ptr, const void* __restrict__ src, uint64_t cache_policy, bool mask)
    {
        asm volatile("{\n  .reg .pred p;\n  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global.L2::cache_hint" L2_CACHEHINT(256) " [%1], [%2], %3, %4;\n"
                     "}\n" ::"r"((int)mask), "r"(smem_ptr), "l"(src), "n"(size), "l"(cache_policy));
    }
    // clang-format on
};

}  // namespace turbomind
