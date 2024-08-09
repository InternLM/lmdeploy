// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
#define TURBOMIND_ARCH_SM70 1
#else
#define TURBOMIND_ARCH_SM70 0
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#define TURBOMIND_ARCH_SM75 1
#else
#define TURBOMIND_ARCH_SM75 0
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define TURBOMIND_ARCH_SM80 1
#else
#define TURBOMIND_ARCH_SM80 0
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define TURBOMIND_ARCH_SM90 1
#else
#define TURBOMIND_ARCH_SM90 0
#endif

#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define PRAGMA_UNROLL _Pragma("unroll")
#define PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define PRAGMA_UNROLL #pragma unroll
#define PRAGMA_NO_UNROLL #pragma unroll 1
#endif
#else
#define PRAGMA_UNROLL
#define PRAGMA_NO_UNROLL
#endif

#if defined(__CUDACC__)
#define TM_HOST_DEVICE __forceinline__ __host__ __device__
#define TM_DEVICE __forceinline__ __device__
#define TM_HOST __forceinline__ __host__
#else
#define TM_HOST_DEVICE inline
#define TM_DEVICE inline
#define TM_HOST inline
#endif

constexpr int WARP_SIZE = 32;

#ifndef uint
using uint = unsigned int;
#endif

#ifndef ushort
using ushort = unsigned short int;
#endif
