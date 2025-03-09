#pragma once
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Macros for error checking
#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = (cmd);                                                    \
    if (r != ncclSuccess && r != ncclInProgress) {                             \
      fprintf(stderr,                                                          \
              "NCCL Error at %s:%d - %d. Refer to "                            \
              "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/" \
              "types.html#c.ncclResult_t for error code meanings\n",           \
              __FILE__, __LINE__, r);                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(e));                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail(const char *__assertion, const char *__file,
                          unsigned int __line, const char *__function) __THROW
    __attribute__((__noreturn__));

#define __ASSERT_FUNCTION __extension__ __PRETTY_FUNCTION__
#define assert_whenever(expr)                                                  \
  (static_cast<bool>(expr)                                                     \
       ? void(0)                                                               \
       : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
}

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
