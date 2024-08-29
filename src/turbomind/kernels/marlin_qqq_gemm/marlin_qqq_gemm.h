// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/allocator.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {
namespace marlin_qqq {

class MarlinQQQGemm {
public:
    // The parameterless constructor is only called for test by Python through pybind
    MarlinQQQGemm()
    {
        auto allocator = std::make_unique<turbomind::Allocator<turbomind::AllocatorType::CUDA>>(0);
        allocator_     = allocator.get();
        // The unique_ptr must be saved, or it will be released after constructing
        allocator_holder_ = std::move(allocator);
    }

    MarlinQQQGemm(IAllocator* allocator): allocator_(allocator) {}

    ~MarlinQQQGemm()
    {
        freeBuffer();
    }

    void Run(half*         D,
             const int8_t* A,
             const uint*   B,
             const float*  s1,
             const float*  s2,
             const half*   s3,
             int           prob_m,
             int           prob_n,
             int           prob_k,
             int           groupsize,
             cudaStream_t  stream);

    void setBuffer(int* reduce_buf, int* workspace_buf)
    {
        reduce_buf_    = reduce_buf;
        workspace_buf_ = workspace_buf;
    }

    std::pair<int*, int*> getBuffer()
    {
        return std::make_pair(reduce_buf_, workspace_buf_);
    }

private:
    // normally the allocation is performed by self-attn or ffn
    // allocateBuffer is only called when testing marlin qqq gemm
    void allocateBuffer(size_t workspace_size, size_t reduce_buf_size);

    void freeBuffer();

    int* workspace_buf_{};
    int* reduce_buf_{};
    bool is_allocate_buffer_{};
    // allocator_holder_ is only for test
    std::unique_ptr<turbomind::Allocator<turbomind::AllocatorType::CUDA>> allocator_holder_;
    IAllocator*                                                           allocator_;
    typedef struct {
        int thread_k;
        int thread_n;
        int num_threads;
    } thread_config_t;

    bool is_valid_config(thread_config_t const& th_config, int prob_m, int prob_n, int prob_k);

    thread_config_t determine_thread_config(int prob_m, int prob_n, int prob_k);

    thread_config_t small_batch_thread_configs[4] = {
        // Ordered by priority

        // thread_k, thread_n, num_threads
        {128, 128, 256},  // Default
        {128, 64, 128},   // Reduce N 2X, same K
        {64, 256, 256},   // Reduce K 2X, increase N 2X
        {64, 128, 128},   // Reduce K 2X, same N
    };

    thread_config_t large_batch_thread_configs[4] = {
        // Ordered by priority

        // thread_k, thread_n, num_threads
        {64, 256, 256},   // Default
        {128, 128, 256},  // Reduce N 2X, increase K 2X
        {64, 128, 128},   // Reduce N 2X, same K
        {128, 64, 128},   // Reduce N 4X, increase K 2X
    };
};
}  // namespace marlin_qqq
}  // namespace turbomind
