// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeMoeGate_a2a(float*       topk_scales,
                       int*         topk_experts,
                       int*         token_idx_in_rank,
                       const float* logits,
                       int          tokens,
                       int          experts,
                       int          ep_size,
                       int          experts_per_token,
                       bool         softmax,
                       bool         norm_topk,
                       float        routed_scale,
                       cudaStream_t stream);

void invokeMoeScan_a2a(int*         f2n,
                       int*         f2E,
                       int*         en2f,
                       int*         offsets,
                       int8_t*      masks,
                       int*         accum,
                       int          token_num,
                       int          token_num_padded,
                       int          local_expert_num,
                       cudaStream_t stream);

void invokeMoeCombine_a2a(Ref<Tensor>   out_,
                          const Tensor& src,
                          const Tensor& bias,
                          const float*  scales,
                          const int*    en2f,
                          const int*    f2E,
                          int           experts_per_token,
                          cudaStream_t  st);

template<typename T>
struct ZeroCopyItem {
    ZeroCopyItem()
    {
        check_cuda_error(cudaMallocHost(&host_, sizeof(T), cudaHostAllocMapped));
        check_cuda_error(cudaHostGetDevicePointer(&mapped_, host_, 0));
    }

    ~ZeroCopyItem()
    {
        check_cuda_error(cudaFreeHost(host_));
    }

    ZeroCopyItem(const ZeroCopyItem&) = delete;
    ZeroCopyItem& operator=(const ZeroCopyItem&) = delete;

    T& operator*()
    {
        return *host_;
    }

    T* mapped()
    {
        return mapped_;
    }

    T* host_{};
    T* mapped_{};
};

}  // namespace turbomind
