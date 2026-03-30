// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/universal_vector.h>

#include "src/turbomind/kernels/unfused_attention_kernels.h"

namespace turbomind {

template<class T>
void invokeApplyRotaryEmbedding(T*           k_cache,
                                int          max_k_len,
                                int          head_num,
                                int          head_dim,
                                float        rope_base,
                                int          rope_dim,
                                int          batch_size,
                                cudaStream_t stream = {});

template<class T>
class Reference {
public:
    explicit Reference(cudaStream_t stream);

    void Reshape(size_t max_q_len,
                 size_t max_k_len,
                 size_t head_num,
                 size_t head_dim,
                 size_t kv_head_num,
                 size_t batch_size,
                 int    window_size);

    void Execute(T*       output,
                 T*       k_cache,
                 T*       v_cache,
                 const T* qkv,
                 const T* qkv_bias,
                 const T* sinks,
                 float    rope_base,
                 int      rope_dim);

    const float* qk() const
    {
        return qk_.data().get();
    }

    const T* pr() const
    {
        return pr_.data().get();
    }

    const T* mask() const
    {
        return mask_.data().get();
    }

private:
    cudaStream_t                    stream_;
    cublasHandle_t                  cublas_;
    thrust::universal_vector<T>     mask_;
    thrust::universal_vector<float> qk_;
    thrust::universal_vector<T>     pr_;
    thrust::universal_vector<T>     q_;
    thrust::universal_vector<T>     out_;

    thrust::universal_vector<T> keys_;
    thrust::universal_vector<T> vals_;

    int max_q_len_{};
    int max_k_len_{};
    int head_num_{};
    int head_dim_{};
    int kv_head_num_{};
    int batch_size_{};
    int window_size_{};
};

}  // namespace turbomind
