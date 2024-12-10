// Copyright (c) OpenMMLab. All rights reserved.
#pragma once
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/allocator.h"

namespace turbomind {

RopeType GetRoPEType(const std::string& type);

struct RotaryEmbeddingV2Param {
    float* rope_theta;
    int*   q_len;
    int*   k_len;
    int*   h_q_len;
    int*   h_k_len;
    int    dc_size;
    int    batch_size;
    int    token_num;
};

struct InnerYarnRopeParam {
    float attention_factor;
    float ramp_inv_factor_div_2;
    float ramp_inv_factor_mul_min;
    float inv_scaling_factor;
};

struct InnerLlama3RopeParam {
    float inv_scaling_factor;
    float alpha;
    float beta;
};

template<typename T>
struct RotaryEmbeddingV2 {

    RotaryEmbeddingV2(const AttentionParam& param, int session_len, cudaStream_t stream, IAllocator* allocator);

    void computeCache(int session_len);

    void updateCache(const RotaryEmbeddingV2Param& params);

    void freeBuffer();

    void allocateBuffer(int token_num);

    ~RotaryEmbeddingV2()
    {
        freeBuffer();
    }

    void updateMapping(const RotaryEmbeddingV2Param& params);

    void forward(const RotaryEmbeddingV2Param& params);

    cudaStream_t const stream_;
    IAllocator* const  allocator_;

    int      dim_;
    RopeType type_;
    float    inv_factor_{1.0};

    union {
        InnerYarnRopeParam   yarn_;
        InnerLlama3RopeParam llama3_;
    };

    int*   q2b_;
    void*  d_temp_storage_{nullptr};
    size_t temp_storage_bytes_{};
    float  rope_base_{};
    int    cached_len_{};

    // output
    T* cos_sin_;  // num_token  x dim, (cos, sin, ...) for dynamic
                  // cached_len x dim, (cos, sin, ...) for other rope
    int* q2p_;
};

};  // namespace turbomind
