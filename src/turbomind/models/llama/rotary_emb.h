// Copyright (c) OpenMMLab. All rights reserved.
#pragma once
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/allocator.h"

namespace turbomind {

RotaryScalingType GetRoPEType(const std::string& type);

struct RotaryEmbeddingV2Params {
    float* rope_theta;
    int*   q_len;
    int*   k_ken;
    int    batch_size;
    int    token_num;
};

struct InnerYarnRopeParam {
    float attention_factor;
    float yarn_ramp_inv_factor_div_2;
    float yarn_ramp_inv_factor_mul_min;
    float yarn_inv_scaling_factor;
};

struct InnerLlama3RopeParam {
    float llama3_inv_scaling_factor;
    float llama3_alpha;
    float llama3_beta;
};

struct RotaryEmbeddingV2 {

    RotaryEmbeddingV2(const AttentionParam& param, cudaStream_t stream, IAllocator* allocator);

    void freeBuffer();

    void allocateBuffer(size_t token_num);

    ~RotaryEmbeddingV2()
    {
        freeBuffer();
    }

    void forward(const RotaryEmbeddingV2Params& params);

    cudaStream_t const stream_;
    IAllocator* const  allocator_;

    int               dim_;
    RotaryScalingType type_;
    float             inv_factor_{1.0};

    union {
        InnerYarnRopeParam   yarn_;
        InnerLlama3RopeParam llama3_;
    };

    // output
    float* cos_sin_;  // num_token x dim, (cos, sin, ...)
};

};  // namespace turbomind
