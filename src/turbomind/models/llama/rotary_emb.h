// Copyright (c) OpenMMLab. All rights reserved.
#pragma once
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/allocator.h"

namespace turbomind {

enum class RotaryScalingType
{
    kDefault,
    kLinear,
    kDynamic,
    kYarn,
    kLlama3,
    kMrope
};

struct RotaryEmbeddingV2Params {
    float* rope_theta;
    int*   q_len;
    int*   k_ken;
    int    batch_size;
    int    token_num;
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

    RotaryScalingType  type_;
    cudaStream_t const stream_;
    IAllocator* const  allocator_;

    // output
    float* cos_sin_;  // num_token x dim, (cos, sin, ...)

    int dim_;
    // default, linear, dynamic
    float attention_factor_;
    float rope_scaling_factor_;
    float inv_scale_factor_;
    // llama3
    float llama3_inv_scaling_factor_;
    float llama3_alpha_;
    float llama3_beta_;
    // yarn
    float yarn_ramp_inv_factor_div_2_;
    float yarn_ramp_inv_factor_mul_min_;
    float yarn_inv_scaling_factor_;
    // mrope
    int3 mrope_section_;
};

};  // namespace turbomind
