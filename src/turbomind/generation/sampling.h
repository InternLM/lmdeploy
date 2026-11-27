
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/generation/base_param.h"
#include "src/turbomind/kernels/speculative_sampling_kernels.h"

namespace turbomind {

struct SamplingData;

class Sampling: public BaseGenerationParam {
public:
    explicit Sampling(const BaseGenerationParam& base, int phases, int tp_rank, int parameter_capacity);

    void Setup(int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env);

    void VerifyTargetBlock(int phase, Tensor_<float> probabilities, VerifyTargetBlockParams params);

    void Fetch(int phase, TensorMap& env);

    void Update(int phase, TensorMap& env);

private:
    void ProcessDistributions(int phase, Tensor_<float> probabilities, Buffer_<int> token_indices);

    const int tp_rank_;
    const int parameter_capacity_;

    std::vector<std::shared_ptr<SamplingData>> data_;

    // host buffer
    Buffer_<int>   kept_;
    Buffer_<int>   top_k_;
    Buffer_<float> top_p_;
    Buffer_<float> min_p_;
    Buffer_<bool>  greedy_;

    Buffer_<float> sampled_logprobs_buf_;
    Buffer_<int>   sampled_indices_buf_;
    Buffer_<int>   sampled_nums_buf_;
};

}  // namespace turbomind
