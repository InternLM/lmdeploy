#pragma once

#include "src/turbomind/core/core.h"

namespace turbomind {

enum class ActivationType
{
    kSilu,
    kSiluGptOss
};

void Activation(
    Ref<Tensor> gate, const Tensor& up, ActivationType type, const int* total_tokens_ptr, cudaStream_t stream);

void Activation(Tensor&             gate_up,  //
                const Tensor&       bias,
                const Buffer_<int>& group_ids,
                ActivationType      type,
                const int*          total_tokens_ptr,
                cudaStream_t        stream);

}  // namespace turbomind
