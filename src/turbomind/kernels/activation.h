#pragma once

#include "src/turbomind/core/core.h"

namespace turbomind {

enum class ActivationType
{
    kSilu,
    kSiluGptOss,
    kGeluPytorchTanh,
    kGelu
};

void Activation(Ref<Tensor> gate, const Tensor& up, ActivationType type, cudaStream_t stream);

// num_valid_tokens points to a device-side valid token count for padded token buffers.
void Activation(
    Ref<Tensor> gate, const Tensor& up, ActivationType type, const int* num_valid_tokens, cudaStream_t stream);

void Activation(Tensor&             gate_up,  //
                const Tensor&       bias,
                const Buffer_<int>& group_ids,
                ActivationType      type,
                cudaStream_t        stream);

// num_valid_tokens points to a device-side valid token count for padded token buffers.
void Activation(Tensor&             gate_up,  //
                const Tensor&       bias,
                const Buffer_<int>& group_ids,
                ActivationType      type,
                const int*          num_valid_tokens,
                cudaStream_t        stream);

// In-place add-bias + unary activation: x <- activation(x + bias).
// `x` is a 2D tensor; `bias` (optional) broadcasts over the last dim.
// Supports kGelu (erf) and kGeluPytorchTanh (tanh approximation).
void invokeAddBiasActivation(Tensor& x, const Tensor& bias, ActivationType type, cudaStream_t stream);

}  // namespace turbomind
