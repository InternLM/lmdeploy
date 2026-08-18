#pragma once

#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream);

}  // namespace turbomind::core
