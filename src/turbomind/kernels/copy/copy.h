#pragma once

#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

// Elementwise device copy between tensors of the same shape and dtype.
// Source broadcast strides are supported. Destination elements and the two
// buffers must not overlap. At most four jointly coalesced axes are supported.
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream);

}  // namespace turbomind::core
