#include "src/turbomind/core/core.h"

namespace turbomind {

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st);

void DequantizeSymm(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st);

void QuantizeSymmBlock(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st);

void DequantizeSymmBlock(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st);

}  // namespace turbomind
