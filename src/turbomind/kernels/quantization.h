#include "src/turbomind/core/core.h"

namespace turbomind {

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st);

void DequantizeSymm(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st);

void QuantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> scale_, const Tensor& src, cudaStream_t st);

void DequantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> src_, const Tensor& scale, cudaStream_t st);

}  // namespace turbomind
