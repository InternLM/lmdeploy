#include "src/turbomind/core/core.h"

namespace turbomind {

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st);

void DequantizeSymm(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st);

void QuantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> scale_, const Tensor& src, cudaStream_t st);

void DequantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> src_, const Tensor& scale, cudaStream_t st);

void QuantizeGroupwise(Tensor            quant,    // (m,k)
                       Tensor            scales,   // (m,k/g)
                       Tensor            zeros,    // (m,k/g)
                       Tensor            dequant,  // (m,k)
                       Tensor            src,      // (m,k)
                       Buffer_<unsigned> rbits,    // (m*k)
                       int               group_size);

}  // namespace turbomind
