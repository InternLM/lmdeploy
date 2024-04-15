#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gemm_template.h"

namespace turbomind::gemm {

template void invoke(half* C, const half* A, const half* B, int m, int n, int k, cudaStream_t st);
template void invoke(half* C, const half* A, const uint4_t* B, int m, int n, int k, cudaStream_t st);

}