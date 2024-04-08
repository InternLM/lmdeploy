
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include <cublas_v2.h>
#include <thrust/universal_vector.h>

using namespace turbomind;
using thrust::universal_vector;

cublasHandle_t cublas_handle{};

void ComputeRefCpu(half* C, const half* A, const half* B, int m, int n, int k)
{
    for (int mm = 0; mm < m; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float c = 0;
            for (int kk = 0; kk < k; ++kk) {
                c += (float)A[mm * k + kk] * (float)B[nn * k + kk];
            }
            C[mm * n + nn] = c;
        }
    }
}

void computeRefCublas(half* C, const half* A, const half* B, int m, int n, int k, cudaStream_t stream)
{
    // cublas
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
    float alpha = 1.f;
    float beta  = 0.f;
    // TNT (A and B are swapped for transposing C)
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 n,
                 m,
                 k,
                 &alpha,
                 B,
                 CUDA_R_16F,
                 k,
                 A,
                 CUDA_R_16F,
                 k,
                 &beta,
                 C,
                 CUDA_R_16F,
                 n,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template<class T>
void Run(int m, int n, int k)
{
    universal_vector<T> a(m * k);
    universal_vector<T> b(n * k);
    universal_vector<T> c(m * n);
    universal_vector<T> c_ref = c;

    std::vector<T> c_cpu(m * n);

    RNG rng;

    rng.GenerateUniform(a.data().get(), a.size());
    rng.GenerateUniform(b.data().get(), b.size());

    cudaDeviceSynchronize();

    // ComputeRefCpu(c_cpu.data(), a.data().get(), b.data().get(), m, n, k);
    computeRefCublas(c_ref.data().get(), a.data().get(), b.data().get(), m, n, k, 0);

    for (int i = 0; i < 10; ++i) {
        gemm::invoke(c.data().get(), a.data().get(), b.data().get(), m, n, k, 0);
    }

    cudaDeviceSynchronize();

    // Compare(c_ref.data().get(), c_cpu.data(), n, n, m, 1);
    Compare(c.data().get(), c_ref.data().get(), n, n, m, 0);
}

int main(int argc, char* argv[])
{
    Run<half>(4096, 4096, 4096);
    return 0;
}