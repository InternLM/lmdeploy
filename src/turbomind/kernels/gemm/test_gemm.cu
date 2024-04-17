
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/transcript.h"
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

template<class T, class Tb>
void Run(int m, int n, int k)
{
    universal_vector<T> a(m * k);
    universal_vector<T> b(n * k);
    universal_vector<T> c(m * n);
    universal_vector<T> c_ref = c;

    universal_vector<unsigned short> b0(n * k);
    universal_vector<Array<Tb, 8>>   b1(n * k / 8);

    universal_vector<T> q((k + 127) / 128 * n * 2);

    std::vector<T> c_cpu(m * n);

    RNG rng;

    rng.GenerateUniform(a.data().get(), a.size(), 1, -0.5);
    if constexpr (!std::is_same_v<Tb, T>) {
        // generate random bytes for 16-bit width
        rng.GenerateUInt((uint*)b0.data().get(), b0.size() / 2);
        cudaDeviceSynchronize();
        for (int i = 0; i < b0.size(); ++i) {
            b0[i] %= (1 << bitsof<Tb>);  // constraint it's range
            // b0[i] = (b0[i] % 15) + 1;
            // b0[i] = 15;
            b[i] = T(b0[i]);  // convert to floating type
        }
        for (int i = 0; i < q.size(); i += 2) {
            q[i]     = T(1);
            q[i + 1] = T(0);
        }
    }
    else {
        rng.GenerateUniform(b.data().get(), b.size(), 1, -0.5);
    }

    cudaDeviceSynchronize();

    // ComputeRefCpu(c_cpu.data(), a.data().get(), b.data().get(), m, n, k);

    if (0) {
        for (int i = 0; i < 1; ++i) {
            gemm::invoke(c.data().get(), a.data().get(), b.data().get(), (T*)nullptr, m, n, k, 0);
        }

        computeRefCublas(c_ref.data().get(), a.data().get(), b.data().get(), m, n, k, 0);

        cudaDeviceSynchronize();

        // Compare(c_ref.data().get(), c_cpu.data(), n, n, m, 1);
        Compare(c.data().get(), c_ref.data().get(), n, n, m, 0);
    }

    if (1) {
        auto B1 = (Tb*)b1.data().get();

        for (int i = 0; i < 1; ++i) {
            if constexpr (std::is_same_v<T, Tb>) {
                gemm::transcript<T>(B1, b.data().get(), n, k, 0);
            }
            else {
                gemm::transcript<T>(B1, b0.data().get(), n, k, 0);
            }
        }

        cudaDeviceSynchronize();
        // for (int i = 0; i < b1.size(); ++i) {
        //     printf("%d %08x\n", i, (uint32_t&)b1[i]);
        // }

        for (int i = 0; i < 10; ++i) {
            gemm::invoke(c.data().get(), a.data().get(), B1, q.data().get(), m, n, k, 0);
        }

        for (int i = 0; i < 5; ++i) {
            computeRefCublas(c_ref.data().get(), a.data().get(), b.data().get(), m, n, k, 0);
        }

        cudaDeviceSynchronize();

        // Compare(c_ref.data().get(), c_cpu.data(), n, n, m, 1);
        Compare(c.data().get(), c_ref.data().get(), n, n, m, 0);
    }

    cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
    // Run<half, uint4_t>(8192, 8192, 8192);
    Run<half, uint4_t>(4096, 4096, 4096);
    // Run<half, uint4_t>(128, 128, 32);
    // Run<half, uint4_t>(128, 128, 128);
    return 0;
}