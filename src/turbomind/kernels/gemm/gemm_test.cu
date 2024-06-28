
#include "src/turbomind/kernels/attention/quantization.h"

#include "src/turbomind/kernels/gemm/cache_utils.h"
#include "src/turbomind/kernels/gemm/convert_v2.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gpu_metric.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/quantization.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/testbed.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <fstream>
#include <limits>
#include <thrust/universal_vector.h>

#include <type_traits>

using namespace turbomind;
using namespace gemm;
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

template<class T>
T& gTestbed()
{
    static T inst{turbomind::gemm::DispatchPolicy::kDefault, "tmp"};
    return inst;
}

template<class T, class Tb>
void Run(int m, int n, int k, int g = 128)
{
    constexpr Pack kPackA = 0;  // HMMA_16816 | OPERAND_A | 1;
    constexpr Pack kPackU = 0;  // HMMA_16816 | OPERAND_U | 1;
    constexpr Pack kPackB = HMMA_16816 | OPERAND_B | 2;
    constexpr Pack kPackV = HMMA_16816 | OPERAND_V | 1;
    auto&          test =
        gTestbed<gemm::Testbed<half, uint4_t, half, kRowMajor, kColMajor, kRowMajor, kPackA, kPackB, kPackU, kPackV>>();

    // constexpr Pack kPackA = 0;
    // constexpr Pack kPackU = 0;
    // constexpr Pack kPackB = HMMA_SIMT | OPERAND_B | 2;
    // constexpr Pack kPackV = HMMA_SIMT | OPERAND_V | 2;
    // auto& test = gTestbed<gemm::Testbed<half, uint4_t, half, kRowMajor, kColMajor, kPackA, kPackB, kPackU,
    // kPackV>>();

    // constexpr Pack kPackA = 0;
    // constexpr Pack kPackU = 0;
    // constexpr Pack kPackB = HMMA_884 | OPERAND_B | 2;
    // constexpr Pack kPackV = HMMA_884 | OPERAND_V | 2;
    // auto& test = gTestbed<gemm::Testbed<half, uint4_t, half, kRowMajor, kColMajor, kPackA, kPackB, kPackU,
    // kPackV>>();

    test.Initialize(m, n, k, g, 0);
    for (int i = 0; i < 10; ++i) {
        test.Run();
    }

    // test.CompareB();
    test.CompareC();

    return;
}

template<class T, class Tb>
void Test(int bsz, int tp)
{
    // Run<T, Tb>(8192 - 64, 8192 , 8192);
    // Run<T, Tb>(bsz, 8192, 8192);
    // Run<T, Tb>(bsz, 4096, 4096);
    // Run<half, uint4_t>(64, 11008, 4096);
    // Run<half, uint4_t>(128, 128, 32);
    // Run<half, uint4_t>(128, 128, 128);

    // llama2-7b
    // Run<T, Tb>(bsz, 2 * 11008 / tp, 4096);  // mlp.up/gate

    // Run<T, Tb>(bsz, 4096, 11008 / tp);  // mlp.down
    // Run<T, Tb>(bsz, 12288 / tp, 4096);  // w_qkv
    // Run<T, Tb>(bsz, 4096, 4096);        // w_o

    // llama2-70b
    // Run<T, Tb>(bsz, 10240 / tp, 8192);  // attn.qkv

    // Run<T, Tb>(8, 128, 512);

    // Run<T, Tb>(16, 16, 64);

    Run<T, Tb>(16384, 16384, 16384);

    // Run<T, Tb>(8192, 8192, 8192);

    // Run<T, Tb>(4096, 4096, 4096);

    // Run<T, Tb>(1024, 1024, 16384);

    // Run<T, Tb>(128, 128 * (2 + 8) * 2, 8192);

    // Run<T, Tb>(16, 4096, 4096);

    // Run<T, Tb>(1, 22016, 4096);

    // Run<T, Tb>(256, 8192, 8192 * 3);

    // Run<T, Tb>(128, 256, 8192);

    // Run<T, Tb>(16, 32, 16384);

    // Run<T, Tb>(16, 16, 16);

    // Run<T, Tb>(16, 32, 16);
}

namespace turbomind::gemm {

Kernel& gKernel();

}

int main(int argc, char* argv[])
{
    // gemm::MeasureL2CacheThroughput();
    // gemm::MeasureMmaThroughput();
    // Test<half, uint4_t>(1, 1);
    // Test<half, uint4_t>(8, 1);
    Test<half, half>(16, 1);
    return 0;
    // Test<half, uint4_t>(32, 1);
    // Test<half, uint4_t>(64, 1);
    // Test<half, uint4_t>(128, 1);
    // Test<half, uint4_t>(256, 1);
    // Test<half, uint4_t>(512, 1);
    // Test<half, uint4_t>(1024, 1);
    // Test<half, uint4_t>(2048, 1);
    // Test<half, uint4_t>(4096, 1);
    // Test<half, uint4_t>(8192, 1);

    const int M = 16;
    const int N = 16;
    const int K = 16;

    universal_vector<half> a(M * K);
    universal_vector<half> p(M * K);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i + j * M] = i + j * M;
        }
    }

    MatrixLayout a_desc{DataType::F16, Order::kColMajor, M, K, M};
    MatrixLayout p_desc{DataType::F16, Order::kColMajor, M, K, 0, HMMA_16816 | OPERAND_A | 1};

    Convert(a.data().get(), a_desc, p.data().get(), p_desc, 0);

    cudaDeviceSynchronize();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            int index = (int)p[i + j * M];
            int row   = index % M;
            int col   = index / M;
            printf("(%2d,%2d) ", row, col);
        }
        printf("\n");
    }

    // universal_vector<half> b(N * K);
    // thrust::fill_n(b.begin(), b.size(), 1);

    // universal_vector<half> c(M * N);

    // Workspace workspace{};

    // const float alpha = 1.f;
    // const float beta  = 0.f;

    // const MatrixLayout c_desc{DataType::F16, Order::kRowMajor, M, N, N};

    // (void)Gemm{}.Run({},
    //                  &alpha,
    //                  p.data().get(),
    //                  p_desc,
    //                  nullptr,
    //                  MatrixLayout{},
    //                  b.data().get(),
    //                  MatrixLayout{DataType::F16, Order::kColMajor, K, N, K},
    //                  nullptr,
    //                  MatrixLayout{},
    //                  &beta,
    //                  c.data().get(),
    //                  c_desc,
    //                  c.data().get(),
    //                  c_desc,
    //                  workspace,
    //                  0);

    // cudaDeviceSynchronize();

    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%2.0f ", (float)c[i * N + j]);
    //     }
    //     printf("\n");
    // }

    return 0;
}