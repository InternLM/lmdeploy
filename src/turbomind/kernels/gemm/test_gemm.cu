
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm/cache_utils.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/quantization.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/testbed.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <fstream>
#include <limits>
#include <thrust/universal_vector.h>
#include <type_traits>

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

template<class T, class Tb>
gemm::Testbed<T, Tb>& gTestbed()
{
    static gemm::Testbed<T, Tb> inst{turbomind::gemm::DispatchPolicy::kUseCached, "tmp"};
    return inst;
}

template<class T, class Tb>
void Run(int m, int n, int k, int g = 128)
{
    auto& test = gTestbed<T, Tb>();

    test.Initialize(m, n, k, g, true, 0);
    for (int i = 0; i < 10; ++i) {
        test.Run();
    }

    test.CompareB();
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
    Run<T, Tb>(bsz, 2 * 11008 / tp, 4096);  // mlp.up/gate
    // Run<T, Tb>(bsz, 4096, 11008 / tp);  // mlp.down
    // Run<T, Tb>(bsz, 12288 / tp, 4096);  // w_qkv
    // Run<T, Tb>(bsz, 4096, 4096);        // w_o

    // llama2-70b
    // Run<T, Tb>(bsz, 10240 / tp, 8192);  // attn.qkv

    // Run<T, Tb>(8, 128, 512);
}

int main(int argc, char* argv[])
{
    Test<half, uint4_t>(1, 1);
    // Test<half, uint4_t>(16, 1);
    // Test<half, uint4_t>(32, 1);
    // Test<half, uint4_t>(64, 1);
    // Test<half, uint4_t>(128, 1);
    // Test<half, uint4_t>(256, 1);
    // Test<half, uint4_t>(512, 1);
    // Test<half, uint4_t>(1024, 1);
    // Test<half, uint4_t>(8192, 1);

    return 0;
}