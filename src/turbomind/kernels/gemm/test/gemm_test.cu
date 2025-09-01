// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"

#include "src/turbomind/kernels/gemm/convert_v2.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gpu_metric.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/test/models.h"
#include "src/turbomind/kernels/gemm/test/quantization.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/test/testbed.h"
#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <fstream>
#include <limits>
#include <thrust/universal_vector.h>

#include <numeric>
#include <random>
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

static int g_check = 0;

void Run(int batch_size, int output_dims, int input_dims, int expert_num = 0, int top_e = 1, int g = 128)
{
    auto& test = get_test();
    int   m    = batch_size;
    int   n    = output_dims;
    int   k    = input_dims;
    if (get_test().kBatchDim == 1) {
        std::swap(m, n);
    }
    std::cerr << "m" << m << "n" << n << "k" << k << "_"
              << "E" << expert_num << "e" << top_e << "\n";
    test.Initialize(m, n, k, g, expert_num, top_e, 0);

    if (g_check) {
        test.Check();
    }
    else {
        for (int i = 0; i < 100; ++i) {
            CacheFlushing::flush();
            test.Run();
            CacheFlushing::flush();
            test.RunCublas();
        }
        test.CompareC();
    }
}

int main(int argc, char* argv[])
{
    g_check = 0;
    // Run(8192, 14336 * 2, 4096);

    // Run(16384, 16384, 16384);
    // Run(18, 14336, 4096, 8, 2);

    // Run(16, 4096, 7168, 8, 2);
    // Run(777, 14336, 4096, 8, 2);

    // Run(256, 14336 * 2, 4096);
    // Run(16, 4096, 14336);

    Run(8192, 16384, 16384);

    // g_check = 1;
    // std::vector<int> bsz(1024);
    // {
    //     std::iota(bsz.begin(), bsz.end(), 1);
    //     std::random_device rd;
    //     std::mt19937       g(rd());
    //     std::shuffle(bsz.begin() + 1, bsz.end(), g);
    // }
    // for (const auto& b : bsz) {
    //     for (const auto& [out, in] : config) {
    //         Run(b, out, in);
    //     }
    // }

    if (auto ec = cudaDeviceSynchronize(); ec != cudaSuccess) {
        std::cerr << "un-clean exit: " << cudaGetErrorString(ec) << "\n";
    }

    return 0;
}
