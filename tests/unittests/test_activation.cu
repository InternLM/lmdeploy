#include <iostream>   // snprintf
#include <string>     // std::string
#include <vector>     // std::vector

#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include "src/turbomind/utils/logger.h"

#include "unittest_utils.h"

using namespace fastertransformer;

#define PRINT_LIMIT 16
#define EPSILON (1e-20)
#define EPSILON_FP16 (1e-10)

struct TestCase {
    std::string name;
    size_t m;
    size_t n;
    size_t ite;

    std::string toString()
    {
        char buf[100];
        snprintf(buf, sizeof(buf), "TestCase[name=%s, m=%ld, n=%ld]", name.c_str(), m, n);
        return buf;
    }

    void print()
    {
        FT_LOG_INFO(toString());
    }
};

template<typename T>
void testActivationKernel(TestCase tc)
{
    const int m = tc.m;
    const int n = tc.n;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    T *output_baseline, *output_opt1, *bias;
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&output_opt1, m * n);
    deviceMalloc(&bias, n);
    cudaD2Dcpy(output_opt1, output_baseline, m * n);
    invokeGenericActivation<GeluActivation>(output_baseline,
                                            (const T*) bias,
                                            (const T*) nullptr,
                                            (const T*) nullptr,
                                            (const int*) nullptr,
                                            (const T*) nullptr,
                                            m,
                                            n,
                                            0,
                                            (const float*) nullptr,
                                            (const float*) nullptr,
                                            stream);
    invokeAddBiasGeluV2(output_opt1, bias, (const int*) nullptr, (const T*) nullptr, m, n, stream);
    bool passed = checkResult(tc.name, output_baseline, output_opt1, m * n, true, true);
    FT_CHECK(passed);

    const int ite = tc.ite;
    CudaTimer cuda_timer_baseline(stream);
    // warmup
    for (int i = 0; i < ite; i++) {
        invokeGenericActivation<GeluActivation>(output_baseline,
                                                (const T*) bias,
                                                (const T*) nullptr,
                                                (const T*) nullptr,
                                                (const int*) nullptr,
                                                (const T*) nullptr,
                                                m,
                                                n,
                                                0,
                                                (const float*) nullptr,
                                                (const float*) nullptr,
                                                stream);
    }
    cuda_timer_baseline.start();
    for (int i = 0; i < ite; i++) {
        invokeGenericActivation<GeluActivation>(output_baseline,
                                                (const T*) bias,
                                                (const T*) nullptr,
                                                (const T*) nullptr,
                                                (const int*) nullptr,
                                                (const T*) nullptr,
                                                m,
                                                n,
                                                0,
                                                (const float*) nullptr,
                                                (const float*) nullptr,
                                                stream);
    }
    float total_time_baseline = cuda_timer_baseline.stop();

    CudaTimer cuda_timer_opt(stream);
    // warmup
    for (int i = 0; i < ite; i++) {
        invokeAddBiasGeluV2(output_baseline, bias, (const int*) nullptr, (const T*) nullptr, m, n, stream);
    }
    cuda_timer_opt.start();
    for (int i = 0; i < ite; i++) {
        invokeAddBiasGeluV2(output_baseline, bias, (const int*) nullptr, (const T*) nullptr, m, n, stream);
    }
    float total_time_opt = cuda_timer_opt.stop();
    FT_LOG_INFO("%s baseline_time: %f us, opt_time: %f us, speedup: %f (ite: %d)",
                tc.toString().c_str(),
                total_time_baseline / ite * 1000.f,
                total_time_opt / ite * 1000.f,
                total_time_baseline / total_time_opt,
                ite);

    deviceFree(output_baseline);
    deviceFree(output_opt1);
    deviceFree(bias);
}

int main()
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    std::vector<TestCase> test_cases{
        // TC: name / m / n
        TestCase{"addBiasGelu", 32, 1024, 1000},
        TestCase{"addBiasGelu", 128, 1024, 1000},
        TestCase{"addBiasGelu", 2048, 1024, 1000},
        TestCase{"addBiasGelu", 32, 3072, 1000},
        TestCase{"addBiasGelu", 128, 3072, 1000},
        TestCase{"addBiasGelu", 2048, 3072, 1000},
        TestCase{"addBiasGelu", 32, 4096, 1000},
        TestCase{"addBiasGelu", 128, 4096, 1000},
        TestCase{"addBiasGelu", 2048, 4096, 1000},
        TestCase{"addBiasGelu", 32, 8192, 1000},
        TestCase{"addBiasGelu", 128, 8192, 1000},
        TestCase{"addBiasGelu", 2048, 8192, 1000},
        TestCase{"addBiasGelu", 32, 49152, 1000},
        TestCase{"addBiasGelu", 128, 49152, 1000},
        TestCase{"addBiasGelu", 2048, 49152, 1000},
        TestCase{"addBiasGelu", 32, 81920, 1000},
        TestCase{"addBiasGelu", 128, 81920, 1000},
        TestCase{"addBiasGelu", 2048, 81920, 1000},
    };

    for (auto& tc : test_cases) {
        // testActivationKernel<float>(tc);
        testActivationKernel<half>(tc);
    }
    FT_LOG_INFO("testActivationKernel done");

    return 0;
}
