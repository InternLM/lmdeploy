#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/kernels/transpose_int8_kernels.h"

#include <algorithm>
#include <iostream>
#include <random>

#include "tests/unittests/gtest_utils.h"

using namespace fastertransformer;

class Int8TestSuite: public FtTestBase {

public:
    void SetUp() override
    {
        FtTestBase::SetUp();
    }
    void TearDown() override
    {
        FtTestBase::TearDown();
    }

protected:
    using FtTestBase::stream;
    using FtTestBase::allocator;

    struct cudaDeviceProp prop;

    void testTransposition();
};

void fill_tensor_random(Tensor a) {
    const size_t num_elems = a.size();
    std::vector<int8_t> host_values(num_elems);
    std::uniform_int_distribution<int8_t> int8_random(-128, 127);
    std::mt19937 rng(0);

    std::generate(host_values.begin(), host_values.end(), [&int8_random, &rng](){ return int8_random(rng); });
    cudaH2Dcpy(a.getPtr<int8_t>(), host_values.data(), num_elems);
}

void reference_transpose_host(std::vector<int8_t>& a_t_host, const Tensor& a)
{
    std::vector<int8_t> a_host(a.size());
    cudaD2Hcpy(a_host.data(), a.getPtr<int8_t>(), a.size());

    for (unsigned int i = 0; i < a.shape[0]; i++) {
        for (unsigned int j = 0; j < a.shape[1]; j++) {
            a_t_host[j * a.shape[0] + i] = a_host[i * a.shape[1] + j];
        }
    }
}

void Int8TestSuite::testTransposition()
{
    const int m = 32;
    const int k = 2048;
    const int n = 2048;

    int8_t *a_data, *a_t_data;

    cudaMalloc(&a_data, m * k * sizeof(int8_t));
    Tensor a {MEMORY_GPU, TYPE_INT8, {32, 2048}, a_data};
    fill_tensor_random(a);

    cudaMalloc(&a_t_data, k * m * sizeof(int8_t));
    Tensor a_t {MEMORY_GPU, TYPE_INT8, {2048, 32}, a_t_data};

    std::vector<int8_t> a_t_host_ref(a_t.size());
    reference_transpose_host(a_t_host_ref, a);

    invokeTransposeInt8Tensor(a_t, a);
    bool result = checkResult("", a_t.getPtr<int8_t>(), a_t_host_ref.data(), a_t.size());

    cudaFree(a_data);
    cudaFree(a_t_data);

    EXPECT_TRUE(result);
}

TEST_F(Int8TestSuite, TranspositionCorrectness)
{
    this->testTransposition();
}
