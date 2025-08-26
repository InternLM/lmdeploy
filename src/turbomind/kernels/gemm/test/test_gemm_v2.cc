

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "testbed_v3.h"

using namespace turbomind;

struct TestParameter: Testbed_v3::Parameter {
    TestParameter(DataType dtype, DataType wtype, DataType itype, int group_size = 128): Testbed_v3::Parameter{}
    {
        data_type   = dtype;
        weight_type = wtype;
        input_type  = itype;

        this->group_size = group_size;
    }
};

int main()
{

    constexpr auto x = (unsigned)0 - (unsigned)64 + (unsigned)8;
    constexpr auto y = (unsigned)1 - x;

    auto stream = core::Stream::create();

    core::ContextGuard ctx{stream, core::Allocator{kCPU}, core::Allocator{stream, false}};

    // TestParameter p{kBfloat16, kBfloat16, kBfloat16};
    // TestParameter p{kBfloat16, kFloat8_e4m3, kFloat8_e4m3, 128};
    // TestParameter p{kHalf, kUint4, kHalf, 128};
    TestParameter p{kBfloat16, kFloat4_e2m1, kBfloat16, 32};

    p.input_dim      = 12288;
    p.output_dim     = 16384;
    p.max_batch_size = 8192;

    // p.input_dim      = 32;
    // p.output_dim     = 32;
    // p.max_batch_size = 1;

    Testbed_v3 test{p};

    test.GetReference();
    test.Run();
    test.Compare();

    cudaDeviceSynchronize();

    return 0;
}
