

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "testbed_v3.h"

using namespace turbomind;

int main()
{
    auto stream = core::Stream::create();

    core::ContextGuard ctx{stream, core::Allocator{kCPU}, core::Allocator{stream, false}};

    Testbed_v3::Parameter p{};

    p.input_dim      = 1024;
    p.output_dim     = 1536;
    p.max_batch_size = 512;

    // std::tie(p.data_type, p.weight_type, p.input_type) = std::tuple{kBfloat16, kBfloat16, kBfloat16};
    // std::tie(p.data_type, p.weight_type, p.input_type) = std::tuple{kBfloat16, kFloat8_e4m3, kFloat8_e4m3};
    std::tie(p.data_type, p.weight_type, p.input_type) = std::tuple{kHalf, kUint4, kHalf};

    Testbed_v3 test{p};

    test.GetReference();
    test.Run();
    test.Compare();

    cudaDeviceSynchronize();

    return 0;
}
