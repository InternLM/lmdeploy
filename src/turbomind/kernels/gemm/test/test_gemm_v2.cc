

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"

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
    auto stream = core::Stream::create();

    core::ContextGuard ctx{stream, core::Allocator{kCPU}, core::Allocator{stream, false}};
    // clang-format off
    // TestParameter p{kHalf, kUint4      , kHalf, 128};
    // TestParameter p{kHalf, kFloat4_e2m1, kHalf,  32};
    // TestParameter p{kHalf, kFloat8_e4m3, kHalf, 128};
    // TestParameter p{kHalf, kHalf       , kHalf};

    // TestParameter p{kBfloat16, kBfloat16   , kBfloat16};
    // TestParameter p{kBfloat16, kFloat8_e4m3, kFloat8_e4m3, 128};
    TestParameter p{kBfloat16, kFloat8_e4m3, kBfloat16   , 128};
    // TestParameter p{kBfloat16, kFloat4_e2m1, kBfloat16   ,  32};
    // clang-format on

    // p.input_dim      = 512;
    // p.output_dim     = 1024;
    // p.max_batch_size = 256;

    // p.input_dim      = 1024;
    // p.output_dim     = 1024;
    // p.max_batch_size = 1024;

    // p.input_dim      = 12288;
    // p.output_dim     = 16384;
    // p.max_batch_size = 8192;

    // p.expert_num        = 1;
    // p.experts_per_token = 1;

    // p.input_dim      = 2880;
    // p.output_dim     = 2880;
    // p.max_batch_size = 64;

    // p.input_dim         = 7168;
    // p.output_dim        = 4096;
    // p.max_batch_size    = 16384;
    // p.expert_num        = 256;
    // p.experts_per_token = 8;

    // Qwen3-MoE
    p.expert_num        = 128;
    p.experts_per_token = 8;
    // 30B
    // p.input_dim  = 2048;
    // p.output_dim = 768 * 2;
    // 235B
    // p.input_dim  = 4096;
    // p.output_dim = 1536 * 2;
    // 480B
    p.input_dim  = 6144;
    p.output_dim = 2560 * 2;

    p.max_batch_size = 256;

    // p.input_dim         = 16384;
    // p.output_dim        = 16384;
    // p.max_batch_size    = 16384;

    // p.input_dim         = 2880;
    // p.output_dim        = 5760;
    // p.max_batch_size    = 16384;
    // p.expert_num        = 32;
    // p.experts_per_token = 4;

    // p.input_dim      = 128;
    // p.output_dim     = 32;
    // p.max_batch_size = 1;

    Testbed_v3 test{p};

    test.GetReference();
    test.Run();
    test.Compare();

    cudaDeviceSynchronize();

    return 0;
}
