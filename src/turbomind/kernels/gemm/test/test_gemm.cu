

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "testbed_v2.h"

#include "cuda_profiler_api.h"

using namespace turbomind;

int main()
{

    auto               stream = core::Stream::create();
    core::ContextGuard ctx{stream, core::Allocator{kCPU}, core::Allocator{stream, false}};

    auto test = gemm::get_test(gemm::TestPreset::kANY_e4m3_e4m3_bf16_TNN);

    // test->Initialize(128, 128, 256, 0, 1, core::Context::stream().handle());

    // test->Initialize(1536 / 4, 8192 * 16, 2048, 0, 1, core::Context::stream().handle());
    // test->Initialize(2048, 8192 * 16, 1536 / 4, 0, 1, core::Context::stream().handle());
    // test->Initialize(384, 128, 1024, core::Context::stream().handle());
    // test->Initialize(8192 * 4, 8192 * 4, 512, 0, 1, core::Context::stream().handle());
    // test->Initialize(8192 * 2, 8192 * 2, 8192 * 2, 0, 1, core::Context::stream().handle());
    // test->Initialize(8192 * 4, 1, 8192 * 4, core::Context::stream().handle());p
    // test->Initialize(1024, 1024, 1024, core::Context::stream().handle());
    // test->Initialize(8192 * 2, 8192 * 2, 2048, 0, 0, core::Context::stream().handle());
    // test->Initialize(12288 * 2, 100, 4096, 0, 0, core::Context::stream().handle());
    // test->Initialize(4096, 100, 12288, 0, 0, core::Context::stream().handle());
    // test->Initialize(6144, 100, 4096, 0, 0, core::Context::stream().handle());
    // test->Initialize(4096, 100, 4096, 0, 0, core::Context::stream().handle());

    // test->Initialize(1536, 32768, 8192, 128, 8, core::Context::stream().handle());

    const int tp = 8;
    const int bs = 1024;

    // deepseek-v3
    test->Initialize(2048 / tp * 2, bs, 7168, 256, 8, core::Context::stream().handle());
    // test->Initialize(7168, bs, 2048 / tp, 256, 8, core::Context::stream().handle());

    // qwen3-30-a3
    // test->Initialize(768 / tp, bs, 2048, 128, 8, core::Context::stream().handle());
    // test->Initialize(2048, bs, 768 / tp, 128, 8, core::Context::stream().handle());

    // qwen3-235-a22
    // test->Initialize(1536 / tp, 16384, 4096, 128, 8, core::Context::stream().handle());
    // test->Initialize(4096, 4096, 1536 / tp, 128, 8, core::Context::stream().handle());

    // test->Initialize(1536, 77, 4096, 8, 2, core::Context::stream().handle());
    // test->Initialize(1536, 256, 4096, 8, 2, core::Context::stream().handle());

    // test->Initialize(1024, 1024, 1024, 8, 2, core::Context::stream().handle());
    // test->Initialize(128, 128, 2048, 8, 2, core::Context::stream().handle());

    test->Ref(0);  // c   <- a   * b
    test->Ref(1);  // c_f <- a_f * b_f

    // cudaProfilerStart();
    test->Run();  // c[2] <- a_q * b_q
    // cudaProfilerStop();

    FC_Header();
    FC_Print(test->Compare(1, 0));

    FC_Print(test->Compare(2, 0));
    FC_Print(test->Compare(2, 1));

    // test->Check();

    return 0;
}