

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "testbed_v2.h"

#include "cuda_profiler_api.h"

using namespace turbomind;

int main()
{

    core::ContextGuard ctx{core::Stream::create(), core::Allocator{kCPU}, core::Allocator{kDEVICE}};

    auto test = gemm::get_test(gemm::TestPreset::kANY_e4m3_e4m3_bf16_TNN);

    // test->Initialize(384, 128, 1024, core::Context::stream().handle());
    // test->Initialize(8192 * 4, 8192 * 4, 8192 * 4, 1, 1, core::Context::stream().handle());
    // test->Initialize(8192 * 4, 1, 8192 * 4, core::Context::stream().handle());
    // test->Initialize(1024, 1024, 1024, core::Context::stream().handle());
    // test->Initialize(8192 * 2, 8192 * 2, 2048, 0, 0, core::Context::stream().handle());
    // test->Initialize(12288 * 2, 3072, 4096, core::Context::stream().handle());
    // test->Initialize(4096, 8192, 12288, core::Context::stream().handle());
    // test->Initialize(6144, 8192, 4096, core::Context::stream().handle());
    // test->Initialize(4096, 8192, 4096, core::Context::stream().handle());

    test->Initialize(1536, 32768, 4096, 128, 8, core::Context::stream().handle());

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