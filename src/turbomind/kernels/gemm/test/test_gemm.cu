

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "testbed_v2.h"

using namespace turbomind;

int main()
{
    core::ContextGuard ctx{core::Stream::create(), core::Allocator{kCPU}, core::Allocator{kDEVICE}};

    auto test = gemm::get_test(gemm::TestPreset::kANY_bf16_bf16_bf16_TNN);

    test->Initialize(8192 * 2, 8192 * 2, 8192 * 2, core::Context::stream().handle());

    test->Ref(0);  // c   <- a   * b
    // test->Ref(1);  // c_f <- a_f * b_f

    test->Run();

    FC_Header();
    FC_Print(test->Compare(2, 0));
    // FC_Print(test->Compare(2, 1));
    // test->Check();

    return 0;
}