#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v3.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::sm90_s64n32_dynamic()
{
    using Kernel = KernelImplSm90<GemmUniversalSm90_v3>;
    Add(std::make_unique<Kernel>());
}

}  // namespace turbomind::gemm