
#include <cuda.h>

#include "src/turbomind/kernels/gemm/registry.h"

// We need modifiable TMA, which is added in 12.3
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3))

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v3.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v5.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::sm90_64n32_8()
{
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kRowMajor, 1, 1, false>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kRowMajor, 2, 1, false>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kRowMajor, 1, 2, false>>>());

    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kColMajor, 1, 1, true>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kColMajor, 2, 1, true>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v5<kColMajor, 1, 2, true>>>());

    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kRowMajor, 1, 1, false>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kRowMajor, 2, 1, false>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kRowMajor, 1, 2, false>>>());

    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kColMajor, 1, 1, true>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kColMajor, 2, 1, true>>>());
    Add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<kColMajor, 1, 2, true>>>());
}

}  // namespace turbomind::gemm

#else

namespace turbomind::gemm {

void Registry::sm90_64n32_8() {}

}  // namespace turbomind::gemm

#endif
