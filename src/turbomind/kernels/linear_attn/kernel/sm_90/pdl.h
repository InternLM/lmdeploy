#pragma once

#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace turbomind::linear_attn::delta_rule::detail {

template<class... KernelArgs, class... CallArgs>
void LaunchPdlKernel(dim3         grid,
                     dim3         block,
                     size_t       dynamic_smem_bytes,
                     cudaStream_t stream,
                     void (*kernel)(KernelArgs...),
                     CallArgs&&... args)
{
    cudaLaunchAttribute attribute{};
    attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = grid;
    config.blockDim         = block;
    config.dynamicSmemBytes = dynamic_smem_bytes;
    config.stream           = stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    TM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...));
}

}  // namespace turbomind::linear_attn::delta_rule::detail
