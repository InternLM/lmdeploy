#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

namespace turbomind {

void invokeQwen2VitWindowReorder(
    Tensor& out, const Tensor& in, const int* window_idx, int merge_unit, int group_count, cudaStream_t stream);

void invokeQwen2VitReverseWindow(
    Tensor& out, const Tensor& in, const int* window_idx, int group_count, cudaStream_t stream);

void invokeQwen2VitBuildWindowMappedIdx(int*         window_mapped_idx,
                                        const int*   mapped_idx,
                                        const int*   window_idx,
                                        int          merge_unit,
                                        int          group_count,
                                        cudaStream_t stream);

}  // namespace turbomind
