// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 int          token_offset,
                                 int          natural_offset,
                                 int          t,
                                 int          h,
                                 int          w,
                                 int          spatial_merge_size,
                                 cudaStream_t stream);

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          spatial_merge_size,
                                 cudaStream_t stream);

}  // namespace turbomind
