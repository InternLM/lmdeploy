#pragma once

#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// Precomputes the 4 bilinear-interpolation gather indices and weights
// used by the subsequent pos-embed merge step in Qwen3-VL.
void invokeFastPosEmbedIdxWeight(int*         idx_out,     // [total_n * 4]
                                 void*        weight_out,  // [total_n * 4]
                                 DataType     dtype,
                                 const int*   grid_thws,     // [num_grids * 3], (t, h, w)
                                 const int*   grid_offsets,  // [num_grids * 2], (t*h*w, h*w)
                                 int          num_grids,
                                 int          total_n,
                                 int          num_grid_per_side,
                                 cudaStream_t stream);

}  // namespace turbomind
