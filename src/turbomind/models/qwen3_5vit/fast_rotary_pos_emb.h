#pragma once

#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// Precomputes the (cos, sin) rotary-embedding table for Qwen3-VL vision tokens.
// Layout per natural flat position (keyed by the same index `mapped_idx` carries):
//   [c_0, s_0, c_1, s_1, ..., c_{head_dim/2-1}, s_{head_dim/2-1}]
// The pair index `k` uses `h_coord` for k < head_dim/4 and `w_coord` otherwise,
// with inv_freq = theta^(-2*(k % (head_dim/4)) / (head_dim/2)).
void invokeQwen3VitRotaryPosEmb(void*        cos_sin,  // [total_hw, head_dim]
                                DataType     dtype,
                                const int*   grid_thws,     // [num_grids * 3], (t, h, w)
                                const int*   grid_offsets,  // [num_grids * 2], (t*h*w, h*w)
                                int          num_grids,
                                int          total_hw,
                                int          head_dim,
                                float        theta,
                                cudaStream_t stream);

}  // namespace turbomind
