#pragma once

#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// Fuses the spatial-merge permutation, the bilinear-weighted sum, and the
// t-expansion of Qwen3-VL ViT pos_embed interpolation into a single pass on
// top of the patch_embed linear output.
void invokeFusedPosEmbedMerge(void*        hidden_states,      // [batch, hidden]
                              const void*  pos_embeds,         // [total_hw * 4, hidden]
                              const void*  pos_embed_weights,  // [total_hw * 4]
                              const int*   mapped_idx,         // [batch]
                              const void*  bias,               // [hidden] or nullptr
                              int          batch,
                              int          hidden,
                              DataType     dtype,
                              cudaStream_t stream);

}  // namespace turbomind
