// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/norm/norm.h"

#include <cuda_runtime.h>

namespace turbomind {

void invokeInternVitPatchify(Tensor&       patches,
                             const Tensor& pixel_values,
                             int           batch_size,
                             int           channels,
                             int           image_h,
                             int           image_w,
                             int           patch_h,
                             int           patch_w,
                             cudaStream_t  stream);

void invokeInternVitAddEmbeddings(Tensor&       hidden,
                                  const Tensor& patch_embeds,
                                  const Tensor& patch_bias,
                                  const Tensor& cls_token,
                                  const Tensor& position_embeddings,
                                  int           batch_size,
                                  int           num_patches,
                                  int           hidden_dim,
                                  cudaStream_t  stream);

void invokeInternVitPreRMSNorm(Tensor& sums, const Tensor& qkv, int local_dim, cudaStream_t stream);

void invokeInternVitPostRMSNorm(Tensor&       qkv,
                                const Tensor& sums,
                                const Tensor& q_weight,
                                const Tensor& k_weight,
                                int           local_dim,
                                int           hidden_dim,
                                float         eps,
                                cudaStream_t  stream);

void invokeInternVitPrepareQKV(Tensor& kv, const Tensor& qkv, int local_head_num, int head_dim, cudaStream_t stream);

void invokeInternVitResidualScaleNorm(Tensor&       hidden_states,
                                      Tensor&       residual,
                                      const Tensor& branch_output,
                                      const Tensor& branch_scale,
                                      const Tensor& branch_bias,
                                      const Tensor& norm_weight,
                                      const Tensor& norm_bias,
                                      float         eps,
                                      NormType      norm_type,
                                      cudaStream_t  stream);

void invokeInternVitPixelShuffle(Tensor& output, const Tensor& hidden, int grid_size, cudaStream_t stream);

}  // namespace turbomind
