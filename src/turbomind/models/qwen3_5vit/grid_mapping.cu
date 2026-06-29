// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/grid_mapping.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace {

__global__ void buildMappedIdxKernel(int* mapped_idx,
                                     int  token_offset,
                                     int  natural_offset,
                                     int  t,
                                     int  h,
                                     int  w,
                                     int  S)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = t * h * w;
    if (idx >= total) {
        return;
    }

    const int hw          = h * w;
    const int merge_unit  = S * S;
    const int local       = idx % hw;
    const int group       = local / merge_unit;
    const int inner       = local - group * merge_unit;
    const int group_cols  = w / S;
    const int h_outer     = group / group_cols;
    const int w_outer     = group - h_outer * group_cols;
    const int h_inner     = inner / S;
    const int w_inner     = inner - h_inner * S;
    const int natural_idx = (h_outer * S + h_inner) * w + (w_outer * S + w_inner);
    mapped_idx[token_offset + idx] = natural_offset + natural_idx;
}

__global__ void buildMappedIdxBatchedKernel(int*       mapped_idx,
                                            const int* grid_thws,
                                            const int* grid_offsets,
                                            int        num_grids,
                                            int        S)
{
    const int grid_id = blockIdx.x;
    if (grid_id >= num_grids) {
        return;
    }

    const int t              = grid_thws[grid_id * 3];
    const int h              = grid_thws[grid_id * 3 + 1];
    const int w              = grid_thws[grid_id * 3 + 2];
    const int token_offset   = grid_offsets[grid_id * 2];
    const int natural_offset = grid_offsets[grid_id * 2 + 1];
    const int total          = t * h * w;
    const int hw             = h * w;
    const int merge_unit     = S * S;
    const int group_cols     = w / S;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int local       = idx % hw;
        const int group       = local / merge_unit;
        const int inner       = local - group * merge_unit;
        const int h_outer     = group / group_cols;
        const int w_outer     = group - h_outer * group_cols;
        const int h_inner     = inner / S;
        const int w_inner     = inner - h_inner * S;
        const int natural_idx = (h_outer * S + h_inner) * w + (w_outer * S + w_inner);
        mapped_idx[token_offset + idx] = natural_offset + natural_idx;
    }
}

}  // namespace

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 int          token_offset,
                                 int          natural_offset,
                                 int          t,
                                 int          h,
                                 int          w,
                                 int          spatial_merge_size,
                                 cudaStream_t stream)
{
    if (t * h * w == 0) {
        return;
    }

    const int total   = t * h * w;
    const int threads = 256;
    buildMappedIdxKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        mapped_idx, token_offset, natural_offset, t, h, w, spatial_merge_size);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          spatial_merge_size,
                                 cudaStream_t stream)
{
    if (num_grids == 0) {
        return;
    }

    const int threads = 256;
    buildMappedIdxBatchedKernel<<<num_grids, threads, 0, stream>>>(
        mapped_idx, grid_thws, grid_offsets, num_grids, spatial_merge_size);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
