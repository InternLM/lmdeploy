// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/mrope_position_ids.h"

namespace turbomind {

namespace {

constexpr int kBlock = 128;

__global__ void mropeScatterKernel(int* pos_ids, int row_stride, const MropeSegment* __restrict__ segs)
{
    const MropeSegment s       = segs[blockIdx.x];
    const int          local_k = blockIdx.y * blockDim.x + threadIdx.x;
    if (local_k >= s.n_tok) {
        return;
    }
    int* dst = pos_ids + s.dst_row * row_stride + 3 * (s.dst_offset + local_k);
    if (s.h2 == 0) {  // text run
        const int p = s.base_pos + local_k;
        dst[0]      = p;
        dst[1]      = p;
        dst[2]      = p;
    }
    else {  // image run — grid math uses the original (un-clipped) k
        const int k  = s.k_offset + local_k;
        const int hw = s.h2 * s.w2;
        dst[0]       = s.base_pos + k / hw;
        dst[1]       = s.base_pos + (k / s.w2) % s.h2;
        dst[2]       = s.base_pos + k % s.w2;
    }
}

}  // namespace

void invokeMropePositionIds(
    int* pos_ids, int row_stride, const MropeSegment* segments, int num_segments, int max_seg_len, cudaStream_t stream)
{
    if (num_segments <= 0 || max_seg_len <= 0) {
        return;
    }
    const int  tiles = (max_seg_len + kBlock - 1) / kBlock;
    const dim3 grid((unsigned)num_segments, (unsigned)tiles);
    mropeScatterKernel<<<grid, kBlock, 0, stream>>>(pos_ids, row_stride, segments);
}

}  // namespace turbomind
