// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

// One descriptor per text / image run, clipped to a prefill chunk's active window.
// `h2 == 0` flags a text run (real image grids always have h2 > 0).
struct MropeSegment {
    int dst_row;     // batch slot in the (bsz, max_active_end, 3) output table
    int dst_offset;  // absolute seq index of the first token written by this segment
    int n_tok;       // tokens to write (already clipped to the active range)
    int base_pos;    // text: position id at local_k = 0; image: image's mm_start
    int h2;          // image grid h after spatial merge (0 ⇒ text)
    int w2;          // image grid w after spatial merge (ignored when h2 == 0)
    int k_offset;    // starting "k" for image grid math (clip-offset within the run); unused for text
};

// Scatter `num_segments` segments into `pos_ids` of shape (bsz, row_stride/3, 3).
// `pos_ids` may be null when num_segments == 0.
void invokeMropePositionIds(int*                pos_ids,
                            int                 row_stride,  // = max_active_end * 3
                            const MropeSegment* segments,    // device
                            int                 num_segments,
                            int                 max_seg_len,
                            cudaStream_t        stream);

}  // namespace turbomind
