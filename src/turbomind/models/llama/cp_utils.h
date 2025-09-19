// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/core.h"

namespace turbomind {

template<typename T>
void invokeCpReduce(T*           out,
                    float*       O,
                    float*       M,
                    float*       L,
                    int          token_num,
                    int          head_num,
                    int          size_per_head,
                    int          cp_size,
                    int          cp_rank,
                    float        exp_scale,
                    cudaStream_t stream);

}  // namespace turbomind
