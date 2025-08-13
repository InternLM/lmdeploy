// Copyright (c) OpenMMLab. All rights reserved.

// #include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
// #include "src/turbomind/kernels/attention/attention_params.h"

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
                    float        exp_scale,
                    cudaStream_t stream);
}  // namespace turbomind
