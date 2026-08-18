// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

void MLACopyQKV(DataType     dtype,
                void*        qkv,
                const void*  q,
                const void*  kv_a,
                int          token_num,
                int          head_num,
                int          kv_lora_rank,
                int          rope_dim,
                cudaStream_t stream);

}  // namespace turbomind
