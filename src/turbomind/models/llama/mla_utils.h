// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

void MLACopyQKV(DataType     dtype,
                void*        qkv,
                const void*  q,
                const void*  kv_a,
                const void*  kv_b,
                int          token_num,
                int          head_num,
                int          nope_dim,
                int          rope_dim,
                int          kv_lora_rank,
                int          v_head_dim,
                cudaStream_t stream);

}  // namespace turbomind
