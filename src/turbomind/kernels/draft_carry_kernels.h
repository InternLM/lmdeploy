// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

void invokeSelectDraftCarry(const void*  local_residual,
                            const int*   selected_local_rows,
                            const bool*  candidate_active,
                            void*        carry,
                            int          local_token_num,
                            int          candidate_count,
                            int          hidden_size,
                            int          element_bits,
                            int          first,
                            int          last,
                            cudaStream_t stream);

}  // namespace turbomind
