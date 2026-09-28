// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

void invokeCaptureTargetHiddenRows(const void*  packed_residual,
                                   void*        captured,
                                   int          owned_begin,
                                   int          owned_row_count,
                                   int          hidden_units,
                                   int          packed_leading_dimension,
                                   int          captured_leading_dimension,
                                   int          tap_ordinal,
                                   int          element_bits,
                                   cudaStream_t stream);

}  // namespace turbomind
