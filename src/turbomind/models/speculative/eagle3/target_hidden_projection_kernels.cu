// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/eagle3/target_hidden_projection_kernels.h"

#include <cstddef>

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
                                   cudaStream_t stream)
{
    if (owned_row_count == 0) {
        return;
    }

    cudaMemcpy2DAsync(static_cast<std::byte*>(captured) + tap_ordinal * hidden_units * element_bits / 8,
                      captured_leading_dimension * element_bits / 8,
                      static_cast<const std::byte*>(packed_residual)
                          + owned_begin * packed_leading_dimension * element_bits / 8,
                      packed_leading_dimension * element_bits / 8,
                      hidden_units * element_bits / 8,
                      owned_row_count,
                      cudaMemcpyDeviceToDevice,
                      stream);
}

}  // namespace turbomind
