#pragma once

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/layout.h"

namespace turbomind::gemm {

struct OutputSpec {
    core::Layout layout;
    DataType     dtype{kNull};
    core::Layout scales_layout;
    DataType     scales_dtype{kNull};
};

}  // namespace turbomind::gemm
