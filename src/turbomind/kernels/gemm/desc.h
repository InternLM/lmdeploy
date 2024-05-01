// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct GemmDesc {
    LayoutType   layout_A;
    LayoutType   layout_B;
    LayoutType   layout_C;
    DataType     type_A;
    DataType     type_B;
    DataType     type_C;
    QuantType    quant_type;
    EpilogueType epilogue;
    int          m;
    int          n;
    int          k;
};

}  // namespace turbomind::gemm