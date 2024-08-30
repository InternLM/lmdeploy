// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

// bytes / second
float MeasureL2CacheThroughput();

// fused multiply-add / second
float MeasureMmaThroughput(int proble_size = 16384);

}  // namespace turbomind::gemm
