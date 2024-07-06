// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm::sm70_mma_simt {

constexpr int OP_M = 2;
constexpr int OP_N = 16;
constexpr int OP_K = 4;

// order wrt (m, n)
// constexpr Order THR_ORDER = kRowMajor;

}  // namespace turbomind::gemm::sm70_mma_simt
