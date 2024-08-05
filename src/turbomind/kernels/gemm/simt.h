// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm::simt {

// constexpr int OP_M = 2;
// constexpr int OP_N = 16;
// constexpr int OP_K = 4;

// constexpr int OP_M = 4;
// constexpr int OP_N = 8;
// constexpr int OP_K = 8;

constexpr int OP_M = 1;
constexpr int OP_N = 32;
constexpr int OP_K = 8;

}  // namespace turbomind::gemm::simt
