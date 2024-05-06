// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

// aggregate that uniquely identifies a GEMM problem
struct GemmDesc {
    DataType   type_a;
    DataType   type_b;
    DataType   type_c;
    LayoutType order_a;
    LayoutType order_b;
    LayoutType order_c;
    QuantDesc  quant_b;
    Epilogue   epilogue;
    int        m;
    int        n;
    int        k;
};

enum class OpClass {
    kSIMT,
    kMMA_884,
    kMMA_81616,
    kMMA_16816,
};

// aggregate that uniquely identifies a kernel
struct KernelDesc {
    DataType   type_a;
    DataType   type_b;
    DataType   type_c;
    LayoutType order_a;
    LayoutType order_b;
    LayoutType order_c;
    QuantDesc  quant_b;
    bool       align_m;
    bool       align_n;
    int3       cta_tile;
    int3       warp_tile;
    bool       split_k;
    int        stages;
    int        swizzle;
    int        arch;
};

}  // namespace turbomind::gemm