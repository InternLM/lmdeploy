// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

// aggregate that uniquely identifies a GEMM problem
struct GemmDesc {
    DataType  type_a;
    DataType  type_b;
    DataType  type_c;
    Order     order_a;
    Order     order_b;
    Order     order_c;
    Pack      pack_a;
    Pack      pack_b;
    QuantDesc quant_a;
    QuantDesc quant_b;
    Epilogue  epilogue;
    int       m;
    int       n;
    int       k;
};

enum class OpClass {
    kSIMT,
    kMMA_884,
    kMMA_81616,
    kMMA_16816,
};

// aggregate that uniquely identifies a kernel
struct KernelDesc {
    DataType  type_a;
    DataType  type_b;
    DataType  type_c;
    Order     order_a;
    Order     order_b;
    Order     order_c;
    Pack      pack_a;
    Pack      pack_b;
    QuantDesc quant_a;
    QuantDesc quant_b;
    int3      cta_tile;
    int3      warp_tile;
    int       stages;
    bool      split_k;
    bool      align_m;
    bool      align_n;
    int       arch;
};

class Kernel;
struct LaunchSpec {
    Kernel* kernel;
    int     swizzle;
    int     splits;
    float   estimated;
    float   measured;
};

}  // namespace turbomind::gemm