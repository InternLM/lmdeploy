// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <array>

namespace turbomind::gemm {

// aggregate that uniquely identifies a GEMM problem
struct GemmDesc {
    int       arch;
    DataType  type_a;
    DataType  type_b;
    DataType  type_c;
    Order     order_a;
    Order     order_b;
    Order     order_c;
    Striding  striding_a;
    Striding  striding_b;
    Striding  striding_c;
    Pack      pack_a;
    Pack      pack_b;
    Pack      pack_u;
    Pack      pack_v;
    QuantDesc quant_a;
    QuantDesc quant_b;
    Epilogue  epilogue;
    int       batch_dim;
    int       sched;
    int       m;
    int       n;
    int       k;
    int       num;
};

inline GemmDesc transpose(GemmDesc d)
{
    std::swap(d.type_a, d.type_b);
    std::swap(d.order_a, d.order_b);
    d.order_a = ~d.order_a;
    d.order_b = ~d.order_b;
    d.order_c = ~d.order_c;
    std::swap(d.striding_a, d.striding_b);
    std::swap(d.pack_a, d.pack_b);
    std::swap(d.pack_u, d.pack_v);
    std::swap(d.quant_a, d.quant_b);
    std::swap(d.m, d.n);
    return d;
}

enum class OpClass
{
    kSIMT,
    kMMA_s884,
    kMMA_s16816,
    kGMMA_s64n16
};

inline const char* to_string(OpClass op)
{
    switch (op) {
        case OpClass::kSIMT:
            return "simt";
        case OpClass::kMMA_s884:
            return "s884";
        case OpClass::kMMA_s16816:
            return "s16816";
        default:
            return "unknown_op_cls";
    }
}

// aggregate that uniquely identifies a kernel
struct KernelDesc {
    int       arch;
    OpClass   op_class;
    DataType  type_a;
    DataType  type_b;
    DataType  type_c;
    Order     order_a;
    Order     order_b;
    Order     order_c;
    Striding  striding_a;
    Striding  striding_b;
    Striding  striding_c;
    Pack      pack_a;
    Pack      pack_b;
    Pack      pack_u;
    Pack      pack_v;
    QuantDesc quant_a;
    QuantDesc quant_b;
    int       policy_a;
    int       policy_b;
    int3      cta_tile;
    int3      mma_tile;
    int2      cluster_shape;
    int3      align;
    int2      c_tile;
    int       stages;
    bool      split_k;
    int       sched;

    // set by `KernelImpl`
    int                max_active_ctas;
    cudaFuncAttributes attr;
};

class Kernel;
struct LaunchSpec {
    Kernel* kernel;
    int     swizzle;
    int     splits;
    float   measured;

    std::array<int64_t, 2> estimated;
};

}  // namespace turbomind::gemm
