// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <array>
#include <tuple>
#include <type_traits>

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

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
    int       group_axis;
    int       m;
    int       n;
    int       k;
    int       num;
};

static_assert(std::is_trivially_copyable_v<GemmDesc>);

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
    d.batch_dim = 1 - d.batch_dim;
    if (d.group_axis >= 0) {
        d.group_axis = 1 - d.group_axis;
    }
    return d;
}

inline std::string to_string(const GemmDesc& d)
{
    std::stringstream ss;
    ss << "sm" << d.arch / 10;
    ss << "_" << to_string(d.type_a);  //
    if (d.quant_a) {
        ss << to_string(d.quant_a);
    }
    ss << "_" << to_string(d.type_b);  //
    if (d.quant_b) {
        ss << to_string(d.quant_b);
    }
    ss << "_" << to_string(d.type_c);
    ss << "_"                                    //
       << (d.order_a == kColMajor ? 'n' : 't')   //
       << (d.order_b == kColMajor ? 'n' : 't')   //
       << (d.order_c == kColMajor ? 'n' : 't');  //
    ss << "_"                                    //
       << to_string(d.striding_a)                //
       << to_string(d.striding_b)                //
       << to_string(d.striding_c);
    ss << "_" << d.m << "x" << d.n << "x" << d.k;
    ss << "_" << d.num;
    return ss.str();
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
    int       group_axis;
    int       backend;
    bool      transpose;
};

static_assert(std::is_trivially_copyable_v<KernelDesc>);

struct KernelInfo {
    int dynamic_smem_size;
    int max_active_ctas;
    int chunk_size_k;

    std::string name;

    cudaFuncAttributes attr;
};

inline KernelDesc transpose(const KernelDesc& d)
{
    KernelDesc k{d};

    k.arch     = d.arch;
    k.op_class = d.op_class;

    k.order_a = ~d.order_b;
    k.order_b = ~d.order_a;
    k.order_c = ~d.order_c;

    k.type_a = d.type_b;
    k.type_b = d.type_a;

    k.striding_a = d.striding_b;
    k.striding_b = d.striding_a;

    k.pack_a = d.pack_b;
    k.pack_b = d.pack_a;
    k.pack_u = d.pack_v;
    k.pack_v = d.pack_u;

    k.quant_a = d.quant_b;
    k.quant_b = d.quant_a;

    k.policy_a = d.policy_b;
    k.policy_b = d.policy_a;

    auto swap = [](auto& v) { std::swap(v.x, v.y); };

    swap(k.cta_tile);
    swap(k.mma_tile);
    swap(k.cluster_shape);
    swap(k.align);
    swap(k.c_tile);

    return k;
}

class Kernel;
struct LaunchSpec {
    Kernel* kernel;
    int     swizzle;
    int     splits;
    float   measured;

    std::array<int64_t, 2> estimated;
};

}  // namespace turbomind::gemm
