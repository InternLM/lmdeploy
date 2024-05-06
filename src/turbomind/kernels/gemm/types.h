// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind::gemm {

enum class LayoutType {
    kColMajor,
    kRowMajor,
    kFragment_884,
    kFragment_81616,
    kFragment_16816,
};

enum class QuantType {
    kNone,
    kSymmetric,
    kAsym_FMA,
    kAsym_SubMul,
};

enum class Epilogue {
    kNone = 0,
    kChannelwiseLinearCombination,  // D'[i,:] = a[i] * D[i,:] + b[i]
    kGatedSilu,                     // D'[:,i] = D[:,i*2] * silu(D[:,i*2+1])
    kGatedGelu,
};

enum class DataType {
    U4,
    U8,
    F16,
    F32,
    BF16,
    TF32,
};

inline const char* to_string(DataType data_type)
{
    switch (data_type) {
        case DataType::U4:
            return "u4";
        case DataType::U8:
            return "u8";
        case DataType::F16:
            return "f16";
        case DataType::F32:
            return "f32";
        case DataType::BF16:
            return "bf16";
        case DataType::TF32:
            return "tf32";
        default:
            return "unknown";
    }
}

template<class T>
struct get_data_type {};

template<>
struct get_data_type<half> {
    static constexpr auto value = DataType::F16;
};

#if ENABLE_BF16
template<>
struct get_data_type<nv_bfloat16> {
    static constexpr auto value = DataType::BF16;
};
#endif

template<>
struct get_data_type<uint4_t> {
    static constexpr auto value = DataType::U4;
};

template<>
struct get_data_type<uint8_t> {
    static constexpr auto value = DataType::U8;
};

template<class T>
inline constexpr auto get_data_type_v = get_data_type<T>::value;

struct QuantDesc {
    QuantType type;
    int       group_size;
};

enum class DispatchType {
    kDefault = 0,
    kCached,
    kMeasure,
};

struct Operation {
    QuantDesc    quant_desc;
    DispatchType dispatch;
    Epilogue     epilogue;
};

struct MatrixLayout {
    DataType   type;
    LayoutType order;
    int        rows;
    int        cols;
    int        ld;
};

}  // namespace turbomind::gemm