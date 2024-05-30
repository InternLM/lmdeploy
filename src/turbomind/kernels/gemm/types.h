// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind::gemm {

enum class Order : int {
    kColMajor = 0,
    kRowMajor = 1,
};

inline constexpr Order kColMajor = Order::kColMajor;
inline constexpr Order kRowMajor = Order::kRowMajor;

using Pack = uint32_t;

typedef enum MMA_Tag {
    HMMA_16816 = 0x100,
    HMMA_1688  = 0x200,
    HMMA_884   = 0x300,
} MMA_Tag;

typedef enum Op_Tag {
    OPERAND_A = 0x010,
    OPERAND_B = 0x020,
    OPERAND_U = 0x030,
    OPERAND_V = 0x040,
} Op_Tag;

constexpr MMA_Tag get_mma_tag(Pack pack)
{
    return static_cast<MMA_Tag>(pack & 0xf00);
}

constexpr Op_Tag get_operand_tag(Pack pack)
{
    return static_cast<Op_Tag>(pack & 0x0f0);
}

constexpr int get_pack_num(Pack pack)
{
    return pack & 0x00f;
}

enum class QuantType : int {
    kNone,
    kSymmetric,
    kAsym_FMA,
    kAsym_SubMul,
};

enum class Epilogue : int {
    kNone = 0,
    kChannelwiseLinearCombination,  // D'[i,:] = a[i] * D[i,:] + b[i]
    kGatedSilu,                     // D'[:,i] = D[:,i*2] * silu(D[:,i*2+1])
    kGatedGelu,
};

enum class DataType : int {
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

template<DataType dtype>
struct get_dtype {};

template<>
struct get_dtype<DataType::F16> {
    using type = half;
};

struct QuantDesc {
    QuantType type;
    int       group_size;
};

enum class DispatchPolicy : int {
    kDefault = 0,
    kMeasure,
    kUseCached,
};

struct Operation {
    DispatchPolicy dispatch;
    Epilogue       epilogue;
    QuantDesc      quant_a;
    QuantDesc      quant_b;
};

struct MatrixLayout {
    DataType type;
    Order    order;
    int      rows;
    int      cols;
    int      ld;
    Pack     pack;
};

struct Workspace {
    void*  barriers;
    size_t barriers_size;
    void*  partials;
    size_t partials_size;
};

}  // namespace turbomind::gemm