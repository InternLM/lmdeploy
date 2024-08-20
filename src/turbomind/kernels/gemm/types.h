// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind::gemm {

enum class Order : int
{
    kColMajor = 0,
    kRowMajor = 1,
};

inline constexpr Order kColMajor = Order::kColMajor;
inline constexpr Order kRowMajor = Order::kRowMajor;

constexpr Order operator~(Order a)
{
    return a == kColMajor ? kRowMajor : kColMajor;
}

using Pack = uint32_t;

typedef enum MMA_Tag
{
    HMMA_16816 = 0x100,  // sm80+
    HMMA_1688  = 0x200,  // sm75
    HMMA_884   = 0x300,  // sm70
    HMMA_SIMT  = 0x400,  // sm75-
} MMA_Tag;

typedef enum Op_Tag
{
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

enum class QuantType : int
{
    kNone,
    kDefault,
};

enum class Epilogue : int
{
    kNone               = 0,
    kChannelCombination = 0x1,
    kGatedSilu          = 0x2,
};

enum class DataType : int
{
    U4,
    U8,
    U16,
    U32,
    U64,
    F8_E4M3,
    F8_E5M2,
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

inline int64_t get_size(DataType type, int64_t size)
{
    if (!size) {
        return 0;
    }
    switch (type) {
        case DataType::U64:
            return size * 8;
        case DataType::F32:
        case DataType::U32:
            return size * 4;
        case DataType::BF16:
        case DataType::F16:
        case DataType::U16:
            return size * 2;
        case DataType::U8:
        case DataType::F8_E4M3:
        case DataType::F8_E5M2:
            return size;
        case DataType::U4:
            return size / 2;
        default:
            // std::cerr << to_string(type) << "\n";
            return -1;
    }
}

template<class T>
struct get_data_type {
};

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
struct get_dtype {
};

template<>
struct get_dtype<DataType::F16> {
    using type = half;
};

template<>
struct get_dtype<DataType::U4> {
    using type = uint4_t;
};

template<>
struct get_dtype<DataType::U8> {
    using type = uint8_t;
};

template<>
struct get_dtype<DataType::U16> {
    using type = uint16_t;
};

template<>
struct get_dtype<DataType::U32> {
    using type = uint32_t;
};

struct QuantDesc {
    QuantType type;
    int       group_size;
};

enum class DispatchPolicy : int
{
    kDefault = 0,
    kMeasure = 1,
    kReuse   = 2,
    kAppend  = 3,
};

constexpr bool operator&(const DispatchPolicy& a, const DispatchPolicy& b)
{
    return ((int)a & (int)b);
}

struct Operation {
    DispatchPolicy dispatch;
    Epilogue       epilogue;
    QuantDesc      quant_a;
    QuantDesc      quant_b;
    int            batch_dim;
    void*          reserved;
};

struct MatrixLayout {
    DataType type;
    Order    order;
    int      rows;
    int      cols;
    int      ld;
    Pack     pack;
};

inline int64_t get_size(const MatrixLayout& m)
{
    return get_size(m.type, (int64_t)m.rows * m.cols);
}

struct Workspace {
    void*  barriers;
    size_t barriers_size;
    void*  partials;
    size_t partials_size;
};

}  // namespace turbomind::gemm
