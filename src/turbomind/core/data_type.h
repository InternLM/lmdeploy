// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/check.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

// forward declarations for CUDA floating point types
struct __half;
struct __nv_bfloat16;
struct __nv_fp8_e4m3;
struct __nv_fp8_e5m2;

namespace turbomind {

// clang-format off

struct uint2_t {};
struct uint4_t {};
struct uint6_t {};

template <int I>
struct int_constant: std::integral_constant<int, I> {};

template <class T>
struct bitsof_t: int_constant<sizeof(T) * 8> {};

template <> struct bitsof_t<uint2_t>: int_constant<2> {};
template <> struct bitsof_t<uint4_t>: int_constant<4> {};
template <> struct bitsof_t<uint6_t>: int_constant<6> {};

template <class T>
inline constexpr bitsof_t<T> bitsof{};

using half_t = __half;
using bfloat16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

struct fp4_e2m1_t {};

template <> struct bitsof_t<fp4_e2m1_t>: int_constant<4> {};


constexpr int encode_data_type(bool sign, int exponent, int mantissa) {
    return ((sign << 16) | (exponent << 8) | mantissa);
}

enum class DataType: int {
    kNull        = 0,
    kBool        = 1,
    kUint8       = encode_data_type(0,  0,  8),
    kUint16      = encode_data_type(0,  0, 16),
    kUint32      = encode_data_type(0,  0, 32),
    kUint64      = encode_data_type(0,  0, 64),
    kInt8        = encode_data_type(1,  0,  8),
    kInt16       = encode_data_type(1,  0, 16),
    kInt32       = encode_data_type(1,  0, 32),
    kInt64       = encode_data_type(1,  0, 64),
    kFloat16     = encode_data_type(1,  5, 10),
    kFloat32     = encode_data_type(1,  8, 23),
    kFloat64     = encode_data_type(1, 11, 52),
    kBfloat16    = encode_data_type(1,  8,  7),
    kFloat4_e2m1 = encode_data_type(1,  2,  1),
    kFloat6_e2m3 = encode_data_type(1,  2,  3),
    kFloat6_e3m2 = encode_data_type(1,  3,  2),
    kFloat8_e4m3 = encode_data_type(1,  4,  3),
    kFloat8_e5m2 = encode_data_type(1,  5,  2),
    kUint2       = encode_data_type(0,  0,  2),
    kUint4       = encode_data_type(0,  0,  4),
    kUint6       = encode_data_type(0,  0,  6),
    kPointer,
    kUint        = kUint32,
    kInt         = kInt32,
    kFloat       = kFloat32,
    kHalf        = kFloat16,
    kDouble      = kFloat64,
    kE2m1        = kFloat4_e2m1,
    kE2m3        = kFloat6_e2m3,
    kE3m2        = kFloat6_e3m2,
    kE4m3        = kFloat8_e4m3,
    kE5m2        = kFloat8_e5m2,
};

inline constexpr DataType kNull = DataType::kNull;
inline constexpr DataType kBool = DataType::kBool;
inline constexpr DataType kPointer = DataType::kPointer;
inline constexpr DataType kUint8  = DataType::kUint8;
inline constexpr DataType kUint16 = DataType::kUint16;
inline constexpr DataType kUint32 = DataType::kUint32;
inline constexpr DataType kUint64 = DataType::kUint64;
inline constexpr DataType kInt8  = DataType::kInt8;
inline constexpr DataType kInt16 = DataType::kInt16;
inline constexpr DataType kInt32 = DataType::kInt32;
inline constexpr DataType kInt64 = DataType::kInt64;
inline constexpr DataType kFloat16 = DataType::kFloat16;
inline constexpr DataType kFloat32 = DataType::kFloat32;
inline constexpr DataType kFloat64 = DataType::kFloat64;
inline constexpr DataType kBfloat16 = DataType::kBfloat16;
inline constexpr DataType kFloat8_e4m3 = DataType::kFloat8_e4m3;
inline constexpr DataType kFloat8_e5m2 = DataType::kFloat8_e5m2;
inline constexpr DataType kFloat4_e2m1 = DataType::kFloat4_e2m1;
inline constexpr DataType kUint2  = DataType::kUint2;
inline constexpr DataType kUint4  = DataType::kUint4;
inline constexpr DataType kUint6  = DataType::kUint6;
inline constexpr DataType kUint = DataType::kUint;
inline constexpr DataType kInt = DataType::kInt;
inline constexpr DataType kHalf = DataType::kHalf;
inline constexpr DataType kFloat = DataType::kFloat;
inline constexpr DataType kDouble = DataType::kDouble;

template <class T>
struct to_data_type;

template <DataType D>
struct from_data_type;

#define CVT_DATA_TYPE(D, T) \
    template <> struct to_data_type<T> { static constexpr auto value = DataType::D; }; \
    template <> struct from_data_type<DataType::D> { using type = T; }

CVT_DATA_TYPE(kNull, void);

CVT_DATA_TYPE(kBool, bool);
CVT_DATA_TYPE( kUint8, uint8_t);
CVT_DATA_TYPE(kUint16, uint16_t);
CVT_DATA_TYPE(kUint32, uint32_t);
CVT_DATA_TYPE(kUint64, uint64_t);

CVT_DATA_TYPE( kInt8, int8_t);  // NOTE: `int8_t` is `signed char` and is different from `char`
CVT_DATA_TYPE(kInt16, int16_t);
CVT_DATA_TYPE(kInt32, int32_t);
CVT_DATA_TYPE(kInt64, int64_t);

CVT_DATA_TYPE(kFloat16, half_t);
CVT_DATA_TYPE(kFloat32, float);
CVT_DATA_TYPE(kFloat64, double);
CVT_DATA_TYPE(kBfloat16, bfloat16_t);
CVT_DATA_TYPE(kFloat4_e2m1, fp4_e2m1_t);
CVT_DATA_TYPE(kFloat8_e4m3, fp8_e4m3_t);
CVT_DATA_TYPE(kFloat8_e5m2, fp8_e5m2_t);

CVT_DATA_TYPE(kUint2, uint2_t);
CVT_DATA_TYPE(kUint4, uint4_t);
CVT_DATA_TYPE(kUint6, uint6_t);

#undef CVT_DATA_TYPE

template <class T> struct to_data_type<T*> { static constexpr auto value = DataType::kPointer; };
template <>  struct from_data_type<DataType::kPointer> { using type = void*; };

template <class T>
inline constexpr auto data_type_v = to_data_type<std::remove_cv_t<T>>::value;

template <DataType D>
using data_type_t = typename from_data_type<D>::type;

constexpr std::ptrdiff_t byte_size(DataType type, std::ptrdiff_t size = 1) {
    switch (type) {
        case kNull: return 0;
        case kBool:
        case kUint8:
        case kInt8:
        case kFloat8_e4m3:
        case kFloat8_e5m2:
            return size;
        case kUint16:
        case kInt16:
        case kFloat16:
        case kBfloat16:
            return size * 2;
        case kUint32:
        case kInt32:
        case kFloat32:
            return size * 4;
        case kUint64:
        case kInt64:
        case kFloat64:
            return size * 8;
        case kUint2: return size * 2 / 8;
        case kUint4:
        case kFloat4_e2m1:
            return size * 4 / 8;
        case kUint6: return size * 6 / 8;
        case kPointer: return size * sizeof(void*);
        default:
            return 0;
    }
    return 0;
}

template <class T>
constexpr std::ptrdiff_t byte_size(std::ptrdiff_t size = 1) { return byte_size(data_type_v<T>, size); }

constexpr std::ptrdiff_t numel(DataType type, std::ptrdiff_t size = 1) {
    switch (type) {
        case kNull: return 0;
        case kBool:
        case kUint8:
        case kInt8:
        case kFloat8_e4m3:
        case kFloat8_e5m2:
            return size;
        case kUint16:
        case kInt16:
        case kFloat16:
        case kBfloat16:
            return size / 2;
        case kUint32:
        case kInt32:
        case kFloat32:
            return size / 4;
        case kUint64:
        case kInt64:
        case kFloat64:
            return size / 8;
        case kUint2: return size * 8 / 2;
        case kUint4:
        case kFloat4_e2m1:
            return size * 8 / 4;
        case kUint6: return size * 8 / 6;
        case kPointer: return size / sizeof(void*);
        default:
            return 0;
    }
    return 0;
}

template <class T>
constexpr std::ptrdiff_t numel(std::ptrdiff_t size) { return numel(data_type_v<T>, size); }

constexpr const char* to_string(DataType type) {
    switch (type) {
        case kNull: return "nil";
        case kBool: return "bool";
        case kUint8: return "u8";
        case kUint16: return "u16";
        case kUint32: return "u32";
        case kUint64: return "u64";
        case kInt8: return "i8";
        case kInt16: return "i16";
        case kInt32: return "i32";
        case kInt64: return "i64";
        case kFloat16: return "f16";
        case kFloat32: return "f32";
        case kFloat64: return "f64";
        case kBfloat16: return "bf16";
        case kFloat8_e4m3: return "e4m3";
        case kFloat8_e5m2: return "e5m2";
        case kFloat4_e2m1: return "e2m1";
        case kUint2: return "u2";
        case kUint4: return "u4";
        case kUint6: return "u8";
        case kPointer: return "pointer";
        default:
            return "unknown";
    }
    return "";
}

inline std::ostream& operator<<(std::ostream& os, DataType type) {
    os << to_string(type);
    return os;
}

/// TODO: mapping with DLPack

// clang-format on

#define TM_PP_NARGS(...) TM_PP_NARGS_IMPL(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define TM_PP_NARGS_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, N, ...) N

#define TM_PP_CAT(a, b) a##b
#define TM_PP_STR(x) #x

#define TM_PP_DISPATCH_N(macro, ...) TM_PP_DISPATCH_N_IMPL(macro, TM_PP_NARGS(__VA_ARGS__))
#define TM_PP_DISPATCH_N_IMPL(macro, x) TM_PP_CAT(macro, x)

#define TM_PP_INVOKE_1(macro, f, _0) macro(f, _0)

#define TM_PP_INVOKE_2(macro, f, _0, _1)                                                                               \
    macro(f, _0);                                                                                                      \
    macro(f, _1)

#define TM_PP_INVOKE_3(macro, f, _0, _1, _2)                                                                           \
    macro(f, _0);                                                                                                      \
    macro(f, _1);                                                                                                      \
    macro(f, _2)

#define TM_PP_INVOKE_4(macro, f, _0, _1, _2, _3)                                                                       \
    macro(f, _0);                                                                                                      \
    macro(f, _1);                                                                                                      \
    macro(f, _2);                                                                                                      \
    macro(f, _3)

#define TM_PP_INVOKE_5(macro, f, _0, _1, _2, _3, _4)                                                                   \
    macro(f, _0);                                                                                                      \
    macro(f, _1);                                                                                                      \
    macro(f, _2);                                                                                                      \
    macro(f, _3);                                                                                                      \
    macro(f, _4)

#define TM_DISPATCH_DTYPE_RET_CASE(f, t)                                                                               \
    case ::turbomind::data_type_v<t>:                                                                                  \
        return f(t{});

#define TM_DISPATCH_DTYPE_CASE(f, t)                                                                                   \
    case ::turbomind::data_type_v<t>:                                                                                  \
        f(t{});                                                                                                        \
        break

// clang-format off
#define TM_DISPATCH_DTYPES_RET(var, f, ...)                                                                            \
    switch (var) {                                                                                                     \
        TM_PP_DISPATCH_N(TM_PP_INVOKE_, __VA_ARGS__)(TM_DISPATCH_DTYPE_RET_CASE, f, __VA_ARGS__);                      \
        default:                                                                                                       \
            TM_CHECK(0) << "unsupported type: "  << to_string(var);                                                    \
            return {};                                                                                                 \
    }

#define TM_DISPATCH_DTYPES(var, f, ...)                                                                                \
    switch (var) {                                                                                                     \
        TM_PP_DISPATCH_N(TM_PP_INVOKE_, __VA_ARGS__)(TM_DISPATCH_DTYPE_CASE, f, __VA_ARGS__);                          \
        default:                                                                                                       \
            TM_CHECK(0) << "unsupported type: "  << to_string(var);                                                    \
    }
// clang-format on

#define TM_PRIMARY_DTYPES_0 ::turbomind::half_t

#if ENABLE_BF16
#define TM_PRIMARY_DTYPES_1 TM_PRIMARY_DTYPES_0, ::turbomind::bfloat16_t
#else
#define TM_PRIMARY_DTYPES_1 TM_PRIMARY_DTYPES_0
#endif

#if ENABLE_FP32
#define TM_PRIMARY_DTYPES TM_PRIMARY_DTYPES_1, float
#else
#define TM_PRIMARY_DTYPES TM_PRIMARY_DTYPES_1
#endif

#define TM_DISPATCH_PRIMARY_DTYPES(var, func) TM_DISPATCH_DTYPES(var, func, TM_PRIMARY_DTYPES)

#define TM_DISPATCH_PRIMARY_DTYPES_RET(var, func) TM_DISPATCH_DTYPES_RET(var, func, TM_PRIMARY_DTYPES)

}  // namespace turbomind
