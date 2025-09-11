// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

constexpr const char* to_string(Order order)
{
    switch (order) {
        case kColMajor:
            return "Col";
        case kRowMajor:
            return "Row";
    }
    return "";
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
    OPERAND_C = 0x050,
    OPERAND_D = 0x060,
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

enum class Striding : int
{
    kFlat,     // [1111,2222,3333]
    kRagged,   // [11,2222222,333]  [0 , 2      , 9  ]
    kIndexed,  // [xx xxxxxxx xxx], [01, 2345678, 9ab]
    kBlocked,  // [11][22222][333]
};

inline const char* to_string(Striding striding)
{
    switch (striding) {
        case Striding::kFlat:
            return "f";
        case Striding::kRagged:
            return "r";
        case Striding::kIndexed:
            return "i";
        case Striding::kBlocked:
            return "b";
        default:
            return "unknown";
    }
}

enum class QuantType : int
{
    kNone    = 0,
    kK       = 1,
    kM       = 2,
    kB       = 3,
    kDefault = kK,
};

inline const char* to_string(QuantType q)
{
    switch (q) {
        case QuantType::kNone:
            return "none";
        case QuantType::kK:
            return "k";
        case QuantType::kM:
            return "m";
        case QuantType::kB:
            return "b";
        default:
            return "unknown";
    }
}

enum class Epilogue : int
{
    kNone               = 0,
    kChannelCombination = 0x1,
    kGatedSilu          = 0x2,
};

struct QuantDesc {
    QuantType type;
    int       group_size;

    operator bool() const noexcept
    {
        return (int)type || group_size;
    }
};

inline std::string to_string(QuantDesc desc)
{
    if (desc) {
        return to_string(desc.type) + std::to_string(desc.group_size);
    }
    else {
        return to_string(desc.type);
    }
}

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

class Kernel;
class Context;

struct Tape {
    int   ctas;
    int   max_num;
    int   max_ctas;
    char* buffer;
    int4* gemm_shapes;
    int4* tiled_shapes;
    int4* tile_offsets;
    int2* iter_k_ranges;
    int*  tile_ids;
};

struct Operation {
    DispatchPolicy dispatch;
    Epilogue       epilogue;
    QuantDesc      quant_a;
    QuantDesc      quant_b;
    int            batch_dim;
    // void*          reserved;
};

inline Operation transpose(Operation o)
{
    std::swap(o.quant_a, o.quant_b);
    o.batch_dim = 1 - o.batch_dim;
    return o;
}

struct MatrixLayout {
    DataType type;
    Order    order;
    int      rows;
    int      cols;
    int      ld;
    Pack     pack;
    int      num;
    int*     offsets;
    int*     idxs;
};

inline std::ostream& operator<<(std::ostream& os, const MatrixLayout& x)
{
    os << x.type << " " << to_string(x.order) << " " << x.rows << " " << x.cols << " " << x.num << " " << x.ld;
    return os;
}

inline int64_t byte_size(const MatrixLayout& m)
{
    return byte_size(m.type, (int64_t)m.rows * m.cols);
}

inline Striding get_mode(const MatrixLayout& m)
{
    if (m.idxs) {
        return Striding::kIndexed;
    }
    else if (m.ld == 0 || m.offsets) {
        return Striding::kBlocked;
    }
    return Striding::kFlat;
}

struct Workspace {
    void*  barriers;
    size_t barriers_size;
    void*  partials;
    size_t partials_size;
    void*  tensormaps;
    size_t tensormaps_size;
    int*   flags;
};

}  // namespace turbomind::gemm
