#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

template<class T>
constexpr WeightType get_default_weight_type()
{
    if constexpr (std::is_same_v<T, half>) {
        return WeightType::kFP16;
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return WeightType::kBF16;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return WeightType::kFP32;
    }
    else {
        static_assert(sizeof(T) != sizeof(T), "not implemented");
        return {};
    }
}

inline size_t getBitSize(WeightType type)
{
    switch (type) {
        case WeightType::kFP32:
            return 32;
        case WeightType::kFP16:
            return 16;
        case WeightType::kFP8:
            return 8;
        case WeightType::kBF16:
            return 16;
        case WeightType::kINT8:
            return 8;
        case WeightType::kINT4:
            return 4;
    }
    return 0;
}

}  // namespace turbomind
