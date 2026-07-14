// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cstdint>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

namespace detail {

struct __uint4_t {
    uint32_t x;
};

}  // namespace detail

template<class T, class SFINAE = void>
struct get_pointer_type_t {
    using type = T*;
};

template<class T>
using get_pointer_type = typename get_pointer_type_t<T>::type;

}  // namespace turbomind
