// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

namespace turbomind::kernel {

template<bool ZeroCentered, class T>
__device__ __forceinline__ T ApplyRMSnorm(T value, float inv_rms, T weight)
{
    if constexpr (ZeroCentered) {
        return static_cast<T>(static_cast<float>(value) * inv_rms * (1.0f + static_cast<float>(weight)));
    }
    else {
        return static_cast<T>(static_cast<float>(value) * inv_rms) * weight;
    }
}

}  // namespace turbomind::kernel
