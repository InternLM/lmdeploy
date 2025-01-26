#pragma once

#ifdef _CLANGD
#define MSCCLPP_DEVICE_COMPILE 1
#define MSCCLPP_DEVICE_CUDA 1

#ifdef MSCCLPP_HOST_DEVICE_INLINE
#undef MSCCLPP_HOST_DEVICE_INLINE
#endif

#define MSCCLPP_HOST_DEVICE_INLINE __host__ __device__ __inline__
#define MSCCLPP_DEVICE_INLINE __device__ __inline__
#endif

#include "mscclpp/sm_channel.hpp"
#include "mscclpp/sm_channel_device.hpp"

#include "src/turbomind/kernels/core/array.h"

namespace turbomind {

struct SmChannels {
    mscclpp::DeviceHandle<mscclpp::SmChannel> data[8];

    __host__ __device__ auto& operator[](int i)
    {
        return data[i];
    }
};

template<class T>
constexpr auto get_proxy_type(T)
{
    if constexpr (sizeof(T) == 16) {
        return int4{};
    }
    else if constexpr (sizeof(T) == 8) {
        return int2{};
    }
    else if constexpr (sizeof(T) == 4) {
        return int{};
    }
    else {
        static_assert(sizeof(T) != sizeof(T), "not supported");
        return int{};
    }
}

template<class T>
using proxy_type = decltype(get_proxy_type(T{}));




}  // namespace turbomind