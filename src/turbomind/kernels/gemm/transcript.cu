// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/format.h"
#include "src/turbomind/kernels/gemm/impl_81616.h"
#include "src/turbomind/kernels/gemm/tile_iterator.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include "src/turbomind/kernels/gemm/transcript_template.h"
#include <type_traits>

namespace turbomind::gemm {

namespace {

template<class Ti, class To>
struct _Converter {
    __device__ _Converter(): impl_(1, 0) {}
    template<class T>
    __device__ auto operator()(T&& t) const
    {
        return impl_((T&&)t);
    }
    ConvertKvCache<Ti, To> impl_;
};

struct BaseConfig {
    static constexpr int CTA_M = 64;
    static constexpr int CTA_N = 64;
    static constexpr int CTA_K = 32;

    static constexpr int WARP_M = 64;
    static constexpr int WARP_N = 64;
    static constexpr int WARP_K = 32;
};

template<class T, class TbI_, class TbO>
struct Config: BaseConfig {};

template<class T, class Tb>
struct Config<T, T, Tb>: BaseConfig {
    using Gemm0  = Impl<MMA_81616, T, T, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Gemm1  = Impl<MMA_81616, T, Tb, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Kernel = Transcript<void, Gemm0, Gemm1, _Converter<T, Tb>, CtaSwizzleMap<0>>;
};

template<class T>
struct Config<T, uint16_t, uint4_t>: BaseConfig {
    static_assert(sizeof(T) == 2);
    using Gemm0  = Impl<MMA_81616, T, T, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Gemm1  = Impl<MMA_81616, T, uint4_t, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Kernel = Transcript<void, Gemm0, Gemm1, Converter<uint16_t, uint4_t>, CtaSwizzleMap<0>>;
};

template<class T>
struct Config<T, uint16_t, uint8_t>: BaseConfig {
    static_assert(sizeof(T) == 2);
    using Gemm0  = Impl<MMA_81616, T, T, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Gemm1  = Impl<MMA_81616, T, uint8_t, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3, 0>;
    using Kernel = Transcript<void, Gemm0, Gemm1, Converter<uint16_t, uint8_t>, CtaSwizzleMap<0>>;
};

}  // namespace

namespace detail {

template<class T>
auto cast(T* p)
{
    if constexpr (bitsof<T> % 8 == 0) {
        return p;
    }
    else {
        return (char*)p;
    }
}

}  // namespace detail

template<class T, class Ti, class To>
void transcript(To* dst, const Ti* src, int n, int k, cudaStream_t st)
{
    using Kernel = typename Config<T, Ti, To>::Kernel;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);
    if constexpr (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(transcript_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    using Map = typename Kernel::CtaMap;

    auto tiles = Map::get_tiled_shape(Kernel::CTA_M, n, k, Kernel::CTA_M, Kernel::CTA_N, 1);
    auto grid  = Map::get_grid_shape(tiles);
    auto block = Kernel::WARP_CNT * WARP_SIZE;

    std::cout << "P_K: " << Kernel::P_K << ", P_N: " << Kernel::P_N << std::endl;

    auto _src = [&] {
        if constexpr (std::is_same_v<Ti, uint16_t>) {
            return (const T*)src;
        }
        else {
            return src;
        }
    }();

    typename Kernel::Param params{nullptr, _src, detail::cast(dst), Kernel::CTA_M, n, k};

    transcript_kernel<Kernel><<<grid, block, kSmemSize, st>>>(params);
}

template void transcript<half>(half* dst, const half* src, int n, int k, cudaStream_t st);

template void transcript<half>(uint4_t* dst, const half* src, int n, int k, cudaStream_t st);
template void transcript<half>(uint8_t* dst, const half* src, int n, int k, cudaStream_t st);

template void transcript<half>(uint4_t* dst, const uint16_t* src, int n, int k, cudaStream_t st);
template void transcript<half>(uint8_t* dst, const uint16_t* src, int n, int k, cudaStream_t st);

}  // namespace turbomind::gemm