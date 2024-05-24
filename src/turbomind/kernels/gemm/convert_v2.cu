// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/convert_v2.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"

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

struct Config {
    static constexpr int CTA_M = 32;
    static constexpr int CTA_K = 32;

    static constexpr int WARP_M = 32;
    static constexpr int MMA_K  = 16;

    static constexpr int WARP_CNT = CTA_M / WARP_M;

    struct OperandA {
        using Dtype      = half;
        using SmemLayout = SmemLayoutV2<CTA_M, CTA_K, 16, 32, Swizzle<2, 3, 3>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_A<half, false>, WARP_M, MMA_K>;
        using GmemIter   = GmemIteratorSm80<Dtype, ThreadMap<CTA_K, CTA_M, 8, WARP_CNT>, SmemLayout, false, true, 0>;
        static constexpr Order Layout     = Order::kColMajor;
        static constexpr bool  is_k_major = true;
    };

    using Kernel = ConvertOperand<CTA_M, CTA_K, 1, OperandA, _Converter<half, half>, half, true>;
};

}  // namespace

int Convert(const void*         S,  //
            const MatrixLayout& Sdesc,
            void*               D,
            const MatrixLayout& Ddesc,
            cudaStream_t        stream)
{
    using Kernel = typename Config::Kernel;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);

    if (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(convert_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    typename Kernel::Param param{
        Sdesc.rows,
        Sdesc.cols,
        (const half*)S,
        Sdesc.ld,
        (half*)D,
        Ddesc.ld,
    };

    constexpr int threads = Kernel::WARP_CNT * WARP_SIZE;
    const int     blocks  = ceil_div(Sdesc.rows, Config::CTA_M);

    convert_kernel<Kernel><<<blocks, threads, kSmemSize, stream>>>(param);

    return 0;
}

}  // namespace turbomind::gemm