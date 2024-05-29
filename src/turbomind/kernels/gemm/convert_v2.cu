// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/codegen/sm80_s16816gemm_f16_f16_nn.h"
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

}  // namespace

int Convert(const void*         S,  //
            const MatrixLayout& Sdesc,
            void*               D,
            const MatrixLayout& Ddesc,
            cudaStream_t        stream)
{
    using T             = half;
    constexpr int CTA_M = 32;
    constexpr int CTA_K = 32;

    auto invoke = [&](auto operand) {
        using Operand = decltype(operand);
        using Kernel  = ConvertOperand<CTA_M, CTA_K, 1, Operand, T, _Converter<T, T>>;

        static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);

        if (kSmemSize > (48 << 10)) {
            cudaFuncSetAttribute(convert_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
        }

        typename Kernel::Param param{
            Sdesc.rows,
            Sdesc.cols,
            (const T*)S,
            Sdesc.ld,
            (T*)D,
            Ddesc.ld,
        };

        constexpr int threads = Kernel::WARP_CNT * WARP_SIZE;
        const int     blocks  = ceil_div(Sdesc.rows, CTA_M);

        convert_kernel<Kernel><<<blocks, threads, kSmemSize, stream>>>(param);
        return 0;
    };

    switch (Ddesc.pack) {
        case Pack::kHMMA_16816_A:
            return invoke(sm80_s16816gemm_f16_f16_nn::OperandA<T, CTA_M, CTA_K, CTA_M, 1, false>{});
        case Pack::kHMMA_16816_B:
            return invoke(sm80_s16816gemm_f16_f16_nn::OperandB<T, CTA_M, CTA_K, CTA_M, 1, false>{});
        default:
            fprintf(stderr, "Not implemented.\n");
    }

    return -1;
}

}  // namespace turbomind::gemm