// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/convert_v2.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/operand.h"
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

// MMA     : H_16816, H_1688, H_884, H_SIMT
// Operand : A, B, U, V
// Order   : row, col
// Dtype   : u16, u8, u4 (u6, u3)
// PackNum : 1, 2, 4

template<MMA_Tag MMA, Op_Tag Op, Order Ord, class Dtype_, int PackNum>
struct Config {
    static constexpr int CTA_M = 32;
    static constexpr int CTA_K = 32;

    static constexpr int BLOCK_SIZE = 32;

    using Operand =
        typename GetOperand<MMA, Op, Ord, false>::Operand::template type<half, CTA_M, CTA_K, CTA_M, 1, false>;

    using Stype = typename Operand::Dtype;
    using Dtype = Dtype_;

    using Kernel = ConvertOperand<CTA_M, CTA_K, PackNum, Operand, half, _Converter<half, Dtype>>;
};

template<class Config>
void Convert_v2_Impl(const void* S, const MatrixLayout& Sdesc, void* D, const MatrixLayout& Ddesc, cudaStream_t stream)
{
    using Kernel = typename Config::Kernel;
    using Stype  = typename Config::Stype;
    using Dtype  = typename Config::Dtype;

    constexpr int CTA_M = 32;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);

    if (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(convert_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    typename Kernel::Param param{Sdesc.rows, Sdesc.cols, (const Stype*)S, Sdesc.ld, (Dtype*)D, Ddesc.ld};

    constexpr int threads = Config::BLOCK_SIZE;
    const int     blocks  = ceil_div(Sdesc.rows, CTA_M);

    convert_kernel<Kernel><<<blocks, threads, kSmemSize, stream>>>(param);
}

int Convert(const void*         S,  //
            const MatrixLayout& _Sdesc,
            void*               D,
            const MatrixLayout& _Ddesc,
            cudaStream_t        stream)
{
    const Op_Tag op_tag = get_operand_tag(_Ddesc.pack);
    const bool   trans  = op_tag == OPERAND_B || op_tag == OPERAND_V;

    // (k, n) -> (n, k)
    const MatrixLayout Sdesc = trans ? transpose(_Sdesc) : _Sdesc;
    const MatrixLayout Ddesc = trans ? transpose(_Ddesc) : _Ddesc;

    auto invoke = [&](auto mma, auto operand, auto order, auto dtype, auto pack_num) -> bool {
        // Make args constexpr explictly, some compilers failed to see const-ness of the args
        constexpr MMA_Tag mma_tag      = mma;
        constexpr Op_Tag  op_tag       = operand;
        constexpr Order   order_tag    = order;
        constexpr int     pack_num_tag = pack_num;

        using Dtype  = typename get_dtype<dtype>::type;
        using Config = Config<mma_tag, op_tag, order_tag, Dtype, pack_num_tag>;

        Convert_v2_Impl<Config>(S, Sdesc, D, Ddesc, stream);

        return true;
    };

    auto dispatch_4 = [&](auto mma, auto operand, auto order, auto dtype) -> bool {
        switch (get_pack_num(Ddesc.pack)) {
            case 1:
                return invoke(mma, operand, order, dtype, constant<1>{});
            // case 2:
            //     return invoke(mma, operand, order, dtype, constant<2>{});
            // case 4:
            //     return invoke(mma, operand, order, dtype, constant<4>{});
            default:
                return false;
        }
    };

    auto dispatch_3 = [&](auto mma, auto operand, auto order) -> bool {
        if constexpr (GetOperand<mma, operand, order, false>::value) {  // is operand exist?
            /// TODO: add U8, U4
            switch (Ddesc.type) {
                case DataType::F16:
                    return dispatch_4(mma, operand, order, constant<DataType::F16>{});
                default:
                    return false;
            }
        }
        return false;
    };

    auto dispatch_2 = [&](auto mma, auto operand) -> bool {
        switch (Ddesc.order) {
            case Order::kRowMajor:
                return dispatch_3(mma, operand, constant<Order::kRowMajor>{});
            case Order::kColMajor:
                return dispatch_3(mma, operand, constant<Order::kColMajor>{});
        }
        return false;
    };

    auto dispatch_1 = [&](auto mma) -> bool {
        /// TODO: add U, V
        switch (get_operand_tag(Ddesc.pack)) {
            case OPERAND_A:
                return dispatch_2(mma, constant<OPERAND_A>{});
            case OPERAND_B:
                return dispatch_2(mma, constant<OPERAND_B>{});
            default:
                return false;
        }
    };

    auto dispatch = [&]() -> bool {
        /// TODO: add HMMA_1688, HMMA_884, HMMA_SIMT
        switch (get_mma_tag(Ddesc.pack)) {
            case HMMA_16816:
                return dispatch_1(constant<HMMA_16816>{});
            default:
                return false;
        }
    };

    // -1 on failure
    return dispatch() - 1;
}

}  // namespace turbomind::gemm