// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/arch/operand_simt.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert_v2.h"
#include "src/turbomind/kernels/gemm/format.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace {

constexpr bool is_AB(Op_Tag op)
{
    if (op == OPERAND_A || op == OPERAND_B) {
        return true;
    }
    else {
        return false;
    }
}

constexpr bool is_UV(Op_Tag op)
{
    return !is_AB(op);
}

template<class Dtype>
constexpr int unit_size(basic_type<Dtype>)
{
    return 1;
}

constexpr int unit_size(basic_type<uint8_t>)
{
    return 4;
}

constexpr int unit_size(basic_type<uint4_t>)
{
    return 8;
}

}  // namespace

// MMA     : H_16816, H_1688, H_884, H_SIMT
// Operand : A, B, U, V
// Order   : row, col
// Dtype   : u16, u8, u4 (u6, u3)
// PackNum : 1, 2, 4

template<class Operand, class Dtype_, int PackNum>
struct Config {
    static constexpr int CTA_M = 64;
    static constexpr int CTA_K = 32;

    static constexpr int BLOCK_SIZE = 32;

    using Stype = typename Operand::Dtype;
    using Dtype = Dtype_;

    using Kernel = ConvertOperand<CTA_M, CTA_K, PackNum, Operand, Dtype, Converter<Stype, Dtype>>;
};

template<class Config>
void Convert_v2_Impl(const void* S, const MatrixLayout& Sdesc, void* D, const MatrixLayout& Ddesc, cudaStream_t stream)
{
    using Kernel = typename Config::Kernel;
    using Stype  = typename Config::Stype;
    using Dtype  = typename Config::Dtype;

    constexpr int CTA_M = Config::CTA_M;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);

    if (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(convert_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    using PointerD = typename Kernel::PtrD;
    typename Kernel::Param param{Sdesc.rows, Sdesc.cols, (const Stype*)S, Sdesc.ld, PointerD{(Dtype*)D}, Ddesc.ld};

    constexpr int threads = Config::BLOCK_SIZE;
    const int     blocks  = ceil_div(Sdesc.rows, CTA_M);

    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // std::cout << __PRETTY_FUNCTION__ << "\nThreadMap:\n";
    // Print(typename Kernel::GmemIter::ThreadMap{});

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
    MatrixLayout Sdesc = trans ? transpose(_Sdesc) : _Sdesc;
    MatrixLayout Ddesc = trans ? transpose(_Ddesc) : _Ddesc;

    auto invoke = [&](auto mma, auto operand, auto order, auto stype, auto dtype, auto pack_num) -> bool {
        using Stype = typename decltype(stype)::type;
        using Dtype = typename decltype(dtype)::type;

        if constexpr (GetOperand<mma, operand, Stype, order, false>::value) {  // is operand exist?

            // Make args constexpr explicitly, some compilers failed to see const-ness of the args
            constexpr int pack_num_tag = pack_num;

            using Operand = typename GetOperand<mma, operand, Stype, order, false>::Operand;

            static constexpr int  kPackSize = Operand::SmemCopyAtom::Frag::size() * pack_num_tag;
            static constexpr bool kIsValid  = kPackSize % unit_size(type_c<Dtype>) == 0;

            if constexpr (kIsValid) {
                Convert_v2_Impl<Config<Operand, Dtype, pack_num_tag>>(S, Sdesc, D, Ddesc, stream);
                return true;
            }

            // std::cerr << __PRETTY_FUNCTION__ << "\n";
            // std::cerr << kPackSize << " " << unit_size(type_c<Dtype>) << "\n";
        }

        return false;
    };

    auto dispatch_4 = [&](auto mma, auto operand, auto order, auto stype, auto dtype) -> bool {
        switch (get_pack_num(Ddesc.pack)) {
            case 1:
                return invoke(mma, operand, order, stype, dtype, constant<1>{});
            case 2:
                return invoke(mma, operand, order, stype, dtype, constant<2>{});
            case 4:
                return invoke(mma, operand, order, stype, dtype, constant<4>{});
            default:
                return false;
        }
    };

    auto dispatch_3 = [&](auto mma, auto operand, auto order) -> bool {
        if constexpr (is_AB(operand)) {
            switch (Ddesc.type) {
                case DataType::F16:
                    return dispatch_4(mma, operand, order, type_c<uint16_t>, type_c<uint16_t>);
                case DataType::U8:
                    return dispatch_4(mma, operand, order, type_c<uint16_t>, type_c<uint8_t>);
                case DataType::U4:
                    return dispatch_4(mma, operand, order, type_c<uint16_t>, type_c<uint4_t>);
                default:
                    return false;
            }
        }
        else {  // UV: U16, U32
            switch (Ddesc.type) {
                case DataType::U32:
                    return dispatch_4(mma, operand, order, type_c<uint32_t>, type_c<uint32_t>);
                default:
                    return false;
            }
        }

        return false;
    };

    auto dispatch_2 = [&](auto mma, auto operand) -> bool {
        switch (Ddesc.order) {
            case Order::kRowMajor:
                return dispatch_3(mma, operand, constant<kRowMajor>{});
            case Order::kColMajor:
                return dispatch_3(mma, operand, constant<kColMajor>{});
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
            case OPERAND_U:
                return dispatch_2(mma, constant<OPERAND_U>{});
            case OPERAND_V:
                return dispatch_2(mma, constant<OPERAND_V>{});
            default:
                return false;
        }
    };

    auto dispatch = [&]() -> bool {
        /// TODO: add HMMA_1688, HMMA_884, HMMA_SIMT
        switch (get_mma_tag(Ddesc.pack)) {
            case HMMA_16816:
                return dispatch_1(constant<HMMA_16816>{});
            case HMMA_SIMT:
                return dispatch_1(constant<HMMA_SIMT>{});
            case HMMA_884:
                return dispatch_1(constant<HMMA_884>{});
            default:
                return false;
        }
    };

    // -1 on failure
    return dispatch() - 1;
}

std::tuple<Order, Pack, Order, Pack> get_weight_and_scales_layout(int sm, bool force_simt)
{
    if (force_simt) {
        return {kColMajor, HMMA_SIMT | OPERAND_B | 1, kRowMajor, HMMA_SIMT | OPERAND_V | 1};
    }
    if (sm >= 80) {
        return {kRowMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
    }
    else if (sm == 75) {
        return {kRowMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
    }
    else if (sm == 70) {
        return {kColMajor, HMMA_884 | OPERAND_B | 1, kRowMajor, HMMA_884 | OPERAND_V | 1};
    }
    else {
        std::cerr << "not implemented: sm_" << sm << std::endl;
        std::abort();
    }
    return {};
}

}  // namespace turbomind::gemm
