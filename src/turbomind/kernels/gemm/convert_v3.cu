
#include <array>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/arch/operand_simt.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h"

namespace turbomind::gemm {

template<class Arch, Order order_, MMA_Tag mma_tag, Op_Tag op_tag, int pack_num, class Stype, class Dtype>
struct LayoutConverterImpl: public LayoutConverter {

    LayoutConverterImpl(): LayoutConverter{}
    {
        this->order = order_;
        this->pack  = mma_tag | op_tag | pack_num;
    }

    int Convert(const void*         S,
                const MatrixLayout& Sdesc_,  // (m,k) / (n,k)
                void*               D,
                MatrixLayout&       Ddesc,  // (m,k) / (n,k)
                cudaStream_t        stream) const override
    {
        // TM_CHECK_EQ(Sdesc.pack, 0U) << "Source must be non-packed format";

        const bool trans = op_tag == OPERAND_B || op_tag == OPERAND_V;
        // (k, n) -> (n, k)
        MatrixLayout Sdesc = trans ? transpose(Sdesc_) : Sdesc_;
        // MatrixLayout Ddesc = trans ? transpose(Ddesc_) : Ddesc_;

        TM_CHECK_NOTNULL(S);
        TM_CHECK_NOTNULL(D);

        using Operand = typename GetOperand<mma_tag, op_tag, Stype, order_, false>::Operand;

        Convert_v2_Impl<Config<Operand, Dtype, pack_num>>(S, Sdesc, D, Ddesc, stream);

        constexpr Pack pack = mma_tag | op_tag | pack_num;

        // Update leading dimension
        Ddesc.ld = mk2cs<order_>(Packing_v2<pack, order_>::apply({Sdesc.rows, Sdesc.cols})).x;

        return 0;
    }
};

template<class Arch, Order order, uint32_t pack, class Stype, class Dtype>
static LayoutConverter* GetImpl()
{
    constexpr auto mma      = get_mma_tag(pack);
    constexpr auto operand  = get_operand_tag(pack);
    constexpr auto pack_num = get_pack_num(pack);

    static LayoutConverterImpl<Arch, order, mma, operand, pack_num, Stype, Dtype> impl{};

    return &impl;
}

template<class Stype, class Dtype>
struct Cvt {
    template<class Arch, Order order, Pack pack>
    LayoutConverter* operator()(Arch, constant<order>, constant<pack>) const
    {
        return GetImpl<Arch, order, pack, Stype, Dtype>();
    }
};

constexpr constant<(Pack)HMMA_16816> s16816h{};
constexpr constant<(Pack)HMMA_884>   s884h{};

template<auto a, auto b>
constexpr auto operator|(constant<a>, constant<b>)
{
    return constant<a | b>{};
}

std::array<const LayoutConverter*, 2> GetConverters(DataType data_type,
                                                    DataType weight_type,  //
                                                    DataType input_type,
                                                    bool     grouped,
                                                    int      sm)
{
    constexpr constant<kRowMajor> kRow{};
    constexpr constant<kColMajor> kCol{};

    constexpr constant<OPERAND_A> A{};
    constexpr constant<OPERAND_B> B{};
    constexpr constant<OPERAND_U> U{};
    constexpr constant<OPERAND_V> V{};

    constexpr constant<1> _1{};
    constexpr constant<2> _2{};

    constexpr Arch<80> sm8_{};
    constexpr Sm75     sm75{};
    constexpr Sm70     sm70{};

    if (weight_type == kHalf || weight_type == kBfloat16) {
        constexpr Cvt<uint16_t, uint16_t> W;
        if (grouped) {
            // clang-format off
            if (sm >= 80) return {W(sm8_, kRow, s16816h | B | _1), {}};
            if (sm == 75) return {W(sm75, kRow, s16816h | B | _1), {}};
            if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), {}};
            // clang-format on
        }
        else {
            return {};  //  trivial case: dense floating point
        }
    }

    // For performance reasons, u4 use different layouts for grouped/non-grouped GEMM
    if (weight_type == kUint4) {
        constexpr Cvt<uint16_t, uint4_t>  W;  // e4m3     weight
        constexpr Cvt<uint32_t, uint32_t> S;  // f16/bf16 scales&zeros
        if (grouped) {
            // clang-format off
            if (sm >= 80) return {W(sm8_, kRow, s16816h | B | _2), S(sm8_, kCol, s16816h | V | _1)};
            if (sm == 75) return {W(sm75, kRow, s16816h | B | _2), S(sm75, kCol, s16816h | V | _1)};
            if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), S(sm70, kCol,   s884h | V | _1)};
            // clang-format on
        }
        else {
            // clang-format off
            if (sm >= 80) return {W(sm8_, kCol, s16816h | B | _2), S(sm8_, kCol, s16816h | V | _1)};
            if (sm == 75) return {W(sm75, kCol, s16816h | B | _2), S(sm75, kCol, s16816h | V | _1)};
            if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), S(sm70, kCol,   s884h | V | _1)};
            // clang-format on
        }
    }

    if (weight_type == kFloat4_e2m1) {
        constexpr Cvt<uint16_t, uint4_t> W;  // e2m1  weight
        constexpr Cvt<uint8_t, uint8_t>  S;  // ue8m0 scales
        // clang-format off
        if (sm >= 80) return {W(sm8_, kCol, s16816h | A | _1), S(sm8_, kCol, s16816h | U | _1)};
        if (sm == 75) return {W(sm75, kCol, s16816h | A | _1), S(sm75, kCol, s16816h | U | _1)};
        if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), S(sm70, kCol,   s884h | V | _1)};
        // clang-format on
    }

    if (weight_type == kFloat8_e4m3) {
        constexpr Cvt<uint16_t, uint8_t>  W;  // e4m3     weight
        constexpr Cvt<uint16_t, uint16_t> S;  // f16/bf16 scales
        // clang-format off
        if (sm >= 80) return {W(sm8_, kCol, s16816h | A | _1), S(sm8_, kCol, s16816h | U | _1)};
        if (sm == 75) return {W(sm75, kCol, s16816h | A | _1), S(sm75, kCol, s16816h | U | _1)};
        if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), S(sm70, kCol,   s884h | V | _1)};
        // clang-format on
    }

    TM_CHECK(0) << "Invalid combination: " << sm << " " << data_type << " " << weight_type << " " << input_type << " "
                << grouped;

    return {};
}

namespace {

template<int N>
struct Param {
    StridedPtr  data[N];
    StridedPtr* ptr;
    int         n;
};

template<int N>
__global__ void fill_strided_ptrs(Param<N> param)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < param.n) {
        param.ptr[idx] = param.data[idx];
    }
}

}  // namespace

void* MakeStridedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream)
{
    constexpr int N = 64;
    Param<N>      param{};
    static_assert(sizeof(param) <= 4096);  // max parameter size for cuda11
    StridedPtr* ptr{};
    cudaMallocAsync(&ptr, sizeof(StridedPtr) * ptrs.size(), stream);
    param.ptr = ptr;
    for (int i = 0; i < (int)ptrs.size(); i += N) {
        const int n = std::min<int>(ptrs.size() - i, N);
        for (int j = 0; j < n; ++j) {
            auto& [p, s]  = ptrs[i + j];
            param.data[j] = StridedPtr{p, s};
        }
        param.n = n;
        fill_strided_ptrs<<<1, N, 0, stream>>>(param);
        param.ptr += N;
    }
    return ptr;
}

namespace {

template<int N>
__global__ void fill_blocked_ptrs(Array<void*, N> src, void** dst, int n)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

}  // namespace

void* MakeBlockedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream)
{
    constexpr int   N = 64;
    Array<void*, N> src{};
    static_assert(sizeof(src) <= 4096);  // max parameter size for cuda11
    void** dst{};
    cudaMallocAsync(&dst, sizeof(void*) * ptrs.size(), stream);
    for (int i = 0; i < (int)ptrs.size(); i += N) {
        const int n = std::min<int>(ptrs.size() - i, N);
        for (int j = 0; j < n; ++j) {
            auto& [p, s] = ptrs[i + j];
            src[j]       = p;
        }
        fill_blocked_ptrs<<<1, N, 0, stream>>>(src, dst, n);
        dst += n;
    }
    return dst - ptrs.size();
}

}  // namespace turbomind::gemm
