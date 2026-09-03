
#include <array>
#include <cstdlib>
#include <numeric>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"

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

int WeightPackEnv()
{
    static const int v = [] {
        const char* p = std::getenv("TM_GEMM_WEIGHT_PACK");
        if (!p) {
            return -1;
        }
        if (p[0] == '0' && p[1] == '\0') {
            return 0;
        }
        if (p[0] == '1' && p[1] == '\0') {
            return 1;
        }
        return -1;
    }();
    return v;
}

bool HasSm90MixedKernel()
{
    return TM_GEMM_HAS_SM90_MIXED;
}

namespace {

template<class Format>
inline constexpr bool is_sm90_mxfp4_fp8_format_v =
    std::is_same_v<Format, Sm90MxFp4Fp8FoldedFormat>
    || std::is_same_v<Format, Sm90MxFp4Fp8UnfoldedFormat>;

template<class Format>
struct Sm90PrepackedWeightConverter: LayoutConverter {
    Sm90PrepackedWeightConverter()
    {
        order        = Format::kConverterOrder;
        pack         = Format::kWeightPack;
        storage_bits = Format::kWeightBits;
    }

    int
    Convert(const void* S, const MatrixLayout& Sdesc, void* D, MatrixLayout& Ddesc, cudaStream_t stream) const override
    {
        TM_CHECK_NOTNULL(S);
        TM_CHECK_NOTNULL(D);
        TM_CHECK_EQ(Sdesc.type, kBfloat16);
        TM_CHECK_EQ(Sdesc.pack, 0U);
        TM_CHECK(Sdesc.order == Format::kConverterOrder);
        if constexpr (Format::kConverterOrder == kRowMajor) {
            TM_CHECK_EQ(Sdesc.ld, Sdesc.cols);
        }
        else {
            TM_CHECK_EQ(Sdesc.ld, Sdesc.rows);
        }
        constexpr int output_alignment = std::is_same_v<Format, Sm90U4Format>
                                             || is_sm90_mxfp4_fp8_format_v<Format> ?
                                             kSm90MixedFragmentN :
                                             kSm90MixedTileN;
        TM_CHECK_EQ(Sdesc.rows % output_alignment, 0);
        constexpr int input_alignment = is_sm90_mxfp4_fp8_format_v<Format> ?
                                            128 :
                                            std::lcm(kSm90MixedTileK, Format::kGroupSize);
        TM_CHECK_EQ(Sdesc.cols % input_alignment, 0);
        TM_CHECK_GE(Sdesc.cols, 128);

        if constexpr (std::is_same_v<Format, Sm90MxFp4Fp8FoldedFormat>) {
            PackSm90MxFp4Fp8FoldedWeight(
                static_cast<uint32_t*>(D), static_cast<const uint16_t*>(S), Sdesc.rows, Sdesc.cols, stream);
        }
        else if constexpr (std::is_same_v<Format, Sm90MxFp4Fp8UnfoldedFormat>) {
            PackSm90MxFp4Fp8UnfoldedWeight(
                static_cast<uint32_t*>(D), static_cast<const uint16_t*>(S), Sdesc.rows, Sdesc.cols, stream);
        }
        else if constexpr (std::is_same_v<Format, Sm90MxFp4Format>
                           || std::is_same_v<Format, Sm90NvFp4Format>) {
            PackSm90Fp4PrmtWeight(
                static_cast<uint32_t*>(D), static_cast<const uint16_t*>(S), Sdesc.rows, Sdesc.cols, stream);
        }
        else if constexpr (Format::kWeightBits == 4) {
            PackSm90U4Weight(
                static_cast<uint32_t*>(D), static_cast<const uint16_t*>(S), Sdesc.rows, Sdesc.cols, stream);
        }
        else {
            static_assert(std::is_same_v<Format, Sm90Fp8E4M3Format>);
            PackSm90Fp8E4M3Weight(
                static_cast<uint32_t*>(D), static_cast<const uint16_t*>(S), Sdesc.rows, Sdesc.cols, stream);
        }

        Ddesc      = Sdesc;
        Ddesc.type = data_type_v<typename Format::WeightType>;
        Ddesc.pack = pack;
        Ddesc.ld   = Sdesc.ld;
        return 0;
    }
};

template<class Format>
struct Sm90Prepacked4QParamConverter: LayoutConverter {
    Sm90Prepacked4QParamConverter()
    {
        order = kColMajor;
        pack  = Format::kQparamPack;
    }

    int
    Convert(const void* S, const MatrixLayout& Sdesc, void* D, MatrixLayout& Ddesc, cudaStream_t stream) const override
    {
        TM_CHECK_NOTNULL(S);
        TM_CHECK_NOTNULL(D);
        TM_CHECK_EQ(Sdesc.type, data_type_v<typename Format::QparamSourceType>);
        TM_CHECK_EQ(Sdesc.pack, 0U);
        TM_CHECK(Sdesc.order == kColMajor);
        TM_CHECK_EQ(Sdesc.ld, Sdesc.rows);
        constexpr int output_alignment = std::is_same_v<Format, Sm90U4Format>
                                             || is_sm90_mxfp4_fp8_format_v<Format> ?
                                             kSm90MixedFragmentN :
                                             kSm90MixedTileN;
        TM_CHECK_EQ(Sdesc.rows % output_alignment, 0);

        // The source is physically [K/group, N]. Pack each OUT64 fragment as
        // the 32 {lo, hi} qparam pairs consumed by the RS operand-A lanes.
        if constexpr (std::is_same_v<Format, Sm90U4Format>) {
            PackSm90U4QParams(static_cast<uint32_t*>(D),
                              static_cast<const uint32_t*>(S),
                              Sdesc.rows,
                              Sdesc.cols,
                              stream);
        }
        else if constexpr (std::is_same_v<Format, Sm90MxFp4Format>) {
            PackSm90MxFp4QParams(static_cast<uint8_t*>(D),
                                 static_cast<const uint8_t*>(S),
                                 Sdesc.rows,
                                 Sdesc.cols,
                                 stream);
        }
        else if constexpr (std::is_same_v<Format, Sm90MxFp4Fp8FoldedFormat>) {
            Sm90MxFp4Fp8FoldedPackStats* stats{};
            const char* stats_env = std::getenv("TM_GEMM_MXFP4_FOLD_STATS");
            const bool collect_stats = stats_env && stats_env[0] == '1' && stats_env[1] == '\0';
            TM_CUDA_CHECK(cudaMallocAsync(&stats, sizeof(*stats), stream));
            TM_CUDA_CHECK(cudaMemsetAsync(stats, 0, sizeof(*stats), stream));
            PackSm90MxFp4Fp8FoldedQParams(static_cast<uint8_t*>(D),
                                          static_cast<const uint8_t*>(S),
                                          Sdesc.rows,
                                          Sdesc.cols,
                                          stream,
                                          stats);
            Sm90MxFp4Fp8FoldedPackStats host{};
            TM_CUDA_CHECK(cudaMemcpyAsync(&host, stats, sizeof(host), cudaMemcpyDeviceToHost, stream));
            TM_CUDA_CHECK(cudaStreamSynchronize(stream));
            TM_CHECK_GT(host.total_records, 0ull);
            // This kernel family has one physical image and one arithmetic
            // schedule: every OUT64 x K128 record must share a base exponent.
            // Reject incompatible checkpoints before publishing descriptors.
            TM_CHECK_EQ(host.foldable_records, host.total_records);
            if (collect_stats) {
                TM_LOG_INFO("SM90 MXFP4 foldable records: {}/{} ({:.2f}%)",
                            host.foldable_records,
                            host.total_records,
                            host.total_records ? 100. * host.foldable_records / host.total_records : 0.);
            }
            TM_CUDA_CHECK(cudaFreeAsync(stats, stream));
        }
        else if constexpr (std::is_same_v<Format, Sm90MxFp4Fp8UnfoldedFormat>) {
            PackSm90MxFp4Fp8UnfoldedQParams(static_cast<uint8_t*>(D),
                                            static_cast<const uint8_t*>(S),
                                            Sdesc.rows,
                                            Sdesc.cols,
                                            stream);
        }
        else {
            static_assert(std::is_same_v<Format, Sm90NvFp4Format>);
            PackSm90Fp4QParams(static_cast<uint8_t*>(D),
                               static_cast<const uint8_t*>(S),
                               Sdesc.rows,
                               Sdesc.cols,
                               stream);
        }

        Ddesc      = Sdesc;
        Ddesc.type = data_type_v<typename Format::QparamType>;
        Ddesc.pack = pack;
        return 0;
    }
};

struct Sm90Fp8E4M3QParamConverter: LayoutConverter {
    Sm90Fp8E4M3QParamConverter()
    {
        order = kRowMajor;
        pack  = Sm90Fp8E4M3Format::kQparamPack;
    }

    int
    Convert(const void* S, const MatrixLayout& Sdesc, void* D, MatrixLayout& Ddesc, cudaStream_t stream) const override
    {
        TM_CHECK_NOTNULL(S);
        TM_CHECK_NOTNULL(D);
        TM_CHECK_EQ(Sdesc.type, kFloat);
        TM_CHECK_EQ(Sdesc.pack, 0U);
        TM_CHECK(Sdesc.order == kRowMajor);
        TM_CHECK_EQ(Sdesc.ld, Sdesc.cols);

        PackSm90Fp8E4M3Scales(
            static_cast<bfloat16_t*>(D), static_cast<const float*>(S), Sdesc.rows, Sdesc.cols, stream);

        Ddesc      = Sdesc;
        Ddesc.type = kBfloat16;
        Ddesc.pack = pack;
        Ddesc.ld   = Sdesc.cols * Sm90Fp8E4M3Format::kQparamValuesTile;
        return 0;
    }
};

}  // namespace

ConverterSet GetConverters(const ConverterRequest& request)
{
    const auto data_type   = request.data_type;
    const auto weight_type = request.weight_type;
    const auto input_type  = request.input_type;
    const auto grouped     = request.grouped;
    const auto sm          = request.sm;

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

    const int pack_env = WeightPackEnv();
    if (pack_env == 0) {
        return {};
    }

    const bool use_sm90_mxfp4_fp8 = HasSm90MixedKernel() && request.sm == 90
                                     && request.data_type == kBfloat16
                                     && request.input_type == kFloat8_e4m3
                                     && request.weight_type == kFloat4_e2m1
                                     && request.group_size == Sm90MxFp4Fp8UnfoldedFormat::kGroupSize
                                     && request.input_dim >= 256
                                     && request.input_dim % 128 == 0 && request.output_dim >= 64
                                     && request.output_dim % 64 == 0;
    if (use_sm90_mxfp4_fp8) {
        const bool fuse_silu = request.epilogue == Epilogue::kGatedSilu;
        if (request.epilogue != Epilogue::kNone && !fuse_silu) {
            return {};
        }
        if (fuse_silu && request.output_dim % 256 != 0) {
            return {};
        }
        static const Sm90PrepackedWeightConverter<Sm90MxFp4Fp8FoldedFormat>  weight_converter;
        static const Sm90Prepacked4QParamConverter<Sm90MxFp4Fp8FoldedFormat> qparam_converter;
        return {&weight_converter, &qparam_converter, QParamEncoding::kMxFp4Fp8Folded};
    }

    // Grouped experts are prepared one at a time before
    // MoeWeight::LinkLinearExperts builds the StridedPtr tables. The native
    // pack is therefore identical for dense and grouped weights.
    const bool use_sm90_u4 = HasSm90MixedKernel() && sm == 90 && data_type == kBfloat16 && input_type == kBfloat16
                             && weight_type == kUint4 && request.group_size == Sm90U4Format::kGroupSize
                             && request.input_dim >= 2 * kSm90MixedTileK
                             && request.input_dim % std::lcm(kSm90MixedTileK, Sm90U4Format::kGroupSize) == 0
                             && request.output_dim % kSm90MixedFragmentN == 0;
    if (use_sm90_u4) {
        static const Sm90PrepackedWeightConverter<Sm90U4Format>  weight_converter;
        static const Sm90Prepacked4QParamConverter<Sm90U4Format> qparam_converter;
        return {&weight_converter, &qparam_converter, QParamEncoding::kBf16ScaleEffZero};
    }

    const bool use_sm90_mxfp4 = HasSm90MixedKernel() && sm == 90 && data_type == kBfloat16 && input_type == kBfloat16
                                && weight_type == kFloat4_e2m1 && request.group_size == Sm90MxFp4Format::kGroupSize
                                && request.input_dim >= 2 * kSm90MixedTileK
                                && request.input_dim % std::lcm(kSm90MixedTileK, Sm90MxFp4Format::kGroupSize) == 0
                                && request.output_dim % kSm90MixedTileN == 0;
    if (use_sm90_mxfp4) {
        static const Sm90PrepackedWeightConverter<Sm90MxFp4Format>  weight_converter;
        static const Sm90Prepacked4QParamConverter<Sm90MxFp4Format> qparam_converter;
        return {&weight_converter, &qparam_converter, QParamEncoding::kMxFp4UnbiasedExponent};
    }

    const bool use_sm90_nvfp4 = HasSm90MixedKernel() && sm == 90 && data_type == kBfloat16 && input_type == kBfloat16
                                && weight_type == kFloat4_e2m1 && request.group_size == Sm90NvFp4Format::kGroupSize
                                && request.input_dim >= 2 * kSm90MixedTileK
                                && request.input_dim % std::lcm(kSm90MixedTileK, Sm90NvFp4Format::kGroupSize) == 0
                                && request.output_dim % kSm90MixedTileN == 0;
    if (use_sm90_nvfp4) {
        static const Sm90PrepackedWeightConverter<Sm90NvFp4Format>  weight_converter;
        static const Sm90Prepacked4QParamConverter<Sm90NvFp4Format> qparam_converter;
        return {&weight_converter, &qparam_converter, QParamEncoding::kNvFp4Scale};
    }

    const bool use_sm90_fp8_e4m3 = HasSm90MixedKernel() && sm == 90 && data_type == kBfloat16 && input_type == kBfloat16
                                   && weight_type == kFloat8_e4m3 && request.group_size == Sm90Fp8E4M3Format::kGroupSize
                                   && request.input_dim >= 2 * kSm90MixedTileK
                                   && request.input_dim % Sm90Fp8E4M3Format::kGroupSize == 0
                                   && request.output_dim % kSm90MixedTileN == 0;
    if (use_sm90_fp8_e4m3) {
        static const Sm90PrepackedWeightConverter<Sm90Fp8E4M3Format> weight_converter;
        static const Sm90Fp8E4M3QParamConverter                      qparam_converter;
        return {&weight_converter, &qparam_converter, QParamEncoding::kBf16BlockScale};
    }

    if (weight_type == kHalf || weight_type == kBfloat16) {
        constexpr Cvt<uint16_t, uint16_t> W;
        if (grouped) {
            if (pack_env != 1) {
                // SM10.x: CublasGroupedKernel expects standard (K,N)
                if (sm >= 100 && sm < 120)
                    return {};
                // SM90: plain B for native GMMA (LinearWeight prepare stores physical (N,K))
                if (sm >= 90 && sm < 100)
                    return {};
            }
            // clang-format off
            if (sm >= 80) return {W(sm8_, kRow, s16816h | B | _1), {}};
            if (sm == 75) return {W(sm75, kRow, s16816h | B | _1), {}};
            if (sm >= 70) return {W(sm70, kRow,   s884h | B | _1), {}};
            // clang-format on
        }
        else {
            return {};  //  trivial case: no quantization
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

    TM_LOG_FATAL("Invalid combination: {} {} {} {} {}", sm, data_type, weight_type, input_type, grouped);

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
    TM_CUDA_CHECK(cudaGetLastError());
    return ptr;
}

}  // namespace turbomind::gemm
