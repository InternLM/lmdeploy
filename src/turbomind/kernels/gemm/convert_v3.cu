#include <cstdlib>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind::gemm {

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

void ApplyWeightBridge(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    if (linear.scales && bridge.convert_scales != kNull) {
        EnsureFloatDtype(linear.scales, bridge.convert_scales);
        TM_CHECK_EQ(linear.scales.dtype(), bridge.convert_scales);
    }
    if (linear.zeros && bridge.convert_zeros != kNull) {
        EnsureFloatDtype(linear.zeros, bridge.convert_zeros);
        TM_CHECK_EQ(linear.zeros.dtype(), bridge.convert_zeros);
    }
    if (linear.scales && (bridge.replicate_scales.x != 1 || bridge.replicate_scales.y != 1)) {
        linear.scales = ReplicateQParams(linear.scales, bridge.replicate_scales, stream);
    }
    if (linear.zeros && (bridge.replicate_scales.x != 1 || bridge.replicate_scales.y != 1)) {
        linear.zeros = ReplicateQParams(linear.zeros, bridge.replicate_scales, stream);
    }
}

void PackWeight(LinearWeight& linear, const LayoutConverter& convert, cudaStream_t stream)
{
    const DataType    source_weight_type = linear.weight_format.dtype;
    const int         bits               = byte_size(source_weight_type, 8);
    Tensor_<uint16_t> tmp{{linear.input_dim, linear.output_dim}, kDEVICE};
    if (bits == 4) {
        extend_to_u16(tmp.data(), (const uint4_t*)linear.weight.raw_data(), tmp.size(), stream);
    }
    else if (bits == 8) {
        extend_to_u16(tmp.data(), (const uint8_t*)linear.weight.raw_data(), tmp.size(), stream);
    }
    else {
        TM_CHECK_EQ(bits, 16);
        TM_CUDA_CHECK(cudaMemcpyAsync(
            tmp.raw_data(), linear.weight.raw_data(), linear.weight.byte_size(), cudaMemcpyDefault, stream));
    }

    const Order order_w = convert.order;
    if (order_w == kRowMajor) {
        Tensor_<uint16_t> trans{{linear.output_dim, linear.input_dim}, kDEVICE};
        invokeTransposeAxis01(trans.data(), tmp.data(), linear.input_dim, linear.output_dim, 1, stream);
        tmp = std::move(trans);
    }
    MatrixLayout w_desc{linear.data_type,
                        order_w,
                        linear.output_dim,
                        linear.input_dim,
                        order_w == kRowMajor ? linear.input_dim : linear.output_dim};
    const bool   is_weight_a = get_operand_tag(convert.pack) == OPERAND_A;
    if (!is_weight_a) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }
    MatrixLayout kd = w_desc;
    kd.type         = source_weight_type;
    kd.pack         = convert.pack;

    TM_CUDA_CHECK(cudaMemsetAsync(linear.weight.raw_data(), 0, linear.weight.byte_size(), stream));
    TM_CHECK_EQ(convert.Convert(tmp.data(), w_desc, linear.weight.raw_data(), kd, stream), 0);
    kd.type = source_weight_type;
    if (is_weight_a) {
        kd = transpose(kd);
    }
    linear.k_desc = kd;
}

void PackWeight(LinearWeight& linear,
                Pack          pack,
                void (*convert)(uint32_t*, const uint16_t*, int, int, cudaStream_t),
                cudaStream_t stream)
{
    const DataType source_weight_type = linear.weight_format.dtype;
    const int      bits               = byte_size(source_weight_type, 8);
    TM_CHECK(bits == 4 || bits == 8);

    Tensor_<uint16_t> tmp{{linear.input_dim, linear.output_dim}, kDEVICE};
    if (bits == 4) {
        extend_to_u16(tmp.data(), (const uint4_t*)linear.weight.raw_data(), tmp.size(), stream);
    }
    else {
        extend_to_u16(tmp.data(), (const uint8_t*)linear.weight.raw_data(), tmp.size(), stream);
    }

    Tensor_<uint16_t> trans{{linear.output_dim, linear.input_dim}, kDEVICE};
    invokeTransposeAxis01(trans.data(), tmp.data(), linear.input_dim, linear.output_dim, 1, stream);
    TM_CUDA_CHECK(cudaMemsetAsync(linear.weight.raw_data(), 0, linear.weight.byte_size(), stream));
    convert(
        static_cast<uint32_t*>(linear.weight.raw_data()), trans.data(), linear.output_dim, linear.input_dim, stream);

    linear.k_desc =
        MatrixLayout{source_weight_type, kColMajor, linear.input_dim, linear.output_dim, linear.input_dim, pack};
}

void PackQParams(LinearWeight& linear, const LayoutConverter& convert, QuantDesc quant, cudaStream_t stream)
{
    TM_CHECK(linear.scales);
    const DataType source_weight_type = linear.weight_format.dtype;
    const bool     is_a               = get_operand_tag(convert.pack) == OPERAND_U;
    Tensor         tmp_q;
    DataType       scale_type{};

    if (linear.zeros) {
        TM_CHECK_EQ(linear.scales.dtype(), kHalf);
        TM_CHECK_EQ(linear.zeros.dtype(), kHalf);
        tmp_q = Tensor{{linear.scales.size(), 2}, kHalf, kDEVICE};
        fuse_scales_and_zeros(
            tmp_q.data<half>(), linear.scales.data<half>(), linear.zeros.data<half>(), linear.scales.size(), stream);
        scale_type    = kUint32;
        linear.zeros  = {};
        linear.scales = empty_like(tmp_q);
    }
    else {
        tmp_q = empty_like(linear.scales);
        TM_CUDA_CHECK(cudaMemcpyAsync(
            tmp_q.raw_data(), linear.scales.raw_data(), linear.scales.byte_size(), cudaMemcpyDefault, stream));
        scale_type = source_weight_type == kFloat8_e4m3 ? kUint16 : kUint8;
    }

    if (linear.data_type == kHalf && source_weight_type == kFloat4_e2m1) {
        AdjustUe8m0ScaleForHalf(tmp_q.data<uint8_t>(), tmp_q.size(), stream);
    }

    MatrixLayout s_desc{
        scale_type, convert.order, linear.output_dim, linear.input_dim / quant.group_size, linear.output_dim};
    if (!is_a) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }
    MatrixLayout qd = s_desc;
    qd.pack         = convert.pack;
    TM_CHECK_EQ(convert.Convert(tmp_q.raw_data(), s_desc, linear.scales.raw_data(), qd, stream), 0);
    linear.q_desc = is_a ? transpose(qd) : qd;
}

void PackQParams(LinearWeight& linear,
                 QuantDesc     quant,
                 Pack          pack,
                 void (*convert)(uint8_t*, const uint8_t*, int, int, cudaStream_t),
                 cudaStream_t stream)
{
    TM_CHECK(linear.scales);
    TM_CHECK_EQ(byte_size(linear.scales.dtype()), 1);
    Tensor tmp_q = empty_like(linear.scales);
    TM_CUDA_CHECK(cudaMemcpyAsync(
        tmp_q.raw_data(), linear.scales.raw_data(), linear.scales.byte_size(), cudaMemcpyDefault, stream));
    Tensor packed_q{{linear.scales.size()}, linear.scales.dtype(), kDEVICE};
    convert(static_cast<uint8_t*>(packed_q.raw_data()),
            static_cast<const uint8_t*>(tmp_q.raw_data()),
            linear.output_dim,
            linear.input_dim / quant.group_size,
            stream);
    linear.scales = std::move(packed_q);
    linear.q_desc = transpose(MatrixLayout{
        kUint8, kColMajor, linear.output_dim, linear.input_dim / quant.group_size, linear.output_dim, pack});
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
    static_assert(sizeof(param) <= 4096);
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
