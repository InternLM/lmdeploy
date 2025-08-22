
#include <cstddef>
#include <cuda.h>
#include <memory>
#include <optional>

#include "src/turbomind/core/core.h"

#include "src/turbomind/kernels/gemm/test/quantization.h"
#include "src/turbomind/kernels/gemm/test/reference.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

#include "src/turbomind/kernels/gpt_kernels.h"

namespace turbomind {

using std::vector;
using std::unique_ptr;

using DenseWeight = LlamaDenseWeight;

using namespace gemm;

struct Parameter {
    int input_dim;
    int output_dim;

    DataType data_type;
    DataType weight_type;
    DataType input_type;

    int group_size;

    int max_batch_size;

    int expert_num;
    int experts_per_token;
};

/// TODO: add a generic copy / casting for non-sub-byte Tensor
static Tensor CopyTransposed(const Tensor& src, Tensor out = {})
{
    if (out) {
        TM_CHECK(out.shapes(0, 1) == src.shapes(1, 0)) << src << " vs " << out;
        TM_CHECK_EQ(out.dtype(), src.dtype());
    }
    else {
        out = {{src.shape(1), src.shape(0)}, src.dtype(), src.device()};
    }

    auto invoke = [&](auto t) {
        using T = decltype(t);
        invokeTransposeAxis01(
            (T*)out.raw_data(), (T*)src.raw_data(), src.shape(0), src.shape(1), 1, core::Context::stream().handle());
    };

    const int bits = byte_size(src.dtype(), 8);
    if (bits == 8) {
        invoke(uint8_t{});
    }
    else if (bits == 16) {
        invoke(uint16_t{});
    }
    else if (bits == 32) {
        invoke(int{});
    }
    else {
        TM_CHECK(0) << "Not implemented. bits = " << bits;
    }

    return out;
}

struct Testbed_v3: Parameter {

    Testbed_v3(const Parameter& param): Parameter{param}, stream_{core::Context::stream().handle()}, linear_{stream_}
    {
        rng_.set_stream(stream_);
        ref_.set_stream(stream_);

        w_original_ = std::make_unique<DenseWeight>();
        w_quant_    = std::make_unique<DenseWeight>();
        w_dequant_  = std::make_unique<DenseWeight>();

        for (int i = 0; i < expert_num; ++i) {
            e_original_.push_back(std::make_unique<DenseWeight>());
            e_quant_.push_back(std::make_unique<DenseWeight>());
            e_dequant_.push_back(std::make_unique<DenseWeight>());
        }

        GenerateWeight();
        GenerateInput();
    }

    void GenerateInput()
    {
        x_original_ = Tensor{{max_batch_size, input_dim}, data_type, kDEVICE};
        rng_.NormalFloat(x_original_, 1., 1.);

        if (input_type == data_type) {
            x_quant_   = empty_like(x_original_);
            x_dequant_ = empty_like(x_original_);
            Copy(x_original_, x_quant_);
            Copy(x_original_, x_dequant_);
        }
        else if (input_type == kFloat8_e4m3) {
            QuantizeSymm(x_quant_, x_scale_, x_original_, stream_);
            DequantizeSymm(x_dequant_, x_quant_, x_scale_, stream_);
        }
        else {
            TM_CHECK(0) << "Not implemented for input type " << to_string(input_type);
        }
    }

    void GenerateWeight()
    {
        if (expert_num) {
            for (size_t i = 0; i < e_original_.size(); ++i) {
                GenerateWeight(*e_original_[i], *e_quant_[i], *e_dequant_[i]);
            }
        }
        else {
            GenerateWeight(*w_original_, *w_quant_, *w_dequant_);
        }
    }

    // - quantize weight
    // - dequantize weight
    void GenerateWeight(LlamaDenseWeight& original, LlamaDenseWeight& quant, LlamaDenseWeight& dequant)
    {
        original.emplace(input_dim, output_dim, data_type, false, data_type, group_size);
        rng_.NormalFloat(original.weight, 1.f, 1.f);

        quant.emplace(input_dim, output_dim, data_type, false, weight_type, group_size);
        dequant.emplace(input_dim, output_dim, data_type, false, data_type, group_size);

        /// Weights are allocated in MN-major, but some quantization requires K-major tensor

        if (weight_type == data_type) {
            Copy(original.weight, quant.weight);
            Copy(original.weight, dequant.weight);
        }
        else if (weight_type == kFloat8_e4m3) {
            QuantizeSymmBlock(quant.weight, quant.scales, original.weight, stream_);
            DequantizeSymmBlock(dequant.weight, quant.weight, quant.scales, stream_);
        }
        else if (weight_type == kUint4) {
            /// Weights are allocated in (M,N), quantization needs K-major tensor
            QuantizeGroupwise(quant.weight.t(),
                              quant.scales.t(),
                              quant.zeros.t(),
                              dequant.weight.t(),
                              original.weight.t(),
                              group_size);
        }
        else if (weight_type == kFloat4_e2m1) {
            QuantizeGroupwise(quant.weight.t(),  //
                              quant.scales.t(),
                              {},
                              dequant.weight.t(),
                              original.weight.t(),
                              group_size);
        }
        else {
            TM_CHECK(0);
        }

        original.prepare(expert_num > 0, 0);
        quant.prepare(expert_num > 0, 0);
        dequant.prepare(expert_num > 0, 0);
    }

    void GetReference()
    {
        if (expert_num) {
            GetReference(x_original_, e_original_, d_original_);
            GetReference(x_dequant_, e_dequant_, d_dequant_);
        }
        else {
            GetReference(x_original_, w_original_, d_original_);
            GetReference(x_dequant_, w_dequant_, d_dequant_);
        }
    }

    void GetReference(const Tensor& x, const unique_ptr<DenseWeight>& dense, Ref<Tensor> d_)
    {
        auto& d = d_.get();
        if (!d) {
            d = Tensor{{x.shape(0), dense->output_dim}, x.dtype(), x.device()};
        }
        /// TODO: refactor reference API
        const MatrixLayout desc_A{x.dtype(), kRowMajor, (int)x.shape(0), (int)x.shape(1), (int)x.stride(0)};  // m,k
        const MatrixLayout desc_D{d.dtype(), kRowMajor, (int)d.shape(0), (int)d.shape(1), (int)d.stride(0)};  // m,n
        ref_.gemm(x.raw_data(), desc_A, dense->weight.raw_data(), dense->k_desc, d.raw_data(), desc_D);
    }

    void GetReference(const Tensor& x, const vector<unique_ptr<DenseWeight>>& experts, Ref<Tensor> d)
    {
        TM_CHECK(0);
    }

    void Run()
    {
        if (expert_num) {
            TM_CHECK(0);
        }
        else {
            // std::cout << x_original_ << " " << w_quant_->weight << " " << d_quant_ << "\n";
            d_quant_ = linear_.Forward(x_original_, *w_quant_);
        }
    }

    void Run(const Tensor& x, const vector<unique_ptr<DenseWeight>>& experts) {}

    void Compare()
    {
        // clang-format off
        printf("%20s", ""); FC_Header();
        printf("%20s", "w_dequant v w_original"); FC_Print(FastCompare(w_dequant_->weight, w_original_->weight, stream_));
        printf("%20s", "quant   vs  dequant"); FC_Print(FastCompare(d_quant_, d_dequant_, stream_));
        printf("%20s", "quant   vs original"); FC_Print(FastCompare(d_quant_, d_original_, stream_));
        printf("%20s", "dequant vs original"); FC_Print(FastCompare(d_dequant_, d_original_, stream_));
        // clang-format on
    }

    cudaStream_t stream_;

    LlamaLinear linear_;

    // ! weights are non-movable
    unique_ptr<DenseWeight> w_original_;
    unique_ptr<DenseWeight> w_quant_;
    unique_ptr<DenseWeight> w_dequant_;

    Tensor x_original_;
    Tensor x_quant_, x_scale_;
    Tensor x_dequant_;

    Tensor d_original_;  // x_original * w_original
    Tensor d_quant_;     // x_original * w_quant, quant for X done by `Linear`
    Tensor d_dequant_;   // x_dequant  * w_dequant

    vector<unique_ptr<DenseWeight>> e_original_;
    vector<unique_ptr<DenseWeight>> e_quant_;
    vector<unique_ptr<DenseWeight>> e_dequant_;

    RNG       rng_;
    Reference ref_;
};

}  // namespace turbomind