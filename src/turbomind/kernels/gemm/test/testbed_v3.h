
#include <memory>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
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
using Linear      = LlamaLinear;

using namespace gemm;

struct Parameter {
    int input_dim;
    int output_dim;

    DataType data_type;
    DataType weight_type;
    DataType input_type;

    int group_size;

    int max_batch_size;

    int  expert_num;
    int  experts_per_token;
    bool combine_experts;
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

        if (auto str = std::getenv("TM_GEMM_IMPORT")) {
            import_file_ = str;
            std::ifstream ifs(import_file_, std::ios::binary);
            auto          n = linear_.Import(ifs);
            std::cout << "Records imported: " << n << "\n";
        }
        if (auto str = std::getenv("TM_GEMM_TUNE"); str && import_file_.empty()) {
            tuning_ = true;
            std::cout << "Enable tuning\n";
        }
        if (auto str = std::getenv("TM_GEMM_EXPORT"); str && import_file_.empty()) {
            export_file_ = str;
        }

        cudaGetDeviceProperties(&prop_, 0);

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

        if (expert_num) {
            LinkExperts([&](int i) { return e_original_[i].get(); }, expert_num, *w_original_);
            LinkExperts([&](int i) { return e_quant_[i].get(); }, expert_num, *w_quant_);
            LinkExperts([&](int i) { return e_dequant_[i].get(); }, expert_num, *w_dequant_);
            Route();
        }
    }

    ~Testbed_v3()
    {
        if (!export_file_.empty()) {
            std::cerr << "export file: " << export_file_ << "\n";
            std::ofstream ofs(export_file_, std::ios::binary);
            if (ofs.is_open()) {
                auto n = linear_.Export(ofs);
                std::cout << "Records exported: " << n << "\n";
            }
        }
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

    void Route()
    {
        const int bsz = max_batch_size;

        std::mt19937 g{};

        /// TODO: Control the distribution
        auto expert_ids = SampleUniform(bsz, expert_num, experts_per_token, g);

        std::uniform_real_distribution<float> dist(1e-3, 1.f);

        Buffer_<float> tmp(experts_per_token, kCPU);
        Buffer_<float> scales(bsz * experts_per_token, kCPU);

        for (int i = 0; i < bsz; ++i) {
            float sum{};
            for (auto& x : tmp) {
                x = dist(g);
                sum += x;
            }
            for (int e = 0; e < experts_per_token; ++e) {
                scales[e * bsz + i] = tmp[e] / sum;
            }
        }

        vector<int>         count(expert_num);
        vector<vector<int>> f2i(expert_num);
        for (int i = 0; i < (int)expert_ids.size(); ++i) {
            ++count[expert_ids[i]];
            f2i[expert_ids[i]].push_back(i);
        }

        Buffer_<int> offsets(expert_num + 1, kCPU);
        offsets[0] = 0;
        for (int i = 0; i < expert_num; ++i) {
            offsets[i + 1] = offsets[i] + count[i];
        }

        for (const auto& x : count) {
            std::cout << x << " ";
        }
        std::cout << "\n";

        Buffer_<int> f2n(expert_ids.size(), kCPU);
        Buffer_<int> en2f(expert_ids.size(), kCPU);
        for (int e = 0, i = 0; e < expert_num; ++e) {
            for (auto x : f2i[e]) {
                f2n[i]   = x / experts_per_token;
                int en   = x % experts_per_token * bsz + x / experts_per_token;
                en2f[en] = i;
                ++i;
            }
        }

        f2n_ = {f2n.size(), kDEVICE};
        Copy(f2n, f2n_);

        en2f_ = {en2f.size(), kDEVICE};
        Copy(en2f, en2f_);

        scales_ = {scales.size(), kDEVICE};
        Copy(scales, scales_);

        offsets_ = {offsets.size(), kDEVICE};
        Copy(offsets, offsets_);
        h_offsets_ = offsets;

        core::Context::stream().Sync();
    }

    void GenerateWeight()
    {
        if (expert_num) {
            for (int i = 0; i < expert_num; ++i) {
                GenerateWeight(*e_original_[i], *e_quant_[i], *e_dequant_[i]);
            }
        }
        else {
            GenerateWeight(*w_original_, *w_quant_, *w_dequant_);
        }
    }

    // - quantize weight
    // - dequantize weight
    void GenerateWeight(DenseWeight& original, DenseWeight& quant, DenseWeight& dequant)
    {
        original.emplace(input_dim, output_dim, data_type, false, data_type, group_size);
        rng_.NormalFloat(original.weight, 1., .1);

        quant.emplace(input_dim, output_dim, data_type, false, weight_type, group_size);
        dequant.emplace(input_dim, output_dim, data_type, false, data_type, group_size);

        Buffer_<unsigned> rbits;
        // rbits = {original.weight.size(), kDEVICE};
        // rng_.RandomBytes(Tensor{rbits});

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
                              {},
                              group_size);
        }
        else if (weight_type == kFloat4_e2m1) {
            QuantizeGroupwise(quant.weight.t(),  //
                              quant.scales.t(),
                              {},
                              dequant.weight.t(),
                              original.weight.t(),
                              rbits,
                              group_size);
        }
        else {
            TM_CHECK(0);
        }

        original.prepare(0);
        quant.prepare(expert_num > 0);
        dequant.prepare(0);
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

    void GetReference(const Tensor& x, const vector<unique_ptr<DenseWeight>>& experts, Ref<Tensor> d_)
    {
        Tensor xe{{x.shape(0) * experts_per_token, input_dim}, data_type, kDEVICE};
        Tensor de{{x.shape(0) * experts_per_token, output_dim}, data_type, kDEVICE};

        invokeMoeDispatch(xe, x, f2n_.data(), experts_per_token, stream_);

        for (int i = 0; i < expert_num; ++i) {
            const int base = h_offsets_[i], size = h_offsets_[i + 1] - base;
            GetReference(xe.slice(base, size), experts[i], de.slice(base, size));
        }

        auto& d = d_.get();
        if (combine_experts) {
            d = Tensor{{x.shape(0), output_dim}, data_type, kDEVICE};
            invokeMoeCombine(d,  //
                             de,
                             {},
                             scales_.data(),
                             en2f_.data(),
                             nullptr,
                             nullptr,
                             experts_per_token,
                             1.,
                             0.,
                             stream_);
        }
        else {
            d = de;
        }
    }

    void Run()
    {
        if (tuning_) {
            linear_.set_measure(true);
        }
        if (expert_num) {
            auto de = linear_.Forward(x_original_, *w_quant_, f2n_, offsets_);
            if (combine_experts) {
                d_quant_ = Tensor{{x_original_.shape(0), output_dim}, data_type, kDEVICE};
                invokeMoeCombine(d_quant_,
                                 de,
                                 {},
                                 scales_.data(),
                                 en2f_.data(),
                                 nullptr,
                                 nullptr,
                                 experts_per_token,
                                 1.,
                                 0.,
                                 stream_);
            }
            else {
                d_quant_ = de;
            }
        }
        else {
            d_quant_ = linear_.Forward(x_original_, *w_quant_);
        }
        if (tuning_) {
            linear_.set_measure(false);
        }
    }

    void Run(const Tensor& x, const vector<unique_ptr<DenseWeight>>& experts) {}

    void Compare()
    {
        // Buffer_<float> h(16 * 16, kCPU);
        // Buffer_<float> x(linear_.buf, 16 * 16, kDEVICE);
        // Copy(x, h);

        // auto y = empty_like(w_dequant_->weight, kCPU);
        // Copy(w_dequant_->weight, y);

        // clang-format off
        printf("%20s", ""); FC_Header();
        if (!expert_num) {
            printf("%20s", "w_dequant v w_origi"); FC_Print(FastCompare(w_dequant_->weight, w_original_->weight, stream_));
        }
        printf("%20s", "quant   vs  dequant"); FC_Print(FastCompare(d_quant_, d_dequant_, stream_));
        printf("%20s", "quant   vs original"); FC_Print(FastCompare(d_quant_, d_original_, stream_));
        printf("%20s", "dequant vs original"); FC_Print(FastCompare(d_dequant_, d_original_, stream_));
        // clang-format on

        // for (int m = 0; m < 16; ++m) {
        //     for (int k = 0; k < 16; ++k) {
        //         printf("%5.1f", h[m * 16 + k]);
        //     }
        //     printf("\n");
        // }

        // printf("\n");

        // for (int m = 0; m < 16; ++m) {
        //     for (int k = 0; k < 16; ++k) {
        //         printf("%5.1f", (float)y.data<bfloat16_t>()[k * output_dim + m]);
        //     }
        //     printf("\n");
        // }
    }

    cudaStream_t stream_;

    cudaDeviceProp prop_;

    Linear linear_;

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

    Buffer_<int> f2n_;
    Buffer_<int> en2f_;

    Buffer_<int>   offsets_;
    Buffer_<float> scales_;

    Buffer_<int> h_offsets_;

    bool tuning_{};

    std::string import_file_;
    std::string export_file_;

    RNG       rng_;
    Reference ref_;
};

}  // namespace turbomind
