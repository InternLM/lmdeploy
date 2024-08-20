// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/test/quantization.h"
#include "src/turbomind/kernels/gemm/test/reference.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <thrust/universal_vector.h>
#include <type_traits>

namespace turbomind::gemm {

using thrust::universal_vector;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "*** Check failed: (%s) @ %s:%d\n", #cond, __FILE__, __LINE__);                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

template<class Ta,
         class Tb,
         class Tc,
         int   batch_dim,
         Order order_a,
         Order order_b,
         Order order_c,
         Pack  pack_a,
         Pack  pack_b,
         Pack  pack_u = 0,
         Pack  pack_v = 0>
class Testbed {
public:
    static constexpr int kBatchDim = batch_dim;

    Testbed(): dispatch_policy_{DispatchPolicy::kDefault} {}

    Testbed(DispatchPolicy dispatch_policy, std::string cache_path):
        dispatch_policy_{dispatch_policy}, cache_path_{cache_path}
    {
        if (dispatch_policy & DispatchPolicy::kReuse) {
            std::ifstream ifs(cache_path);
            if (ifs.is_open()) {
                gemm_.Import(ifs);
            }
            else {
                std::cerr << "failed to import dispatch cache from \"" << cache_path << "\"" << std::endl;
            }
        }
    }

    ~Testbed()
    {
        if (dispatch_policy_ & DispatchPolicy::kMeasure) {
            std::ofstream ofs(cache_path_);
            if (ofs.is_open()) {
                gemm_.Export(ofs);
            }
            else {
                std::cerr << "failed to export dispatch cache to \"" << cache_path_ << "\"" << std::endl;
            }
        }
    }

    void Initialize(int m, int n, int k, int g, cudaStream_t stream)
    {
        rng_.set_stream(stream);
        reference_.set_stream(stream);
        stream_ = stream;

        m_ = m;
        n_ = n;
        k_ = k;

        a_.resize(m * k);
        b_.resize(n * k);
        c_.resize(m * n);

        a_desc_ = MatrixLayout{get_data_type_v<Tc>, order_a, m, k, mk2cs<order_a>(m, k).x};
        b_desc_ = MatrixLayout{get_data_type_v<Tc>, order_b, k, n, _kn2cs<order_b>(k, n).x};
        c_desc_ = MatrixLayout{get_data_type_v<Tc>, order_c, m, n, mk2cs<order_c>(m, n).x};

        c_f_.resize(c_.size());
        c_ref_.resize(c_.size());

        // a_q_.resize(a_.size());
        // b_q_.resize(b_.size());

        // u_.resize(a_.size());
        // v_.resize(b_.size());

        // a_f_.resize(a_.size());
        // b_f_.resize(b_.size());

        /// TODO: Revise packed format
        a_pack_.resize(a_.size() / kVecSize);
        b_pack_.resize(b_.size() / kVecSize);

        barriers_.resize(Gemm::kBarriersSize);
        partials_.resize(Gemm::kPartialsSize);

        rng_.GenerateUniform(a_.data().get(), a_.size(), 1, -.5f);
        rng_.GenerateUniform(b_.data().get(), b_.size(), 1, -.5f);

        for (int i = 0; i < n; ++i) {
            // for (int j = 0; j < k; ++j) {
            //     b_[i * k + j] = i * k + j;
            // }
            // for (int j = 0; j < k; j += 2) {
            //     b_[i * k + j]     = i;
            //     b_[i * k + j + 1] = j;
            // }
        }

        a_f_ = a_;
        b_f_ = b_;

        a_pack_desc_ = a_desc_;
        b_pack_desc_ = b_desc_;
        u_pack_desc_ = {};
        v_pack_desc_ = {};

        constexpr bool is_quant_a = !std::is_same_v<Ta, Tc>;
        constexpr bool is_quant_b = !std::is_same_v<Tb, Tc>;

        if constexpr (is_quant_a) {
            static_assert(pack_a && pack_u);
            Quantize<Ta>(a_, m, k, order_a, g, a_f_, a_q_, u_, stream);
            u_pack_desc_ = u_desc_ = {DataType::U32, kColMajor, m, ceil_div(k, g), m};
            u_pack_desc_.pack      = pack_u;
            u_pack_.resize(u_.size());
            CHECK(!Convert(u_.data().get(), u_desc_, u_pack_.data().get(), u_pack_desc_, stream_));
            quant_a_ = {QuantType::kDefault, g};

            // cudaDeviceSynchronize();

            // for (int i = 0; i < u_pack_.size(); ++i) {
            //     std::cout << (float)u_pack_[i] << " ";
            // }
            // std::cout << "\n";
        }

        // b (k, n) -> v is always row major
        if constexpr (is_quant_b) {
            static_assert(pack_b && pack_v);
            constexpr Order _order_b = transpose(order_b);
            Quantize<Tb>(b_, n, k, _order_b, g, b_f_, b_q_, v_, stream);
            v_pack_desc_ = v_desc_ = {DataType::U32, kRowMajor, ceil_div(k, g), n, n};
            v_pack_desc_.pack      = pack_v;
            v_pack_.resize(v_.size());
            CHECK(!Convert(v_.data().get(), v_desc_, v_pack_.data().get(), v_pack_desc_, stream_));
            quant_b_ = {QuantType::kDefault, g};

            // cudaDeviceSynchronize();

            // for (int i = 0; i < v_pack_.size(); ++i) {
            //     std::cout << (float)v_pack_[i] << " ";
            // }
            // std::cout << "\n";
        }

        if constexpr (pack_a) {
            a_pack_desc_.type = get_data_type_v<Ta>;
            a_pack_desc_.pack = pack_a;
            const auto a_data = is_quant_a ? (void*)a_q_.data().get() : (void*)a_.data().get();
            CHECK(!Convert(a_data, a_desc_, a_pack_.data().get(), a_pack_desc_, stream_));
        }
        else {
            cudaMemcpyAsync(
                (Ta*)a_pack_.data().get(), a_.data().get(), sizeof(Ta) * a_.size(), cudaMemcpyDefault, stream);
        }

        if constexpr (pack_b) {
            b_pack_desc_.type = get_data_type_v<Tb>;
            b_pack_desc_.pack = pack_b;
            const auto b_data = is_quant_b ? (void*)b_q_.data().get() : (void*)b_.data().get();
            CHECK(!Convert(b_data, b_desc_, b_pack_.data().get(), b_pack_desc_, stream_));

            // {
            //     cudaDeviceSynchronize();
            //     for (int i = 0; i < n; ++i) {
            //         for (int j = 0; j < k; j += 2) {
            //             // int index = (int)((Tb*)b_pack_.data().get())[i * k + j];
            //             // int row   = index / k;
            //             // int col   = index % k;
            //             int row = (int)((Tb*)b_pack_.data().get())[i * k + j];
            //             int col = (int)((Tb*)b_pack_.data().get())[i * k + j + 1];
            //             printf("(%2d,%2d) ", row, col);
            //         }
            //         printf("\n");
            //     }
            // }
        }
        else {
            cudaMemcpyAsync(
                (Tb*)b_pack_.data().get(), b_.data().get(), sizeof(Tb) * b_.size(), cudaMemcpyDefault, stream);
        }
    }

    void Run(void* ctx = {})
    {
        const Operation operation{
            dispatch_policy_,
            Epilogue::kNone,
            quant_a_,
            quant_b_,
            kBatchDim,
            ctx,
        };

        const Workspace workspace{barriers_.data().get(), barriers_.size(), partials_.data().get(), partials_.size()};

        auto status = gemm_.Run(operation,
                                1.f,
                                a_pack_.data().get(),
                                a_pack_desc_,
                                u_pack_.data().get(),
                                u_pack_desc_,
                                b_pack_.data().get(),
                                b_pack_desc_,
                                v_pack_.data().get(),
                                v_pack_desc_,
                                0.f,
                                c_.data().get(),
                                c_desc_,
                                c_.data().get(),
                                c_desc_,
                                workspace,
                                stream_);

        if (!ctx && status) {
            std::cerr << "Run failed, code =" << status << "\n";
            std::abort();
        }
    }

    void RunCublas()
    {
        reference_.gemm(a_f_.data().get(),  //
                        a_desc_,
                        b_f_.data().get(),
                        b_desc_,
                        c_f_.data().get(),
                        c_desc_);
    }

    void CompareB()
    {
        cudaDeviceSynchronize();
        Compare(b_f_.data().get(), b_.data().get(), k_, k_, n_);
    }

    void CompareC()
    {
        for (int i = 0; i < 10; ++i) {
            reference_.gemm(a_f_.data().get(),  //
                            a_desc_,
                            b_f_.data().get(),
                            b_desc_,
                            c_ref_.data().get(),
                            c_desc_);
        }

        // c_f_.resize(m_ * n_);
        // computeRefCublas(c_f_.data().get(), a_.data().get(), b_f_.data().get(), m_, n_, k_, stream_);
        // RunCublas();

        cudaDeviceSynchronize();

        // Compare(c_f_.data().get(), c_ref_.data().get(), n_, n_, m_, 0);

        // Compare(c_.data().get(), c_f_.data().get(), n_, n_, m_, 0);

        int dims = m_, bsz = n_;
        if (order_c == kRowMajor) {
            std::swap(dims, bsz);
        }
        Compare(c_.data().get(), c_ref_.data().get(), dims, dims, bsz, 0);
    }

    void Check()
    {
        reference_.gemm(a_f_.data().get(),  //
                        a_desc_,
                        b_f_.data().get(),
                        b_desc_,
                        c_ref_.data().get(),
                        c_desc_);

        std::vector<std::function<LaunchSpec()>> cases;
        Run(&cases);

        int dims = m_, bsz = n_;
        if (order_c == kRowMajor) {
            std::swap(dims, bsz);
        }

        max_vals_.resize(7);

        auto infnan  = [](float x) { return std::isinf(x) || std::isnan(x); };
        auto greater = [](auto& a, auto& b) {
            // skip abs(src) & abs(ref)
            for (int i = 2; i < (int)b.size(); ++i) {
                if (a[i] > b[i]) {
                    return true;
                }
            }
            return false;
        };

        for (const auto& c : cases) {
            const auto spec = c();
            auto       diff = FastCompare(c_.data().get(), c_ref_.data().get(), dims, bsz, stream_);
            if (greater(diff, max_vals_) || std::any_of(diff.begin(), diff.end(), infnan)) {
                std::cout << spec.kernel->name() << " " << spec.splits << " " << spec.swizzle      //
                          << " " << diff[0] << " " << diff[1] << " " << diff[2] << " " << diff[3]  //
                          << " " << diff[4] << " " << diff[5] << " " << diff[6] << "\n";
                for (int i = 0; i < (int)max_vals_.size(); ++i) {
                    max_vals_[i] = std::max(max_vals_[i], diff[i]);
                }
            }
        }
    }

    int64_t global_memory_reads()
    {
        return get_size(a_pack_desc_) + get_size(b_pack_desc_) + get_size(u_pack_desc_) + get_size(v_pack_desc_);
    }

    int64_t ref_global_memory_reads()
    {
        return get_size(a_desc_) + get_size(b_desc_);
    }

private:
    int m_{};
    int n_{};
    int k_{};
    int g_{};

    universal_vector<Tc> a_;      // A in fp
    universal_vector<Tc> b_;      // B in fp
    universal_vector<Tc> c_ref_;  // reference C
    universal_vector<Tc> c_;      // buffer for C

    // shared with `*_f_` variants
    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;

    universal_vector<uint16_t> a_q_;  // quantized a
    universal_vector<uint16_t> b_q_;  // quantized B
    universal_vector<Tc>       u_;    // quant param of `a_q_`
    universal_vector<Tc>       v_;    // quant param of `b_q_`

    // descs for converting to packed format
    MatrixLayout a_q_desc_;
    MatrixLayout b_q_desc_;
    MatrixLayout u_desc_;
    MatrixLayout v_desc_;

    universal_vector<Tc> a_f_;  // dequant `a_q_` back to fp
    universal_vector<Tc> b_f_;  // dequant `b_q_` back to fp
    universal_vector<Tc> c_f_;  // ref C computed by `b_f_`

    static constexpr int kVecSize = 8;

    universal_vector<Array<Ta, kVecSize>> a_pack_;  // packed A
    universal_vector<Array<Tb, kVecSize>> b_pack_;  // packed B

    universal_vector<Tc> u_pack_;  // packed U
    universal_vector<Tc> v_pack_;  // packed V

    MatrixLayout a_pack_desc_;
    MatrixLayout b_pack_desc_;
    MatrixLayout u_pack_desc_;
    MatrixLayout v_pack_desc_;

    QuantDesc quant_a_{};
    QuantDesc quant_b_{};

    universal_vector<char> barriers_;
    universal_vector<char> partials_;

    cudaStream_t stream_;

    RNG rng_;

    Gemm           gemm_;
    Reference      reference_;
    DispatchPolicy dispatch_policy_;
    std::string    cache_path_;

    std::vector<float> max_vals_;
};

template<class T>
T& gTestbed()
{
    static auto policy = [&] {
        const auto str    = std::getenv("TM_GEMM_TEST_POLICY");
        auto       policy = turbomind::gemm::DispatchPolicy::kDefault;
        using namespace turbomind::gemm;
        if (str) {
            using namespace std::string_view_literals;
            if (str == "measure"sv) {
                policy = DispatchPolicy::kMeasure;
            }
            else if (str == "reuse"sv) {
                policy = DispatchPolicy::kReuse;
            }
            else if (str == "append"sv) {
                policy = DispatchPolicy::kAppend;
            }
            else {
                std::cerr << "unrecognized policy: " << std::quoted(str) << ", default policy will be used.\n";
            }
        }
        return policy;
    }();

    static T inst{policy, "tm_cache"};
    return inst;
}

inline decltype(auto) get_test()
{
    if constexpr (0) {
        // native
        return gTestbed<gemm::Testbed<half, half, half, 0, kRowMajor, kColMajor, kRowMajor, 0, 0, 0, 0>>();
    }
    else if constexpr (0) {
        // sm80 / sm75
        constexpr Pack kPackA = HMMA_16816 | OPERAND_A | 2;
        constexpr Pack kPackU = HMMA_16816 | OPERAND_U | 1;
        return gTestbed<gemm::Testbed<uint4_t, half, half, 1, kColMajor, kColMajor, kColMajor, kPackA, 0, kPackU, 0>>();
    }
    else if constexpr (1) {
        // sm80 / sm75
        constexpr Pack kPackB = HMMA_16816 | OPERAND_B | 2;
        constexpr Pack kPackV = HMMA_16816 | OPERAND_V | 1;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kRowMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        // sm70
        constexpr Pack kPackB = HMMA_884 | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_884 | OPERAND_V | 1;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        // simt
        constexpr Pack kPackB = HMMA_SIMT | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_SIMT | OPERAND_V | 1;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
}

}  // namespace turbomind::gemm
