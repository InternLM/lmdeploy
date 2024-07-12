// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/quantization.h"
#include "src/turbomind/kernels/gemm/reference.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <fstream>
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
         Order order_a,
         Order order_b,
         Order order_c,
         Pack  pack_a,
         Pack  pack_b,
         Pack  pack_u = 0,
         Pack  pack_v = 0>
class Testbed {
public:
    static constexpr size_t kMaxSplits = 16;

    Testbed(): dispatch_policy_{DispatchPolicy::kDefault} {}

    Testbed(DispatchPolicy dispatch_policy, std::string cache_path):
        dispatch_policy_{dispatch_policy}, cache_path_{cache_path}
    {
        if (dispatch_policy == DispatchPolicy::kUseCached) {
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
        if (dispatch_policy_ == DispatchPolicy::kMeasure) {
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

        barriers_.resize(m_ * n_);
        partials_.resize(kMaxSplits * m_ * n_);

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

        constexpr bool is_quant_a = !std::is_same_v<Ta, Tc>;
        constexpr bool is_quant_b = !std::is_same_v<Tb, Tc>;

        if constexpr (is_quant_a) {
            static_assert(pack_a && pack_u);
            Quantize<Ta>(a_, m, k, order_a, g, a_f_, a_q_, u_, stream);
            u_pack_desc_ = u_desc_ = {DataType::U32, kColMajor, m, ceil_div(k, g), m};
            u_pack_desc_.pack      = pack_u;
            u_pack_.resize(u_.size());
            CHECK(!Convert(u_.data().get(), u_desc_, u_pack_.data().get(), u_pack_desc_, stream_));
            quant_a_ = {QuantType::kAsym_FMA, g};

            // cudaDeviceSynchronize();

            // for (int i = 0; i < u_pack_.size(); ++i) {
            //     std::cout << (float)u_pack_[i] << " ";
            // }
            // std::cout << "\n";
        }

        if constexpr (is_quant_b) {
            static_assert(pack_b && pack_v);
            constexpr Order _order_b = transpose(order_b);
            Quantize<Tb>(b_, n, k, _order_b, g, b_f_, b_q_, v_, stream);
            v_pack_desc_ = v_desc_ = {DataType::U32, kColMajor, n, ceil_div(k, g), n};
            v_pack_desc_.pack      = pack_v;
            v_pack_.resize(v_.size());
            CHECK(!Convert(v_.data().get(), v_desc_, v_pack_.data().get(), v_pack_desc_, stream_));
            quant_b_ = {QuantType::kAsym_FMA, g};

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

    void Run()
    {
        const Operation operation{
            dispatch_policy_,
            Epilogue::kNone,
            quant_a_,
            quant_b_,
        };

        const Workspace workspace{barriers_.data().get(),
                                  sizeof(int) * barriers_.size(),
                                  partials_.data().get(),
                                  sizeof(float) * partials_.size()};

        auto status = gemm_.Run(operation,
                                nullptr,
                                a_pack_.data().get(),
                                a_pack_desc_,
                                u_pack_.data().get(),
                                u_pack_desc_,
                                b_pack_.data().get(),
                                b_pack_desc_,
                                v_pack_.data().get(),
                                v_pack_desc_,
                                nullptr,
                                c_.data().get(),
                                c_desc_,
                                c_.data().get(),
                                c_desc_,
                                workspace,
                                stream_);

        if (status) {
            std::cerr << "Run failed, code =" << status << "\n";
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

        if (order_c == kRowMajor) {
            Compare(c_.data().get(), c_ref_.data().get(), n_, n_, m_, 0);
        }
        else {
            Compare(c_.data().get(), c_ref_.data().get(), m_, m_, n_, 0);
        }
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

    universal_vector<int>   barriers_;
    universal_vector<float> partials_;

    cudaStream_t stream_;

    RNG rng_;

    Gemm           gemm_;
    Reference      reference_;
    DispatchPolicy dispatch_policy_;
    std::string    cache_path_;
};

template<class T>
T& gTestbed()
{
    static T inst{turbomind::gemm::DispatchPolicy::kDefault, "tm_cache"};
    return inst;
}

inline decltype(auto) get_test()
{
    if constexpr (0) {
        // native
        constexpr Pack kPackA = 0;
        constexpr Pack kPackU = 0;
        constexpr Pack kPackB = 0;
        constexpr Pack kPackV = 0;
        return gTestbed<
            gemm::Testbed<half, half, half, kRowMajor, kColMajor, kRowMajor, kPackA, kPackB, kPackU, kPackV>>();
    }
    else if constexpr (1) {
        // sm80 / sm75
        constexpr Pack kPackA = 0;  // HMMA_16816 | OPERAND_A | 1;
        constexpr Pack kPackU = 0;  // HMMA_16816 | OPERAND_U | 1;
        constexpr Pack kPackB = 0;
        constexpr Pack kPackV = 0;
        return gTestbed<
            gemm::Testbed<half, half, half, kColMajor, kColMajor, kColMajor, kPackA, kPackB, kPackU, kPackV>>();
    }
    else if constexpr (0) {
        // sm80 / sm75
        constexpr Pack kPackA = 0;
        constexpr Pack kPackU = 0;
        constexpr Pack kPackB = HMMA_16816 | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_16816 | OPERAND_V | 1;
        return gTestbed<
            gemm::Testbed<half, uint4_t, half, kRowMajor, kRowMajor, kRowMajor, kPackA, kPackB, kPackU, kPackV>>();
    }
    else if constexpr (0) {
        // sm70
        constexpr Pack kPackA = 0;
        constexpr Pack kPackU = 0;
        constexpr Pack kPackB = HMMA_884 | OPERAND_B | 2;
        constexpr Pack kPackV = HMMA_884 | OPERAND_V | 2;
        return gTestbed<
            gemm::Testbed<half, uint4_t, half, kRowMajor, kColMajor, kRowMajor, kPackA, kPackB, kPackU, kPackV>>();
    }
    else if constexpr (0) {
        // simt
        constexpr Pack kPackA = 0;
        constexpr Pack kPackU = 0;
        constexpr Pack kPackB = HMMA_SIMT | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_SIMT | OPERAND_V | 1;
        return gTestbed<
            gemm::Testbed<half, uint4_t, half, kRowMajor, kColMajor, kRowMajor, kPackA, kPackB, kPackU, kPackV>>();
    }
}

}  // namespace turbomind::gemm
