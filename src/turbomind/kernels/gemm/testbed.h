// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/quantization.h"
#include "src/turbomind/kernels/gemm/reference.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include <fstream>
#include <thrust/universal_vector.h>

namespace turbomind::gemm {

using thrust::universal_vector;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "*** Check failed: (%s) @ %s:%d\n", #cond, __FILE__, __LINE__);                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

template<class Ta, class Tb, class Tc, Order order_a, Order order_b>
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

        a_desc_ = MatrixLayout{get_data_type_v<Ta>, order_a, m, k, mk2cs<order_a>(m, k).x};
        b_desc_ = MatrixLayout{get_data_type_v<Tb>, order_b, k, n, _kn2cs<order_a>(k, n).x};
        c_desc_ = MatrixLayout{get_data_type_v<Tc>, order_c_, m, n, mk2cs<order_c_>(m, n).x};

        c_f_.resize(c_.size());
        c_ref_.resize(c_.size());

        a_q_.resize(a_.size());
        b_q_.resize(b_.size());

        u_.resize(a_.size());
        v_.resize(b_.size());

        a_f_.resize(a_.size());
        b_f_.resize(b_.size());

        /// TODO: Revise packed format
        a_pack_.resize(a_.size() / kVecSize);
        b_pack_.resize(b_.size() / kVecSize);

        u_pack_.resize(u_.size());
        v_pack_.resize(v_.size());

        barriers_.resize(m_ * n_);
        partials_.resize(kMaxSplits * m_ * n_);

        rng_.GenerateUniform(a_.data().get(), a_.size(), 1, -.5f);
        rng_.GenerateUniform(b_.data().get(), b_.size(), 1, -.5f);

        a_pack_desc_ = a_desc_;
        b_pack_desc_ = b_desc_;

        /// TODO: Revise packing condition
        if constexpr (0) {
            a_pack_desc_.pack = HMMA_16816 | OPERAND_A | 1;
            Convert(a_.data().get(), a_desc_, a_pack_.data().get(), a_pack_desc_, stream_);
        }
        else {
            cudaMemcpyAsync(
                (Ta*)a_pack_.data().get(), a_.data().get(), sizeof(Ta) * a_.size(), cudaMemcpyDefault, stream);
        }

        if constexpr (0) {
            b_pack_desc_.pack = HMMA_16816 | OPERAND_B | 1;
            Convert(b_.data().get(), b_desc_, b_pack_.data().get(), b_pack_desc_, stream_);
        }
        else {
            cudaMemcpyAsync(
                (Tb*)b_pack_.data().get(), b_.data().get(), sizeof(Ta) * b_.size(), cudaMemcpyDefault, stream);
        }
    }

    void Run()
    {
        const Operation operation{
            dispatch_policy_,
            Epilogue::kNone,
            QuantDesc{QuantType::kNone, 0},
            QuantDesc{QuantType::kNone, 0},
        };

        const Workspace workspace{barriers_.data().get(),
                                  sizeof(int) * barriers_.size(),
                                  partials_.data().get(),
                                  sizeof(float) * partials_.size()};

        auto status = gemm_.Run(operation,
                                nullptr,
                                a_pack_.data().get(),
                                a_pack_desc_,
                                u_.data().get(),
                                u_pack_desc_,
                                b_pack_.data().get(),
                                b_pack_desc_,
                                v_.data().get(),
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
            reference_.gemm(a_.data().get(),  //
                            a_desc_,
                            b_.data().get(),
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

        Compare(c_.data().get(), c_ref_.data().get(), n_, n_, m_, 0);
    }

private:
    void Quantize(int g)
    {
        // b_q_.resize(n_ * k_);
        // q_.resize(k_ / g * n_ * 2);

        // g_ = g;

        // universal_vector<Array<T, 2>> minmax(k_ / g_ * n_);

        // const int  threads = std::min(256, n_);
        // const dim3 blocks((n_ + threads - 1) / threads, k_ / g_);

        // find_stats<<<blocks, threads, 0, stream_>>>(minmax.data().get(),  //
        //                                             b_.data().get(),
        //                                             n_,
        //                                             k_,
        //                                             g);

        // find_params<Tb, true><<<(minmax.size() + 255) / 256, 256, 0, stream_>>>(q_.data().get(),  //
        //                                                                         minmax.data().get(),
        //                                                                         minmax.size());

        // // universal_vector<T> b_f(b_.size());
        // b_f_.resize(b_.size());
        // quantize<Tb><<<blocks, threads, 0, stream_>>>(b_q_.data().get(),  //
        //                                               b_f_.data().get(),
        //                                               b_.data().get(),
        //                                               q_.data().get(),
        //                                               n_,
        //                                               k_,
        //                                               g);
    }

private:
private:
    int m_{};
    int n_{};
    int k_{};
    int g_{};

    static constexpr Order order_c_ = Order::kRowMajor;

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

    universal_vector<int>   barriers_;
    universal_vector<float> partials_;

    cudaStream_t stream_;

    RNG rng_;

    Gemm           gemm_;
    Reference      reference_;
    DispatchPolicy dispatch_policy_;
    std::string    cache_path_;
};

}  // namespace turbomind::gemm