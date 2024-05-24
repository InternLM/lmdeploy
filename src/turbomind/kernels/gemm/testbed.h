// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/quantization.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <cublas_v2.h>
#include <fstream>
#include <thrust/universal_vector.h>

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "*** Check failed: (%s) @ %s:%d\n", #cond, __FILE__, __LINE__);                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace turbomind::gemm {

using thrust::universal_vector;

template<class T, class Tb>
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

        if (cublas_) {
            cublasDestroy(cublas_);
            cublas_ = {};
        }
    }

    void Initialize(int m, int n, int k, int g, bool reuse, cudaStream_t stream)
    {
        bool flag_a = !reuse;
        bool flag_b = !reuse;

        if (m != m_ || k != k_) {
            a_.resize(m * k);
            flag_a = true;
        }

        if (n != n_ || k != k_) {
            b_.resize(n * k);
            flag_b = true;
        }

        if (m != m_ || n != n_) {
            c_.resize(m * n);
            c_f_.resize(c_.size());
            c_ref_.resize(c_.size());
        }

        m_ = m;
        n_ = n;
        k_ = k;

        rng_.set_stream(stream);
        stream_ = stream;

        if (flag_a) {
            rng_.GenerateUniform(a_.data().get(), a_.size(), 1, -.5f);
            // thrust::fill_n(a_.begin(), a_.size(), 0);
        }

        if (flag_b) {
            rng_.GenerateUniform(b_.data().get(), b_.size(), 1, -.5f);
            // thrust::fill_n(b_.begin(), b_.size(), 0);
        }

        bool flag_p = flag_b;

        if constexpr (!std::is_same_v<T, Tb>) {
            if (flag_b || g != g_) {
                Quantize(g);
                flag_p = true;
            }
        }

        if (flag_p) {
            Pack();
        }

        barriers_.resize(m_ * n_);
        partials_.resize(kMaxSplits * m_ * n_);
    }

    void Run()
    {
        // const Operation operation{
        //     QuantDesc{QuantType::kAsym_FMA, g_},
        //     Epilogue::kNone,
        //     dispatch_policy_,
        // };

        const Operation operation{
            QuantDesc{QuantType::kNone, 0},
            Epilogue::kNone,
            dispatch_policy_,
        };

        const MatrixLayout c_desc{get_data_type_v<T>, Order::kRowMajor, m_, n_, n_};

        const Workspace workspace{barriers_.data().get(),
                                  sizeof(int) * barriers_.size(),
                                  partials_.data().get(),
                                  sizeof(float) * partials_.size()};

        // auto status = gemm_.Run(operation,
        //                         nullptr,
        //                         a_.data().get(),
        //                         MatrixLayout{get_data_type_v<T>, Order::kRowMajor, m_, k_, k_},
        //                         b_pack_.data().get(),
        //                         MatrixLayout{get_data_type_v<Tb>, Order::kFragment_81616, k_, n_, k_},
        //                         q_pack_.data().get(),
        //                         MatrixLayout{get_data_type_v<T>, Order::kColMajor, k_ / g_, n_, n_},
        //                         nullptr,
        //                         c_.data().get(),
        //                         c_desc,
        //                         c_.data().get(),
        //                         c_desc,
        //                         workspace,
        //                         stream_);

        auto status = gemm_.Run(operation,
                                nullptr,
                                a_.data().get(),
                                MatrixLayout{get_data_type_v<T>, Order::kColMajor, m_, k_, m_},
                                b_.data().get(),
                                MatrixLayout{get_data_type_v<Tb>, Order::kColMajor, k_, n_, k_},
                                nullptr,
                                MatrixLayout{get_data_type_v<T>, Order::kRowMajor, 0, 0, 0},
                                nullptr,
                                c_.data().get(),
                                c_desc,
                                c_.data().get(),
                                c_desc,
                                workspace,
                                stream_);

        if (status) {
            std::cerr << "Run failed, code =" << status << "\n";
        }
    }

    void RunCublas()
    {
        computeRefCublas(c_f_.data().get(), a_.data().get(), b_f_.data().get(), m_, n_, k_, stream_);
    }

    void CompareB()
    {
        cudaDeviceSynchronize();
        Compare(b_f_.data().get(), b_.data().get(), k_, k_, n_);
    }

    void CompareC()
    {
        for (int i = 0; i < 10; ++i) {
            computeRefCublas(c_ref_.data().get(), a_.data().get(), b_.data().get(), m_, n_, k_, stream_);
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
        b_q_.resize(n_ * k_);
        q_.resize(k_ / g * n_ * 2);

        g_ = g;

        universal_vector<Array<T, 2>> minmax(k_ / g_ * n_);

        const int  threads = std::min(256, n_);
        const dim3 blocks((n_ + threads - 1) / threads, k_ / g_);

        find_stats<<<blocks, threads, 0, stream_>>>(minmax.data().get(),  //
                                                    b_.data().get(),
                                                    n_,
                                                    k_,
                                                    g);

        find_params<Tb, true><<<(minmax.size() + 255) / 256, 256, 0, stream_>>>(q_.data().get(),  //
                                                                                minmax.data().get(),
                                                                                minmax.size());

        // universal_vector<T> b_f(b_.size());
        b_f_.resize(b_.size());
        quantize<Tb><<<blocks, threads, 0, stream_>>>(b_q_.data().get(),  //
                                                      b_f_.data().get(),
                                                      b_.data().get(),
                                                      q_.data().get(),
                                                      n_,
                                                      k_,
                                                      g);
    }

    void Pack()
    {
        // b_pack_.resize(n_ * k_ / 8);
        // q_pack_.resize(q_.size());

        // constexpr int kQ = !std::is_same_v<T, Tb>;
        // gemm::transcript<T>((Tb*)b_pack_.data().get(),  //
        //                     kQ ? q_pack_.data().get() : nullptr,
        //                     b_q_.data().get(),
        //                     kQ ? q_.data().get() : nullptr,
        //                     n_,
        //                     k_,
        //                     g_,
        //                     stream_);

        // Convert(a_.data(), {}, )
    }

    void computeRefCublas(half* C, const half* A, const half* B, int m, int n, int k, cudaStream_t stream)
    {
        // cublas
        if (!cublas_) {
            cublasCreate(&cublas_);
        }
        float alpha = 1.f;
        float beta  = 0.f;
        // TNT (A and B are swapped for transposing C)
        cublasGemmEx(cublas_,
                     CUBLAS_OP_T,
                     CUBLAS_OP_T,
                     n,
                     m,
                     k,
                     &alpha,
                     B,
                     CUDA_R_16F,
                     k,
                     A,
                     CUDA_R_16F,
                     m,
                     &beta,
                     C,
                     CUDA_R_16F,
                     n,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

private:
    int m_{};
    int n_{};
    int k_{};
    int g_{};

    universal_vector<T> a_;      // A in fp
    universal_vector<T> b_;      // B in fp
    universal_vector<T> c_ref_;  // reference C

    universal_vector<uint16_t> b_q_;  // quantized B
    universal_vector<T>        b_f_;  // dequant `b_q_` back to fp
    universal_vector<T>        q_;    // quant param `b_q_`
    universal_vector<T>        c_f_;  // ref C computed by `b_f_`

    universal_vector<Array<T, 8>>  a_pack_;  // packed A
    universal_vector<Array<Tb, 8>> b_pack_;  // packed B
    universal_vector<T>            q_pack_;  // packed Q

    universal_vector<T> c_;  // output C

    universal_vector<int>   barriers_;
    universal_vector<float> partials_;

    cudaStream_t stream_;

    RNG            rng_;
    cublasHandle_t cublas_;

    Gemm           gemm_;
    DispatchPolicy dispatch_policy_;
    std::string    cache_path_;
};

}  // namespace turbomind::gemm