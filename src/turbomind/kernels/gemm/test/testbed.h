// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include <thrust/universal_vector.h>

#include "src/turbomind/core/core.h"

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/test/quantization.h"
#include "src/turbomind/kernels/gemm/test/reference.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

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
        Initialize(m, n, k, g, 1, 1, stream);
    }

    void Initialize(int m, int n, int k, int g, int experts, int top_e, cudaStream_t stream) noexcept
    {
        rng_.set_stream(stream);
        reference_.set_stream(stream);
        stream_ = stream;

        cudaGetDeviceProperties(&prop_, 0);

        m_ = m;
        n_ = n;
        k_ = k;

        batch_size_  = batch_dim == 0 ? m_ : n_;
        input_dims_  = k;
        output_dims_ = (size_t)m_ * n_ / batch_size_;

        const size_t E = std::max(1, experts);

        a_.resize(m * k);
        b_.resize(n * k * E);
        c_.resize(m * n);

        a_desc_ = MatrixLayout{data_type_v<Tc>, order_a, m, k, mk2cs<order_a>(m, k).x, 0};
        b_desc_ = MatrixLayout{data_type_v<Tc>, order_b, k, n, _kn2cs<order_b>(k, n).x, 0};
        c_desc_ = MatrixLayout{data_type_v<Tc>, order_c, m, n, mk2cs<order_c>(m, n).x, 0};

        c_f_.resize(c_.size());
        c_ref_.resize(c_.size());

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
            u_pack_desc_ = u_desc_ = {kUint32, kColMajor, m, ceil_div(k, g), m};
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
            Quantize<Tb>(b_, n * E, k, _order_b, g, b_f_, b_q_, v_, stream);
            quant_b_ = {QuantType::kDefault, g};

            v_pack_desc_ = v_desc_ = {kUint32, kRowMajor, ceil_div(k, g), n, int(n * E)};
            v_pack_desc_.pack      = pack_v;
            v_pack_.resize(v_.size());
            auto v_src_data = (uint32_t*)v_.data().get();
            auto v_dst_data = (uint32_t*)v_pack_.data().get();
            std::cout << "pre-pack: " << v_pack_desc_.ld << "\n";
            for (size_t e = 0; e < E; ++e) {
                CHECK(!Convert(v_src_data, v_desc_, v_dst_data, v_pack_desc_, stream_));
                v_src_data += n;
                v_dst_data += (size_t)v_desc_.rows * v_desc_.cols;
            }
            std::cout << "post-pack: " << v_pack_desc_.ld << "\n";

            // cudaDeviceSynchronize();

            // for (int i = 0; i < v_pack_.size(); ++i) {
            //     std::cout << (float)v_pack_[i] << " ";
            // }
            // std::cout << "\n";
        }

        if constexpr (pack_a) {
            a_pack_desc_.type = data_type_v<Ta>;
            a_pack_desc_.pack = pack_a;
            const auto a_data = is_quant_a ? (void*)a_q_.data().get() : (void*)a_.data().get();
            CHECK(!Convert(a_data, a_desc_, a_pack_.data().get(), a_pack_desc_, stream_));
        }
        else {
            cudaMemcpyAsync(
                (Ta*)a_pack_.data().get(), a_.data().get(), sizeof(Ta) * a_.size(), cudaMemcpyDefault, stream);
        }

        if constexpr (pack_b) {
            // CHECK(experts == 0);
            b_pack_desc_.type = data_type_v<Tb>;
            b_pack_desc_.pack = pack_b;
            // clang-format off
            auto b_src_data = [&] {
                // MSVC does not recognise `is_quant_b` as compile time constant
                constexpr bool is_quant = !std::is_same_v<Tb, Tc>;
                if constexpr (is_quant) return b_q_.data().get(); else return b_.data().get();
            }();
            // clang-format on
            get_pointer_type<Tb> b_dst_data{(Tb*)b_pack_.data().get()};
            const size_t         numel = (size_t)b_desc_.rows * b_desc_.cols;
            std::cout << "pre-pack: " << b_pack_desc_.ld << "\n";
            for (size_t e = 0; e < E; ++e) {
                CHECK(!Convert((Tb*)b_src_data, b_desc_, (Tb*)b_dst_data, b_pack_desc_, stream_));
                // NOTE: This is not correct when b is quantized in n-major
                b_src_data = b_src_data + numel;
                b_dst_data = b_dst_data + numel;
            }
            std::cout << "post-pack: " << b_pack_desc_.ld << "\n";

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

        // ctx_ = std::make_unique<DynamicGemmContext>(prop_, stream_);

        InitMoE(batch_size_, experts, top_e);
    }

    void InitMoE(int batch_size, int experts, int top_e)
    {
        experts_     = experts;
        exp_per_tok_ = top_e;

        if (experts == 0) {
            return;
        }

        ctx_ = nullptr;  // std::make_unique<MoeGemmContext>(experts_, top_e, prop_, stream_);

        std::vector<int> r(experts);
        std::iota(r.begin(), r.end(), 0);

        // Sample `top_e` experts per token
        std::mt19937 g{};
        expert_ids_ = SampleBalanced(batch_size_, experts_, top_e, g);

        std::uniform_real_distribution<float> dist(1e-3, 1.f);
        std::vector<float>                    tmp(top_e);
        moe_scales_.resize(top_e * batch_size_);
        for (int i = 0; i < batch_size_; ++i) {
            float inv{};
            for (auto& x : tmp) {
                x = dist(g);
                inv += x;
            }
            inv = 1.f / inv;
            for (int e = 0; e < top_e; ++e) {
                moe_scales_[e * batch_size_ + i] = tmp[e] * inv;
            }
        }

        moe_cnt_.resize(experts);
        std::fill_n(moe_cnt_.begin(), moe_cnt_.size(), 0);
        std::vector<std::vector<int>> f2i(experts_);
        for (int i = 0; i < (int)expert_ids_.size(); ++i) {
            ++moe_cnt_[expert_ids_[i]];
            f2i[expert_ids_[i]].push_back(i);  // i ~ [n, e]
        }

        moe_m_offsets_.resize(experts_ + 1);
        moe_m_offsets_[0] = 0;
        for (int i = 0; i < experts_; ++i) {
            moe_m_offsets_[i + 1] = moe_m_offsets_[i] + moe_cnt_[i];
        }

        moe_n_offsets_.resize(experts_ + 1);
        moe_n_offsets_[0] = 0;
        for (int i = 0; i < experts_; ++i) {
            moe_n_offsets_[i + 1] = moe_n_offsets_[i] + output_dims_;
        }

        if (1) {
            moe_n_ptrs_.resize(experts_);
            const size_t         numel = (size_t)input_dims_ * output_dims_;
            get_pointer_type<Tb> p{(Tb*)b_pack_.data().get()};
            for (int i = 0; i < experts_; ++i) {
                moe_n_ptrs_[i] = StridedPtr{static_cast<Tb*>(p + i * numel), b_pack_desc_.ld};
            }
        }
        if (1) {
            moe_v_ptrs_.resize(experts_);
            const size_t numel = (size_t)v_desc_.rows * v_desc_.cols;
            const auto   p     = (uint32_t*)v_pack_.data().get();
            for (int i = 0; i < experts_; ++i) {
                moe_v_ptrs_[i] = StridedPtr{p + i * numel, v_pack_desc_.ld};
            }
        }

        std::cout << expert_ids_.size() << "\n";

        // for (auto x : expert_ids_) {
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        for (auto x : moe_cnt_) {
            std::cout << x << " ";
        }
        std::cout << "\n";

        for (auto x : moe_m_offsets_) {
            std::cout << x << " ";
        }
        std::cout << "\n";

        for (auto x : moe_n_offsets_) {
            std::cout << x << " ";
        }
        std::cout << "\n";

        moe_f2n_.resize(expert_ids_.size());
        moe_f2en_.resize(expert_ids_.size());
        moe_en2f_.resize(expert_ids_.size());
        for (int e = 0, i = 0; e < experts_; ++e) {
            for (const auto& x : f2i[e]) {
                moe_f2n_[i] = x / top_e;
                // [n, e] -> [e, n]
                const int en  = x % top_e * batch_size_ + x / top_e;
                moe_f2en_[i]  = en;
                moe_en2f_[en] = i;
                ++i;
            }
        }

        CHECK(batch_dim == 0);
        CHECK(a_desc_.order == kRowMajor);

        a_e_.resize(a_f_.size() * top_e);
        c_e_.resize(c_f_.size() * top_e);
        c_e_ref_.resize(c_e_.size());

        for (int i = 0; i < 10; ++i) {
            invokeMoeDispatch(Tensor{a_e_.data().get(), {top_e * batch_size_, input_dims_}, kDEVICE},
                              Tensor{a_f_.data().get(), {batch_size_, input_dims_}, kDEVICE},
                              moe_f2n_.data().get(),
                              top_e,
                              stream_);
        }

        a_pack_desc_.num = b_pack_desc_.num = c_desc_.num = experts_;

        a_pack_desc_.rows = a_desc_.rows = c_desc_.rows = expert_ids_.size();
        a_pack_desc_.offsets = c_desc_.offsets = moe_m_offsets_.data().get();

        a_pack_desc_.idxs = moe_f2n_.data().get();

        if (!moe_n_ptrs_.empty()) {
            b_pack_desc_.ld = 0;
        }
        // b_pack_desc_.offsets = moe_n_offsets_.data().get();

        v_pack_desc_.num = b_pack_desc_.num;
        if (!moe_v_ptrs_.empty()) {
            v_pack_desc_.ld = 0;
        }

        cudaMemPrefetchAsync(moe_m_offsets_.data().get(), sizeof(int) * moe_m_offsets_.size(), 0, stream_);
        cudaMemPrefetchAsync(moe_n_offsets_.data().get(), sizeof(int) * moe_n_offsets_.size(), 0, stream_);
        cudaMemPrefetchAsync(moe_f2n_.data().get(), sizeof(int) * moe_f2n_.size(), 0, stream_);
    }

    void Run(void* ctx = {})
    {
        const Operation operation{
            dispatch_policy_,
            Epilogue::kNone,
            quant_a_,
            quant_b_,
            kBatchDim,
            // ctx_.get(),
            // ctx,
        };

        const Workspace workspace{barriers_.data().get(), barriers_.size(), partials_.data().get(), partials_.size()};

        void* A = a_pack_.data().get();
        void* B = b_pack_.data().get();
        void* V = v_pack_.data().get();
        void* C = c_.data().get();

        if (experts_) {
            C = c_e_.data().get();
            if (!moe_n_ptrs_.empty()) {
                B = moe_n_ptrs_.data().get();
            }
            if (!moe_v_ptrs_.empty()) {
                V = moe_v_ptrs_.data().get();
            }
        }

        auto status = gemm_.Run(operation,  //
                                1.f,
                                A,
                                a_pack_desc_,
                                u_pack_.data().get(),
                                u_pack_desc_,
                                B,
                                b_pack_desc_,
                                V,
                                v_pack_desc_,
                                0.f,
                                C,
                                c_desc_,
                                C,
                                c_desc_,
                                workspace,
                                stream_);
        // auto status = 0;

        if (!ctx && status) {
            std::cerr << "Run failed, code =" << status << "\n";
            std::abort();
        }
    }

    void RunCublas()
    {
        if (experts_ == 0) {
            // reference_.gemm(a_f_.data().get(),  //
            //                 a_desc_,
            //                 b_f_.data().get(),
            //                 b_desc_,
            //                 c_f_.data().get(),
            //                 c_desc_);

            reference_.gemm(a_f_.data().get(),  //
                            a_desc_,
                            b_f_.data().get(),
                            b_desc_,
                            c_ref_.data().get(),
                            c_desc_);
        }
        else {  // [e_i, k] -> [k, n / E] -> [e_i, n / E]
            auto a_desc = a_desc_;
            auto b_desc = b_desc_;
            auto c_desc = c_desc_;

            CHECK(a_desc.order == kRowMajor);  // k-major, T
            // CHECK(b_desc.order == kColMajor);  // k-major, N
            CHECK(c_desc.order == kRowMajor);  // n-major, T

            auto a = a_e_.data().get();
            auto b = b_f_.data().get();
            auto c = c_e_ref_.data().get();

            for (int e = 0; e < experts_; ++e) {
                // Set input size for current expert
                c_desc.rows = a_desc.rows = moe_cnt_[e];

                reference_.gemm(a, a_desc, b, b_desc, c, c_desc);

                // Move to next expert
                a += moe_cnt_[e] * input_dims_;
                b += output_dims_ * input_dims_;
                c += moe_cnt_[e] * output_dims_;
            }
        }
    }

    void CompareB()
    {
        cudaDeviceSynchronize();
        Compare(b_f_.data().get(), b_.data().get(), k_, k_, n_);
    }

    void CompareC()
    {
        if (experts_ == 0) {

            cudaDeviceSynchronize();

            int dims = m_, bsz = n_;
            if (order_c == kRowMajor) {
                std::swap(dims, bsz);
            }
            Compare(c_.data().get(), c_ref_.data().get(), dims, dims, bsz, 0);
        }
        else {
            invokeMoeCombine(Tensor{c_.data().get(), {batch_size_, output_dims_}, kDEVICE},
                             Tensor{c_e_.data().get(), {(int)expert_ids_.size(), output_dims_}, kDEVICE},
                             moe_scales_.data().get(),
                             moe_en2f_.data().get(),
                             nullptr,
                             expert_ids_.size() / batch_size_,
                             0.f,
                             stream_);

            invokeMoeCombine(Tensor{c_ref_.data().get(), {batch_size_, output_dims_}, kDEVICE},
                             Tensor{c_e_ref_.data().get(), {(int)expert_ids_.size(), output_dims_}, kDEVICE},
                             moe_scales_.data().get(),
                             moe_en2f_.data().get(),
                             nullptr,
                             expert_ids_.size() / batch_size_,
                             0.f,
                             stream_);

            cudaDeviceSynchronize();

            Compare(c_e_.data().get(), c_e_ref_.data().get(), output_dims_, output_dims_, expert_ids_.size(), 0);
            Compare(c_.data().get(), c_ref_.data().get(), output_dims_, output_dims_, batch_size_, 0);
        }
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

    int64_t get_global_memory_reads()
    {
        if (experts_ == 0) {
            return byte_size(a_pack_desc_) + byte_size(b_pack_desc_) + byte_size(u_pack_desc_)
                   + byte_size(v_pack_desc_);
        }
        else {
            size_t    size = byte_size(a_pack_desc_) + byte_size(u_pack_desc_);
            const int nnz =
                std::accumulate(moe_cnt_.begin(), moe_cnt_.end(), 0, [](auto a, auto x) { return a + (x > 0); });
            size += nnz * (byte_size(b_pack_desc_) + byte_size(v_pack_desc_));
            return size;
        }
    }

    int64_t get_ref_global_memory_reads()
    {
        if (experts_ == 0) {
            return byte_size(a_desc_) + byte_size(b_desc_);
        }
        else {
            size_t    size = byte_size(a_desc_);
            const int nnz =
                std::accumulate(moe_cnt_.begin(), moe_cnt_.end(), 0, [](auto a, auto x) { return a + (x > 0); });
            size += nnz * byte_size(b_desc_);
            return size;
        }
    }

    int64_t get_element_count()
    {
        if (experts_ == 0) {
            return (int64_t)m_ * n_ * k_ * 2;
        }
        else {
            int64_t count = 0;
            for (const auto& m : moe_cnt_) {
                count += (int64_t)m * output_dims_ * input_dims_;
            }
            return count * 2;
        }
    }

    // private:
    int m_{};
    int n_{};
    int k_{};
    int g_{};

    int batch_size_{};
    int input_dims_{};
    int output_dims_{};
    int experts_{};
    int exp_per_tok_{};

    /// MoE buffers
    universal_vector<Tc> a_e_;
    universal_vector<Tc> c_e_;
    universal_vector<Tc> c_e_ref_;

    /// MoE utils
    std::vector<int>             expert_ids_;  // f(batch_idx * top_e) -> expert_id
    std::vector<int>             moe_cnt_;
    universal_vector<int>        moe_f2n_;
    universal_vector<int>        moe_f2en_;
    universal_vector<int>        moe_en2f_;
    universal_vector<int>        moe_m_offsets_;
    universal_vector<int>        moe_n_offsets_;
    universal_vector<StridedPtr> moe_n_ptrs_;
    universal_vector<StridedPtr> moe_v_ptrs_;
    universal_vector<float>      moe_scales_;

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

    cudaDeviceProp           prop_;
    std::unique_ptr<Context> ctx_;

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
        // sm70 int4
        constexpr Pack kPackB = HMMA_884 | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_884 | OPERAND_V | 1;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        // sm70 half
        constexpr Pack kPackB = HMMA_884 | OPERAND_B | 1;
        return gTestbed<gemm::Testbed<half, half, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, 0>>();
    }
    else if constexpr (0) {
        // simt
        constexpr Pack kPackB = HMMA_SIMT | OPERAND_B | 1;
        constexpr Pack kPackV = HMMA_SIMT | OPERAND_V | 1;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        constexpr Pack kPackB = HMMA_16816 | OPERAND_B | 1;
        // constexpr Pack kPackB = 0;
        constexpr Pack kPackV = 0;
        return gTestbed<gemm::Testbed<half, half, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        constexpr Pack kPackB = HMMA_16816 | OPERAND_B | 2;
        // constexpr Pack kPackB = 0;
        constexpr Pack kPackV = HMMA_16816 | OPERAND_V | 1;
        // constexpr Pack kPackV = 0;
        return gTestbed<gemm::Testbed<half, uint4_t, half, 0, kRowMajor, kColMajor, kRowMajor, 0, kPackB, 0, kPackV>>();
    }
    else if constexpr (0) {
        // constexpr Pack kPackA = HMMA_16816 | OPERAND_A | 1;
        constexpr Pack kPackA = 0;
        return gTestbed<gemm::Testbed<half, half, half, 1, kColMajor, kColMajor, kColMajor, kPackA, 0, 0, 0>>();
    }
}

}  // namespace turbomind::gemm
