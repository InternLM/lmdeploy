

#pragma once

#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>

#include <cuda.h>
#include <cuda_bf16.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/test/reference.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/quantization.h"

namespace turbomind::gemm {

// A is input, B is weight

// A = Uniform(m, k)
// B = Uniform(n, k)

// A_q, A_s, A' = quant(A)
// B_q, B_s, B' = quant(B)

// A = pack(A or A_q)
// B = pack(B or B_q)

// C_ref = A  * B
// C'    = A' * B'

// [m, k]

using std::tie;

class Testbed_v2 {
public:
    struct Config {
        DataType ta;
        DataType tb;
        DataType tc;
        Order    oa;
        Order    ob;
        Order    oc;
        Pack     pa;
        Pack     pb;
        Pack     pu;
        Pack     pv;
    };

    DispatchPolicy get_dispatch_policy()
    {
        static DispatchPolicy policy = [] {
            if (std::getenv("TM_GEMM_TUNE")) {
                return DispatchPolicy::kMeasure;
            }
            return DispatchPolicy::kDefault;
        }();
        return policy;
    }

    Testbed_v2(Config c)
    {
        tie(Ta, Tb, Tc) = tie(c.ta, c.tb, c.tc);
        tie(Oa, Ob, Oc) = tie(c.oa, c.ob, c.oc);
        tie(Pa, Pu)     = tie(c.pa, c.pu);
        tie(Pb, Pv)     = tie(c.pb, c.pv);

        workspace.tensormaps_size = 4096 * sizeof(CUtensorMap);
        cudaMalloc(&workspace.tensormaps, workspace.tensormaps_size);
    }

    auto trans_(const Tensor& x, bool pred)
    {
        return pred ? x.t() : x;
    }

    auto create_(int rows, int cols, DataType type, Order order, int num = 1)
    {
        MatrixLayout m{};
        m.type  = type;
        m.order = order;
        m.rows  = rows;
        m.cols  = cols;
        m.num   = num;  // added in strided dim
        Tensor t{};
        if (order == kRowMajor) {
            m.ld = cols;
            t    = {{{rows * num, cols}, {cols, 1}}, type, kDEVICE};
        }
        else {
            m.ld = rows;
            t    = {{{rows, cols * num}, {1, rows}}, type, kDEVICE};
        }
        return std::make_pair(t, m);
    }

    template<class T>
    auto prod_(const std::vector<T>& x)
    {
        return std::accumulate(x.begin(), x.end(), 1, std::multiplies<>{});
    }

    void Initialize(int m, int n, int k, int expert_num, int e, cudaStream_t stream)
    {
        stream_ = stream;
        rng_.set_stream(stream);
        reference_.set_stream(stream);

        tie(M, N, K) = tie(m, n, k);

        expert_num_ = expert_num;

        E  = std::max(expert_num, 1);
        e_ = std::max(e, 1);

        tie(a_, a_desc_) = create_(M, K, Tc, Oa, E);
        tie(b_, b_desc_) = create_(K, N, Tc, Ob);
        tie(c_, c_desc_) = create_(M, N, Tc, Oc);

        tie(a_x_, a_desc_x_) = std::make_tuple(&a_, &a_desc_);
        tie(b_x_, b_desc_x_) = std::make_tuple(&b_, &b_desc_);

        c_x_ = empty_like(c_);

        if (1) {
            std::cout << "A " << a_ << "\n";
            std::cout << "B " << b_ << "\n";
            std::cout << "C " << c_ << "\n";
        }

        rng_.NormalFloat(a_, 1., 1.);
        rng_.NormalFloat(b_, 1., 1.);

        if (Ta == kFloat8_e4m3) {
            QuantizeSymmBlock(a_q_, a_s_, a_, stream);
            DequantizeSymmBlock(a_f_, a_q_, a_s_, stream);
            a_q_desc_ = {a_q_.dtype(), kRowMajor, (int)a_q_.shape(0), (int)a_q_.shape(1), (int)prod_(a_q_.stride())};
            u_desc_   = {a_s_.dtype(), kRowMajor, (int)a_s_.shape(0), (int)a_s_.shape(1), (int)prod_(a_s_.stride())};
            tie(a_x_, a_desc_x_) = std::make_tuple(&a_q_, &a_q_desc_);
            u_desc_.num = a_q_desc_.num = a_desc_.num;
            if (1) {
                std::cout << "a_q " << a_q_ << "\n";
                std::cout << "a_s " << a_s_ << "\n";
                std::cout << "a_f " << a_f_ << "\n";
            }
        }

        if (Tb == kFloat8_e4m3) {  // b is k-major & b_s is n-major
            QuantizeSymm(b_q_, b_s_, trans_(b_, Ob == kColMajor), stream);
            DequantizeSymm(b_f_, b_q_, b_s_, stream);
            if (Ob == kColMajor) {
                b_q_ = b_q_.t();
                b_s_ = b_s_.t();
                b_f_ = b_f_.t();
            }
            b_q_desc_ = {b_q_.dtype(), kColMajor, (int)b_q_.shape(0), (int)b_q_.shape(1), (int)prod_(b_q_.stride())};
            v_desc_   = {b_s_.dtype(), kRowMajor, (int)b_s_.shape(0), (int)b_s_.shape(1), (int)prod_(b_s_.stride())};
            tie(b_x_, b_desc_x_) = std::make_tuple(&b_q_, &b_q_desc_);
            v_desc_.num = b_q_desc_.num = b_desc_.num;
            if (1) {
                std::cout << "b_q " << b_q_ << "\n";
                std::cout << "b_s " << b_s_ << "\n";
                std::cout << "b_f " << b_f_ << "\n";
            }
        }

        if (expert_num) {
            InitializeExperts();
        }
    }

    void InitializeExperts()
    {
        std::vector<int> r(expert_num_);
        std::iota(r.begin(), r.end(), 0);

        const int bsz = N;

        // Sample `top_e` experts per token
        std::mt19937 g{};

        auto expert_ids = SampleBalanced(bsz, expert_num_, e_, g);

        std::vector<float> tmp(e_);
        std::vector<float> scales(bsz * e_);

        std::uniform_real_distribution<float> dist(1e-3, 1.f);

        for (int i = 0; i < bsz; ++i) {
            float inv{};
            for (auto& x : tmp) {
                x = dist(g);
                inv += x;
            }
            inv = 1.f / inv;
            for (int e = 0; e < e_; ++e) {
                scales[e * bsz + i] = tmp[e] * inv;
            }
        }

        std::vector<int>              cnt(expert_num_);
        std::vector<std::vector<int>> f2i(expert_num_);

        for (int i = 0; i < (int)expert_ids.size(); ++i) {
            ++cnt[expert_ids[i]];
            f2i[expert_ids[i]].push_back(i);  // i ~ [n, e]
        }

        std::vector<int> n_offset(expert_num_ + 1);  // bsz
        for (int i = 0; i < expert_num_; ++i) {
            n_offset[i + 1] = n_offset[i] + cnt[i];
        }

        std::vector<int> m_offset(expert_num_ + 1);  // output dim
        for (int i = 0; i < expert_num_; ++i) {
            m_offset[i + 1] = m_offset[i] + M;
        }

        std::vector<int> f2n(expert_ids.size());
        std::vector<int> f2en(expert_ids.size());
        std::vector<int> en2f(expert_ids.size());

        for (int e = 0, i = 0; e < expert_num_; ++e) {
            for (const auto x : f2i[e]) {
                f2n[i]   = x / e_;
                int en   = x % e_ * bsz + x / e_;
                f2en[i]  = en;
                en2f[en] = i;
                ++i;
            }
        }

        moe_cnt_ = cnt;

        f2n_ = {(int)f2n.size(), kDEVICE};
        Copy(Buffer_{f2n.data(), f2n_.size(), kCPU}, f2n_);

        m_offset_ = {(int)m_offset.size(), kDEVICE};
        Copy(Buffer_{m_offset.data(), m_offset_.size(), kCPU}, m_offset_);

        n_offset_ = {(int)n_offset.size(), kDEVICE};
        Copy(Buffer_{n_offset.data(), n_offset_.size(), kCPU}, m_offset_);

        en2f_ = {(int)en2f.size(), kDEVICE};
        Copy(Buffer_{en2f.data(), en2f_.size(), kCPU}, en2f_);

        moe_scales_ = {(int)scales.size(), kDEVICE};
        Copy(Buffer_{scales.data(), moe_scales_.size(), kCPU}, moe_scales_);

        // b_e_   = create_(K, N, Tb, Ob, e_).first;
        // b_e_f_ = empty_like(b_e_);
        // invokeMoeDispatch(b_e_, b_, f2n_.data(), e_, stream_);
        // invokeMoeDispatch(b_e_f_, b_f_, f2n_.data(), e_, stream_);
    }

    void Run()
    {
        const Operation operation{get_dispatch_policy(),  //
                                  Epilogue::kNone,
                                  {QuantType::kDefault, 128},
                                  {QuantType::kDefault, 128},
                                  0};

        // const auto& a      = a_q_ ? a_q_ : a_;
        // const auto& b      = b_q_ ? b_q_ : b_;
        // const auto& a_desc = a_q_ ? a_q_desc_ : a_desc_;
        // const auto& b_desc = b_q_ ? b_q_desc_ : b_desc_;

        FT_CHECK(a_x_ && a_desc_x_);
        FT_CHECK(b_x_ && b_desc_x_);

        std::cout << *a_desc_x_ << " " << *b_desc_x_ << " " << c_desc_ << "\n";

        auto status = gemm_.Run(operation,
                                1.f,
                                a_x_->raw_data(),
                                *a_desc_x_,
                                a_s_.data_or((void*)nullptr),
                                u_desc_,
                                b_x_->raw_data(),
                                *b_desc_x_,
                                b_s_.data_or((void*)nullptr),
                                v_desc_,
                                0.f,
                                c_x_.raw_data(),
                                c_desc_,
                                c_x_.raw_data(),
                                c_desc_,
                                workspace,
                                stream_);

        TM_CHECK_EQ(status, 0);

        // Tensor h_c = empty_like(c_o_.t(), kCPU);
        // Copy(c_o_.t(), h_c);
        // core::Context::stream().Sync();
        // TM_CHECK(h_c.shapes(0, 1) == std::make_tuple(128, 128));
        // for (int i = 0; i < 128; ++i) {
        //     for (int j = 0; j < 128; ++j) {
        //         printf("%4d", (int)h_c.data<nv_bfloat16>()[i * 128 + j]);
        //     }
        //     printf("\n");
        // }
    }

    void Ref(bool f)
    {
        auto& a = f ? a_f_ : a_;
        auto& b = f ? b_f_ : b_;
        auto& c = f ? c_f_ : c_;

        auto a_desc = a_desc_;
        auto b_desc = b_desc_;
        auto c_desc = c_desc_;

        if (c.shape() != c_.shape()) {
            c = Tensor{c_.layout(), c_.dtype(), c_.device()};
        }

        if (expert_num_ == 0) {

            reference_.gemm(a.raw_data(),  //
                            a_desc_,
                            b.raw_data(),
                            b_desc_,
                            c.raw_data(),
                            c_desc_);
        }
        else {
            auto c_e = create_(M, N, Tc, Oc, e_).first;
            auto b_e = create_(K, N, b.dtype(), Ob, e_).first;

            invokeMoeDispatch(b_e, b, f2n_.data(), e_, stream_);

            auto a_ptr = (char*)a.raw_data();
            auto b_ptr = (char*)b_e.raw_data();
            auto c_ptr = (char*)c_e.raw_data();

            const auto& cnt = moe_cnt_;

            TM_CHECK(Oa == kRowMajor);
            TM_CHECK(Ob == kColMajor);
            TM_CHECK(Oc == kColMajor);

            for (int i = 0; i < expert_num_; ++i) {
                c_desc.cols = b_desc.cols = cnt[i];
                // std::cout << "A: " << a_desc << "\n";
                // std::cout << "B: " << b_desc << "\n";
                // std::cout << "C: " << c_desc << "\n";
                reference_.gemm(a_ptr, a_desc, b_ptr, b_desc, c_ptr, c_desc);
                a_ptr += byte_size(a_desc.type, K) * M;
                b_ptr += byte_size(b_desc.type, K) * cnt[i];
                c_ptr += byte_size(c_desc.type, M) * cnt[i];
            }
            if (!c) {
                c = Tensor{c_.layout(), c_.dtype(), c_.device()};
            }
            // std::cout << "C : " << c << "\n";
            // std::cout << "Ce: " << c_e << "\n";
            invokeMoeCombine(
                c, trans_(c_e, Oc == kColMajor), moe_scales_.data(), en2f_.data(), nullptr, e_, 0.f, stream_);
        }
    }

    auto Compare(int x, int r)
    {
        std::vector c{&c_, &c_f_, &c_x_};
        return FastCompare(trans_(*c[x], Oc == kColMajor), trans_(*c[r], Oc == kColMajor), stream_);
    }

    auto Check()
    {
        auto h_c   = empty_like(c_, kCPU);
        auto h_c_f = empty_like(c_, kCPU);

        Copy(c_, h_c);
        Copy(c_f_, h_c_f);

        core::Context::stream().Sync();

        turbomind::Compare(h_c_f.raw_data(), h_c.raw_data(), Tc, c_desc_.ld, N, M, true);
    }

private:
    DataType Ta{};
    DataType Tb{};
    DataType Tc{};
    Order    Oa{};
    Order    Ob{};
    Order    Oc{};
    Pack     Pa{};
    Pack     Pb{};
    Pack     Pu{};
    Pack     Pv{};
    int      M{};  // batch_size
    int      N{};  // output_dim
    int      K{};  // input_dim
    int      E{};  // expert num

    int expert_num_;
    int e_;

    Buffer_<int>   m_offset_;
    Buffer_<int>   n_offset_;
    Buffer_<int>   f2n_;   // for `dispatch`
    Buffer_<int>   en2f_;  // for `combine`
    Buffer_<float> moe_scales_;

    std::vector<int> moe_cnt_;

    cudaStream_t stream_;

    Tensor a_;
    Tensor b_;
    Tensor c_;

    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;
    MatrixLayout u_desc_;
    MatrixLayout v_desc_;
    MatrixLayout a_q_desc_;
    MatrixLayout b_q_desc_;

    Tensor a_q_;  // quant(a)
    Tensor a_f_;  // dequant(a_q)
    Tensor a_s_;

    Tensor b_q_;  // quant(b)
    Tensor b_f_;  // dequant(b_q)
    Tensor b_s_;

    Tensor c_f_;  //
    Tensor c_x_;  // test output

    Tensor*       a_x_;
    Tensor*       b_x_;
    MatrixLayout* a_desc_x_;
    MatrixLayout* b_desc_x_;

    Workspace workspace{};

    RNG rng_;

    Gemm      gemm_;
    Reference reference_;
};

enum class TestPreset : int {
    kANY_bf16_bf16_bf16_TNN,
    kANY_e4m3_e4m3_bf16_TNN,
};

inline std::unique_ptr<Testbed_v2> get_test(TestPreset preset)
{
    Testbed_v2::Config config{};
    switch (preset) {
        case TestPreset::kANY_bf16_bf16_bf16_TNN:
            config = {kBfloat16, kBfloat16, kBfloat16, kRowMajor, kColMajor, kColMajor};
            break;
        case TestPreset::kANY_e4m3_e4m3_bf16_TNN:
            config = {kFloat8_e4m3, kFloat8_e4m3, kBfloat16, kRowMajor, kColMajor, kColMajor};
            break;
        default:
            TM_CHECK(0) << "not implemented";
    }
    return std::make_unique<Testbed_v2>(config);
}

}  // namespace turbomind::gemm