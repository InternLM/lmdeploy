

#pragma once

#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>

#include <cuda.h>
#include <cuda_bf16.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
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

        workspace.tensormaps_size = 8192 * sizeof(CUtensorMap);
        cudaMalloc(&workspace.tensormaps, workspace.tensormaps_size);
        cudaMalloc(&workspace.flags, sizeof(int));
        // TM_CHECK_NOTNULL(workspace.flags);
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
            t    = {{{cols * num, rows}, {rows, 1}}, type, kDEVICE};
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
            if (expert_num_ == 0) {
                QuantizeSymmBlock(a_q_, a_s_, a_, stream);
                DequantizeSymmBlock(a_f_, a_q_, a_s_, stream);
            }
            else {
                a_q_          = empty_like(a_, kFloat8_e4m3);
                a_f_          = empty_like(a_);
                const int m_s = cdiv(M, 128);
                a_s_          = Tensor_<float>({m_s * expert_num_, cdiv(K, 128)}, kDEVICE);
                for (int i = 0; i < expert_num_; ++i) {
                    auto a_s = a_s_.slice(i * m_s, m_s);
                    QuantizeSymmBlock(a_q_.slice(i * M, M), a_s, a_.slice(i * M, M), stream);
                    DequantizeSymmBlock(a_f_.slice(i * M, M), a_q_.slice(i * M, M), a_s, stream);
                }
            }

            a_q_desc_ = {a_q_.dtype(), kRowMajor, M, K, (int)a_q_.stride(0)};
            u_desc_   = {a_s_.dtype(), kRowMajor, (int)a_s_.shape(0), (int)a_s_.shape(1), (int)a_s_.stride(0)};
            tie(a_x_, a_desc_x_) = std::make_tuple(&a_q_, &a_q_desc_);
            u_desc_.num = a_q_desc_.num = a_desc_.num;
            if (1) {
                std::cout << "a_q " << a_q_ << "\n";
                std::cout << "a_s " << a_s_ << "\n";
                std::cout << "a_f " << a_f_ << "\n";
            }

            quant_a_ = {QuantType::kB, 128};
        }

        if (Tb == kFloat8_e4m3) {  // b is k-major & b_s is n-major
            QuantizeSymm(b_q_, b_s_, b_, stream);
            DequantizeSymm(b_f_, b_q_, b_s_, stream);
            b_q_desc_ = {b_q_.dtype(), kColMajor, K, N, (int)b_q_.stride(0)};
            v_desc_   = {b_s_.dtype(), kRowMajor, (int)b_s_.shape(0), (int)b_s_.shape(1), (int)b_s_.stride(0)};
            tie(b_x_, b_desc_x_) = std::make_tuple(&b_q_, &b_q_desc_);
            v_desc_.num = b_q_desc_.num = b_desc_.num;
            if (1) {
                std::cout << "b_q " << b_q_ << "\n";
                std::cout << "b_s " << b_s_ << "\n";
                std::cout << "b_f " << b_f_ << "\n";
            }

            quant_b_ = {QuantType::kK, 128};
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

        auto expert_ids = SampleUniform(bsz, expert_num_, e_, g);

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

        std::vector<int> m_offset(expert_num_ + 1);  // output dim
        for (int i = 0; i < expert_num_; ++i) {
            m_offset[i + 1] = m_offset[i] + M;
        }

        Buffer_<uint64_t> a_ptrs(expert_num_, kCPU);
        Buffer_<uint64_t> u_ptrs(expert_num_, kCPU);

        for (int i = 0; i < expert_num_; ++i) {
            a_ptrs[i] = reinterpret_cast<uint64_t>(a_x_->slice(m_offset[i]).raw_data());
            if (a_s_) {
                const int m_s = cdiv(M, 128);
                u_ptrs[i]     = reinterpret_cast<uint64_t>(a_s_.slice(i * m_s).raw_data());
            }
        }

        std::vector<int> n_offset(expert_num_ + 1);  // bsz
        for (int i = 0; i < expert_num_; ++i) {
            n_offset[i + 1] = n_offset[i] + cnt[i];
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

        if (1) {
            // std::cout << "f2n: ";
            // for (size_t i = 0; i < f2n.size(); ++i) {
            //     std::cout << f2n[i] << " ";
            // }
            // std::cout << "\n";
            std::cout << "m_offset: ";
            for (size_t i = 0; i < m_offset.size(); ++i) {
                std::cout << m_offset[i] << " ";
            }
            std::cout << "\n";
            std::cout << "n_offset: ";
            for (size_t i = 0; i < n_offset.size(); ++i) {
                std::cout << n_offset[i] << " ";
            }
            std::cout << "\n";
        }

        moe_cnt_ = cnt;

        f2n_ = {(int)f2n.size(), kDEVICE};
        Copy(Buffer_{f2n.data(), f2n_.size(), kCPU}, f2n_);
        // std::cout << "f2n: " << f2n_ << "\n";

        m_offset_ = {(int)m_offset.size(), kDEVICE};
        Copy(Buffer_{m_offset.data(), m_offset_.size(), kCPU}, m_offset_);
        // std::cout << "m_offset: " << m_offset_ << "\n";

        n_offset_ = {(int)n_offset.size(), kDEVICE};
        Copy(Buffer_{n_offset.data(), n_offset_.size(), kCPU}, n_offset_);
        // std::cout << "n_offset: " << n_offset_ << "\n";

        en2f_ = {(int)en2f.size(), kDEVICE};
        Copy(Buffer_{en2f.data(), en2f_.size(), kCPU}, en2f_);
        // std::cout << "en2f: " << en2f_ << "\n";

        moe_scales_ = {(int)scales.size(), kDEVICE};
        Copy(Buffer_{scales.data(), moe_scales_.size(), kCPU}, moe_scales_);
        // std::cout << "moe_scales: " << moe_scales_ << "\n";

        a_ptrs_ = {a_ptrs.size(), kDEVICE};
        Copy(a_ptrs, a_ptrs_);

        u_ptrs_ = {u_ptrs.size(), kDEVICE};
        Copy(u_ptrs, u_ptrs_);

        b_q_e_ = create_(K, N, b_q_.dtype(), Ob, e_).first;
        std::cout << "Bqe: " << b_q_e_ << "\n";
        invokeMoeDispatch(b_q_e_, b_q_, f2n_.data(), e_, stream_);
        invokeMoeDispatchScales(v_e_, b_s_, f2n_.data(), e_, stream_);

        std::cout << "Ve: " << v_e_ << std::endl;
        b_q_desc_.num = v_desc_.num = expert_num_;
        b_q_desc_.offsets = v_desc_.offsets = n_offset_.data();

        v_desc_.ld = v_e_.stride(0);

        b_x_ = &b_q_e_;
    }

    void Run()
    {
        const Operation operation{get_dispatch_policy(),  //
                                  Epilogue::kNone,
                                  quant_a_,
                                  quant_b_,
                                  1};

        auto C = &c_x_;
        auto V = &b_s_;

        auto a_desc = *a_desc_x_;
        auto u_desc = u_desc_;
        auto b_desc = *b_desc_x_;
        auto v_desc = v_desc_;
        auto c_desc = c_desc_;

        Tensor C_e;
        if (expert_num_) {
            c_desc.num     = expert_num_;
            c_desc.offsets = n_offset_.data();
            C_e            = create_(M, N, Tc, Oc, e_).first;
            C              = &C_e;
            V              = &v_e_;
            a_desc.offsets = m_offset_.data();
            u_desc.offsets = m_offset_.data();
            c_desc.cols *= e_;
            b_desc.cols *= e_;
            v_desc.cols *= e_;
        }

        FT_CHECK(a_x_ && a_desc_x_);
        FT_CHECK(b_x_ && b_desc_x_);

        auto status = gemm_.Run(operation,
                                1.f,
                                expert_num_ ? a_ptrs_.raw_data() : a_x_->raw_data(),
                                a_desc,
                                expert_num_ ? u_ptrs_.raw_data() : a_s_.data_or((void*)nullptr),
                                u_desc,
                                b_x_->raw_data(),
                                b_desc,
                                V->data_or((void*)nullptr),
                                v_desc,
                                0.f,
                                C->raw_data(),
                                c_desc,
                                C->raw_data(),
                                c_desc,
                                workspace,
                                stream_);

        TM_CHECK_EQ(status, 0);

        if (expert_num_) {
            invokeMoeCombine(c_x_,  //
                             C_e,
                             moe_scales_.data(),
                             en2f_.data(),
                             nullptr,
                             e_,
                             0,
                             stream_);
        }

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

            std::cout << b << " " << b_e << " " << c_e << "\n";
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
            invokeMoeCombine(c, c_e, moe_scales_.data(), en2f_.data(), nullptr, e_, 0.f, stream_);
        }
    }

    auto Compare(int x, int r)
    {
        std::vector c{&c_, &c_f_, &c_x_};
        return FastCompare(*c[x], *c[r], stream_);
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

    QuantDesc quant_a_{};
    QuantDesc quant_b_{};

    Buffer_<int>   m_offset_;
    Buffer_<int>   n_offset_;
    Buffer_<int>   f2n_;   // for `dispatch`
    Buffer_<int>   en2f_;  // for `combine`
    Buffer_<float> moe_scales_;

    Buffer_<uint64_t> a_ptrs_;
    Buffer_<uint64_t> u_ptrs_;

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

    Tensor b_q_e_;
    Tensor v_e_;

    Tensor*       a_x_;
    Tensor*       b_x_;
    MatrixLayout* a_desc_x_;
    MatrixLayout* b_desc_x_;

    Workspace workspace{};

    RNG rng_;

    Gemm      gemm_;
    Reference reference_;
};

enum class TestPreset : int
{
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
