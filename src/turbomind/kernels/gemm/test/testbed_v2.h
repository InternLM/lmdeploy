

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/gemm.h"
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

    Testbed_v2(Config c)
    {
        tie(Ta, Tb, Tc) = tie(c.ta, c.tb, c.tc);
        tie(Oa, Ob, Oc) = tie(c.oa, c.ob, c.oc);
        tie(Pa, Pu)     = tie(c.pa, c.pu);
        tie(Pb, Pv)     = tie(c.pb, c.pv);
    }

    auto trans_(const Tensor& x, bool pred)
    {
        return pred ? x.t() : x;
    }

    auto create_(int rows, int cols, DataType type, Order order)
    {
        MatrixLayout m{};
        m.type  = type;
        m.order = order;
        m.rows  = rows;
        m.cols  = cols;
        Tensor t{};
        if (order == kRowMajor) {
            m.ld = cols;
            t    = {{{rows, cols}, {cols, 1}}, type, kDEVICE};
        }
        else {
            m.ld = rows;
            t    = {{{rows, cols}, {1, rows}}, type, kDEVICE};
        }
        return std::make_pair(t, m);
    }

    void Initialize(int m, int n, int k, cudaStream_t stream)
    {
        stream_ = stream;
        rng_.set_stream(stream);
        reference_.set_stream(stream);

        tie(M, N, K) = tie(m, n, k);

        tie(a_, a_desc_) = create_(M, K, Tc, Oa);
        tie(b_, b_desc_) = create_(K, N, Tc, Ob);
        tie(c_, c_desc_) = create_(M, N, Tc, Oc);

        c_o_ = empty_like(c_);

        if (1) {
            std::cout << "A " << a_ << "\n";
            std::cout << "B " << b_ << "\n";
            std::cout << "C " << c_ << "\n";
        }

        rng_.UniformFloat(a_, 1.);
        rng_.UniformFloat(b_, 1.);

        if (Ta == kFloat8_e4m3) {
            QuantizeSymm(a_q_, a_s_, a_, stream);
            DequantizeSymm(a_f_, a_q_, a_s_, stream);
            if (0) {
                std::cout << "a_q " << a_q_ << "\n";
                std::cout << "a_s " << a_s_ << "\n";
                std::cout << "a_f " << a_f_ << "\n";
            }
        }

        if (Tb == kFloat8_e4m3) {
            QuantizeSymmBlock(b_q_, b_s_, trans_(b_, Ob == kColMajor), stream);
            DequantizeSymmBlock(b_f_, b_q_, b_s_, stream);
            if (Ob == kColMajor) {
                b_q_ = b_q_.t();
                b_s_ = b_s_.t();
                b_f_ = b_f_.t();
            }
            if (0) {
                std::cout << "b_q " << b_q_ << "\n";
                std::cout << "b_s " << b_s_ << "\n";
                std::cout << "b_f " << b_f_ << "\n";
            }
        }
    }

    void Run()
    {
        const Operation operation{DispatchPolicy::kDefault,  //
                                  Epilogue::kNone,
                                  {},
                                  {},
                                  0};

        auto status = gemm_.Run(operation,
                                1.f,
                                a_.raw_data(),
                                a_desc_,
                                a_s_.data_or((void*)nullptr),
                                u_desc_,
                                b_.raw_data(),
                                b_desc_,
                                b_s_.data_or((void*)nullptr),
                                v_desc_,
                                0.f,
                                c_o_.raw_data(),
                                c_desc_,
                                c_o_.raw_data(),
                                c_desc_,
                                {},
                                stream_);

        TM_CHECK_EQ(status, 0);
    }

    void Ref(bool f)
    {
        auto& a = f ? a_f_ : a_;
        auto& b = f ? b_f_ : b_;
        auto& c = f ? c_f_ : c_;
        if (!c) {
            c = Tensor{c_.layout(), c_.dtype(), c_.device()};
        }
        reference_.gemm(a.raw_data(),  //
                        a_desc_,
                        b.raw_data(),
                        b_desc_,
                        c.raw_data(),
                        c_desc_);
    }

    auto Compare(int x, int r)
    {
        std::vector c{&c_, &c_f_, &c_o_};
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

    cudaStream_t stream_;

    Tensor a_;
    Tensor b_;
    Tensor c_;

    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;
    MatrixLayout u_desc_;
    MatrixLayout v_desc_;

    Tensor a_q_;  // quant(a)
    Tensor a_f_;  // dequant(a_q)
    Tensor a_s_;

    Tensor b_q_;  // quant(b)
    Tensor b_f_;  // dequant(b_q)
    Tensor b_s_;

    Tensor c_f_;  // a_f * b_f
    Tensor c_o_;  // a_q * b_q

    RNG rng_;

    Gemm      gemm_;
    Reference reference_;
};

enum class TestPreset : int {
    kANY_bf16_bf16_bf16_TNN,
    kANY_e4m3_e4m3_bf16_TTT,
};

inline std::unique_ptr<Testbed_v2> get_test(TestPreset preset)
{
    Testbed_v2::Config config{};
    switch (preset) {
        case TestPreset::kANY_bf16_bf16_bf16_TNN:
            config = {kBfloat16, kBfloat16, kBfloat16, kRowMajor, kColMajor, kColMajor};
            break;
        case TestPreset::kANY_e4m3_e4m3_bf16_TTT:
            config = {kFloat8_e4m3, kFloat8_e4m3, kBfloat16, kRowMajor, kRowMajor, kRowMajor};
            break;
        default:
            TM_CHECK(0) << "not implemented";
    }
    return std::make_unique<Testbed_v2>(config);
}

}  // namespace turbomind::gemm