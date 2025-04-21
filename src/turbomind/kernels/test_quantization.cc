

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/stream.h"

#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/quantization.h"

using namespace turbomind;

static void Header()
{
    printf("%12s%12s%12s%12s%12s%12s%12s\n",
           "amean",
           "amean_ref",
           "absdiff",
           "absdiff_max",
           "reldiff",
           "reldiff_max",
           "#outlier");
}
static void Print(const std::vector<float>& d)
{
    printf("%12f%12f%12f%12f%12f%12f%12f\n", d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
}

int main()
{
    core::ContextGuard ctx{core::Stream::create(), core::Allocator{kCPU}, core::Allocator{kDEVICE}};

    auto stream = core::Context::stream().handle();

    const int m = 5120, n = 8192, gs = 128;

    Tensor_<bfloat16_t> h_x{{m, n}, kCPU};
    Tensor_<bfloat16_t> h_x_f{{m, n}, kCPU};

    Tensor_<bfloat16_t> x{{m, n}, kDEVICE};
    Tensor_<bfloat16_t> x_f{{m, n}, kDEVICE};
    Tensor_<fp8_e4m3_t> x_q{{m, n}, kDEVICE};

    Tensor_<float> x_s{{m, n / gs}, kDEVICE};

    RNG r;
    r.set_stream(stream);

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(x))
    r.UniformFloat(x, 2.f, -1.f);  // [-1, +1]
    Copy(x.buffer(), h_x.buffer());
    QuantizeSymm(x_q, x_s, x, stream);
    DequantizeSymm(x_f, x_q, x_s, stream);
    Copy(x_f.buffer(), h_x_f.buffer());
    Header();
    Print(FastCompare(x_f.data(), x.data(), n, m, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(dequant(quant(x)))), aligned representable values
    Copy(x_f.buffer(), x.buffer());
    QuantizeSymm(x_q, x_s, x, stream);
    DequantizeSymm(x_f, x_q, x_s, stream);
    Print(FastCompare(x_f.data(), x.data(), n, m, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(x))
    x_s = {{m / gs, n / gs}, kDEVICE};
    r.UniformFloat(x, 2.f, -1.f);  // [-1, +1]
    Copy(x.buffer(), h_x.buffer());
    QuantizeSymmBlock(x_q, x_s, x, stream);
    DequantizeSymmBlock(x_f, x_q, x_s, stream);
    Print(FastCompare(x_f.data(), x.data(), n, m, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(dequant(quant(x)))), aligned representable values
    Copy(x_f.buffer(), x.buffer());
    QuantizeSymmBlock(x_q, x_s, x, stream);
    DequantizeSymmBlock(x_f, x_q, x_s, stream);
    Print(FastCompare(x_f.data(), x.data(), n, m, stream));

    return 0;
}