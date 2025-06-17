

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/stream.h"

#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/quantization.h"

using namespace turbomind;

int main()
{
    core::ContextGuard ctx{core::Stream::create(), core::Allocator{kCPU}, core::Allocator{kDEVICE}};

    auto stream = core::Context::stream().handle();

    const int m = 1024, n = 2048, gs = 128;

    Tensor_<bfloat16_t> h_x{{m, n}, kCPU};
    Tensor_<bfloat16_t> h_x_f{{m, n}, kCPU};

    Tensor_<bfloat16_t> x{{m, n}, kDEVICE};
    Tensor_<bfloat16_t> x_f{{m, n}, kDEVICE};
    Tensor_<fp8_e4m3_t> x_q{{m, n}, kDEVICE};

    // Tensor_<float> x_s{{{m, n / gs}, {1, round_up(m, 4)}}, kDEVICE};
    Tensor_<float> x_s;

    RNG r;
    r.set_stream(stream);

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(x))
    r.UniformFloat(x, 2.f, 2.f);  // [-1, +1]
    Copy(x, h_x);
    QuantizeSymm(x_q, x_s, x, stream);
    DequantizeSymm(x_f, x_q, x_s, stream);
    Copy(x_f, h_x_f);
    FC_Header();
    FC_Print(FastCompare(x_f, x, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(dequant(quant(x)))), aligned representable values
    Copy(x_f, x);
    Clear(x_f);
    QuantizeSymm(x_q, x_s, x, stream);
    DequantizeSymm(x_f, x_q, x_s, stream);
    FC_Print(FastCompare(x_f, x, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(x))
    // x_s = {{cdiv(m, gs), cdiv(n, gs)}, kDEVICE};
    x_s = {};
    r.UniformFloat(x, 2.f, 2.f);  // [-1, +1]
    Copy(x, h_x);
    QuantizeSymmBlock(x_q, x_s, x, stream);
    DequantizeSymmBlock(x_f, x_q, x_s, stream);
    FC_Print(FastCompare(x_f, x, stream));

    /////////////////////////////////////////////////////////////////////////////////////
    // round trip of dequant(quant(dequant(quant(x)))), aligned representable values
    Copy(x_f, x);
    Clear(x_f);
    QuantizeSymmBlock(x_q, x_s, x, stream);
    DequantizeSymmBlock(x_f, x_q, x_s, stream);
    FC_Print(FastCompare(x_f, x, stream));

    return 0;
}
