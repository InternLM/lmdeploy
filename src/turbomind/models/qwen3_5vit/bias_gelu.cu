#include "src/turbomind/models/qwen3_5vit/bias_gelu.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/activation_ops.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

#include <type_traits>

namespace turbomind {

namespace {

template<int vec_size, typename T, template<typename> class Activation>
__global__ void biasActivationKernel(T* data, const T* __restrict__ bias, int64_t stride, int num, int dim)
{
    const int ti = blockIdx.x;
    const int di = (threadIdx.x + blockIdx.y * blockDim.x) * vec_size;

    if (ti >= num || di >= dim) {
        return;
    }

    Array<T, vec_size> x_vec;
    Load(x_vec, data + ti * stride + di);

    auto x = cast<float>(x_vec);

    if (bias) {
        Array<T, vec_size> bias_vec;
        Ldg(bias_vec, bias + di);
        using namespace ops;
        x = x + cast<float>(bias_vec);
    }

    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        x[i] = Activation<float>::apply(x[i]);
    }

    Store(data + ti * stride + di, cast<T>(x));
}

}  // namespace

void invokeQwen3_5VitBiasActivation(Tensor& x, const Tensor& bias, ActivationType type, cudaStream_t stream)
{
    if (x.size() == 0) {
        return;
    }

    TM_CHECK_EQ(x.ndim(), 2);
    if (bias) {
        TM_CHECK_EQ(bias.shape(-1), x.shape(-1));
        TM_CHECK_EQ(bias.dtype(), x.dtype());
    }

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        constexpr int max_vec = sizeof(uint4) / sizeof(T);
        constexpr int threads = 512;

        const int     num    = x.shape(0);
        const int     dim    = x.shape(1);
        const int64_t stride = x.stride(0);

        int best_vec_size = 1;
        for (int v = max_vec; v >= 1; v >>= 1) {
            if (dim % v == 0 && stride % v == 0) {
                best_vec_size = v;
                break;
            }
        }

        auto launch = [&](auto vec_size_) {
            constexpr int vec_size = decltype(vec_size_)::value;
            const dim3    grid(num, cdiv(dim, threads * vec_size));
            if (type == ActivationType::kGeluPytorchTanh) {
                biasActivationKernel<vec_size, T, GeluActivation>
                    <<<grid, threads, 0, stream>>>(x.data<T>(), bias.data_or((T*)nullptr), stride, num, dim);
            }
            else if (type == ActivationType::kGelu) {
                biasActivationKernel<vec_size, T, GeluErfActivation>
                    <<<grid, threads, 0, stream>>>(x.data<T>(), bias.data_or((T*)nullptr), stride, num, dim);
            }
            else {
                TM_LOG_FATAL("unsupported Qwen3.5 ViT bias activation type: {}", (int)type);
            }
        };

        switch (best_vec_size) {
            case 8:
                return launch(std::integral_constant<int, 8>{});
            case 4:
                return launch(std::integral_constant<int, 4>{});
            case 2:
                return launch(std::integral_constant<int, 2>{});
            default:
                return launch(std::integral_constant<int, 1>{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(x.dtype(), invoke);
}

}  // namespace turbomind
