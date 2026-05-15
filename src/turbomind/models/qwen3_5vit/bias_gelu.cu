#include "src/turbomind/models/qwen3_5vit/bias_gelu.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

namespace {

struct GeluPytorchTanh {
    __device__ __forceinline__ float operator()(float x) const
    {
        constexpr float kAlpha = 0.7978845608028654f;
        constexpr float kBeta  = 0.044715f;
        return 0.5f * x * (1.f + tanhf(kAlpha * (x + kBeta * x * x * x)));
    }
};

struct Gelu {
    __device__ __forceinline__ float operator()(float x) const
    {
        constexpr float kInvSqrt2 = 0.70710678118654752440f;
        return 0.5f * x * (1.f + erff(x * kInvSqrt2));
    }
};

template<int vec_size, typename T, class Activation>
__global__ void
biasActivationKernel(T* data, const T* __restrict__ bias, int64_t stride, Activation activation, int num, int dim)
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
        x[i] = activation(x[i]);
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
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        constexpr int threads  = 512;

        const int num = x.shape(0);
        const int dim = x.shape(1);
        TM_CHECK_EQ(dim % vec_size, 0);

        const dim3 grid(num, cdiv(dim, threads * vec_size));

        if (type == ActivationType::kGeluPytorchTanh) {
            biasActivationKernel<vec_size, T><<<grid, threads, 0, stream>>>(
                x.data<T>(), bias.data_or((T*)nullptr), x.stride(0), GeluPytorchTanh{}, num, dim);
        }
        else if (type == ActivationType::kGelu) {
            biasActivationKernel<vec_size, T>
                <<<grid, threads, 0, stream>>>(x.data<T>(), bias.data_or((T*)nullptr), x.stride(0), Gelu{}, num, dim);
        }
        else {
            TM_LOG_FATAL("unsupported Qwen3.5 ViT bias activation type: {}", (int)type);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(x.dtype(), invoke);
}

}  // namespace turbomind
