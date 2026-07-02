
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/activation_ops.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <type_traits>

namespace turbomind {

template<class T>
struct SiluGptOss {
    __device__ T operator()(T gate, T up) const noexcept
    {
        gate = __hmin((T)7.f, gate);
        up   = __hmax((T)-7.f, __hmin((T)7.f, up));
        return static_cast<T>(fdividef((float)gate, 1.f + expf((float)-gate * 1.702f)) * (1.f + (float)up));
    }
};

template<class T>
struct Silu {
    __device__ T operator()(T gate, T up) const noexcept
    {
        return static_cast<T>(fdividef((float)gate, 1.f + expf(-(float)gate)) * (float)up);
    }
};

template<int vec_size, class Activation, class T>
__global__ void ActivationKernel(
    T* gate_buf, const T* __restrict__ up_buf, Activation activation, int64_t stride, int token_num, int dim)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int di = threadIdx.x + blockIdx.y * blockDim.x;
        const int ti = blockIdx.x;

        dim /= vec_size;

        if (di >= dim) {
            return;
        }

        using Vec = Array<T, vec_size>;

        auto p_gate = reinterpret_cast<Vec*>(gate_buf + ti * stride);
        auto p_up   = reinterpret_cast<const Vec*>(up_buf + ti * stride);

        Vec gate;
        Load(gate, (const T*)&p_gate[di]);

        Vec up;
        Ldg(up, (T*)&p_up[di]);

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            gate[i] = activation(gate[i], up[i]);
        }

        Store((T*)&p_gate[di], gate);
    }
}

void Activation(Ref<Tensor> gate_, const Tensor& up, ActivationType type, cudaStream_t stream)
{
    auto& gate = gate_.get();

    TM_CHECK(gate.shape() == up.shape());

    int num, dim;
    std::tie(num, dim) = gate.shapes(0, 1);

    auto invoke = [&](auto t, auto act) {
        using T = decltype(t);

        constexpr int vec_size = 4;
        constexpr int threads  = 512;

        const dim3 blocks(num, cdiv(dim, threads * vec_size));

        ActivationKernel<vec_size><<<blocks, threads, 0, stream>>>(gate.data<T>(),  //
                                                                   up.data<T>(),
                                                                   act,
                                                                   gate.stride(0),
                                                                   num,
                                                                   dim);
    };

    auto dispatch = [&](auto t) {
        using T = decltype(t);
        if (type == ActivationType::kSilu) {
            return invoke(t, Silu<T>{});
        }
        else if (type == ActivationType::kSiluGptOss) {
            return invoke(t, SiluGptOss<T>{});
        }
        else {
            TM_LOG_FATAL("unknown activation type: {}", (int)type);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(gate.dtype(), dispatch);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<int vec_size, class Activation, class T>
__global__ void ActivationKernel(
    T* gate_up, const T* bias, const int* group_ids, int64_t stride, Activation activation, int token_num, int dim)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int di = (threadIdx.x + blockIdx.y * blockDim.x) * vec_size;
        const int ti = blockIdx.x;
        const int gi = group_ids ? group_ids[ti] : 0;

        if (di >= dim) {
            return;
        }

        using Vec = Array<T, vec_size>;

        Vec gate_bias{}, up_bias{};
        Ldg(gate_bias, &bias[gi * stride + di]);
        Ldg(up_bias, &bias[gi * stride + dim + di]);

        Vec gate, up;
        Load(gate, &gate_up[ti * stride + di]);
        Load(up, &gate_up[ti * stride + dim + di]);

        {
            using namespace ops;
            gate = gate + gate_bias;
            up   = up + up_bias;
        }

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            gate[i] = activation(gate[i], up[i]);
        }

        Store(&gate_up[ti * stride + di], gate);
    }
}

void Activation(Tensor&             gate_up,  //
                const Tensor&       bias,
                const Buffer_<int>& group_ids,
                ActivationType      type,
                cudaStream_t        stream)
{
    const int num = gate_up.shape(0);
    const int dim = gate_up.shape(1) / 2;

    if (!bias) {
        Activation(gate_up.slice({0, 0}, {-1, dim}),  //
                   gate_up.slice({0, dim}, {-1, -1}),
                   type,
                   stream);
        return;
    }

    TM_CHECK_EQ(gate_up.shape(-1), bias.shape(-1));

    auto invoke = [&](auto t, auto act) {
        using T = decltype(t);

        constexpr int vec_size = 4;
        constexpr int threads  = 512;

        const dim3 blocks(num, cdiv(dim, threads * vec_size));

        ActivationKernel<vec_size><<<blocks, threads, 0, stream>>>(gate_up.data<T>(),  //
                                                                   bias.data_or((T*)nullptr),
                                                                   group_ids.data_or(nullptr),
                                                                   gate_up.stride(0),
                                                                   act,
                                                                   num,
                                                                   dim);
    };

    auto dispatch = [&](auto t) {
        using T = decltype(t);
        if (type == ActivationType::kSilu) {
            return invoke(t, Silu<T>{});
        }
        else if (type == ActivationType::kSiluGptOss) {
            return invoke(t, SiluGptOss<T>{});
        }
        else {
            TM_LOG_FATAL("unknown activation type: {}", (int)type);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(gate_up.dtype(), dispatch);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<int vec_size, typename T, template<typename> class Activation>
__global__ void AddBiasActivationKernel(T* data, const T* __restrict__ bias, int64_t stride, int num, int dim)
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

void invokeAddBiasActivation(Tensor& x, const Tensor& bias, ActivationType type, cudaStream_t stream)
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
                AddBiasActivationKernel<vec_size, T, GeluActivation>
                    <<<grid, threads, 0, stream>>>(x.data<T>(), bias.data_or((T*)nullptr), stride, num, dim);
            }
            else if (type == ActivationType::kGelu) {
                AddBiasActivationKernel<vec_size, T, GeluErfActivation>
                    <<<grid, threads, 0, stream>>>(x.data<T>(), bias.data_or((T*)nullptr), stride, num, dim);
            }
            else {
                TM_LOG_FATAL("unsupported add-bias activation type: {}", (int)type);
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
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
