
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

template<class T>
struct SiluGptOss {
    __device__ T operator()(T gate, T up) const noexcept
    {
        if constexpr (TURBOMIND_ARCH_HAS_BF16 || data_type_v<T> != kBfloat16) {
            gate = __hmin((T)7.f, gate);
            up   = __hmax((T)-7.f, __hmin((T)7.f, up));
            return static_cast<T>(fdividef((float)gate, 1.f + expf((float)-gate * 1.702f)) * (1.f + (float)up));
        }
        else {
            return {};
        }
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
            TM_CHECK(0) << "unknown activation type: " << (int)type;
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(gate.dtype(), dispatch);
}

}  // namespace turbomind
