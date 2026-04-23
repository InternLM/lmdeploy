
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

#include "src/turbomind/utils/cuda_utils.h"

#include <algorithm>

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
__global__ void ActivationKernel(T* gate_buf,
                                 const T* __restrict__ up_buf,
                                 Activation activation,
                                 int64_t    stride,
                                 const int* __restrict__ total_tokens_ptr,
                                 int token_num,
                                 int dim)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int total = total_tokens_ptr ? __ldg(total_tokens_ptr) : token_num;

        const int di = threadIdx.x + blockIdx.y * blockDim.x;

        dim /= vec_size;

        if (di >= dim) {
            return;
        }

        using Vec = Array<T, vec_size>;

        for (int ti = blockIdx.x; ti < total; ti += gridDim.x) {
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
}

void Activation(
    Ref<Tensor> gate_, const Tensor& up, ActivationType type, const int* total_tokens_ptr, cudaStream_t stream)
{
    auto& gate = gate_.get();

    TM_CHECK(gate.shape() == up.shape());

    int num, dim;
    std::tie(num, dim) = gate.shapes(0, 1);

    auto invoke = [&](auto t, auto act) {
        using T = decltype(t);

        constexpr int vec_size = 4;
        constexpr int threads  = 512;

        static const int sm_count = getSMCount();
        const int        grid_x   = std::min<int>(num, sm_count * 4);
        const dim3       blocks(grid_x, cdiv(dim, threads * vec_size));

        ActivationKernel<vec_size><<<blocks, threads, 0, stream>>>(gate.data<T>(),  //
                                                                   up.data<T>(),
                                                                   act,
                                                                   gate.stride(0),
                                                                   total_tokens_ptr,
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

template<int vec_size, class Activation, class T>
__global__ void ActivationKernel(T*         gate_up,
                                 const T*   bias,
                                 const int* group_ids,
                                 int64_t    stride,
                                 Activation activation,
                                 const int* __restrict__ total_tokens_ptr,
                                 int token_num,
                                 int dim)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int total = total_tokens_ptr ? __ldg(total_tokens_ptr) : token_num;

        const int di = (threadIdx.x + blockIdx.y * blockDim.x) * vec_size;

        if (di >= dim) {
            return;
        }

        using Vec = Array<T, vec_size>;

        for (int ti = blockIdx.x; ti < total; ti += gridDim.x) {
            const int gi = group_ids ? group_ids[ti] : 0;

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
}

void Activation(Tensor&             gate_up,  //
                const Tensor&       bias,
                const Buffer_<int>& group_ids,
                ActivationType      type,
                const int*          total_tokens_ptr,
                cudaStream_t        stream)
{
    const int num = gate_up.shape(0);
    const int dim = gate_up.shape(1) / 2;

    if (!bias) {
        Activation(gate_up.slice({0, 0}, {-1, dim}),  //
                   gate_up.slice({0, dim}, {-1, -1}),
                   type,
                   total_tokens_ptr,
                   stream);
        return;
    }

    TM_CHECK_EQ(gate_up.shape(-1), bias.shape(-1));

    auto invoke = [&](auto t, auto act) {
        using T = decltype(t);

        constexpr int vec_size = 4;
        constexpr int threads  = 512;

        static const int sm_count = getSMCount();
        const int        grid_x   = std::min<int>(num, sm_count * 4);
        const dim3       blocks(grid_x, cdiv(dim, threads * vec_size));

        ActivationKernel<vec_size><<<blocks, threads, 0, stream>>>(gate_up.data<T>(),  //
                                                                   bias.data_or((T*)nullptr),
                                                                   group_ids.data_or(nullptr),
                                                                   gate_up.stride(0),
                                                                   act,
                                                                   total_tokens_ptr,
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

    TM_DISPATCH_PRIMARY_DTYPES(gate_up.dtype(), dispatch);
}

}  // namespace turbomind
