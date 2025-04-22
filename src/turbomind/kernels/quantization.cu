#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/quantization.cuh"
#include "src/turbomind/kernels/quantization.h"

#include <cub/block/block_reduce.cuh>

namespace turbomind {

template<int vec_size, int group_size, class Tout, class Tscale, class T>
__global__ void quant_symm_row(Tout* out, Tscale* scales, const T* src, Tscale qmax, int64_t n)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    n /= vec_size;
    static_assert(group_size % vec_size == 0);
    constexpr int threads = group_size / vec_size;
    for (int64_t i = idx; i < n; i += gridDim.x * blockDim.x) {
        Array<T, vec_size> vec;
        Ldg(vec, src + i * vec_size);
        auto         absmax    = static_cast<Tscale>(find_absmax<threads>(vec));
        const Tscale scale     = absmax / qmax;
        const Tscale inv_scale = qmax / absmax;
        if (threadIdx.x % threads == 0) {
            scales[i / threads] = scale;
        }
        Array<Tout, vec_size> tmp;
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            tmp[c] = Tout(static_cast<Tscale>(vec[c]) * inv_scale);
        }
        Store(out + i * vec_size, tmp);
    }
}

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st)
{
    TM_CHECK_EQ(src.ndim(), 2);
    TM_CHECK(src.is_contiguous());

    const auto [num, dim] = src.shapes(0, 1);

    using T      = bfloat16_t;
    using Tout   = fp8_e4m3_t;
    using Tscale = float;

    constexpr int group_size = 128;
    constexpr int vec_size   = 8;

    if (!out) {
        out = Tensor_<Tout>{src.layout(), kDEVICE};
    }
    else {
        TM_CHECK(out.layout() == src.layout());
    }

    if (!scale) {
        scale = Tensor_<Tscale>({num, dim / group_size}, kDEVICE);
    }
    else {
        TM_CHECK(std::make_tuple(num, dim / group_size) == scale.shapes(0, 1));
    }

    constexpr int block_dim = 512;
    int           grid_dim  = (int)cdiv<int64_t>(src.size(), block_dim * vec_size);

    quant_symm_row<vec_size, group_size><<<grid_dim, block_dim, 0, st>>>(out.data<Tout>(),  //
                                                                         scale.data<Tscale>(),
                                                                         src.data<T>(),
                                                                         448.f,
                                                                         src.size());
}

template<int vec_size, int group_size, class Tout, class Tscale, class T>
__global__ void dequant_symm_row(Tout* out, const T* src, const Tscale* scales, int64_t n)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    n /= vec_size;
    static_assert(group_size % vec_size == 0);
    constexpr int threads = group_size / vec_size;
    for (int64_t i = idx; i < n; i += gridDim.x * blockDim.x) {
        Array<T, vec_size> vec;
        Ldg(vec, src + i * vec_size);
        const auto            scale = __ldg(&scales[i / threads]);
        Array<Tout, vec_size> tmp;
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            tmp[c] = Tout(static_cast<Tscale>(vec[c]) * scale);
        }
        Store(out + i * vec_size, tmp);
    }
}

void DequantizeSymm(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st)
{
    using T      = fp8_e4m3_t;
    using Tout   = bfloat16_t;
    using Tscale = float;

    if (!out) {
        out = Tensor_<Tout>{src.layout(), kDEVICE};
    }
    else {
        TM_CHECK(out.layout() == src.layout());
    }

    constexpr int group_size = 128;
    constexpr int vec_size   = 8;

    constexpr int block_dim = 512;
    int           grid_dim  = (int)cdiv<int64_t>(src.size(), block_dim * vec_size);

    dequant_symm_row<vec_size, group_size, Tout, Tscale, T><<<grid_dim, block_dim, 0, st>>>(out.data<Tout>(),  //
                                                                                            src.data<T>(),
                                                                                            scale.data<Tscale>(),
                                                                                            src.size());
}

template<int vec_size, int cta_size, int block_size, class Tout, class Tscale, class T>
__global__ void quant_symm_block(Tout* out, Tscale* scales, const T* src, Tscale qmax, int num, int dim)
{
    static_assert(block_size % vec_size == 0);
    constexpr int threads = block_size / vec_size;

    static_assert(cta_size % threads == 0);
    constexpr int rows = cta_size / threads;

    constexpr int S = cdiv(block_size, rows);

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    using BlockReduce = cub::BlockReduce<T, cta_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T                                 shared_inv_scale;

    Array<T, vec_size>    xs[S];
    Array<Tout, vec_size> ys[S];
    for (int tx = bx; tx < dim / block_size; tx += gridDim.x) {
        for (int ty = by; ty < num / block_size; ty += gridDim.y) {
            const int col = threadIdx.x % threads;
            const int row = threadIdx.x / threads;
            T         absmax{};
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                Ldg(xs[s], src + int64_t(ty * block_size + s * rows + row) * dim + tx * block_size + col * vec_size);
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    absmax = __hmax(absmax, __habs(xs[s][i]));
                }
            }
            absmax = BlockReduce{temp_storage}.Reduce(absmax, [](auto a, auto b) { return __hmax(a, b); });
            if (threadIdx.x == 0) {
                auto maxval                        = static_cast<Tscale>(absmax);
                scales[ty * dim / block_size + tx] = maxval / qmax;
                shared_inv_scale                   = qmax / maxval;
            }
            __syncthreads();
            const Tscale inv_scale = shared_inv_scale;
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    ys[s][i] = Tout(static_cast<Tscale>(xs[s][i]) * inv_scale);
                }
                Store(out + int64_t(ty * block_size + s * rows + row) * dim + tx * block_size + col * vec_size, ys[s]);
            }
        }
    }
}

void QuantizeSymmBlock(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st)
{
    TM_CHECK(src.is_contiguous());
    TM_CHECK_EQ(src.ndim(), 2);

    using T      = bfloat16_t;
    using Tout   = fp8_e4m3_t;
    using Tscale = float;

    constexpr int block_size = 128;
    constexpr int vec_size   = 8;

    const auto [num, dim] = src.shapes(0, 1);

    constexpr int cta_size = 1024;
    const dim3    grid(dim / block_size, num / block_size);

    if (!out) {
        out = Tensor_<Tout>{src.layout(), kDEVICE};
    }
    else {
        TM_CHECK(out.layout() == src.layout());
    }

    if (!scale) {
        scale = Tensor_<Tscale>({num / block_size, dim / block_size}, kDEVICE);
    }
    else {
        TM_CHECK(std::make_tuple(num / block_size, dim / block_size) == scale.shapes(0, 1));
    }

    quant_symm_block<vec_size, cta_size, block_size><<<grid, cta_size, 0, st>>>(  //
        out.data<Tout>(),
        scale.data<Tscale>(),
        src.data<T>(),
        448.f,
        num,
        dim);
}

template<int vec_size, int cta_size, int block_size, class Tout, class Tscale, class T>
__global__ void dequant_symm_block(Tout* out, const T* src, const Tscale* scales, int num, int dim)
{
    static_assert(block_size % vec_size == 0);
    constexpr int threads = block_size / vec_size;
    static_assert(cta_size % threads == 0);
    constexpr int rows = cta_size / threads;
    constexpr int S    = cdiv(block_size, rows);
    const int     bx   = blockIdx.x;
    const int     by   = blockIdx.y;
    for (int tx = bx; tx < dim / block_size; tx += gridDim.x) {
        for (int ty = by; ty < num / block_size; ty += gridDim.y) {
            const int  col   = threadIdx.x % threads;
            const int  row   = threadIdx.x / threads;
            const auto scale = __ldg(&scales[ty * dim / block_size + tx]);
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                Array<T, vec_size> x;
                Ldg(x, src + int64_t(ty * block_size + s * rows + row) * dim + tx * block_size + col * vec_size);
                Array<Tout, vec_size> y;
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    y[i] = Tout(static_cast<Tscale>(x[i]) * scale);
                }
                Store(out + int64_t(ty * block_size + s * rows + row) * dim + tx * block_size + col * vec_size, y);
            }
        }
    }
}

void DequantizeSymmBlock(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st)
{
    using T      = fp8_e4m3_t;
    using Tout   = bfloat16_t;
    using Tscale = float;

    constexpr int block_size = 128;
    constexpr int vec_size   = 8;

    if (!out) {
        out = Tensor_<Tout>{src.layout(), kDEVICE};
    }
    else {
        TM_CHECK(out.layout() == src.layout());
    }

    const auto [num, dim] = src.shapes(0, 1);

    constexpr int cta_size = 1024;
    const dim3    grid(dim / block_size, num / block_size);

    dequant_symm_block<vec_size, cta_size, block_size><<<grid, cta_size, 0, st>>>(  //
        out.data<Tout>(),
        src.data<T>(),
        scale.data<Tscale>(),
        num,
        dim);
}

}  // namespace turbomind