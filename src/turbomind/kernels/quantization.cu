
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cub/block/block_reduce.cuh>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/quantization.cuh"
#include "src/turbomind/kernels/quantization.h"

namespace turbomind {

template<int vec_size, int group_size, class Tout, class Tscale, class T>
__global__ void quant_symm_row(
    Tout* out, int out_ld, Tscale* scales, int scales_ld, const T* src, int src_ld, int num, int dim, Tscale qmax)
{
    static_assert(group_size % vec_size == 0);
    constexpr int threads = group_size / vec_size;
    for (int ti = blockIdx.x; ti < num; ti += gridDim.x) {
        for (int di = threadIdx.x * vec_size; di < dim; di += blockDim.x * vec_size) {
            Array<T, vec_size> vec;
            Ldg(vec, src + ti * src_ld + di);
            auto         absmax    = static_cast<Tscale>(find_absmax<threads>(vec));
            const Tscale scale     = absmax / qmax;
            const Tscale inv_scale = qmax / absmax;
            if (threadIdx.x % threads == 0) {
                // column-major
                scales[(di / group_size) * scales_ld + ti] = scale;
            }
            Array<Tout, vec_size> tmp;
            PRAGMA_UNROLL
            for (int c = 0; c < vec_size; ++c) {
                tmp[c] = Tout(static_cast<Tscale>(vec[c]) * inv_scale);
            }
            Store(out + ti * out_ld + di, tmp);
        }
    }
}

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st)
{
    TM_CHECK_EQ(src.ndim(), 2);
    TM_CHECK_EQ(src.stride(1), 1);  // row-major

    const auto [num, dim] = src.shapes(0, 1);

    using T      = bfloat16_t;
    using Tout   = fp8_e4m3_t;
    using Tscale = float;

    constexpr int group_size = 128;
    constexpr int vec_size   = 8;

    constexpr int alignment = 16 / sizeof(Tscale);

    if (!out) {
        out = Tensor_<Tout>{src.shape(), kDEVICE};
    }
    else {
        TM_CHECK(out.shape() == src.shape());
    }

    const int aligned_num = round_up<int>(num, alignment);

    if (!scale) {
        scale = Tensor_<Tscale>({{dim / group_size, num}, {aligned_num, 1}}, kDEVICE);
    }
    else {
        TM_CHECK(std::make_tuple(dim / group_size, num) == scale.shapes(0, 1));
        TM_CHECK(scale.stride(1) == 1);
        TM_CHECK(scale.stride(0) % alignment == 0);
    }

    constexpr int block_dim = 512;

    quant_symm_row<vec_size, group_size><<<num, block_dim, 0, st>>>(out.data<Tout>(),  //
                                                                    out.stride(0),
                                                                    scale.data<Tscale>(),
                                                                    scale.stride(0),
                                                                    src.data<T>(),
                                                                    src.stride(0),
                                                                    num,
                                                                    dim,
                                                                    448.f);
}

template<int vec_size, int group_size, class Tout, class Tscale, class T>
__global__ void
dequant_symm_row(Tout* out, int out_ld, const T* src, int src_ld, const Tscale* scales, int scales_ld, int num, int dim)
{
    static_assert(group_size % vec_size == 0);
    for (int ti = blockIdx.x; ti < num; ti += gridDim.x) {
        for (int di = threadIdx.x * vec_size; di < dim; di += blockDim.x * vec_size) {
            Array<T, vec_size> vec;
            Ldg(vec, src + ti * src_ld + di);
            const auto            scale = __ldg(&scales[(di / group_size) * scales_ld + ti]);
            Array<Tout, vec_size> tmp;
            PRAGMA_UNROLL
            for (int c = 0; c < vec_size; ++c) {
                tmp[c] = Tout(static_cast<Tscale>(vec[c]) * scale);
            }
            Store(out + ti * out_ld + di, tmp);
        }
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

    auto [num, dim] = src.shapes(0, 1);

    constexpr int group_size = 128;
    constexpr int vec_size   = 8;

    constexpr int block_dim = 512;

    dequant_symm_row<vec_size, group_size, Tout, Tscale, T><<<num, block_dim, 0, st>>>(out.data<Tout>(),  //
                                                                                       out.stride(0),
                                                                                       src.data<T>(),
                                                                                       src.stride(0),
                                                                                       scale.data<Tscale>(),
                                                                                       scale.stride(0),
                                                                                       num,
                                                                                       dim);
}

template<int vec_size, int cta_size, int block_size, class Tout, class Tscale, class T>
__global__ void quant_symm_block(Tout* out, Tscale* scales, const T* src, Tscale qmax, int num, int dim)
{
    static_assert(block_size % vec_size == 0);
    constexpr int threads = block_size / vec_size;

    static_assert(cta_size % threads == 0);
    constexpr int rows = cta_size / threads;

    constexpr int S = cdiv(block_size, rows);

    using BlockReduce = cub::BlockReduce<T, cta_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T                                 shared_inv_scale;

    const int ti  = blockIdx.x * block_size;
    const int di  = blockIdx.y * block_size;
    const int col = threadIdx.x % threads;
    const int row = threadIdx.x / threads;

    T                  absmax{};
    Array<T, vec_size> xs[S]{};
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        if (auto r = ti + s * rows + row; r < num) {
            Ldg(xs[s], src + (int64_t)r * dim + di + col * vec_size);
        }
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            absmax = __hmax(absmax, __habs(xs[s][i]));
        }
    }

    absmax = BlockReduce{temp_storage}.Reduce(absmax, [](auto a, auto b) { return __hmax(a, b); });
    if (threadIdx.x == 0) {
        auto maxval                                 = static_cast<Tscale>(absmax);
        scales[blockIdx.x * gridDim.y + blockIdx.y] = maxval / qmax;
        shared_inv_scale                            = qmax / maxval;
    }
    __syncthreads();
    const Tscale inv_scale = shared_inv_scale;

    Array<Tout, vec_size> ys[S];
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            ys[s][i] = Tout(static_cast<Tscale>(xs[s][i]) * inv_scale);
        }
        if (auto r = ti + s * rows + row; r < num) {
            Store(out + (int64_t)r * dim + di + col * vec_size, ys[s]);
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

    const int bnum = cdiv<int>(num, block_size);
    const int bdim = cdiv<int>(dim, block_size);

    constexpr int cta_size = 1024;
    const dim3    grid(bnum, bdim);

    if (!out) {
        out = Tensor_<Tout>{src.layout(), kDEVICE};
    }
    else {
        TM_CHECK(out.layout() == src.layout());
    }

    if (!scale) {
        scale = Tensor_<Tscale>({bnum, bdim}, kDEVICE);
    }
    else {
        TM_CHECK(std::make_tuple(bnum, bdim) == scale.shapes(0, 1));
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
    constexpr int rows  = cta_size / threads;
    constexpr int S     = cdiv(block_size, rows);
    const int     col   = threadIdx.x % threads;
    const int     row   = threadIdx.x / threads;
    const auto    scale = __ldg(&scales[blockIdx.x * gridDim.y + blockIdx.y]);
    const auto    di    = blockIdx.y * block_size + col * vec_size;
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        const auto ti = blockIdx.x * block_size + s * rows + row;
        if (ti < num) {
            Array<T, vec_size> x;
            Ldg(x, src + (int64_t)ti * dim + di);
            Array<Tout, vec_size> y;
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                y[i] = Tout(static_cast<Tscale>(x[i]) * scale);
            }
            Store(out + (int64_t)ti * dim + di, y);
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

    const int bnum = cdiv<int>(num, block_size);
    const int bdim = cdiv<int>(dim, block_size);

    constexpr int cta_size = 1024;
    const dim3    grid(bnum, bdim);

    dequant_symm_block<vec_size, cta_size, block_size><<<grid, cta_size, 0, st>>>(  //
        out.data<Tout>(),
        src.data<T>(),
        scale.data<Tscale>(),
        num,
        dim);
}

}  // namespace turbomind