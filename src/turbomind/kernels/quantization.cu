

#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/quantization.cuh"
#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/kernels/attention/quantization.h"

namespace turbomind {

template<int vec_size, int group_size, class Tout, class Tscale, class T>
__global__ void quant_symm_row(
    Tout* out, int out_ld, Tscale* scales, int scales_ld, const T* src, int src_ld, int num, int dim, Tscale qmax)
{
#if TURBOMIND_ARCH_SM90
    static_assert(group_size % vec_size == 0);
    constexpr int threads = group_size / vec_size;
    const int     dim1    = round_up(dim, WARP_SIZE * vec_size);
    for (int ti = blockIdx.x; ti < num; ti += gridDim.x) {
        for (int di = threadIdx.x * vec_size; di < dim1; di += blockDim.x * vec_size) {
            Array<T, vec_size> vec{};
            if (di < dim) {
                Ldg(vec, src + ti * src_ld + di);
            }
            auto         absmax    = fmaxf(static_cast<Tscale>(find_absmax<threads>(vec)), 1e-8f);
            const Tscale scale     = absmax / qmax;
            const Tscale inv_scale = qmax / absmax;
            if (threadIdx.x % threads == 0 && di < dim) {
                // column-major
                scales[(di / group_size) * scales_ld + ti] = scale;
            }
            Array<Tout, vec_size> tmp;
            PRAGMA_UNROLL
            for (int c = 0; c < vec_size; ++c) {
                tmp[c] = Tout(static_cast<Tscale>(vec[c]) * inv_scale);
            }
            if (di < dim) {
                Store(out + ti * out_ld + di, tmp);
            }
        }
    }
#endif
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

    const int s_dim = cdiv<ssize_t>(dim, group_size);

    if (!scale) {
        scale = Tensor_<Tscale>({{s_dim, num}, {aligned_num, 1}}, kDEVICE);
    }
    else {
        TM_CHECK(std::make_tuple(s_dim, num) == scale.shapes(0, 1));
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
#if TURBOMIND_ARCH_SM90
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
#endif
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
#if TURBOMIND_ARCH_SM90
    static_assert(block_size % vec_size == 0);
    constexpr int threads = block_size / vec_size;

    static_assert(cta_size % threads == 0);
    constexpr int rows = cta_size / threads;

    constexpr int S = cdiv(block_size, rows);

    using BlockReduce = cub::BlockReduce<T, cta_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T                                 shared_inv_scale;

    const int row = threadIdx.x / threads;
    const int col = threadIdx.x % threads;
    const int ti  = blockIdx.x * block_size;
    const int di  = blockIdx.y * block_size + col * vec_size;

    T                  absmax{};
    Array<T, vec_size> xs[S]{};
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        if (auto r = ti + s * rows + row; r < num && di < dim) {
            Ldg(xs[s], src + (int64_t)r * dim + di);
        }
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            absmax = __hmax(absmax, __habs(xs[s][i]));
        }
    }

    absmax = BlockReduce{temp_storage}.Reduce(absmax, [](auto a, auto b) { return __hmax(a, b); });
    if (threadIdx.x == 0) {
        auto maxval                                 = fmaxf(static_cast<Tscale>(absmax), 1e-8f);
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
        if (auto r = ti + s * rows + row; r < num && di < dim) {
            Store(out + (int64_t)r * dim + di, ys[s]);
        }
    }
#endif
}

void QuantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> scale_, const Tensor& src, cudaStream_t st)
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

    auto& out   = out_.get();
    auto& scale = scale_.get();

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
#if TURBOMIND_ARCH_SM90
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
        if (ti < num && di < dim) {
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
#endif
}

void DequantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> src_, const Tensor& scale, cudaStream_t st)
{
    using T      = fp8_e4m3_t;
    using Tout   = bfloat16_t;
    using Tscale = float;

    constexpr int block_size = 128;
    constexpr int vec_size   = 8;

    auto& out = out_.get();
    auto& src = src_.get();

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

template<int vec_size, int bits, class Q, class S, class X>
__global__ void QuantizeFloatAsymmKernel(Q*            q,
                                         S*            s,
                                         S*            z,
                                         X*            d,
                                         const X*      x,
                                         Array<int, 2> stride_q,
                                         Array<int, 2> stride_s,
                                         Array<int, 2> stride_d,
                                         Array<int, 2> stride_x,
                                         int           M,
                                         int           K,
                                         int           G)
{
    static_assert(bits <= bitsof<Q>);

    int m = blockIdx.x;
    int k = threadIdx.x + blockIdx.y * blockDim.x;

    k *= vec_size;

    const int threads_per_group = G / vec_size;
    const int warp_k            = WARP_SIZE * vec_size;

    for (; k < round_up(K, warp_k); k += gridDim.y * blockDim.x * vec_size) {

        Array<float, vec_size> f_vec;

        float minval = std::numeric_limits<float>::infinity();
        float maxval = -minval;

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            if (k + i < K) {
                f_vec[i] = x[stride_x[0] * m + stride_x[1] * (k + i)];
                minval   = fminf(minval, f_vec[i]);
                maxval   = fmaxf(maxval, f_vec[i]);
            }
        }

        for (int offset = threads_per_group / 2; offset >= 1; offset /= 2) {
            minval = fminf(minval, __shfl_xor_sync((uint32_t)-1, minval, offset));
            maxval = fmaxf(maxval, __shfl_xor_sync((uint32_t)-1, maxval, offset));
        }

        constexpr int max_q = (1 << bits) - 1;

        auto clamp = [](int x, int a, int b) { return max(a, min(b, x)); };

        auto scale = fdividef(fmaxf(maxval - minval, 1e-5f), max_q);
        int  zero  = clamp(-round<int32_t>(minval / scale), 0, max_q);

        Array<Q, vec_size> q_vec;
        Array<X, vec_size> d_vec;
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            q_vec[i] = clamp(round<int32_t>(f_vec[i] / scale) + zero, 0, max_q);
            d_vec[i] = (X)((int)q_vec[i] - zero) * (X)scale;
        }

        if (k < K) {
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                if (k + i < K) {
                    q[stride_q[0] * m + stride_q[1] * (k + i)] = q_vec[i];
                    d[stride_d[0] * m + stride_d[1] * (k + i)] = d_vec[i];
                }
            }
            if (threadIdx.x % threads_per_group == 0) {
                s[stride_s[0] * m + stride_s[1] * (k / G)] = (S)scale;
                z[stride_s[0] * m + stride_s[1] * (k / G)] = (S)zero;
            }
        }
    }
}

template<int start_bit, int end_bit, class D, class T>
__global__ void Pack1DKernel(D* d, const T* s, int n)
{
    constexpr int bits     = end_bit - start_bit;
    constexpr int vec_size = bitsof<D> / bits;

    const auto idx = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;

    if (idx * vec_size >= n) {
        return;
    }

    Array<T, vec_size> s_vec;

    Load(s_vec, &s[idx * vec_size]);

    constexpr T mask = ((1 << bits) - 1) << start_bit;

    D pack{};

    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        pack |= ((s_vec[i] & mask) >> start_bit) << (i * bits);
    }

    d[idx] = pack;
}

void QuantizeAsymm(Tensor   quant,    // (m,k)
                   Tensor   scales,   // (m,k/g)
                   Tensor   zeros,    // (m,k/g)
                   Tensor   dequant,  // (m,k)
                   Tensor   src,      // (m,k)
                   DataType data_type,
                   int      group_size)
{
    // std::cout << quant << std::endl;
    // std::cout << scales << std::endl;
    // std::cout << zeros << std::endl;
    // std::cout << dequant << std::endl;
    // std::cout << src << std::endl;

    TM_CHECK(scales.layout() == zeros.layout());
    TM_CHECK(quant.shape() == dequant.shape());
    TM_CHECK(quant.size() == quant.layout().cosize());

    Tensor_<uint16_t> u16 = empty_like(quant, kUint16);

    auto stream = core::Context::stream().handle();

    auto stride_2d = [](const Tensor& t) {
        TM_CHECK_EQ(t.ndim(), 2);
        auto [a, b] = t.strides(0, 1);
        return Array<int, 2>{(int)a, (int)b};
    };

    const int m = src.shape(0);
    const int k = src.shape(1);

    // std::cout << "m" << m << "k" << k << "\n";

    using T = half;
    using Q = uint4_t;

    constexpr int bits = bitsof<Q>;
    constexpr int vec  = 16 / sizeof(half);

    const int threads = round_up(std::min(cdiv(k, vec), 1024), WARP_SIZE);
    QuantizeFloatAsymmKernel<vec, bits><<<m, threads, 0, stream>>>(u16.data(),
                                                                   scales.data<T>(),
                                                                   zeros.data<T>(),
                                                                   dequant.data<T>(),
                                                                   src.data<T>(),
                                                                   stride_2d(u16),
                                                                   stride_2d(scales),
                                                                   stride_2d(dequant),
                                                                   stride_2d(src),
                                                                   m,
                                                                   k,
                                                                   group_size);

    Pack1DKernel<0, bits><<<cdiv((int)quant.size(), 512), 512, 0, stream>>>(
        (uint32_t*)quant.raw_data(), (uint16_t*)u16.raw_data(), quant.size());
}

}  // namespace turbomind
