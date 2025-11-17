

#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <type_traits>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/floating_point.h"
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
    if constexpr (TURBOMIND_ARCH_BF16_GUARD(data_type_v<T>)) {
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
    }
}

void QuantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> scale_, const Tensor& src, cudaStream_t st)
{
    TM_CHECK(src.is_contiguous());
    TM_CHECK_EQ(src.ndim(), 2);

    auto invoke = [&](auto t) {
        using T      = decltype(t);
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
    };

    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), invoke);
}

template<int vec_size, int cta_size, int block_size, class Tout, class Tscale, class T>
__global__ void dequant_symm_block(Tout* out, const T* src, const Tscale* scales, int num, int dim)
{
    if constexpr (TURBOMIND_ARCH_BF16_GUARD(data_type_v<T>)) {
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
    }
}

void DequantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> src_, const Tensor& scale, cudaStream_t st)
{
    auto invoke = [&](auto tout) {
        using T      = fp8_e4m3_t;
        using Tout   = decltype(tout);
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
    };

    if (!out_.get()) {
        return invoke(nv_bfloat16{});
    }

    TM_DISPATCH_PRIMARY_DTYPES(out_.get().dtype(), invoke);
}

template<int start_bit, int end_bit, class D, class T>
__global__ void Compact1D_Kernel(D* d, const T* s, int n)
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

template<class T_, int bits_, class Q_>
struct IntegralQuantizer {

    using T = T_;
    using Q = Q_;

    using Scale = T;
    using Zero  = T;

    static constexpr int bits  = bits_;
    static constexpr int max_q = (1 << bits) - 1;

    template<class T, int N, class R>
    __device__ void operator()(const Array<T, N>&    x,  //
                               const Array<bool, N>& pred,
                               const R&              rbits,
                               Array<Q, N>&          q,
                               Array<T, N>&          d,
                               T&                    scale,
                               T&                    zero,
                               int                   threads) const
    {
        auto f = cast<float>(x);

        float minval = std::numeric_limits<float>::infinity();
        float maxval = -minval;

        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            if (pred[i]) {
                minval = fminf(minval, f[i]);
                maxval = fmaxf(maxval, f[i]);
            }
        }

        for (int offset = threads / 2; offset >= 1; offset /= 2) {
            minval = fminf(minval, __shfl_xor_sync((uint32_t)-1, minval, offset));
            maxval = fmaxf(maxval, __shfl_xor_sync((uint32_t)-1, maxval, offset));
        }

        auto clamp = [](int x, int a, int b) { return max(a, min(b, x)); };

        float scale_ = fmaxf(maxval - minval, 1e-5f) / (float)max_q;
        int   zero_  = clamp(-round<int32_t>(minval / scale_), 0, max_q);

        scale = (T)scale_;
        zero  = (T)zero_;

        // T sz = zero_ * scale_;

        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            q[i] = clamp(round<int32_t>(f[i] / scale_) + zero_, 0, max_q);
            d[i] = (T)((int)q[i] - zero_) * (T)scale_;
            // d[i] = __hfma((T)q[i], (T)scale_, -sz);
        }
    }
};

template<class T_, int E, int M, class Q_>
struct FloatingPointQuantizer {

    using T = T_;
    using Q = Q_;

    using Scale = uint8_t;
    using Zero  = void;

    using traits = FloatingPoint<E, M>;

    static constexpr int bits = traits::bits;

    float pre_rounding_scale_;

    __host__ __device__ FloatingPointQuantizer(float pre_rounding_scale = 1.f): pre_rounding_scale_{pre_rounding_scale}
    {
    }

    template<int N, class Z, class R>
    __device__ void operator()(const Array<T, N>&    x,  //
                               const Array<bool, N>& pred,
                               const R&              rbits,
                               Array<Q, N>&          q,
                               Array<T, N>&          d,
                               Scale&                scale,
                               Z                     ignore,
                               int                   threads) const
    {
        auto f = cast<float>(x);

        float absmax = 0.f;

        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            if (pred[i]) {
                absmax = fmaxf(absmax, fabsf(f[i]));
            }
        }

        for (int offset = threads / 2; offset >= 1; offset /= 2) {
            absmax = fmaxf(absmax, __shfl_xor_sync((uint32_t)-1, absmax, offset));
        }

        auto get_exponent = [](float x) -> int { return (__float_as_uint(x) >> 23U) & 0xFFU; };

        int scale_i32 = get_exponent(absmax) - (traits::exponent_bias + 1);

        // int scale_i32 = 127;

        if (scale_i32 < 0) {  // absmax(group) < 2*2^-125, flush to zero
            scale_i32 = 0;
            f         = {};
        }

        scale = scale_i32;

        float scale_f32 = __uint_as_float((uint32_t)scale_i32 << 23U);

        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            q[i] = traits::from_f32((f[i] * pre_rounding_scale_) / scale_f32, rbits[i]);
            d[i] = (traits::to_f32(q[i]) * scale_f32) / pre_rounding_scale_;
        }
    }
};

template<int vec_size,
         class Quantizer,
         class T = typename Quantizer::T,
         class Q = typename Quantizer::Q,
         class S = typename Quantizer::Scale,
         class Z = typename Quantizer::Zero>
__global__ void QuantizeGroupwise_Kernel(Quantizer       quantizer,
                                         Q*              q,
                                         S*              s,
                                         Z*              z,
                                         T*              d,
                                         const T*        x,
                                         const unsigned* r,
                                         Array<int, 2>   stride_q,
                                         Array<int, 2>   stride_s,
                                         Array<int, 2>   stride_d,
                                         Array<int, 2>   stride_x,
                                         int             M,
                                         int             K,
                                         int             G)
{
    if constexpr (TURBOMIND_ARCH_BF16_GUARD(data_type_v<T>)) {
        static constexpr bool has_zero = !std::is_void_v<Z>;

        int m = blockIdx.x;
        int k = threadIdx.x + blockIdx.y * blockDim.x;

        const int threads_per_group = G / vec_size;
        const int warp_k            = WARP_SIZE * vec_size;

        k *= vec_size;

        for (; k < round_up(K, warp_k); k += gridDim.y * blockDim.x * vec_size) {

            Array<T, vec_size>    x_vec;
            Array<bool, vec_size> p_vec;

            Array<unsigned, vec_size> r_vec;

            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                p_vec[i] = k + i < K;
                x_vec[i] = p_vec[i] ? x[stride_x[0] * m + stride_x[1] * (k + i)] : T{0};
                if (r) {
                    r_vec[i] = p_vec[i] ? r[m * K + k] : 0;
                }
            }

            Array<Q, vec_size> q_vec;
            Array<T, vec_size> d_vec;

            S                                    scale;
            std::conditional_t<has_zero, Z, int> zero{};

            auto invoke = [&](auto rbits) {
                quantizer(x_vec, p_vec, rbits, q_vec, d_vec, scale, zero, threads_per_group);
            };

            r ? invoke(r_vec) : invoke(Array<char, vec_size>{});

            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                const auto idx = stride_q[0] * m + stride_q[1] * (k + i);
                if (p_vec[i]) {
                    q[idx] = q_vec[i];
                    d[idx] = d_vec[i];
                }
            }
            if (threadIdx.x % threads_per_group == 0) {
                const auto idx = stride_s[0] * m + stride_s[1] * (k / G);
                if (p_vec[0]) {
                    s[idx] = (S)scale;
                    if constexpr (has_zero) {
                        z[idx] = (S)zero;
                    }
                }
            }
        }
    }
}

void QuantizeGroupwise(Tensor            quant,    // (m,k)
                       Tensor            scales,   // (m,k/g)
                       Tensor            zeros,    // (m,k/g)
                       Tensor            dequant,  // (m,k)
                       Tensor            src,      // (m,k)
                       Buffer_<unsigned> rbits,    // (m*k)
                       int               group_size)
{
    // std::cout << quant << std::endl;
    // std::cout << scales << std::endl;
    // std::cout << zeros << std::endl;
    // std::cout << dequant << std::endl;
    // std::cout << src << std::endl;

    if (zeros) {
        TM_CHECK(scales.layout() == zeros.layout());
    }
    TM_CHECK(quant.shape() == dequant.shape());
    TM_CHECK(quant.size() == quant.layout().cosize());

    auto stream = core::Context::stream().handle();

    auto stride_2d = [](const Tensor& t) {
        TM_CHECK_EQ(t.ndim(), 2);
        auto [a, b] = t.strides(0, 1);
        return Array<int, 2>{(int)a, (int)b};
    };

    const int m = src.shape(0);
    const int k = src.shape(1);

    // std::cout << "m" << m << "k" << k << "\n";

    auto invoke = [&](auto quantizer) {
        using Quantizer = decltype(quantizer);

        using T = typename Quantizer::T;
        using Q = typename Quantizer::Q;
        using S = typename Quantizer::Scale;
        using Z = typename Quantizer::Zero;

        constexpr int bits = Quantizer::bits;

        Tensor_<Q> proxy = empty_like(quant, data_type_v<Q>);

        constexpr int vec = 8;

        TM_CHECK((group_size & (group_size - 1)) == 0);
        TM_CHECK_GE(group_size, vec);
        TM_CHECK_LE(group_size, WARP_SIZE * vec);

        const int threads = round_up(std::min(cdiv(k, vec), 1024), WARP_SIZE);

        QuantizeGroupwise_Kernel<vec><<<m, threads, 0, stream>>>(quantizer,
                                                                 proxy.data(),
                                                                 scales.data<S>(),
                                                                 zeros.data_or((Z*)nullptr),
                                                                 dequant.data<T>(),
                                                                 src.data<T>(),
                                                                 rbits.data_or(nullptr),
                                                                 stride_2d(proxy),
                                                                 stride_2d(scales),
                                                                 stride_2d(dequant),
                                                                 stride_2d(src),
                                                                 m,
                                                                 k,
                                                                 group_size);

        Compact1D_Kernel<0, bits><<<cdiv((int)quant.size(), 512), 512, 0, stream>>>(
            (uint32_t*)quant.raw_data(), (Q*)proxy.raw_data(), quant.size());
    };

    if (0) {}
    else if (src.dtype() == kHalf && quant.dtype() == kUint4) {
        invoke(IntegralQuantizer<half_t, 4, uint16_t>{});
    }
    else if (src.dtype() == kBfloat16 && quant.dtype() == kFloat4_e2m1) {
        invoke(FloatingPointQuantizer<bfloat16_t, 2, 1, uint16_t>{});
    }
    else if (src.dtype() == kHalf && quant.dtype() == kFloat4_e2m1) {
        invoke(FloatingPointQuantizer<half_t, 2, 1, uint16_t>{});
    }
    else {
        TM_CHECK(0) << "Unsupported types: " << to_string(src.dtype()) << ", " << to_string(quant.dtype());
    }
}

}  // namespace turbomind
