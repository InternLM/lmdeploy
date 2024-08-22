// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace turbomind {

cublasHandle_t cublas_handle{};
cudaStream_t   cublas_stream{};

template<typename T>
void Compare(const T* src, const T* ref, size_t stride, int dims, int bsz, bool show, float rtol, float atol)
{
    float asums{};
    float rsums{};
    int   outliers{};
    for (int nn = 0; nn < bsz; ++nn) {
        float abs_diff_sum{};
        float rel_diff_sum{};
        for (int mm = 0; mm < dims; ++mm) {
            auto x = float(src[nn * stride + mm]);
            auto y = float(ref[nn * stride + mm]);
            // if (show) {
            //     std::cout << x << "\t" << y << std::endl;
            // }
            auto abs_diff = std::abs(x - y);
            auto rel_diff = abs_diff / (std::max(std::abs(y), std::abs(x)) + 1e-8f);
            if (!(abs_diff <= atol + rtol * std::abs(y))) {
                ++outliers;
                if (show) {
                    std::cout << nn << "," << mm << "\t" << x << "\t" << y << std::endl;
                }
            }
            abs_diff_sum += abs_diff;
            rel_diff_sum += rel_diff;
        }
        asums += abs_diff_sum / dims;
        rsums += rel_diff_sum / dims;
    }
    const float abs_diff = asums / bsz;
    const float rel_diff = rsums / bsz;
    const float outlier  = outliers / (float)bsz;
    std::cout << "abs_diff = " << abs_diff << " rel_diff = " << rel_diff << " outliers = " << outlier << std::endl;
}

template void
Compare(const half* src, const half* ref, size_t stride, int dims, int bsz, bool show, float rtol, float atol);
template void
Compare(const float* src, const float* ref, size_t stride, int dims, int bsz, bool show, float rtol, float atol);
#if ENABLE_BF16
template void Compare(const nv_bfloat16* src,
                      const nv_bfloat16* ref,
                      size_t             stride,
                      int                dims,
                      int                bsz,
                      bool               show,
                      float              rtol,
                      float              atol);
#endif

template<class T>
std::vector<float>
FastCompare(const T* src, const T* ref, int dims, int bsz, cudaStream_t stream, float rtol, float atol)
{
    auto       zip_iter = thrust::make_zip_iterator(src, ref);
    const auto count    = (size_t)dims * bsz;
    // nvcc-11.8: __host__ __device__ lambda can't be generic
    using Tuple = thrust::tuple<float, float, float, float, float, float, int64_t>;
    auto res    = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        zip_iter,
        zip_iter + count,
        [=] __device__(auto tup) {
            float   s        = thrust::get<0>(tup);
            float   r        = thrust::get<1>(tup);
            float   abs_diff = fabsf(s - r);
            float   abs_s    = fabsf(s);
            float   abs_r    = fabsf(r);
            float   rel_diff = abs_diff / (fmaxf(abs_r, abs_s) + 1e-8f);
            int64_t outlier  = !(abs_diff <= (atol + rtol * abs_r));
            return thrust::make_tuple(abs_s, abs_r, abs_diff, abs_diff, rel_diff, rel_diff, outlier);
        },
        thrust::make_tuple(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0LL),
        [] __host__ __device__(const Tuple& a, const Tuple& b) {  // `__host__`: compiler needs the return type
            return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                      thrust::get<1>(a) + thrust::get<1>(b),
                                      thrust::get<2>(a) + thrust::get<2>(b),
                                      fmaxf(thrust::get<3>(a), thrust::get<3>(b)),
                                      thrust::get<4>(a) + thrust::get<4>(b),
                                      fmaxf(thrust::get<5>(a), thrust::get<5>(b)),
                                      thrust::get<6>(a) + thrust::get<6>(b));
        });
    return {thrust::get<0>(res) / dims / bsz,   // avg abs src
            thrust::get<1>(res) / dims / bsz,   // avg abs ref
            thrust::get<2>(res) / dims / bsz,   // avg abs diff
            thrust::get<3>(res),                // max abs diff
            thrust::get<4>(res) / dims / bsz,   // avg rel diff
            thrust::get<5>(res),                // max rel diff
            (float)thrust::get<6>(res) / bsz};  // outlier count
}

template std::vector<float>
FastCompare(const half* src, const half* ref, int dims, int bsz, cudaStream_t stream, float rtol, float atol);

void LoadBinary(const std::string& path, size_t size, void* dst)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << "failed to open " << path << "\n";
        std::abort();
    }
    ifs.seekg(0, ifs.end);
    auto actual_size_in_bytes = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (size != actual_size_in_bytes) {
        std::cerr << "[warning] file " << path << " has " << actual_size_in_bytes << " bytes, while " << size
                  << " bytes is requested\n";
    }
    ifs.read((char*)dst, size);
    std::cerr << "[info] " << path << " " << size << "\n";
}

namespace cg = cooperative_groups;

__global__ void curand_init(curandState* state)
{
    auto tid = cg::this_grid().thread_rank();
    curand_init(0xe4c45822e90461ddULL, tid, 0, state + tid);
}

template<typename T>
__global__ void curand_uniform(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_uniform(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

template<typename T>
__global__ void curand_normal(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_normal(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

__global__ void curand_bytes(curandState* state, size_t count, uint* result)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        result[i] = curand(state + grid.thread_rank());
    }
}

struct RNG::Impl {

    curandState* states{};

    Impl()
    {
        cudaMalloc(&states, sizeof(curandState) * 64 * 64);
        curand_init<<<64, 64>>>(states);
    }

    ~Impl()
    {
        cudaFree(states);
    }

    void GenerateUInt(uint* out, size_t count)
    {
        curand_bytes<<<64, 64, 0, stream_>>>(states, count, out);
    }

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale, float shift)
    {
        curand_uniform<<<64, 64, 0, stream_>>>(states, count, out, scale, shift);
    }

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale, float shift)
    {
        curand_normal<<<64, 64, 0, stream_>>>(states, count, out, scale, shift);
    }

    cudaStream_t stream_{};
};

RNG::RNG(): impl_(std::make_unique<Impl>()) {}

RNG::~RNG() = default;

void RNG::GenerateUInt(uint* out, size_t count)
{
    impl_->GenerateUInt(out, count);
}

template<typename T>
void RNG::GenerateUniform(T* out, size_t count, float scale, float shift)
{
    impl_->GenerateUniform(out, count, scale, shift);
}

template<typename T>
void RNG::GenerateNormal(T* out, size_t count, float scale, float shift)
{
    impl_->GenerateNormal(out, count, scale, shift);
}

cudaStream_t RNG::stream() const
{
    return impl_->stream_;
}

void RNG::set_stream(cudaStream_t stream)
{
    impl_->stream_ = stream;
}

template void RNG::GenerateUniform(half* out, size_t count, float scale, float shift);
template void RNG::GenerateUniform(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateUniform(nv_bfloat16* out, size_t count, float scale, float shift);
#endif

template void RNG::GenerateNormal(half* out, size_t count, float scale, float shift);
template void RNG::GenerateNormal(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateNormal(nv_bfloat16* out, size_t count, float scale, float shift);
#endif

}  // namespace turbomind
