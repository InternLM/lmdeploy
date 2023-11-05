// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <vector>

namespace turbomind {

CmpMode compare_mode = kCmpNone;

template<typename T>
struct abs_diff_t {
    using type = T;
};

template<>
struct abs_diff_t<half> {
    using type = float;
};

template<typename T>
struct abs_diff: public thrust::unary_function<thrust::tuple<T, T>, typename abs_diff_t<T>::type> {
    __host__ __device__ float operator()(thrust::tuple<T, T> x) const
    {
        using R = typename abs_diff_t<T>::type;
        auto r  = R(thrust::get<0>(x)) - R(thrust::get<1>(x));
        return r < R(0) ? -r : r;
    }
};

template<typename T>
void CheckNan(const T* ptr, size_t size, std::string key, cudaStream_t stream)
{
    std::vector<T> h_data(size);
    cudaMemcpyAsync(h_data.data(), ptr, sizeof(T) * size, cudaMemcpyDefault, stream);

    check_cuda_error(cudaStreamSynchronize(stream));

    size_t nan_cnt = 0;
    for (const auto& x : h_data) {
        nan_cnt += std::isnan(static_cast<float>(x));
    }
    if (nan_cnt) {
        std::cerr << key << ": NaN count " << nan_cnt << "\n";
    }
}

template<typename T>
void CmpRead(T* ptr, size_t size, std::string key, cudaStream_t stream, std::string msg)
{
    // wait for b
    check_cuda_error(cudaStreamSynchronize(stream));
    // read a from file
    thrust::host_vector<T> h_a(size);
    {
        const auto    filename = "tmp/" + key + ".cmp";
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << key << ": failed to open " + filename << "\n";
            return;
        }
        ifs.seekg(0, ifs.end);
        const auto actual_size_in_bytes = ifs.tellg();
        ifs.seekg(0, ifs.beg);
        const auto expect_size_in_bytes = sizeof(T) * size;
        if (actual_size_in_bytes != expect_size_in_bytes) {
            std::cerr << key << ": file size in bytes mismatch, expect " << expect_size_in_bytes << ", got "
                      << actual_size_in_bytes << "\n";
            return;
        }
        ifs.read((char*)h_a.data(), sizeof(T) * h_a.size());
    }
    // copy a to device
    thrust::device_vector<T> a = h_a;
    // create abs(a - b) iterator
    thrust::device_ptr<T> dev_ptr(ptr);
    auto                  zip_iter       = thrust::make_zip_iterator(thrust::make_tuple(a.begin(), dev_ptr));
    auto                  transform_iter = thrust::make_transform_iterator(zip_iter, abs_diff<T>{});
    // sum(abs(a - b))
    auto asum = thrust::reduce(thrust::device, transform_iter, transform_iter + size);
    std::cerr << key << msg << ": " << asum << " " << asum / size << "\n";
}

template<typename T>
void CmpWrite(T* ptr, size_t size, std::string key, cudaStream_t stream)
{
    std::vector<T> a(size);
    // copy a to host
    check_cuda_error(cudaMemcpyAsync(a.data(), ptr, sizeof(T) * size, cudaMemcpyDefault, stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    // write to file
    {
        std::ofstream ofs("tmp/" + key + ".cmp", std::ios::binary);
        ofs.write((char*)a.data(), sizeof(T) * a.size());
    }
}

template<typename T>
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream, std::string msg)
{
    // std::cerr << "Comparing " << key << "\n";
    if (mode == kCmpRead) {
        CmpRead(ptr, size, key, stream, msg);
    }
    else if (mode == kCmpWrite) {
        CmpWrite(ptr, size, key, stream);
    }
    else {
        // kCmpNone
    }
}

template void Compare(int* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream, std::string msg);
template void Compare(float* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream, std::string msg);
template void Compare(half* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream, std::string msg);

template void CheckNan(const float* ptr, size_t size, std::string key, cudaStream_t stream);
template void CheckNan(const half* ptr, size_t size, std::string key, cudaStream_t stream);

std::string format(const std::pair<std::string, Tensor>& p)
{
    std::stringstream ss;
    ss << p.first << " [";
    bool first = true;
    for (const auto& x : p.second.shape) {
        ss << (first ? "" : ", ") << x;
        first = false;
    }
    ss << "]";
    return ss.str();
}

size_t curandStateGetSize()
{
    return sizeof(curandState_t);
}

bool isDebug()
{
    static const bool is_debug = [] {
        const auto level = std::getenv("TM_DEBUG_LEVEL");
        if (level && level == std::string("DEBUG")) {
            return true;
        }
        return false;
    }();
    return is_debug;
}

template<int kWarpCount, typename T>
inline __device__ T blockSum(T val, int warp_id, int lane_id)
{
    __shared__ T smem_red[32];

    for (int mask = 32 >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync((uint32_t)-1, val, mask);
    }
    if (lane_id == 0) {
        smem_red[warp_id] = val;
    }

    __syncthreads();

    val = lane_id < kWarpCount ? smem_red[lane_id] : T{};

    for (int mask = kWarpCount >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync((uint32_t)-1, val, mask);
    }
    val = __shfl_sync((uint32_t)-1, val, 0);

    return val;
}

template<int kWarpCount, typename T>
__global__ void CountInfNan(const T* data, uint32_t* g_inf_count, uint32_t* g_nan_count, size_t count)
{
    uint32_t inf_count = 0;
    uint32_t nan_count = 0;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            inf_count += __isinf(data[i]);
            nan_count += __isnan(data[i]);
        }
        else if constexpr (std::is_same_v<T, half>) {
            inf_count += __hisinf(data[i]);
            nan_count += __hisnan(data[i]);
        }
    }
    inf_count = blockSum<kWarpCount>(inf_count, threadIdx.x / 32, threadIdx.x % 32);
    nan_count = blockSum<kWarpCount>(nan_count, threadIdx.x / 32, threadIdx.x % 32);
    if (threadIdx.x == 0) {
        if (inf_count) {
            atomicAdd(g_inf_count, inf_count);
        }
        if (nan_count) {
            atomicAdd(g_nan_count, nan_count);
        }
    }
}
namespace {
struct Info {
    char data[256];
};

}  // namespace

__global__ void ReportInfNan(uint32_t* g_inf_count, uint32_t* g_nan_count, Info info)
{
    auto inf_count = *g_inf_count;
    auto nan_count = *g_nan_count;
    if (inf_count || nan_count) {
        printf("[TM][ERROR] [%s] Inf=%u, NaN=%u\n", info.data, inf_count, nan_count);
    }
    // reset the counters for later use
    *g_inf_count = 0;
    *g_nan_count = 0;
}

template<typename T>
void CheckValues(const T* data, int count, const std::string& msg, cudaStream_t stream)
{
    thread_local uint32_t* counters = [] {
        uint32_t* ptr{};
        cudaMalloc(&ptr, sizeof(uint32_t) * 2);
        cudaMemset(ptr, 0, sizeof(uint32_t) * 2);
        cudaDeviceSynchronize();
        return ptr;
    }();

    if (data == nullptr && count == 0) {
        return;
    }

    const auto g_inf_count = counters;
    const auto g_nan_count = g_inf_count + 1;

    FT_CHECK(msg.size() < sizeof(Info) - 1);

    CountInfNan<4><<<256, 128, 0, stream>>>(data, g_inf_count, g_nan_count, count);

    Info info;
    strncpy(info.data, msg.c_str(), sizeof(info) - 1);

    ReportInfNan<<<1, 1, 0, stream>>>(g_inf_count, g_nan_count, info);
}

template void CheckValues(const half* data, int count, const std::string& msg, cudaStream_t stream);
template void CheckValues(const float* data, int count, const std::string& msg, cudaStream_t stream);

Barrier*& model_instance_barrier()
{
    thread_local Barrier* p{};
    return p;
}

}  // namespace turbomind
