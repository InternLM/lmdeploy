// Copyright (c) OpenMMLab. All rights reserved.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

CmpMode compare_mode = kCmpRead;
// CmpMode compare_mode = kCmpWrite;

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
void CmpRead(T* ptr, size_t size, std::string key, cudaStream_t stream)
{
    // read a from file
    std::vector<T> h_a(size);
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
    std::vector<T> h_b(size);
    check_cuda_error(cudaMemcpyAsync(h_b.data(), ptr, sizeof(T) * size, cudaMemcpyDefault, stream));
    check_cuda_error(cudaStreamSynchronize(stream));

    using Tacc         = std::conditional_t<std::is_integral_v<T>, int64_t, float>;
    constexpr Tacc eps = std::is_integral_v<T> ? 1 : 1e-8f;

    Tacc asum{};
    Tacc rsum{};
    Tacc amean_r{};
    Tacc amean_x{};
    for (size_t i = 0; i < size; ++i) {
        Tacc x        = (Tacc)h_b[i];
        Tacc r        = (Tacc)h_a[i];
        Tacc abs_diff = std::abs(x - r);
        Tacc rel_diff = abs_diff / std::max(std::max(std::abs(r), std::abs(x)), eps);
        asum += abs_diff;
        rsum += rel_diff;
        amean_x += std::abs(x);
        amean_r += std::abs(r);
    }

    fprintf(stderr,
            "%15s%15f%15f%15f%15f%15f\n",
            key.c_str(),
            (float)amean_x / (float)size,
            (float)amean_r / (float)size,
            (float)asum,
            (float)asum / (float)size,
            (float)rsum / (float)size);

    check_cuda_error(cudaMemcpyAsync(ptr, h_a.data(), sizeof(T) * h_a.size(), cudaMemcpyDefault, stream));
    check_cuda_error(cudaStreamSynchronize(stream));
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
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream)
{
    // std::cerr << "Comparing " << key << "\n";
    if (mode == kCmpRead) {
        CmpRead(ptr, size, key, stream);
    }
    else if (mode == kCmpWrite) {
        CmpWrite(ptr, size, key, stream);
    }
    else {
        // kCmpNone
    }
}

template void Compare(int* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);
template void Compare(float* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);
template void Compare(half* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);
template void Compare(__nv_bfloat16* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);

template void CheckNan(const float* ptr, size_t size, std::string key, cudaStream_t stream);
template void CheckNan(const half* ptr, size_t size, std::string key, cudaStream_t stream);

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

int64_t& gSequenceIds(int batch_idx)
{
    thread_local std::vector<int64_t> ids{};
    if (batch_idx >= ids.size()) {
        ids.resize(batch_idx + 1, -1);
    }
    return ids.at(batch_idx);
}

}  // namespace turbomind
