// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/gpu_metric.h"
#include <thrust/device_vector.h>

#include <cublas_v2.h>

namespace turbomind::gemm {

using thrust::device_vector;

namespace {

template<int BLOCK_NUM, int BLOCK_DIM, int LOG_TILE>
__global__ void l2_bw(float* dsink, const float* array, int count)
{
    int    tid = threadIdx.x + (blockIdx.x >> LOG_TILE) * blockDim.x;
    float4 sink{};

    constexpr int NUM_THREADS = BLOCK_NUM * BLOCK_DIM;

    for (int i = 0; i < count; i += NUM_THREADS * 4) {
        const float* ptr    = array + i;
        const int    offset = tid * 4;
        float4       data   = __ldcg(reinterpret_cast<const float4*>(ptr + offset));
        sink.x += data.x;
        sink.y += data.y;
        sink.z += data.z;
        sink.w += data.w;
    }

    dsink[threadIdx.x] = sink.x + sink.y + sink.z + sink.w;
}

}  // namespace

float MeasureL2CacheThroughput()
{
    cudaDeviceProp prop{};
    int            device{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    size_t size = static_cast<size_t>(prop.l2CacheSize) * 64;

    std::cout << size << std::endl;

    constexpr int BLOCK_X  = 128;  // blocks participating single sweep
    constexpr int BLOCK_Y  = 128;  // full sweep iters
    constexpr int LOG_TILE = 5;    // swizzling factor to bring up L2 hit rate, set to 0 will minimize hit rate

    constexpr int BLOCK_DIM = 256;

    constexpr int CHUNK_SIZE = BLOCK_X * BLOCK_DIM * 4;  // x4 for float4 load pattern

    device_vector<float> data(ceil_div(size, sizeof(float)) / CHUNK_SIZE * CHUNK_SIZE);
    device_vector<float> dsink(BLOCK_DIM);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemsetAsync(data.data().get(), 0, sizeof(float) * data.size(), stream);

    cudaEvent_t ev_start, ev_end;

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start, stream);

    l2_bw<BLOCK_X, BLOCK_DIM, LOG_TILE><<<dim3(BLOCK_X << LOG_TILE, BLOCK_Y >> LOG_TILE), BLOCK_DIM, 0, stream>>>(
        dsink.data().get(), data.data().get(), data.size());

    cudaEventRecord(ev_end, stream);

    cudaEventSynchronize(ev_end);

    float ms{};
    cudaEventElapsedTime(&ms, ev_start, ev_end);

    size_t bytes = BLOCK_Y * sizeof(float) * data.size();

    const float bytes_per_second = bytes / ms * 1e3;
    std::cout << bytes_per_second / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    cudaStreamDestroy(stream);

    return bytes_per_second;
}

float MeasureMmaThroughput(int problem_size)
{
    device_vector<half> a(problem_size * problem_size);
    device_vector<half> b(a.size());
    device_vector<half> c(a.size());

    cublasHandle_t cublas{};
    cublasCreate(&cublas);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasSetStream(cublas, stream);

    cudaEvent_t ev_start, ev_end;

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start, stream);

    float alpha = 1.f;
    float beta  = 0.f;
    cublasGemmEx(cublas,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 problem_size,
                 problem_size,
                 problem_size,
                 &alpha,
                 a.data().get(),
                 CUDA_R_16F,
                 problem_size,
                 b.data().get(),
                 CUDA_R_16F,
                 problem_size,
                 &beta,
                 c.data().get(),
                 CUDA_R_16F,
                 problem_size,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(ev_end, stream);

    cudaEventSynchronize(ev_end);

    float ms{};
    cudaEventElapsedTime(&ms, ev_start, ev_end);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    cudaStreamDestroy(stream);

    cublasDestroy(cublas);

    const size_t ops = (size_t)problem_size * problem_size * problem_size;

    float fma_per_second = ops / ms * 1e3;

    std::cout << 2 * fma_per_second / 1e9 << " FLOPS/s" << std::endl;

    return fma_per_second;
}

}  // namespace turbomind::gemm
