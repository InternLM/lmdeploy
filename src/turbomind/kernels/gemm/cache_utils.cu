

#include "src/turbomind/kernels/gemm/cache_utils.h"
#include <iostream>

namespace turbomind::gemm {

CacheFlushing::CacheFlushing()
{
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);

    static constexpr int scale = 4;

    size_ = props.l2CacheSize * scale / sizeof(uint32_t);

    std::cout << "L2 flushing size: " << size_ * sizeof(uint32_t) / (1 << 20) << " MB\n";

    cudaMalloc(&buffer_, sizeof(uint32_t) * size_);
}

void CacheFlushing::flush(cudaStream_t stream)
{
    static CacheFlushing inst{};
    inst(stream);
}

__global__ void flush_kernel(uint32_t* buffer, int size, uint32_t pattern)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        buffer[i] ^= pattern;
    }
}

void CacheFlushing::operator()(cudaStream_t stream) const
{
    int threads = 512;
    int blocks  = 512;
    flush_kernel<<<blocks, threads, 0, stream>>>(buffer_, size_, uint32_t(-1));
}

}  // namespace turbomind::gemm