

#include "src/turbomind/kernels/gemm/cache_utils.h"
#include <iostream>

namespace turbomind::gemm {

CacheFlushing::CacheFlushing()
{
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);

    size_ = props.l2CacheSize;

    std::cout << "L2 flushing size: " << (size_ >> 20) << " MB\n";

    cudaMalloc(&buffer_, size_);
}

void CacheFlushing::flush(cudaStream_t stream)
{
    thread_local CacheFlushing inst{};
    inst(stream);
}

void CacheFlushing::operator()(cudaStream_t stream) const
{
    cudaMemsetAsync(buffer_, 0, size_, stream);
}

}  // namespace turbomind::gemm