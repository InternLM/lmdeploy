// Copyright (c) OpenMMLab. All rights reserved.

// Run with compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99.
// Leak checking must include thread-local destruction after main returns.
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"

namespace {

void Check(cudaError_t status)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void Flush()
{
    // Exercise reuse on one thread, then destruction after its stream retires.
    cudaStream_t stream{};
    Check(cudaStreamCreate(&stream));

    try {
        turbomind::gemm::CacheFlushing::flush(stream);
        turbomind::gemm::CacheFlushing::flush(stream);
        Check(cudaGetLastError());
        Check(cudaStreamSynchronize(stream));
    }
    catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }

    Check(cudaStreamDestroy(stream));
}

}  // namespace

int main()
{
    try {
        Check(cudaSetDevice(0));

        for (int i = 0; i < 2; ++i) {
            std::exception_ptr error;
            std::thread        worker([&] {
                try {
                    Check(cudaSetDevice(0));
                    Flush();
                }
                catch (...) {
                    error = std::current_exception();
                }
            });

            worker.join();

            if (error) {
                std::rethrow_exception(error);
            }
        }

        // Also check main-thread TLS cleanup. Do not mask leaks with device reset.
        Flush();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
