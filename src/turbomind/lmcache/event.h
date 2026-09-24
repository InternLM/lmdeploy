#pragma once

#include "src/turbomind/core/logger.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace turbomind::lmcache {

#define LMCACHE_CUDA_CHECK(expr) Check((expr), #expr, __FILE__, __LINE__)

class CudaEvent final {
public:
    static CudaEvent Create()
    {
        CudaEvent event;
        LMCACHE_CUDA_CHECK(cudaEventCreateWithFlags(&event.event_, cudaEventInterprocess | cudaEventDisableTiming));
        return event;
    }

    static CudaEvent Open(const cudaIpcEventHandle_t& handle)
    {
        CudaEvent event;
        LMCACHE_CUDA_CHECK(cudaIpcOpenEventHandle(&event.event_, handle));
        return event;
    }

    CudaEvent(CudaEvent&& other) noexcept: event_{std::exchange(other.event_, {})} {}
    CudaEvent& operator=(CudaEvent&&) noexcept = delete;

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent()
    {
        Reset();
    }

    void Record(cudaStream_t stream)
    {
        LMCACHE_CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    cudaIpcEventHandle_t handle() const
    {
        cudaIpcEventHandle_t handle{};
        LMCACHE_CUDA_CHECK(cudaIpcGetEventHandle(&handle, event_));
        return handle;
    }

    cudaError_t Query() const noexcept
    {
        return cudaEventQuery(event_);
    }

private:
    CudaEvent() = default;

    static void Check(cudaError_t status, const char* expression, const char* file, int line)
    {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error ") + file + ":" + std::to_string(line) + " '" + expression
                                     + "': " + cudaGetErrorString(status));
        }
    }

    void Reset() noexcept
    {
        if (!event_) {
            return;
        }
        if (const auto status = cudaEventDestroy(event_); status != cudaSuccess) {
            TM_LOG_ERROR("cudaEventDestroy: {}", cudaGetErrorString(status));
        }
        event_ = {};
    }

    cudaEvent_t event_{};
};

static_assert(std::is_nothrow_move_constructible_v<CudaEvent>);
static_assert(!std::is_copy_constructible_v<CudaEvent>);
static_assert(!std::is_move_assignable_v<CudaEvent>);

#undef LMCACHE_CUDA_CHECK

}  // namespace turbomind::lmcache
