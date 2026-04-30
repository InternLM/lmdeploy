
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"

namespace turbomind::core {

AllocatorImpl::~AllocatorImpl() = default;

Stream AllocatorImpl::stream() const noexcept
{
    return Stream{};
}

class CudaMemPoolAllocator: public AllocatorImpl {
public:
    CudaMemPoolAllocator(Stream stream, bool use_default_pool):
        pool_{}, stream_{stream}, device_{kDEVICE}, use_default_pool_{use_default_pool}
    {
        TM_CUDA_CHECK(cudaGetDevice(&device_.id));
        if (use_default_pool_) {
            TM_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, device_.id));
        }
        else {
            cudaMemPoolProps props{};
            props.allocType     = cudaMemAllocationTypePinned;
            props.handleTypes   = cudaMemHandleTypeNone;
            props.location.type = cudaMemLocationTypeDevice;
            props.location.id   = device_.id;
            TM_CUDA_CHECK(cudaMemPoolCreate(&pool_, &props));
            cuuint64_t thres = (cuuint64_t)-1;
            TM_CUDA_CHECK(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &thres));
        }
    }

    ~CudaMemPoolAllocator() override
    {
        if (!use_default_pool_) {
            TM_CUDA_CHECK(cudaMemPoolDestroy(pool_));
        }
        pool_ = {};
    }

    void* allocate(ssize_t size) override
    {
        void* ptr{};
        TM_CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, size, pool_, stream_.handle()));
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        TM_CUDA_CHECK(cudaFreeAsync(p, stream_.handle()));
    }

    Device device() const noexcept override
    {
        return device_;
    }

    Stream stream() const noexcept override
    {
        return stream_;
    }

    void trim(size_t bytes_to_keep)
    {
        TM_CUDA_CHECK(cudaMemPoolTrimTo(pool_, bytes_to_keep));
    }

private:
    cudaMemPool_t pool_;
    Stream        stream_;
    Device        device_;
    bool          use_default_pool_;
};

class CudaAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        void* ptr{};
        TM_CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        TM_CUDA_CHECK(cudaFree(p));
    }

    Device device() const noexcept override
    {
        return kDEVICE;
    }
};

class CudaHostAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        void* ptr{};
        TM_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        TM_CUDA_CHECK(cudaFreeHost(p));
    }

    Device device() const noexcept override
    {
        return kCPUpinned;
    }
};

class HostAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        return ::operator new(size);
    }

    void deallocate(void* p, ssize_t) override
    {
        ::operator delete(p);
    }

    Device device() const noexcept override
    {
        return kCPU;
    }
};

Allocator::Allocator(DeviceType type)
{
    impl_ = [&]() -> shared_ptr<AllocatorImpl> {
        switch (type) {
            case kCPU:
                return std::make_shared<HostAllocator>();
            case kDEVICE:
                return std::make_shared<CudaAllocator>();
            case kCPUpinned:
                return std::make_shared<CudaHostAllocator>();
        }
        return {};
    }();
    TM_CHECK_NOTNULL(impl_);
}

Allocator::Allocator(Stream stream, bool use_default_pool)
{
    impl_ = std::make_shared<CudaMemPoolAllocator>(std::move(stream), use_default_pool);
}

}  // namespace turbomind::core
