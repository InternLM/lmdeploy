#pragma once

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/common.h"
#include "src/turbomind/core/stream.h"

#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

enum class MemoryType : int {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
};

inline constexpr MemoryType MEMORY_CPU        = MemoryType::MEMORY_CPU;
inline constexpr MemoryType MEMORY_CPU_PINNED = MemoryType::MEMORY_CPU_PINNED;
inline constexpr MemoryType MEMORY_GPU        = MemoryType::MEMORY_GPU;

}  // namespace turbomind

namespace turbomind::core {

struct MemoryLocation {
    MemoryType type;
    int        id;

    MemoryLocation(): MemoryLocation{MEMORY_CPU} {}
    MemoryLocation(MemoryType type_): type{type_}, id{-1} {}
    MemoryLocation(MemoryType type_, int device_): type{type_}, id{device_} {}

    friend bool operator==(const MemoryLocation& a, const MemoryLocation& b)
    {
        return a.type == b.type && a.id == b.id;
    }

    friend bool operator!=(const MemoryLocation& a, const MemoryLocation& b)
    {
        return !(a == b);
    }
};

using MemLoc = MemoryLocation;

class AllocatorImpl {
public:
    virtual ~AllocatorImpl();

    virtual void* allocate(ssize_t size) = 0;

    virtual void deallocate(void* p, ssize_t size) = 0;

    // Returns invalid stream by default
    virtual Stream stream() const noexcept;

    virtual MemoryLocation device() const noexcept = 0;
};

class Allocator {
public:
    Allocator() = default;

    explicit Allocator(MemoryType type);

    Allocator(Stream stream, bool use_default_pool);

    Allocator(shared_ptr<AllocatorImpl> impl): impl_{std::move(impl)} {};

    AllocatorImpl* operator->() const
    {
        TM_CHECK_NOTNULL(impl_);
        return impl_.get();
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    friend bool operator==(const Allocator& a, const Allocator& b)
    {
        return a.impl_ == b.impl_;
    }

    friend bool operator!=(const Allocator& a, const Allocator& b)
    {
        return !(a == b);
    }

    template<class T, class... Args>
    shared_ptr<T> adapt(Args&&... args) const
    {
        return {std::make_shared<T>(impl_, ((Args&&)args)...)};
    }

private:
    shared_ptr<AllocatorImpl> impl_;
};

class StackAllocatorImpl: public AllocatorImpl {
public:
    static constexpr ssize_t kAlignment = 256;

    explicit StackAllocatorImpl(shared_ptr<AllocatorImpl> underlying_impl): underlying_impl_{std::move(underlying_impl)}
    {
    }

    ~StackAllocatorImpl() override
    {
        if (cached_beg_) {
            underlying_impl_->deallocate(cached_beg_, cached_end_ - cached_beg_);
        }
    }

    void* allocate(ssize_t size) override
    {
        size = round_up(size, kAlignment);

        void* p{};
        if (cached_ptr_ + size <= cached_end_) {
            p = cached_ptr_;
            cached_ptr_ += size;
        }
        else {
            TM_CHECK(!cached_beg_);
            p = underlying_impl_->allocate(size);
        }

        // TM_LOG_ERROR("allocate %p, %ld", p, size);

        size_ += size;
        ++num_;
        max_size_ = std::max(size_, max_size_);
        num_      = std::max(num_, max_num_);
        return p;
    }

    void deallocate(void* p, ssize_t size) override
    {
        size = round_up(size, kAlignment);

        // TM_LOG_ERROR("deallocate %p, %p, %ld", p, cached_ptr_, size);

        if ((char*)p + size == cached_ptr_) {
            cached_ptr_ -= size;
        }
        else {
            TM_CHECK(!cached_beg_);
            underlying_impl_->deallocate(p, size);
        }
        size_ -= size;
        --num_;
    }

    Stream stream() const noexcept override
    {
        return underlying_impl_->stream();
    }

    MemoryLocation device() const noexcept override
    {
        return underlying_impl_->device();
    }

    void iter()
    {
        TM_CHECK_EQ((void*)cached_beg_, (void*)cached_ptr_);
        auto excpected = max_size_ + kAlignment * max_num_;
        if (cached_end_ - cached_beg_ < excpected) {
            if (cached_beg_) {
                underlying_impl_->deallocate(cached_beg_, cached_end_ - cached_beg_);
            }
            cached_ptr_ = cached_beg_ = (char*)underlying_impl_->allocate(excpected);
            cached_end_               = cached_beg_ + excpected;
        }
        size_ = num_ = max_size_ = max_num_ = 0;
    }

private:
    ssize_t size_{};
    ssize_t num_{};
    ssize_t max_size_{};
    ssize_t max_num_{};

    char* cached_beg_{};
    char* cached_end_{};
    char* cached_ptr_{};

    std::shared_ptr<AllocatorImpl> underlying_impl_;
};

class SimpleAllocator: public AllocatorImpl {
public:
    template<class Alloc, class Dealloc>
    static Allocator Create(Alloc&& alloc, Dealloc&& dealloc, MemLoc device)
    {
        return Allocator{std::make_shared<SimpleAllocator>((Alloc&&)alloc, (Dealloc&&)dealloc, device)};
    }

    template<class Alloc, class Dealloc>
    SimpleAllocator(Alloc&& alloc, Dealloc&& dealloc, MemLoc device):
        alloc_{std::move(alloc)}, dealloc_{std ::move(dealloc)}, device_{device}
    {
    }

    void* allocate(ssize_t size) override
    {
        return alloc_(size);
    };

    void deallocate(void* p, ssize_t size) override
    {
        return dealloc_(p, size);
    }

    MemoryLocation device() const noexcept override
    {
        return device_;
    }

private:
    std::function<void*(ssize_t)>       alloc_;
    std::function<void(void*, ssize_t)> dealloc_;
    MemLoc                              device_;
};

}  // namespace turbomind::core