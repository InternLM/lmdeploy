
#include <stack>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind::core {

namespace {

struct ContextStorage {
    enum {
        stream_bit       = 1,
        host_alloc_bit   = 2,
        device_alloc_bit = 4,
        pinned_alloc_bit = 8,
    };

    std::stack<Stream>    stream_;
    std::stack<Allocator> host_alloc_;
    std::stack<Allocator> device_alloc_;
    std::stack<Allocator> pinned_alloc_;
    std::stack<int>       mask_;

    ContextStorage()
    {
        push(Allocator{MEMORY_CPU});
    }

    void push(const Stream& stream)
    {
        int mask{};
        if (stream) {
            stream_.push(stream);
            mask = stream_bit;
        }
        mask_.push(mask);
    }

    void push(const Allocator& alloc)
    {
        int mask{};
        if (alloc) {
            const auto type = alloc->device().type;
            if (type == MEMORY_CPU) {
                mask = host_alloc_bit;
                host_alloc_.push(alloc);
            }
            else if (type == MEMORY_GPU) {
                mask = device_alloc_bit;
                device_alloc_.push(alloc);
            }
            else if (type == MEMORY_CPU_PINNED) {
                mask = pinned_alloc_bit;
                pinned_alloc_.push(alloc);
            }
        }
        mask_.push(mask);
    }

    void pop()
    {
        if (mask_.top() & stream_bit) {
            stream_.pop();
        }
        if (mask_.top() & host_alloc_bit) {
            host_alloc_.pop();
        }
        if (mask_.top() & device_alloc_bit) {
            device_alloc_.pop();
        }
        if (mask_.top() & pinned_alloc_bit) {
            pinned_alloc_.pop();
        }
        mask_.pop();
    }

    static ContextStorage& instance()
    {
        thread_local ContextStorage inst{};
        return inst;
    }
};

}  // namespace

void Context::push(const Stream& stream)
{
    ContextStorage::instance().push(stream);
}

void Context::push(const Allocator& alloc)
{
    ContextStorage::instance().push(alloc);
}

void Context::pop()
{
    ContextStorage::instance().pop();
}

Stream& Context::stream()
{
    auto& stream_ = ContextStorage::instance().stream_;
    TM_CHECK(!stream_.empty());
    return stream_.top();
}

Allocator& Context::host_alloc()
{
    auto& host_alloc_ = ContextStorage::instance().host_alloc_;
    TM_CHECK(!host_alloc_.empty());
    return host_alloc_.top();
}

Allocator& Context::device_alloc()
{
    auto& device_alloc_ = ContextStorage::instance().device_alloc_;
    TM_CHECK(!device_alloc_.empty());
    return device_alloc_.top();
}

Allocator& Context::pinned_alloc()
{
    auto& pinned_alloc_ = ContextStorage::instance().pinned_alloc_;
    TM_CHECK(!pinned_alloc_.empty());
    return pinned_alloc_.top();
}

Allocator& Context::alloc(MemoryLocation device)
{
    switch (device.type) {
        case MEMORY_GPU:
            return device_alloc();
        case MEMORY_CPU:
            return host_alloc();
        case MEMORY_CPU_PINNED:
            return pinned_alloc();
    }
    TM_UNREACHABLE;
}

}  // namespace turbomind::core