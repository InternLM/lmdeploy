#pragma once

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/common.h"
#include "src/turbomind/core/stream.h"

namespace turbomind::core {

class Context {
public:
    static Stream&    stream();
    static Allocator& host_alloc();
    static Allocator& device_alloc();
    static Allocator& pinned_alloc();
    static Allocator& alloc(Device device);

private:
    friend class ContextGuard;
    static void push(const Stream& stream);
    static void push(const Allocator& alloc);
    static void pop();
};

class ContextGuard {
public:
    template<class... Args>
    explicit ContextGuard(Args&&... args): n_{}
    {
        (Context::push((Args &&) args), ...);
        n_ = sizeof...(Args);
    }
    ~ContextGuard()
    {
        for (int i = 0; i < n_; ++i) {
            Context::pop();
        }
    }

private:
    int n_;
};

}  // namespace turbomind::core
