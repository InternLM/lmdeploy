
#include <filesystem>
#include <fmt/format.h>
#include <ostream>
#include <stack>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"

namespace turbomind::core {

namespace {

std::string StripNamespace(std::string_view name)
{
    static constexpr std::string_view ns = "turbomind::";
    std::string                       result;

    for (auto it = name.find(ns); it != std::string_view::npos; it = name.find(ns)) {
        result.append(name.substr(0, it));
        name.remove_prefix(it + ns.size());
    }

    result.append(name);

    return result;
}

std::string StripPathPrefix(std::string_view file)
{
    static const char* flag = std::getenv("TM_SRC_FULL_PATH");
    if (flag) {
        return std::string{file};
    }

    // Return only the filename (last component of the path)
    std::filesystem::path path{file};
    return path.filename().string();
}

struct ContextStorage {
    enum
    {
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

    std::vector<ScopeEntry> scope_;

    ContextStorage()
    {
        push(Allocator{kCPU});
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
            if (type == kCPU) {
                mask = host_alloc_bit;
                host_alloc_.push(alloc);
            }
            else if (type == kDEVICE) {
                mask = device_alloc_bit;
                device_alloc_.push(alloc);
            }
            else if (type == kCPUpinned) {
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

    void push_scope(const ScopeEntry& entry)
    {
        scope_.push_back(entry);
    }

    void pop_scope()
    {
        TM_CHECK(!scope_.empty()) << "pop_scope called with empty scope stack";
        scope_.pop_back();
    }

    int scope_depth() const
    {
        return static_cast<int>(scope_.size());
    }

    std::string scope_trace() const
    {
        if (scope_.empty()) {
            return {};
        }
        std::ostringstream oss;
        oss << std::hex << std::this_thread::get_id();
        std::string s = fmt::format("*** stacktrace of thread 0x{} ***\n", oss.str());
        for (size_t i = 0; i < scope_.size(); ++i) {
            const int r    = static_cast<int>(scope_.size()) - i - 1;
            auto      name = StripNamespace(scope_[r].name);
            if (scope_[r].type == scope_type::function) {
                name = StripFunctionSignature(name);
            }
            s += fmt::format("  [{:>2}] {} @ {}:{}\n", i, name, StripPathPrefix(scope_[r].file), scope_[r].line);
        }
        return s;
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

void Context::push_scope(const ScopeEntry& entry)
{
    ContextStorage::instance().push_scope(entry);
}

void Context::pop_scope()
{
    ContextStorage::instance().pop_scope();
}

std::string Context::scope_trace()
{
    return ContextStorage::instance().scope_trace();
}

int Context::scope_depth()
{
    return ContextStorage::instance().scope_depth();
}

Stream& Context::stream()
{
    auto& stream_ = ContextStorage::instance().stream_;
    TM_CHECK(!stream_.empty()) << "No STREAM available in current context";
    return stream_.top();
}

Allocator& Context::host_alloc()
{
    auto& host_alloc_ = ContextStorage::instance().host_alloc_;
    TM_CHECK(!host_alloc_.empty()) << "No HOST memory allocator available in current context";
    return host_alloc_.top();
}

Allocator& Context::device_alloc()
{
    auto& device_alloc_ = ContextStorage::instance().device_alloc_;
    TM_CHECK(!device_alloc_.empty()) << "No DEVICE memory allocator available in current context";
    return device_alloc_.top();
}

Allocator& Context::pinned_alloc()
{
    auto& pinned_alloc_ = ContextStorage::instance().pinned_alloc_;
    TM_CHECK(!pinned_alloc_.empty()) << "No PINNED memory allocator available in current context";
    return pinned_alloc_.top();
}

Allocator& Context::alloc(Device device)
{
    switch (device.type) {
        case kDEVICE:
            return device_alloc();
        case kCPU:
            return host_alloc();
        case kCPUpinned:
            return pinned_alloc();
    }
    TM_UNREACHABLE;
}

}  // namespace turbomind::core
