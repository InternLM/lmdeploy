#pragma once

#include <cstdint>
#include <cstdlib>
#include <utility>

namespace turbomind {

class Monotonic {
public:
    Monotonic(void* base, size_t alignment = 256): ptr_{base}, alignment_{alignment}
    {
        ptr_ = align(ptr_);
    }

    template<class T>
    void operator()(T** ptr, size_t numel) noexcept
    {
        *ptr = (T*)std::exchange(ptr_, align((T*)ptr_ + numel));
    }

    void* ptr() const noexcept
    {
        return ptr_;
    }

private:
    template<class T>
    void* align(T* p)
    {
        static_assert(sizeof(T*) == sizeof(uintptr_t));
        auto x = reinterpret_cast<uintptr_t>(p);
        if (auto remainder = x % alignment_) {
            x += alignment_ - remainder;
        }
        return reinterpret_cast<void*>(x);
    }

    void*  ptr_;
    size_t alignment_;
};

}  // namespace turbomind
