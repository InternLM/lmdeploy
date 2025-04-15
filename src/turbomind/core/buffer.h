#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <optional>
#include <type_traits>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/common.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

class Buffer {
public:
    Buffer(): data_{}, base_{}, size_{}, device_{}, dtype_{} {}

    // Typed empty buffer
    explicit Buffer(DataType dtype): Buffer()
    {
        dtype_ = dtype;
    }

    // Reference into `data` buffer
    template<class T>
    Buffer(T* data, ssize_t size, MemoryLocation device):
        data_{data, [](auto) {}}, base_{}, size_{size}, device_{device}, dtype_{data_type_v<T>}
    {
    }

    Buffer(void* data, ssize_t size, DataType dtype, MemoryLocation device):
        data_{data, [](auto) {}}, base_{}, size_{size}, device_{device}, dtype_{dtype}
    {
    }

    // Share ownership of `data`
    Buffer(shared_ptr<void> data, ssize_t size, DataType dtype, MemoryLocation device):
        data_{std::move(data)}, base_{}, size_{size}, device_{device}, dtype_{dtype}
    {
    }

    // Create from the allocator
    Buffer(ssize_t size, DataType dtype, Allocator& alloc):
        base_{}, size_{size}, device_{alloc->device()}, dtype_{dtype}
    {
        auto bytes = bytesize(dtype, size);
        data_      = {alloc->allocate(bytes), [=](auto p) { alloc->deallocate(p, bytes); }};
    }

    Buffer(ssize_t size, DataType dtype, MemLoc device): Buffer{size, dtype, Context::alloc(device)} {}

    template<class T>
    T* data()
    {
        TM_CHECK_EQ(data_type_v<T>, dtype_);
        return static_cast<T*>(raw_data());
    }

    template<class T>
    const T* data() const
    {
        return const_cast<Buffer*>(this)->data<T>();
    }

    void* raw_data(ssize_t offset = 0)
    {
        return TM_CHECK_NOTNULL((char*)data_.get()) + bytesize(dtype_, base_ + offset);
    }

    const void* raw_data(ssize_t offset = 0) const
    {
        return const_cast<Buffer*>(this)->raw_data(offset);
    }

    template<class T = void>
    T* unsafe_data() const
    {
        return (T*)((char*)data_.get() + bytesize(dtype_, base_));
    }

    DataType dtype() const
    {
        return dtype_;
    }

    MemoryLocation device() const
    {
        return device_;
    }

    ssize_t size() const
    {
        return size_;
    }

    ssize_t byte_size() const
    {
        return bytesize(dtype_, size_);
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(data_);
    }

    Buffer view(DataType dtype) const;

    template<class T>
    Buffer view() const
    {
        return view(data_type_v<T>);
    }

    Buffer slice(ssize_t base, ssize_t size) const;

    Buffer borrow() const
    {
        return Buffer{const_cast<void*>(raw_data()), size_, dtype_, device_};
    }

    friend bool operator==(const Buffer& a, const Buffer& b);

    friend bool operator!=(const Buffer& a, const Buffer& b);

private:
    auto as_tuple() const
    {
        return std::tie(data_, base_, size_, dtype_, device_);
    }

    shared_ptr<void> data_;
    ssize_t          base_;
    ssize_t          size_;
    MemoryLocation   device_;
    DataType         dtype_;
};

inline bool operator==(const Buffer& a, const Buffer& b)
{
    return a.as_tuple() == b.as_tuple();
}

inline bool operator!=(const Buffer& a, const Buffer& b)
{
    return !(a == b);
}

///////////////////////////////////////////////////////////
// fill

void Fill(Buffer& b, const void* v);

void Fill(Buffer&& b, const void* v);

void Fill(Buffer& b, const void* v, const Stream& stream);

void Fill(Buffer&& b, const void* v, const Stream& stream);

template<class T>
struct Buffer_: public Buffer {

    Buffer_(): Buffer{data_type_v<T>} {}

    Buffer_(T* data, ssize_t size, MemLoc device): Buffer{data, size, device} {}

    Buffer_(shared_ptr<void> data, ssize_t size, MemLoc device): Buffer{std::move(data), size, data_type_v<T>, device}
    {
    }

    Buffer_(ssize_t size, Allocator& alloc): Buffer{size, data_type_v<T>, alloc} {}

    Buffer_(ssize_t size, MemLoc device): Buffer{size, data_type_v<T>, device} {}

    Buffer_(const Buffer_&)            = default;
    Buffer_& operator=(const Buffer_&) = default;

    Buffer_(Buffer_&&) noexcept            = default;
    Buffer_& operator=(Buffer_&&) noexcept = default;

    Buffer_(const Buffer& b)
    {
        *static_cast<Buffer*>(this) = ensure_dtype(b);
    }
    Buffer_(Buffer&& b) noexcept
    {
        *static_cast<Buffer*>(this) = ensure_dtype(std::move(b));
    }

    T* data()
    {
        return static_cast<T*>(raw_data());
    }

    const T* data() const
    {
        return static_cast<const T*>(raw_data());
    }

    T* begin()
    {
        return data();
    }

    const T* begin() const
    {
        return data();
    }

    T* end()
    {
        return begin() + size();
    }

    const T* end() const
    {
        return begin() + size();
    }

    T& operator[](ssize_t i)
    {
        return data()[i];
    }

    const T& operator[](ssize_t i) const
    {
        return data()[i];
    }

    T& at(ssize_t i)
    {
        TM_CHECK_LT(i, size());
        return data()[i];
    }

    T& at(ssize_t i) const
    {
        TM_CHECK_LT(i, size());
        return data()[i];
    }

    constexpr DataType dtype() const noexcept
    {
        return data_type_v<T>;
    }

private:
    template<class U>
    static decltype(auto) ensure_dtype(U&& u) noexcept
    {
        TM_CHECK_EQ(u.dtype(), data_type_v<T>);
        return (U&&)u;
    }
};

template<class T>
class Ref {
public:
    Ref(T& x): ref_{x} {}
    Ref(T&& x): ref_{x} {}

    operator T&()
    {
        return ref_;
    }

    T& get()
    {
        return ref_;
    }

private:
    T& ref_;
};

void Copy(const Buffer& a, ssize_t n, Ref<Buffer> b_, const Stream& stream);

void Copy(const Buffer& a, ssize_t n, Ref<Buffer> b_);

void Copy(const Buffer& a, Ref<Buffer> b_, const Stream& stream);

void Copy(const Buffer& a, Ref<Buffer> b_);

// Static type checking
template<class T>
inline void Copy_(const Buffer_<T>& a, ssize_t n, Buffer_<T>& b_)
{
    Copy((const Buffer&)a, n, (Buffer&)b_);
}

void* Copy(const void* a, ssize_t n, void* b, const Stream& stream);

template<class T>
inline T* Copy(const T* a, ssize_t n, T* b, const Stream& stream)
{
    return (T*)Copy((const void*)a, sizeof(T) * n, (void*)b, stream);
}

template<class T>
inline T* Copy(const T* a, ssize_t n, T* b)
{
    return Copy(a, n, b, Context::stream());
}

void Clear(Ref<Buffer> b_, const Stream& stream);

void Clear(Ref<Buffer> b_);

}  // namespace turbomind::core