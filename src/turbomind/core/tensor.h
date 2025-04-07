#pragma once

#include <string>
#include <unordered_map>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/layout.h"

#include "src/turbomind/utils/Tensor.h"

namespace turbomind::core {

class Tensor {
public:
    Tensor() = default;

    Tensor(Layout layout, DataType dtype, MemLoc device): Tensor{layout, dtype, Context::alloc(device)} {}

    Tensor(Layout layout, DataType dtype, Allocator& alloc): layout_{std::move(layout)}
    {
        buffer_ = Buffer(layout_.cosize(), dtype, alloc);
    }

    Tensor(Buffer buffer, Layout layout): layout_{std::move(layout)}, buffer_{std::move(buffer)}
    {
        TM_CHECK_LE(layout_.cosize(), buffer_.size());
    }

    Tensor(Buffer buffer): layout_{buffer.size()}, buffer_{buffer} {}

    Tensor(void* data, Layout layout, DataType dtype, MemLoc device):
        Tensor{Buffer{data, layout.cosize(), dtype, device}, layout}
    {
    }

    template<class T>
    Tensor(T* data, Layout layout, MemLoc device): Tensor{Buffer{data, layout.cosize(), device}, layout}
    {
    }

    static Tensor empty_like(const Tensor& tensor, std::optional<MemLoc> device = {})
    {
        return Tensor{tensor.layout_, tensor.dtype(), device ? *device : tensor.device()};
    }

    Buffer& buffer() noexcept
    {
        return buffer_;
    }

    const Buffer& buffer() const noexcept
    {
        return buffer_;
    }

    DataType dtype() const
    {
        return buffer_.dtype();
    }

    MemLoc device() const
    {
        return buffer_.device();
    }

    ssize_t size() const noexcept
    {
        return layout_.size();
    }

    ssize_t byte_size() const noexcept
    {
        return get_byte_size(dtype(), size());
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(buffer_);
    }

    template<class T>
    T* data()
    {
        return buffer_.data<T>();
    }

    template<class T>
    const T* data() const
    {
        return const_cast<Tensor*>(this)->data<T>();
    }

    void* raw_data()
    {
        return buffer_.raw_data();
    }

    const void* raw_data() const
    {
        return const_cast<Tensor*>(this)->raw_data();
    }

    Tensor view(std::vector<ssize_t> shape) const
    {
        return Tensor{buffer_, layout_.view(std::move(shape))};
    }

    auto& layout() const noexcept
    {
        return layout_;
    }

    auto& shape() const noexcept
    {
        return layout_.shape();
    }

    auto shape(int i) const
    {
        return layout_.shape(i);
    }

    template<class... Is>
    auto shapes(Is&&... is) const
    {
        return layout_.shapes(((Is&&)is)...);
    }

    auto& stride() const noexcept
    {
        return layout_.stride();
    }

    auto stride(int i) const
    {
        return layout_.stride(i);
    }

    bool is_contiguous() const noexcept
    {
        return layout().is_contiguous();
    }

    Tensor slice(std::vector<ssize_t> base, std::vector<ssize_t> shape) const
    {
        auto&& [layout, offset] = layout_.slice(base, std::move(shape));
        const auto cosize       = layout.cosize();
        return Tensor{buffer_.slice(offset, cosize), std::move(layout)};
    }

    // The outermost dimension
    Tensor slice(ssize_t base, ssize_t size = 1) const
    {
        vector<ssize_t> bases(shape().size());
        bases.front() = base;
        vector<ssize_t> sizes{this->shape()};
        sizes.front() = size;
        return slice(bases, sizes);
    }

    Tensor squeeze(int dim) const
    {
        return Tensor{buffer_, layout_.squeeze(dim)};
    }

    int ndim() const noexcept
    {
        return layout_.rank();
    }

private:
    Layout layout_;
    Buffer buffer_;
};

#if 0
void Copy(const Tensor& src, Tensor& dst, Stream& stream);

void Copy(const Tensor& src, Tensor&& dst, Stream& stream);

// Launch a kernel to perform the complicated copying
void GenericCopy(const Tensor& src, Tensor& dst, Stream& stream);

Tensor Reshape(const Tensor& t, vector<ssize_t> shape);

Tensor Transpoe(const Tensor& t, int dim0, int dim1);

Tensor Permute(const Tensor& t, vector<int> dims);

Tensor Contiguous(const Tensor& t);
#endif

template<class T>
struct Tensor_: public Tensor {
    Tensor_() = default;

    Tensor_(Layout layout, MemLoc device): Tensor{std::move(layout), getTensorType<T>(), device} {}

    Tensor_(Layout layout, Allocator& alloc): Tensor{std::move(layout), getTensorType<T>(), alloc} {}

    Tensor_(Buffer buffer, Layout layout): Tensor{ensure_dtype(std::move(buffer)), std::move(layout)} {}

    Tensor_(T* data, Layout layout, MemLoc device): Tensor{data, std::move(layout), device} {}

    Tensor_(shared_ptr<void> data, Layout layout, MemLoc device):
        Tensor{Buffer{std::move(data), layout.cosize(), getTensorType<T>(), device}, layout}
    {
    }

    Tensor_(const Tensor_&)            = default;
    Tensor_& operator=(const Tensor_&) = default;

    Tensor_(Tensor_&&) noexcept            = default;
    Tensor_& operator=(Tensor_&&) noexcept = default;

    Tensor_(const Tensor& other)
    {
        *static_cast<Tensor*>(this) = ensure_dtype(other);
    }
    Tensor_(Tensor&& other) noexcept
    {
        *static_cast<Tensor*>(this) = ensure_dtype(std::move(other));
    }

    ssize_t offset(const vector<ssize_t>& idxs)
    {
        return layout().offset(idxs);
    }

    T* data() noexcept
    {
        return static_cast<T*>(raw_data());
    }

    const T* data() const noexcept
    {
        return static_cast<const T*>(raw_data());
    }

    constexpr DataType dtype() const noexcept
    {
        return dtype_;
    }

private:
    static constexpr DataType dtype_ = getTensorType<T>();

    template<class U>
    static decltype(auto) ensure_dtype(U&& u)
    {
        TM_CHECK_EQ(u.dtype(), dtype_);
        return (U&&)u;
    }
};

class TensorMap: public std::unordered_map<std::string, Tensor> {
public:
    using std::unordered_map<std::string, Tensor>::unordered_map;

    Tensor& at(const std::string& key);

    const Tensor& at(const std::string& key) const
    {
        return const_cast<TensorMap*>(this)->at(key);
    }

private:
    std::string get_out_of_range_msg(const std::string& key) const;
};

}  // namespace turbomind::core