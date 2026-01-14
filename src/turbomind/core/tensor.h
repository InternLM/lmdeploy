#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/layout.h"

namespace turbomind::core {

class Tensor {
public:
    Tensor() = default;

    Tensor(Layout layout, DataType dtype, Device device): Tensor{layout, dtype, Context::alloc(device)} {}

    Tensor(Layout layout, DataType dtype, Allocator& alloc): layout_{std::move(layout)}
    {
        buffer_ = Buffer(layout_.cosize(), dtype, alloc);
    }

    Tensor(Buffer buffer, Layout layout): layout_{std::move(layout)}, buffer_{buffer.slice(0, layout_.cosize())} {}

    Tensor(Buffer buffer): layout_{buffer.size()}, buffer_{buffer} {}

    Tensor(void* data, Layout layout, DataType dtype, Device device):
        Tensor{Buffer{data, layout.cosize(), dtype, device}, layout}
    {
    }

    Tensor(std::shared_ptr<void> data, Layout layout, DataType dtype, Device device):
        Tensor{Buffer{data, layout.cosize(), dtype, device}, layout}
    {
    }

    template<class T>
    Tensor(T* data, Layout layout, Device device): Tensor{Buffer{data, layout.cosize(), device}, layout}
    {
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

    Device device() const
    {
        return buffer_.device();
    }

    ssize_t size() const noexcept
    {
        return layout_.size();
    }

    ssize_t byte_size() const noexcept
    {
        return turbomind::byte_size(dtype(), size());
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

    template<class T>
    T* data_or(T* other)
    {
        return buffer_.data_or(other);
    }

    template<class T>
    const T* data_or(T* other) const
    {
        return buffer_.data_or(other);
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
        return layout_.shapes(((Is &&) is)...);
    }

    auto& stride() const noexcept
    {
        return layout_.stride();
    }

    auto stride(int i) const
    {
        return layout_.stride(i);
    }

    template<class... Is>
    auto strides(Is&&... is) const
    {
        return layout_.strides(((Is &&) is)...);
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

    Tensor borrow() const
    {
        return Tensor{buffer_.borrow(), layout_};
    }

    Tensor squeeze(int dim) const
    {
        return Tensor{buffer_, layout_.squeeze(dim)};
    }

    Tensor transpose(int a, int b) const
    {
        return Tensor{buffer_, layout_.transpose(a, b)};
    }

    Tensor t() const
    {
        TM_CHECK_EQ(ndim(), 2);
        return transpose(0, 1);
    }

    int ndim() const noexcept
    {
        return layout_.rank();
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    Layout layout_;
    Buffer buffer_;
};

inline Tensor empty_like(const Tensor& tensor)
{
    return Tensor{tensor.layout(), tensor.dtype(), tensor.device()};
}

inline Tensor empty_like(const Tensor& tensor, Device device)
{
    return Tensor{tensor.layout(), tensor.dtype(), device};
}

inline Tensor empty_like(const Tensor& tensor, DataType dtype)
{
    return Tensor{tensor.layout(), dtype, tensor.device()};
}

void Copy(const Tensor& src, Ref<Tensor> dst_, const Stream& stream);

void Copy(const Tensor& src, Ref<Tensor> dst_);

void Clear(Ref<Tensor> a_, const Stream& stream);

void Clear(Ref<Tensor> a_);

#if 0

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

    Tensor_(Layout layout, Device device): Tensor{std::move(layout), data_type_v<T>, device} {}

    Tensor_(Layout layout, Allocator& alloc): Tensor{std::move(layout), data_type_v<T>, alloc} {}

    Tensor_(Buffer buffer, Layout layout): Tensor{ensure_dtype(std::move(buffer)), std::move(layout)} {}

    Tensor_(T* data, Layout layout, Device device): Tensor{data, std::move(layout), device} {}

    Tensor_(shared_ptr<void> data, Layout layout, Device device):
        Tensor{Buffer{std::move(data), layout.cosize(), data_type_v<T>, device}, layout}
    {
    }

    Tensor_(const Tensor_&) = default;
    Tensor_& operator=(const Tensor_&) = default;

    Tensor_(Tensor_&&) noexcept = default;
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
        return Tensor::data<T>();
    }

    const T* data() const noexcept
    {
        return Tensor::data<T>();
    }

    T* data_or(T* other)
    {
        return Tensor::data_or<T>(other);
    }

    const T* data_or(T* other) const
    {
        return Tensor::data_or<T>(other);
    }

    constexpr DataType dtype() const noexcept
    {
        return data_type_v<T>;
    }

private:
    template<class U>
    static decltype(auto) ensure_dtype(U&& u)
    {
        TM_CHECK_EQ(u.dtype(), data_type_v<T>);
        return (U &&) u;
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

    Tensor* try_(const std::string& key);

    const Tensor* try_(const std::string& key) const
    {
        return const_cast<TensorMap*>(this)->try_(key);
    }

    bool contains(const std::string& key) const
    {
        return find(key) != end();
    }

    void produce(const std::string& key, Tensor value)
    {
        TM_CHECK(emplace(key, std::move(value)).second);
    }

    Tensor try_consume(const std::string& key)
    {
        if (auto it = find(key); it != end()) {
            auto value = std::move(it->second);
            erase(it);
            return value;
        }
        return Tensor{};
    }

    Tensor consume(const std::string& key)
    {
        auto value = try_consume(key);
        TM_CHECK(value) << get_out_of_range_msg(key);
        return value;
    }

private:
    std::string get_out_of_range_msg(const std::string& key) const;
};

// clang-format off
template<class Archive, class T, std::enable_if_t<std::is_same_v<Tensor, T>, int> = 0>
void save(Archive& ar, const T& tensor)
{
    TM_CHECK(tensor.size() == 0 || tensor.is_contiguous());
    ar & tensor.buffer(); // implicit convert to tensor
    ar & tensor.layout();
}

template<class Archive>
void load(Archive& ar, Tensor& tensor)
{
    Buffer buffer;
    Layout layout;
    ar & buffer;
    ar & layout;
    tensor = Tensor{std::move(buffer), std::move(layout)};
}


template<class Archive>
void save(Archive& ar, const TensorMap& map)
{
    ar & map.size();
    for (const auto& [k, t]: map) {
        ar & k;
        ar & t;
    }
}

template<class Archive>
void load(Archive& ar, TensorMap& map)
{
    map.clear();
    decltype(map.size()) size;
    ar & size;
    for (int i = 0; i < size; ++i) {
        std::string k;
        Tensor   t;
        ar & k;
        ar & t;
        map.emplace(std::move(k), std::move(t));
    }
}
// clang-format on

}  // namespace turbomind::core
