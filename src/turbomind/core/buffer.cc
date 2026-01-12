
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/stream.h"
namespace turbomind::core {

Buffer Buffer::view(DataType dtype) const
{
    auto b = *this;
    if (dtype == dtype_) {
        return b;
    }
    b.dtype_ = dtype;
    b.size_  = numel(dtype, byte_size());
    if (base_) {
        b.base_ = numel(dtype, turbomind::byte_size(dtype_, base_));
    }
    return b;
}

Buffer Buffer::slice(ssize_t base, ssize_t size) const
{
    TM_CHECK_LE(base + size, size_);
    auto b = *this;
    b.base_ += base;
    if (size == -1) {
        b.size_ -= base;
    }
    else {
        b.size_ = size;
    }
    return b;
}

std::ostream& operator<<(std::ostream& os, const Buffer& b)
{
    os << b.dtype() << "[" << b.size() << "]@" << b.data_;
    if (b.base_) {
        os << "+" << b.base_;
    }
    return os;
}

void Copy(const Buffer& a, ssize_t n, Ref<Buffer> b_, const Stream& stream)
{
    auto& b = b_.get();
    TM_CHECK_EQ(a.dtype(), b.dtype());
    TM_CHECK_LE(n, a.size());
    TM_CHECK_LE(n, b.size());
    if (auto size = byte_size(a.dtype(), n)) {
        check_cuda_error(cudaMemcpyAsync(b.raw_data(), a.raw_data(), size, cudaMemcpyDefault, stream.handle()));
    }
}

void Copy(const Buffer& a, ssize_t n, Ref<Buffer> b_)
{
    Copy(a, n, b_, Context::stream());
}

void Copy(const Buffer& a, Ref<Buffer> b_, const Stream& stream)
{
    TM_CHECK_EQ(a.size(), b_.get().size());
    Copy(a, a.size(), b_, stream);
}

void Copy(const Buffer& a, Ref<Buffer> b_)
{
    Copy(a, b_, Context::stream());
}

namespace detail {

void* Copy(const void* a, ssize_t n, void* b, const Stream& stream)
{
    if (n) {
        check_cuda_error(cudaMemcpyAsync(b, a, n, cudaMemcpyDefault, stream.handle()));
    }
    return (uint8_t*)b + n;
}

}  // namespace detail

void Clear(Ref<Buffer> b_, const Stream& stream)
{
    auto& b = b_.get();
    if (auto size = b.byte_size()) {
        check_cuda_error(cudaMemsetAsync(b.raw_data(), 0, b.byte_size(), stream.handle()));
    }
}

void Clear(Ref<Buffer> b_)
{
    Clear(b_, Context::stream());
}

}  // namespace turbomind::core
