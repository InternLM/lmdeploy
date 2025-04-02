
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/stream.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind::core {

Buffer Buffer::view(DataType dtype) const
{
    auto b = *this;
    if (dtype == dtype_) {
        return b;
    }
    b.dtype_ = dtype;
    b.size_  = get_elem_num(byte_size(), dtype);
    if (base_) {
        b.base_ = get_elem_num(get_byte_size(dtype_, base_), dtype);
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

void Copy(const Buffer& a, ssize_t n, Ref<Buffer> b_, const Stream& stream)
{
    auto& b = b_.get();
    TM_CHECK_EQ(a.dtype(), b.dtype());
    TM_CHECK_LE(n, a.size());
    TM_CHECK_LE(n, b.size());
    check_cuda_error(
        cudaMemcpyAsync(b.raw_data(), a.raw_data(), get_byte_size(a.dtype(), n), cudaMemcpyDefault, stream.handle()));
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

std::byte* Copy(const std::byte* a, ssize_t n, std::byte* b, const Stream& stream)
{
    check_cuda_error(cudaMemcpyAsync(b, a, n, cudaMemcpyDefault, stream.handle()));
    return b + n;
}

void Clear(Ref<Buffer> b_, const Stream& stream)
{
    auto& b = b_.get();
    check_cuda_error(cudaMemsetAsync(b.raw_data(), 0, b.byte_size(), stream.handle()));
}

void Clear(Ref<Buffer> b_)
{
    Clear(b_, Context::stream());
}

}  // namespace turbomind::core