
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/stream.h"

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
    b.size_ = size;
    return b;
}

void Copy(const Buffer& src, Buffer& dst, const Stream& stream)
{
    TM_CHECK_EQ(src.dtype(), dst.dtype());
    TM_CHECK_EQ(src.size(), dst.size());
    check_cuda_error(
        cudaMemcpyAsync(dst.raw_data(), src.raw_data(), src.byte_size(), cudaMemcpyDefault, stream.handle()));
}

void Copy(const Buffer& src, Buffer&& dst, const Stream& stream)
{
    ContextGuard g{stream};
    return Copy(src, dst, stream);
}

void Copy(const Buffer& src, Buffer&& dst)
{
    TM_CHECK(0) << "Not implemented";
}

}  // namespace turbomind::core