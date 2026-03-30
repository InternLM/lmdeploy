
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/stream.h"

namespace turbomind::core {

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << t.dtype() << "[" << t.layout() << "]@" << t.buffer_.data_or((void*)nullptr);
    return os;
}

Tensor& TensorMap::at(const std::string& key)
{
    auto it = find(key);
    TM_CHECK(it != end()) << get_out_of_range_msg(key);
    return it->second;
}

std::string TensorMap::get_out_of_range_msg(const std::string& key) const
{
    std::ostringstream oss;
    oss << "Cannot find a tensor of name '" << key << "' in the tensor map (keys: ";
    auto sep = "";
    for (const auto& [k, _] : *this) {
        oss << std::exchange(sep, ", ") << k;
    }
    oss << ")";
    return oss.str();
}

Tensor* TensorMap::try_(const std::string& key)
{
    auto it = find(key);
    if (it != end()) {
        return &it->second;
    }
    return nullptr;
}

void Copy(const Tensor& src, Ref<Tensor> dst_, const Stream& stream)
{
    auto& dst = dst_.get();
    TM_CHECK(src.dtype() == dst.dtype());
    TM_CHECK(src.shape() == dst.shape());
    TM_CHECK(src.is_contiguous());
    TM_CHECK(dst.is_contiguous());
    if (auto size = src.byte_size()) {
        check_cuda_error(cudaMemcpyAsync(dst.raw_data(), src.raw_data(), size, cudaMemcpyDefault, stream.handle()));
    }
}

void Copy(const Tensor& src, Ref<Tensor> dst_)
{
    Copy(src, dst_, Context::stream());
}

void Clear(Ref<Tensor> a_, const Stream& stream)
{
    auto& a = a_.get();
    TM_CHECK(a.is_contiguous());
    if (auto size = a.byte_size()) {
        check_cuda_error(cudaMemsetAsync(a.raw_data(), 0, size, stream.handle()));
    }
}

void Clear(Ref<Tensor> a_)
{
    Clear(a_, Context::stream());
}

#if 0

void Copy(const Tensor& src, Tensor& dst, Stream& stream)
{
    TM_CHECK(src.dtype() == dst.dtype());
    TM_CHECK(src.shape() == dst.shape());

    const DataType dtype = src.dtype();

    auto trivial = [&] {
        const ssize_t bytesize = get_byte_size(dtype, src.size());
        check_cuda_error(cudaMemcpyAsync(dst.raw_data(), src.raw_data(), bytesize, cudaMemcpyDefault, stream.handle()));
    };

    if (src.layout().is_contiguous() && dst.layout().is_contiguous()) {
        return trivial();
    }

    auto a = src.layout();
    auto b = dst.layout();

    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
        return a.stride()[j] < a.stride()[i];
    });

    // innermost dim is not contiguous
    if (a.stride(idxs.back()) > 1 || b.stride(idxs.back()) > 1) {
        return GenericCopy(src, dst, stream);
    }

    a = a.reorder(idxs);
    b = b.reorder(idxs);

    // trivial after reorder (e.g. transposed matrices)
    if (a.is_contiguous() && b.is_contiguous()) {
        return trivial();
    }

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (rank > 3) {
        return GenericCopy(src, dst, stream);
    }

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(b.shape());
    }

    if (rank == 2) {
        check_cuda_error(cudaMemcpy2DAsync(dst.raw_data(),
                                           get_byte_size(dtype, b.stride(0)),
                                           src.raw_data(),
                                           get_byte_size(dtype, a.stride(0)),
                                           get_byte_size(dtype, a.shape(1)),
                                           a.shape(0),
                                           cudaMemcpyDefault,
                                           stream.handle()));
        return;
    }

    auto [a0, a1] = a.strides(0, 1);
    auto [b0, b1] = b.strides(0, 1);

    // make sure the underlying space is actually a cube [x % (y * z) == 0]
    if (rank == 3 && a0 % a1 == 0 && b0 % b1 == 0) {
        const auto xsz_a = get_byte_size(dtype, a.stride(1));
        const auto xsz_b = get_byte_size(dtype, b.stride(1));
        const auto ysz_a = a0 / a1;
        const auto ysz_b = b0 / b1;

        cudaMemcpy3DParms param{};
        param.srcPtr = make_cudaPitchedPtr((void*)src.raw_data(), xsz_a, xsz_a, ysz_a);
        param.dstPtr = make_cudaPitchedPtr((void*)dst.raw_data(), xsz_b, xsz_b, ysz_b);
        param.extent = make_cudaExtent(get_byte_size(dtype, a.shape(2)), a.shape(1), a.shape(0));
        param.kind   = cudaMemcpyDefault;

        if (auto ec = cudaMemcpy3DAsync(&param, stream.handle()); ec == cudaSuccess) {
            TM_LOG_WARNING(cudaGetErrorString(ec));
            return;
        }
    }

    return GenericCopy(src, dst, stream);
}

void Copy(const Tensor& src, Tensor&& dst, Stream& stream)
{
    return Copy(src, dst, stream);
}

#endif

}  // namespace turbomind::core
