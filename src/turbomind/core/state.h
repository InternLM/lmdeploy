
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/layout.h"
#include "src/turbomind/core/tensor.h"
#include <algorithm>

namespace turbomind {

// Goals:
// 1. constant number of cudaMemcpy / kernel launches
// 2. single stream synchronization / iteration

struct State {

    Tensor data_[2];

    State() = default;

    State(const Layout& layout, DataType dtype, const core::Device& device)
    {
        data_[0] = {layout, dtype, device};
        data_[1] = {layout, dtype, device};
    }

    Tensor& front()
    {
        return data_[0];
    }

    Tensor& back()
    {
        return data_[1];
    }

    void Swap()
    {
        std::swap(data_[0], data_[1]);
    }
};

template<class Copy>
void Warp(const Tensor& a0, int size0, const Buffer_<int>& perm, Tensor b1, Copy& copy)
{
    auto a0_ptr = (const uint8_t*)a0.raw_data();
    auto b1_ptr = (uint8_t*)b1.raw_data();

    const auto vec_size = byte_size(a0.dtype(), a0.stride(0));

    for (int i = 0; i < perm.size(); ++i) {
        if (const int j = perm[i]; TM_LIKELY(j < size0)) {
            copy(a0_ptr + j * vec_size, vec_size, b1_ptr + i * vec_size);
        }
    }
}

template<class Copy>
void Warp(const Tensor& a0, const Tensor& b1, int size0, const Buffer_<int>& perm, Tensor c1, Copy& copy)
{
    auto a0_ptr = (const uint8_t*)a0.raw_data();
    auto b1_ptr = (const uint8_t*)b1.raw_data();
    auto c1_ptr = (uint8_t*)c1.raw_data();

    const auto vec_size = byte_size(a0.dtype(), a0.stride(0));

    for (int i = 0; i < perm.size(); ++i) {
        const uint8_t* src_ptr = TM_LIKELY(perm[i] < size0) ? a0_ptr + perm[i] * vec_size : b1_ptr + i * vec_size;
        copy(src_ptr, vec_size, c1_ptr + i * vec_size);
    }
}

template<class Copy>
void Warp(const Tensor&       src0,
          const Buffer_<int>& offset0,
          int                 size0,
          const Tensor&       src1,
          const Buffer_<int>& offset1,
          const Buffer_<int>& perm0,
          Tensor              dst,
          Buffer_<int>        offsetd,
          Copy&               copy)
{
    auto p_src0 = (const uint8_t*)src0.raw_data();
    auto p_src1 = (const uint8_t*)src1.raw_data();

    const ssize_t vec_size = byte_size(src0.dtype(), src0.stride(0));

    auto p_dst = (uint8_t*)dst.raw_data();

    offsetd[0] = 0;

    for (int i = 0; i < perm0.size(); ++i) {
        const uint8_t* p_src;
        ssize_t        n;
        if (const int j = perm0[i]; TM_LIKELY(j < size0)) {
            p_src = p_src0 + offset0[j] * vec_size;
            n     = offset0[j + 1] - offset0[j];
        }
        else {
            p_src = p_src1 + offset1[i] * vec_size;
            n     = offset1[i + 1] - offset1[i];
        }
        offsetd[i + 1] = offsetd[i] + n;
        copy(p_src, n * vec_size, p_dst + offsetd[i] * vec_size);
    }
}

// d1[i] = a0[perm[i]]:b0[perm[i]] if perm[i] < size0 else c1[i]
// where `a0` has variable size with fixed stride
//       `b0` has fixed size (1)
//       `a1` has variable size
//       `c1` has variable size with fixed stride
template<class Copy>
void Append(const Tensor&       a0,
            const Buffer_<int>& a0_size,
            const Tensor&       b0,
            const Tensor&       c1,
            const Buffer_<int>& c1_offset,
            const Buffer_<int>& perm,
            int                 size0,
            Tensor              d1,
            Buffer_<int>        d1_size,
            Copy&               copy)
{
    auto a0_ptr = (const uint8_t*)a0.raw_data();
    auto b0_ptr = (const uint8_t*)b0.raw_data();
    auto c1_ptr = (const uint8_t*)c1.raw_data();

    auto d1_ptr = (uint8_t*)d1.raw_data();

    TM_CHECK_EQ(a0.stride(0), d1.stride(0));

    const auto stride   = byte_size(a0.dtype(), a0.stride(0));
    const auto vec_size = byte_size(a0.dtype(), a0.stride(1));

    for (int i = 0; i < perm.size(); ++i) {
        if (const int j = perm[i]; TM_LIKELY(j < size0)) {
            uint8_t* out = copy(a0_ptr + j * stride, vec_size * a0_size[j], d1_ptr + i * stride);
            copy(b0_ptr + j * vec_size, vec_size, out);
            d1_size[i] = a0_size[j] + 1;
        }
        else {
            const auto n = c1_offset[i + 1] - c1_offset[i];
            copy(c1_ptr + c1_offset[i] * vec_size, n * vec_size, d1_ptr + i * stride);
            d1_size[i] = n;
        }
    }
}

}  // namespace turbomind
