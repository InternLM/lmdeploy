

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

namespace turbomind::core {

#if 0

namespace kernel {

// This is going to be slow for transposing the innermost dim
template<class T, class Index, int D>
__global__ void GenericCopy(const T*          a,
                            T*                b,
                            Array<int64_t, D> stride_a,
                            Array<int64_t, D> stride_b,
                            Array<Index, D>   shape,
                            int               ndim,
                            int64_t           size)
{
    Index idx = threadIdx.x + (Index)blockIdx.x * blockDim.x;

    if (idx >= size) {
        return;
    }

    Array<int64_t, D> coord;
    PRAGMA_UNROLL
    for (int i = 0; i < D; ++i) {
        if (i < ndim) {
            auto div = idx / shape[i];
            auto mod = idx % shape[i];
            coord[i] = mod;
            idx      = div;
        }
    }

    int64_t idx_a = 0;
    int64_t idx_b = 0;

    PRAGMA_UNROLL
    for (int i = 0; i < D; ++i) {
        if (i < ndim) {
            idx_a += coord[i] * stride_a[i];
            idx_b += coord[i] * stride_b[i];
        }
    }

    b[idx_b] = a[idx_a];
}

}  // namespace kernel

void GenericCopy(const Tensor& src, Tensor& dst, Stream& stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    // Sort strides ascending
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(b.shape());
    }

    const DataType dtype = src.dtype();

    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = get_byte_size(dtype);
    }

    align(get_byte_size(dtype, a.shape(0)));

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(get_byte_size(dtype, a.stride(i)));
        align(get_byte_size(dtype, b.stride(i)));
    }

    const auto vec_size = get_elem_num(alignment, dtype);

    const auto size = a.size() / vec_size;

    int device{};
    check_cuda_error(cudaGetDevice(&device));
    int sm_num{};
    check_cuda_error(cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, device));

    auto invoke = [&](auto vec_t, auto index_t, auto d) {
        using T         = decltype(vec_t);
        using Index     = decltype(index_t);
        constexpr int D = d.value;

        Array<Index, D> shape;
        std::fill(shape.begin() + rank, shape.end(), 1);
        std::copy_n(a.shape().data(), rank, shape.data());

        Array<int64_t, D> stride_a{};
        Array<int64_t, D> stride_b{};
        std::copy_n(a.stride().data(), rank, stride_a.data());
        std::copy_n(b.stride().data(), rank, stride_b.data());

        if (vec_size > 1) {
            shape[0] /= vec_size;
            for (int i = 0; i < rank; ++i) {
                stride_a[i] /= vec_size;
                stride_b[i] /= vec_size;
            }
        }

        auto func = kernel::GenericCopy<T, Index, D>;

        int min_waves  = INT_MAX;
        int block_size = 0;
        int grid_size  = 0;

        for (int threads = 256; threads <= 1024; threads *= 2) {
            int blocks = cdiv<ssize_t>(size, block_size);
            int n_active{};
            check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_active, func, block_size, 0));
            int waves = cdiv(blocks, n_active * sm_num);
            if (waves < min_waves) {
                min_waves  = waves;
                block_size = threads;
                grid_size  = blocks;
            }
        }

        func<<<grid_size, block_size, 0, stream.handle()>>>(
            (const T*)data_a, (T*)data_b, stride_a, stride_b, shape, rank, a.size());
    };

    auto invoke_d = [&](auto vec_t, auto idx_t) {
        if (rank <= 2) {
            invoke(vec_t, idx_t, constant<2>{});
        }
        else if (rank <= 4) {
            invoke(vec_t, idx_t, constant<4>{});
        }
        else if (rank <= 8) {
            invoke(vec_t, idx_t, constant<8>{});
        }
        else {
            throw std::runtime_error("not implemented");
        }
    };

    auto invoke_i = [&](auto vec_t) {
        if (size < INT_MAX) {
            invoke_d(vec_t, int{});
        }
        else {
            invoke_d(vec_t, int64_t{});
        }
    };

    switch (alignment) {
        case 16:
            return invoke_i(uint4{});
        case 8:
            return invoke_i(uint2{});
        case 4:
            return invoke_i(uint{});
        case 2:
            return invoke_i(ushort{});
        default:
            return invoke_i(char{});
    }
}

#endif

}  // namespace turbomind::core
