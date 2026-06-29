#include "src/turbomind/models/qwen2_vit/qwen2_vit_kernels.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <type_traits>

namespace turbomind {
namespace {

template<int vec_size, typename T>
__global__ void windowReorderKernel(T*         out,
                                    const T*   in,
                                    const int* window_idx,
                                    int64_t    out_stride,
                                    int64_t    in_stride,
                                    int        merge_unit,
                                    int        group_count,
                                    int        dim)
{
    const int dst_group = blockIdx.x;
    const int inner     = blockIdx.y;
    const int di        = (threadIdx.x + blockIdx.z * blockDim.x) * vec_size;
    if (di >= dim) {
        return;
    }

    const int src_group = window_idx[dst_group];
    using Vec           = Array<T, vec_size>;
    Vec x;
    Load(x, in + ((int64_t)src_group * merge_unit + inner) * in_stride + di);
    Store(out + ((int64_t)dst_group * merge_unit + inner) * out_stride + di, x);
}

template<int vec_size, typename T>
__global__ void reverseWindowKernel(
    T* out, const T* in, const int* window_idx, int64_t out_stride, int64_t in_stride, int group_count, int dim)
{
    const int src_group = blockIdx.x;
    const int di        = (threadIdx.x + blockIdx.y * blockDim.x) * vec_size;
    if (di >= dim) {
        return;
    }

    const int dst_group = window_idx[src_group];
    using Vec           = Array<T, vec_size>;
    Vec x;
    Load(x, in + (int64_t)src_group * in_stride + di);
    Store(out + (int64_t)dst_group * out_stride + di, x);
}

__global__ void buildWindowMappedIdxKernel(
    int* window_mapped_idx, const int* mapped_idx, const int* window_idx, int merge_unit, int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int dst_group    = idx / merge_unit;
    const int inner        = idx - dst_group * merge_unit;
    const int src_group    = window_idx[dst_group];
    window_mapped_idx[idx] = mapped_idx[src_group * merge_unit + inner];
}

}  // namespace

void invokeQwen2VitWindowReorder(
    Tensor& out, const Tensor& in, const int* window_idx, int merge_unit, int group_count, cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int dim     = in.shape(1);
    const int threads = 256;

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        constexpr int max_vec = sizeof(uint4) / sizeof(T);

        int best_vec_size = 1;
        for (int v = max_vec; v >= 1; v >>= 1) {
            if (dim % v == 0 && in.stride(0) % v == 0 && out.stride(0) % v == 0) {
                best_vec_size = v;
                break;
            }
        }

        auto launch = [&](auto vec_size_c) {
            constexpr int vec_size = decltype(vec_size_c)::value;
            const dim3    grid(group_count, merge_unit, cdiv(dim, threads * vec_size));
            windowReorderKernel<vec_size, T><<<grid, threads, 0, stream>>>(
                out.data<T>(), in.data<T>(), window_idx, out.stride(0), in.stride(0), merge_unit, group_count, dim);
        };

        switch (best_vec_size) {
            case 8:
                return launch(std::integral_constant<int, 8>{});
            case 4:
                return launch(std::integral_constant<int, 4>{});
            case 2:
                return launch(std::integral_constant<int, 2>{});
            default:
                return launch(std::integral_constant<int, 1>{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(in.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwen2VitReverseWindow(
    Tensor& out, const Tensor& in, const int* window_idx, int group_count, cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int dim     = in.shape(1);
    const int threads = 256;

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        constexpr int max_vec = sizeof(uint4) / sizeof(T);

        int best_vec_size = 1;
        for (int v = max_vec; v >= 1; v >>= 1) {
            if (dim % v == 0 && in.stride(0) % v == 0 && out.stride(0) % v == 0) {
                best_vec_size = v;
                break;
            }
        }

        auto launch = [&](auto vec_size_c) {
            constexpr int vec_size = decltype(vec_size_c)::value;
            const dim3    grid(group_count, cdiv(dim, threads * vec_size));
            reverseWindowKernel<vec_size, T><<<grid, threads, 0, stream>>>(
                out.data<T>(), in.data<T>(), window_idx, out.stride(0), in.stride(0), group_count, dim);
        };

        switch (best_vec_size) {
            case 8:
                return launch(std::integral_constant<int, 8>{});
            case 4:
                return launch(std::integral_constant<int, 4>{});
            case 2:
                return launch(std::integral_constant<int, 2>{});
            default:
                return launch(std::integral_constant<int, 1>{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(in.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwen2VitBuildWindowMappedIdx(int*         window_mapped_idx,
                                        const int*   mapped_idx,
                                        const int*   window_idx,
                                        int          merge_unit,
                                        int          group_count,
                                        cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int total   = group_count * merge_unit;
    const int threads = 256;
    buildWindowMappedIdxKernel<<<cdiv(total, threads), threads, 0, stream>>>(
        window_mapped_idx, mapped_idx, window_idx, merge_unit, total);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
