#include "src/turbomind/models/qwen3_5vit/fast_pos_embed.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

namespace {

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline half from_float<half>(float x)
{
    return __float2half(x);
}

#ifdef ENABLE_BF16
template<>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x)
{
    return __float2bfloat16(x);
}
#endif

// `num_grids` is tiny (usually 1..a few) so a linear scan is fine.
__device__ inline int find_grid(const int* offsets, int num_grids, int pos)
{
    int g = 0;
    for (int i = 1; i < num_grids; ++i) {
        if (offsets[i * 2 + 1] <= pos) {
            g = i;
        }
        else {
            break;
        }
    }
    return g;
}

template<typename T>
__global__ void fastPosEmbedIdxWeightKernel(
    int* idx_out, T* weight_out, const int* grid_thws, const int* grid_offsets, int num_grids, int total_n, int G)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_n) {
        return;
    }

    const int g      = find_grid(grid_offsets, num_grids, pos);
    const int grid_h = grid_thws[g * 3 + 1];
    const int grid_w = grid_thws[g * 3 + 2];
    const int local  = pos - grid_offsets[g * 2 + 1];
    const int i      = local / grid_w;
    const int j      = local % grid_w;

    // torch.linspace(0, G-1, n) uses the halfway-symmetric formulation so
    // that both endpoints are exact:
    //   step      = (end - start) / (n - 1)
    //   halfway   = n / 2
    //   out[i<hw] = start + step * i
    //   out[i>=hw]= end   - step * (n - 1 - i)
    // For n == 1 the single element is `start` (== 0 here); the formula
    // below collapses to 0 since hw_h == 0 is bypassed via grid_h > 1.
    const float end    = (float)(G - 1);
    const float step_h = (grid_h > 1) ? end / (float)(grid_h - 1) : 0.f;
    const float step_w = (grid_w > 1) ? end / (float)(grid_w - 1) : 0.f;

    const int hw_h = grid_h / 2;
    const int hw_w = grid_w / 2;

    const float h_val = (grid_h == 1) ? 0.f : ((i < hw_h) ? step_h * (float)i : end - step_h * (float)(grid_h - 1 - i));
    const float w_val = (grid_w == 1) ? 0.f : ((j < hw_w) ? step_w * (float)j : end - step_w * (float)(grid_w - 1 - j));

    // torch.Tensor.int() truncates toward zero; h_val, w_val are non-negative
    // and bounded above by G-1, so (int) cast is in [0, G-1].
    const int h_floor = (int)h_val;
    const int w_floor = (int)w_val;
    const int h_ceil  = min(h_floor + 1, G - 1);
    const int w_ceil  = min(w_floor + 1, G - 1);

    const float dh = h_val - (float)h_floor;
    const float dw = w_val - (float)w_floor;

    const int base_h      = h_floor * G;
    const int base_h_ceil = h_ceil * G;

    Array<int, 4> idx;
    idx[0] = base_h + w_floor;
    idx[1] = base_h + w_ceil;
    idx[2] = base_h_ceil + w_floor;
    idx[3] = base_h_ceil + w_ceil;

    Array<T, 4> weight;
    weight[0] = from_float<T>((1.f - dh) * (1.f - dw));
    weight[1] = from_float<T>((1.f - dh) * dw);
    weight[2] = from_float<T>(dh * (1.f - dw));
    weight[3] = from_float<T>(dh * dw);

    const int out_base = pos * 4;
    Store(idx_out + out_base, idx);
    Store(weight_out + out_base, weight);
}

}  // namespace

void invokeFastPosEmbedIdxWeight(int*         idx_out,
                                 void*        weight_out,
                                 DataType     dtype,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          total_n,
                                 int          num_grid_per_side,
                                 cudaStream_t stream)
{
    if (total_n <= 0 || num_grids <= 0) {
        return;
    }
    const int block = 256;
    const int grid  = (total_n + block - 1) / block;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        fastPosEmbedIdxWeightKernel<T><<<grid, block, 0, stream>>>(
            idx_out, (T*)weight_out, grid_thws, grid_offsets, num_grids, total_n, num_grid_per_side);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

}  // namespace turbomind
