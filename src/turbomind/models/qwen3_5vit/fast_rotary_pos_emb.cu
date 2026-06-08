#include "src/turbomind/models/qwen3_5vit/fast_rotary_pos_emb.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

namespace {

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
__global__ void fastRotaryPosEmbKernel(T*         cos_sin_out,
                                       const int* grid_thws,
                                       const int* grid_offsets,
                                       int        num_grids,
                                       int        total_hw,
                                       int        head_dim,
                                       float      scale)  // -log2(theta) / (head_dim/4)
{
    const int pair_count = head_dim / 2;  // e.g. 36
    const int freq_half  = head_dim / 4;  // e.g. 18

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos    = tid / pair_count;
    const int pair_k = tid % pair_count;
    if (pos >= total_hw) {
        return;
    }

    const int g      = find_grid(grid_offsets, num_grids, pos);
    const int grid_w = grid_thws[g * 3 + 2];
    const int local  = pos - grid_offsets[g * 2 + 1];
    const int i      = local / grid_w;  // h_coord
    const int j      = local % grid_w;  // w_coord

    // Pairs [0, freq_half) rotate in h; pairs [freq_half, 2*freq_half) rotate in w.
    const int   freq_idx = pair_k % freq_half;
    const int   coord    = (pair_k < freq_half) ? i : j;
    const float inv_freq = exp2f((float)freq_idx * scale);

    float c, s;
    sincosf((float)coord * inv_freq, &s, &c);

    Array<T, 2> cs{(T)c, (T)s};
    Store(cos_sin_out + (size_t)pos * head_dim + pair_k * 2, cs);
}

}  // namespace

void invokeQwen3VitRotaryPosEmb(void*        cos_sin,
                                DataType     dtype,
                                const int*   grid_thws,
                                const int*   grid_offsets,
                                int          num_grids,
                                int          total_hw,
                                int          head_dim,
                                float        theta,
                                cudaStream_t stream)
{
    if (total_hw <= 0 || num_grids <= 0 || head_dim <= 0) {
        return;
    }
    TM_CHECK(head_dim % 4 == 0) << "head_dim must be divisible by 4, got " << head_dim;

    const int   total = total_hw * (head_dim / 2);
    const int   block = 256;
    const int   grid  = (total + block - 1) / block;
    const float scale = -log2f(theta) / (float)(head_dim / 4);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        fastRotaryPosEmbKernel<T>
            <<<grid, block, 0, stream>>>((T*)cos_sin, grid_thws, grid_offsets, num_grids, total_hw, head_dim, scale);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

}  // namespace turbomind
