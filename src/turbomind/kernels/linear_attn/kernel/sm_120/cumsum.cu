#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kCumsumWarpSize = 32;

template<int ChunkSize>
struct ChunkCumsumPolicy;

template<>
struct ChunkCumsumPolicy<32> {
    static constexpr int kHeadsPerBlock = 8;
};

__device__ __forceinline__ float WarpInclusiveScan(float value, int lane)
{
#pragma unroll
    for (int step = 1; step < kCumsumWarpSize; step <<= 1) {
        const float addend = __shfl_up_sync(0xffffffffu, value, step);
        if (lane >= step) {
            value += addend;
        }
    }
    return value;
}

template<int ChunkSize, int HeadsPerBlock>
__global__ void ParallelChunkLocalCumsumKernel(const float* __restrict__ g,
                                               const int32_t* __restrict__ q_offsets,
                                               float* __restrict__ g_cumsum,
                                               int     sequence_num,
                                               int     token_num,
                                               int     hv,
                                               int64_t input_gate_stride,
                                               int64_t input_gate_batch_stride,
                                               int64_t output_gate_stride,
                                               int64_t output_gate_batch_stride)
{
    static_assert(ChunkSize == 32);
    static_assert(ChunkSize % kCumsumWarpSize == 0);
    static_assert(ChunkSize * HeadsPerBlock <= 1024);

    __shared__ float lower_totals[HeadsPerBlock];

    const int global_chunk_id = static_cast<int>(blockIdx.x);
    const int head_group      = static_cast<int>(blockIdx.y);
    const int token_lane      = static_cast<int>(threadIdx.x) & (ChunkSize - 1);
    const int head_lane       = static_cast<int>(threadIdx.x) / ChunkSize;
    const int warp_lane       = token_lane & (kCumsumWarpSize - 1);
    const int hv_id           = head_group * HeadsPerBlock + head_lane;

    int sequence_id    = 0;
    int local_chunk_id = global_chunk_id;

    for (int b = 0; b < sequence_num; ++b) {
        const int seq_start  = q_offsets[b];
        const int seq_end    = q_offsets[b + 1];
        const int seq_chunks = cdiv(seq_end - seq_start, ChunkSize);
        if (local_chunk_id < seq_chunks) {
            sequence_id = b;
            break;
        }
        local_chunk_id -= seq_chunks;
    }

    const int seq_start   = q_offsets[sequence_id];
    const int seq_end     = q_offsets[sequence_id + 1];
    const int token0      = seq_start + local_chunk_id * ChunkSize;
    const int remaining   = seq_end - token0;
    const int token_count = remaining < ChunkSize ? remaining : ChunkSize;

    float      value         = 0.0f;
    int64_t    input_offset  = 0;
    int64_t    output_offset = 0;
    const bool valid         = hv_id < hv && token_lane < token_count;
    if (valid) {
        const int flat_token = token0 + token_lane;
        const int batch_id   = flat_token / token_num;
        const int token      = flat_token - batch_id * token_num;
        input_offset         = static_cast<int64_t>(batch_id) * input_gate_batch_stride
                       + static_cast<int64_t>(token) * input_gate_stride + hv_id;
        output_offset = static_cast<int64_t>(batch_id) * output_gate_batch_stride
                        + static_cast<int64_t>(token) * output_gate_stride + hv_id;
        value = g[input_offset];
    }

    value = WarpInclusiveScan(value, warp_lane);
    if (token_lane == kCumsumWarpSize - 1) {
        lower_totals[head_lane] = value;
    }
    __syncthreads();

    if (token_lane >= kCumsumWarpSize) {
        value += lower_totals[head_lane];
    }
    if (valid) {
        g_cumsum[output_offset] = value;
    }
}

template<int ChunkSize>
void LaunchChunkCumsum(const core::Tensor& g,
                       const core::Tensor& q_offsets,
                       core::Tensor&       g_cumsum,
                       const Problem&      problem,
                       cudaStream_t        stream)
{
    static_assert(ChunkSize == 32);

    const float* g_ptr   = g.data<float>();
    float*       out_ptr = g_cumsum.data<float>();

    if (problem.total_chunks == 0) {
        return;
    }

    constexpr int kHeadsPerBlock = ChunkCumsumPolicy<ChunkSize>::kHeadsPerBlock;
    constexpr int kThreads       = ChunkSize * kHeadsPerBlock;
    dim3          grid(problem.total_chunks, cdiv(problem.hv, kHeadsPerBlock));
    ParallelChunkLocalCumsumKernel<ChunkSize, kHeadsPerBlock><<<grid, kThreads, 0, stream>>>(g_ptr,
                                                                                             q_offsets.data<int32_t>(),
                                                                                             out_ptr,
                                                                                             problem.sequence_num,
                                                                                             problem.token_num,
                                                                                             problem.hv,
                                                                                             g.stride(1),
                                                                                             g.stride(0),
                                                                                             g_cumsum.stride(1),
                                                                                             g_cumsum.stride(0));
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace detail {

void LaunchChunk32LocalCumsum(const core::Tensor& g,
                              const core::Tensor& q_offsets,
                              core::Tensor&       g_cumsum,
                              const Problem&      problem,
                              cudaStream_t        stream)
{
    LaunchChunkCumsum<32>(g, q_offsets, g_cumsum, problem, stream);
}

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
