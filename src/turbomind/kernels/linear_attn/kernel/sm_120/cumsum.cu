#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/tma_desc_prepare.h"

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<int ChunkSize, bool PrepareDirectDescriptors = false>
struct Sm120ChunkLocalCumsum {
    static_assert(ChunkSize == 32);

    using DescriptorPrepare = Sm120GdrTmaDescPrepare<float, ChunkSize>;
    using ChunkedKktTma     = typename DescriptorPrepare::ChunkedKktTma;
    template<class T>
    using StridedTensorBase = typename DescriptorPrepare::template StridedTensorBase<T>;

    static constexpr int kWarpSize       = 32;
    static constexpr int kWarps          = 4;
    static constexpr int kHeadsPerBlock  = 32;
    static constexpr int kThreads        = kWarps * kWarpSize;
    static constexpr int kStageRowStride = ChunkSize + 1;

    struct TmaTemplates {
        CUtensorMap kkt_k{};
        CUtensorMap kkt_resolvent{};
        CUtensorMap fused_q{};
        CUtensorMap fused_k{};
        CUtensorMap fused_v{};
        CUtensorMap fused_g{};
        CUtensorMap fused_resolvent{};
        CUtensorMap fused_out{};
    };

    struct DescriptorBases {
        StridedTensorBase<const __nv_bfloat16> q{};
        StridedTensorBase<const __nv_bfloat16> k{};
        StridedTensorBase<const __nv_bfloat16> v{};
        StridedTensorBase<const float>         g_cumsum{};
        StridedTensorBase<__nv_bfloat16>       resolvent{};
        StridedTensorBase<__nv_bfloat16>       out{};
    };

    static __device__ __forceinline__ float WarpInclusiveScan(float value, int lane)
    {
#pragma unroll
        for (int step = 1; step < kWarpSize; step <<= 1) {
            const float addend = __shfl_up_sync(0xffffffffu, value, step);
            if (lane >= step) {
                value += addend;
            }
        }
        return value;
    }

    static __device__ __forceinline__ void Run(const float* __restrict__ g,
                                               const int32_t* __restrict__ q_offsets,
                                               float* __restrict__ g_cumsum,
                                               int                    sequence_num,
                                               int                    token_num,
                                               int                    hv,
                                               int64_t                input_gate_stride,
                                               int64_t                input_gate_batch_stride,
                                               int64_t                output_gate_stride,
                                               int64_t                output_gate_batch_stride,
                                               const TmaTemplates*    descriptor_templates,
                                               const DescriptorBases* descriptor_bases,
                                               CUtensorMap*           kkt_desc_workspace,
                                               CUtensorMap*           fused_desc_workspace,
                                               CUtensorMap*           descriptor_stage)
    {
        static_assert(ChunkSize % kWarpSize == 0);
        static_assert(kThreads <= 1024);

        __shared__ float stage[kHeadsPerBlock][kStageRowStride];
        __shared__ int   sequence_id_shared;
        __shared__ int   local_chunk_id_shared;
        __shared__ int   seq_start_shared;
        __shared__ int   seq_end_shared;

        const int global_chunk_id = static_cast<int>(blockIdx.x);
        const int head_base       = static_cast<int>(blockIdx.y) * kHeadsPerBlock;
        const int tid             = static_cast<int>(threadIdx.x);
        const int warp_id         = tid / kWarpSize;
        const int warp_lane       = tid & (kWarpSize - 1);
        const int head_count      = min(kHeadsPerBlock, hv - head_base);

        int sequence_id    = 0;
        int local_chunk_id = global_chunk_id;
        int seq_start      = 0;
        int seq_end        = 0;
        if (tid == 0) {
            for (int b = 0; b < sequence_num; ++b) {
                const int candidate_start  = q_offsets[b];
                const int candidate_end    = q_offsets[b + 1];
                const int candidate_chunks = cdiv(candidate_end - candidate_start, ChunkSize);
                if (local_chunk_id < candidate_chunks) {
                    sequence_id = b;
                    seq_start   = candidate_start;
                    seq_end     = candidate_end;
                    break;
                }
                local_chunk_id -= candidate_chunks;
            }
            sequence_id_shared    = sequence_id;
            local_chunk_id_shared = local_chunk_id;
            seq_start_shared      = seq_start;
            seq_end_shared        = seq_end;
        }
        __syncthreads();
        sequence_id    = sequence_id_shared;
        local_chunk_id = local_chunk_id_shared;
        seq_start      = seq_start_shared;
        seq_end        = seq_end_shared;

        const int token0      = seq_start + local_chunk_id * ChunkSize;
        const int remaining   = seq_end - token0;
        const int token_count = remaining < ChunkSize ? remaining : ChunkSize;

        if constexpr (PrepareDirectDescriptors) {
            if (local_chunk_id == 0 && blockIdx.y == 0) {
                const int   physical_batch  = seq_start / token_num;
                const int   local_seq_start = seq_start - physical_batch * token_num;
                const int   seq_len         = seq_end - seq_start;
                const auto& templates       = *descriptor_templates;
                const auto& bases           = *descriptor_bases;
                ChunkedKktTma::Build(&kkt_desc_workspace[sequence_id * DescriptorPrepare::kKktTmaDescCount],
                                     descriptor_stage,
                                     templates.kkt_k,
                                     templates.kkt_resolvent,
                                     bases.k,
                                     bases.resolvent,
                                     tid,
                                     local_seq_start,
                                     physical_batch,
                                     seq_len);
                DescriptorPrepare::BuildSequenceDataTmaDescriptors(
                    &fused_desc_workspace[sequence_id * DescriptorPrepare::kFusedGdrDataDescCount],
                    descriptor_stage,
                    templates.fused_q,
                    templates.fused_q,
                    templates.fused_k,
                    templates.fused_k,
                    templates.fused_v,
                    templates.fused_g,
                    templates.fused_resolvent,
                    templates.fused_out,
                    bases.q,
                    bases.k,
                    bases.v,
                    bases.g_cumsum,
                    StridedTensorBase<const __nv_bfloat16>{
                        bases.resolvent.ptr, bases.resolvent.batch_stride, bases.resolvent.token_stride},
                    bases.out,
                    tid,
                    local_seq_start,
                    physical_batch,
                    seq_len);
            }
        }

        // Consecutive threads cover consecutive heads before moving to the next token,
        // matching the physical [token, head] gate layout instead of issuing one
        // head-strided transaction per warp. Common head counts are multiples of four,
        // so use one naturally aligned 16-byte transaction per participating thread.
        if ((head_count & 3) == 0) {
            const int vector_count = ChunkSize * head_count / 4;
#pragma unroll
            for (int vector_id = tid; vector_id < vector_count; vector_id += kThreads) {
                const int linear     = vector_id * 4;
                const int token_lane = linear / head_count;
                const int head_lane  = linear - token_lane * head_count;
                float4    values     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (token_lane < token_count) {
                    const int     flat_token   = token0 + token_lane;
                    const int     batch_id     = flat_token / token_num;
                    const int     token        = flat_token - batch_id * token_num;
                    const int64_t input_offset = static_cast<int64_t>(batch_id) * input_gate_batch_stride
                                                 + static_cast<int64_t>(token) * input_gate_stride + head_base
                                                 + head_lane;
                    values = *reinterpret_cast<const float4*>(g + input_offset);
                }
                stage[head_lane + 0][token_lane] = values.x;
                stage[head_lane + 1][token_lane] = values.y;
                stage[head_lane + 2][token_lane] = values.z;
                stage[head_lane + 3][token_lane] = values.w;
            }
        }
        else {
#pragma unroll
            for (int linear = tid; linear < ChunkSize * kHeadsPerBlock; linear += kThreads) {
                const int token_lane = linear / kHeadsPerBlock;
                const int head_lane  = linear - token_lane * kHeadsPerBlock;
                float     value      = 0.0f;
                if (head_lane < head_count && token_lane < token_count) {
                    const int     flat_token   = token0 + token_lane;
                    const int     batch_id     = flat_token / token_num;
                    const int     token        = flat_token - batch_id * token_num;
                    const int64_t input_offset = static_cast<int64_t>(batch_id) * input_gate_batch_stride
                                                 + static_cast<int64_t>(token) * input_gate_stride + head_base
                                                 + head_lane;
                    value = g[input_offset];
                }
                stage[head_lane][token_lane] = value;
            }
        }
        __syncthreads();

        // Each warp scans one or more complete 32-token head rows.
#pragma unroll
        for (int head_lane = warp_id; head_lane < kHeadsPerBlock; head_lane += kWarps) {
            float value                 = stage[head_lane][warp_lane];
            value                       = WarpInclusiveScan(value, warp_lane);
            stage[head_lane][warp_lane] = value;
        }
        __syncthreads();

        // Transpose back to the contiguous [token, head] store order.
        if ((head_count & 3) == 0) {
            const int vector_count = ChunkSize * head_count / 4;
#pragma unroll
            for (int vector_id = tid; vector_id < vector_count; vector_id += kThreads) {
                const int linear     = vector_id * 4;
                const int token_lane = linear / head_count;
                const int head_lane  = linear - token_lane * head_count;
                if (token_lane < token_count) {
                    const int     flat_token    = token0 + token_lane;
                    const int     batch_id      = flat_token / token_num;
                    const int     token         = flat_token - batch_id * token_num;
                    const int64_t output_offset = static_cast<int64_t>(batch_id) * output_gate_batch_stride
                                                  + static_cast<int64_t>(token) * output_gate_stride + head_base
                                                  + head_lane;
                    const float4 values                                  = make_float4(stage[head_lane + 0][token_lane],
                                                      stage[head_lane + 1][token_lane],
                                                      stage[head_lane + 2][token_lane],
                                                      stage[head_lane + 3][token_lane]);
                    *reinterpret_cast<float4*>(g_cumsum + output_offset) = values;
                }
            }
        }
        else {
#pragma unroll
            for (int linear = tid; linear < ChunkSize * kHeadsPerBlock; linear += kThreads) {
                const int token_lane = linear / kHeadsPerBlock;
                const int head_lane  = linear - token_lane * kHeadsPerBlock;
                if (head_lane < head_count && token_lane < token_count) {
                    const int     flat_token    = token0 + token_lane;
                    const int     batch_id      = flat_token / token_num;
                    const int     token         = flat_token - batch_id * token_num;
                    const int64_t output_offset = static_cast<int64_t>(batch_id) * output_gate_batch_stride
                                                  + static_cast<int64_t>(token) * output_gate_stride + head_base
                                                  + head_lane;
                    g_cumsum[output_offset] = stage[head_lane][token_lane];
                }
            }
        }
    }
};

template<int ChunkSize>
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
    Sm120ChunkLocalCumsum<ChunkSize>::Run(g,
                                          q_offsets,
                                          g_cumsum,
                                          sequence_num,
                                          token_num,
                                          hv,
                                          input_gate_stride,
                                          input_gate_batch_stride,
                                          output_gate_stride,
                                          output_gate_batch_stride,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr);
}

template<int ChunkSize>
__global__ void ParallelChunkLocalCumsumAndPrepareDirectKernel(
    const float* __restrict__ g,
    const int32_t* __restrict__ q_offsets,
    float* __restrict__ g_cumsum,
    int                                                                                      sequence_num,
    int                                                                                      token_num,
    int                                                                                      hv,
    int64_t                                                                                  input_gate_stride,
    int64_t                                                                                  input_gate_batch_stride,
    int64_t                                                                                  output_gate_stride,
    int64_t                                                                                  output_gate_batch_stride,
    const __grid_constant__ typename Sm120ChunkLocalCumsum<ChunkSize, true>::TmaTemplates    descriptor_templates,
    const __grid_constant__ typename Sm120ChunkLocalCumsum<ChunkSize, true>::DescriptorBases descriptor_bases,
    CUtensorMap*                                                                             kkt_desc_workspace,
    CUtensorMap*                                                                             fused_desc_workspace)
{
    using Kernel            = Sm120ChunkLocalCumsum<ChunkSize, true>;
    using DescriptorPrepare = typename Kernel::DescriptorPrepare;
    __shared__ __align__(128) CUtensorMap descriptor_stage[DescriptorPrepare::kFusedGdrDataDescCount];
    Sm120ChunkLocalCumsum<ChunkSize, true>::Run(g,
                                                q_offsets,
                                                g_cumsum,
                                                sequence_num,
                                                token_num,
                                                hv,
                                                input_gate_stride,
                                                input_gate_batch_stride,
                                                output_gate_stride,
                                                output_gate_batch_stride,
                                                &descriptor_templates,
                                                &descriptor_bases,
                                                kkt_desc_workspace,
                                                fused_desc_workspace,
                                                descriptor_stage);
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

    using Kernel = Sm120ChunkLocalCumsum<ChunkSize>;
    dim3 grid(problem.total_chunks, cdiv(problem.hv, Kernel::kHeadsPerBlock));
    ParallelChunkLocalCumsumKernel<ChunkSize><<<grid, Kernel::kThreads, 0, stream>>>(g_ptr,
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

template<int BlockDv, int ChunkSize>
void LaunchChunkCumsumAndPrepareDirect(const core::Tensor& q,
                                       const core::Tensor& k,
                                       const core::Tensor& v,
                                       const core::Tensor& g,
                                       const core::Tensor& q_offsets,
                                       core::Tensor&       g_cumsum,
                                       core::Tensor&       resolvent,
                                       core::Tensor&       out,
                                       void*               kkt_desc_workspace,
                                       void*               fused_desc_workspace,
                                       const Problem&      problem,
                                       cudaStream_t        stream)
{
    static_assert(ChunkSize == 32);
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    if (problem.total_chunks == 0) {
        return;
    }

    using Kernel            = Sm120ChunkLocalCumsum<ChunkSize, true>;
    using DescriptorPrepare = typename Kernel::DescriptorPrepare;
    using ChunkedKktTma     = typename Kernel::ChunkedKktTma;
    const typename Kernel::TmaTemplates descriptor_templates{
        ChunkedKktTma::MakeKDesc(k),
        ChunkedKktTma::MakeResolventDesc(resolvent),
        DescriptorPrepare::MakeFusedGdrQkTmaDesc(q),
        DescriptorPrepare::MakeFusedGdrQkTmaDesc(k),
        DescriptorPrepare::MakeFusedGdrValueTmaDesc(v, BlockDv),
        DescriptorPrepare::MakeFusedGdrGateTmaDesc(g_cumsum),
        DescriptorPrepare::MakeFusedGdrResolventTmaDesc(resolvent),
        DescriptorPrepare::MakeFusedGdrOutputTmaDesc(out, BlockDv),
    };
    typename Kernel::DescriptorBases descriptor_bases{};
    descriptor_bases.q         = DescriptorPrepare::template MakeStridedTensorBase<__nv_bfloat16>(q);
    descriptor_bases.k         = DescriptorPrepare::template MakeStridedTensorBase<__nv_bfloat16>(k);
    descriptor_bases.v         = DescriptorPrepare::template MakeStridedTensorBase<__nv_bfloat16>(v);
    descriptor_bases.g_cumsum  = {g_cumsum.data<float>(), g_cumsum.stride(0), g_cumsum.stride(1)};
    descriptor_bases.resolvent = DescriptorPrepare::template MakeStridedTensorBase<__nv_bfloat16>(resolvent);
    descriptor_bases.out       = DescriptorPrepare::template MakeStridedTensorBase<__nv_bfloat16>(out);

    dim3 grid(problem.total_chunks, cdiv(problem.hv, Kernel::kHeadsPerBlock));
    ParallelChunkLocalCumsumAndPrepareDirectKernel<ChunkSize>
        <<<grid, Kernel::kThreads, 0, stream>>>(g.data<float>(),
                                                q_offsets.data<int32_t>(),
                                                g_cumsum.data<float>(),
                                                problem.sequence_num,
                                                problem.token_num,
                                                problem.hv,
                                                g.stride(1),
                                                g.stride(0),
                                                g_cumsum.stride(1),
                                                g_cumsum.stride(0),
                                                descriptor_templates,
                                                descriptor_bases,
                                                reinterpret_cast<CUtensorMap*>(kkt_desc_workspace),
                                                reinterpret_cast<CUtensorMap*>(fused_desc_workspace));
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

template<int BlockDv>
void LaunchChunk32LocalCumsumAndPrepareDirect(const core::Tensor& q,
                                              const core::Tensor& k,
                                              const core::Tensor& v,
                                              const core::Tensor& g,
                                              const core::Tensor& q_offsets,
                                              core::Tensor&       g_cumsum,
                                              core::Tensor&       resolvent,
                                              core::Tensor&       out,
                                              void*               kkt_desc_workspace,
                                              void*               fused_desc_workspace,
                                              const Problem&      problem,
                                              cudaStream_t        stream)
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    LaunchChunkCumsumAndPrepareDirect<BlockDv, 32>(
        q, k, v, g, q_offsets, g_cumsum, resolvent, out, kkt_desc_workspace, fused_desc_workspace, problem, stream);
}

#define TM_INSTANTIATE_SM120_CUMSUM_PREPARE(BLOCK_DV)                                                                  \
    template void LaunchChunk32LocalCumsumAndPrepareDirect<BLOCK_DV>(const core::Tensor&,                              \
                                                                     const core::Tensor&,                              \
                                                                     const core::Tensor&,                              \
                                                                     const core::Tensor&,                              \
                                                                     const core::Tensor&,                              \
                                                                     core::Tensor&,                                    \
                                                                     core::Tensor&,                                    \
                                                                     core::Tensor&,                                    \
                                                                     void*,                                            \
                                                                     void*,                                            \
                                                                     const Problem&,                                   \
                                                                     cudaStream_t)

TM_INSTANTIATE_SM120_CUMSUM_PREPARE(kContextParallelGdrBlockDv);
TM_INSTANTIATE_SM120_CUMSUM_PREPARE(kFusedGdrBlockDv);

#undef TM_INSTANTIATE_SM120_CUMSUM_PREPARE

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
