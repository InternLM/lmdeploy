#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_90/common.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include <stdexcept>
#include <string>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kChunkedKktTileDim = kHeadDim / 2;

template<class T>
constexpr CUtensorMapDataType FusedGdrTmaDataType()
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template<>
constexpr CUtensorMapDataType FusedGdrTmaDataType<float>()
{
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
}

inline CUtensorMapSwizzle FusedGdrBf16TmaSwizzle(int block_dv)
{
    if (block_dv == kContextParallelGdrBlockDv) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }
    if (block_dv == kFusedGdrBlockDv || block_dv == kWideGdrBlockDv) {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
    return CU_TENSOR_MAP_SWIZZLE_128B;
}

template<class StateT>
inline CUtensorMapSwizzle ContextParallelStateTmaSwizzle(int block_dv)
{
    static_assert(std::is_same_v<StateT, __nv_bfloat16> || std::is_same_v<StateT, float>);
    if (block_dv != kContextParallelGdrBlockDv) {
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
    if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }
    else {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

template<int ChunkSize>
constexpr CUtensorMapSwizzle FusedGdrSquareTmaSwizzle()
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    if constexpr (ChunkSize == 32) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }
    else {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

enum ChunkedKktTmaDescIndex : int
{
    kChunkedKktKDesc         = 0,
    kChunkedKktResolventDesc = 3,
};

static_assert(kChunkedKktTileDim == 64);

template<class K>
constexpr CUtensorMapDataType ChunkedKktTmaDataType()
{
    static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT descriptor prep supports only bf16 K tensors");
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template<class T>
struct StridedTensorBase {
    T*      ptr{};
    int64_t batch_stride{};
    int64_t token_stride{};
};

template<class T>
StridedTensorBase<T> MakeStridedTensorBase(core::Tensor& tensor)
{
    return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
}

template<class T>
StridedTensorBase<const T> MakeStridedTensorBase(const core::Tensor& tensor)
{
    return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
}

template<class T>
CUtensorMap MakeQkTmaDesc(const core::Tensor& tensor, const uint32_t (&box_dims)[5], CUtensorMapSwizzle swizzle)
{
    const uint64_t global_dims[5] = {
        64u,
        2u,
        static_cast<uint64_t>(tensor.shape(2)),
        static_cast<uint64_t>(tensor.shape(1)),
        static_cast<uint64_t>(tensor.shape(0)),
    };
    const uint64_t global_strides[4] = {
        64u * sizeof(T),
        static_cast<uint64_t>(tensor.stride(2)) * sizeof(T),
        static_cast<uint64_t>(tensor.stride(1)) * sizeof(T),
        static_cast<uint64_t>(tensor.stride(0)) * sizeof(T),
    };
    return MakeTmaDesc(
        const_cast<T*>(tensor.data<T>()), FusedGdrTmaDataType<T>(), 5, global_dims, global_strides, box_dims, swizzle);
}

template<int ChunkSize>
CUtensorMap MakeChunkedKktTmaDesc(const core::Tensor& k)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint32_t box_dims[5] = {64u, 1u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeQkTmaDesc<__nv_bfloat16>(k, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap MakeChunkedKktTmaDesc(const core::Tensor& k)
{
    return MakeChunkedKktTmaDesc<kChunkSize>(k);
}

template<int ChunkSize>
inline CUtensorMap MakeFusedGdrHGateTmaDesc(const core::Tensor& gate)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[3] = {
        static_cast<uint64_t>(gate.stride(1)),
        static_cast<uint64_t>(gate.shape(1)),
        static_cast<uint64_t>(gate.shape(0)),
    };
    const uint64_t global_strides[2] = {
        static_cast<uint64_t>(gate.stride(1)) * sizeof(float),
        static_cast<uint64_t>(gate.stride(0)) * sizeof(float),
    };
    const uint32_t box_dims[3] = {4u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<float*>(gate.data<float>()),
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       3,
                       global_dims,
                       global_strides,
                       box_dims,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
}

inline CUtensorMap MakeFusedGdrHGateTmaDesc(const core::Tensor& gate)
{
    return MakeFusedGdrHGateTmaDesc<kChunkSize>(gate);
}

template<int ChunkSize>
CUtensorMap MakeChunkedKktResolventTmaDesc(const core::Tensor& resolvent)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(ChunkSize),
        static_cast<uint64_t>(resolvent.shape(2)),
        static_cast<uint64_t>(resolvent.shape(1)),
        static_cast<uint64_t>(resolvent.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(resolvent.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(resolvent.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(resolvent.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(ChunkSize), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<__nv_bfloat16*>(resolvent.data<__nv_bfloat16>()),
                       ChunkedKktTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrSquareTmaSwizzle<ChunkSize>());
}

inline CUtensorMap MakeChunkedKktResolventTmaDesc(const core::Tensor& resolvent)
{
    return MakeChunkedKktResolventTmaDesc<kChunkSize>(resolvent);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrQkTmaDesc(const core::Tensor& q_or_k)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint32_t box_dims[5] = {64u, 1u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeQkTmaDesc<__nv_bfloat16>(q_or_k, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap MakeFusedGdrQkTmaDesc(const core::Tensor& q_or_k)
{
    return MakeFusedGdrQkTmaDesc<kChunkSize>(q_or_k);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrValueTmaDesc(const core::Tensor& v, int block_dv)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const int      tma_block_dv   = block_dv == kWideGdrBlockDv ? kFusedGdrBlockDv : block_dv;
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(v.shape(2)),
        static_cast<uint64_t>(v.shape(1)),
        static_cast<uint64_t>(v.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(v.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(v.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(v.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(tma_block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<__nv_bfloat16*>(v.data<__nv_bfloat16>()),
                       FusedGdrTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrBf16TmaSwizzle(tma_block_dv));
}

inline CUtensorMap MakeFusedGdrValueTmaDesc(const core::Tensor& v, int block_dv)
{
    return MakeFusedGdrValueTmaDesc<kChunkSize>(v, block_dv);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrOutputTmaDesc(core::Tensor& out, int block_dv)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const int      tma_block_dv   = block_dv == kWideGdrBlockDv ? kFusedGdrBlockDv : block_dv;
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(out.shape(2)),
        static_cast<uint64_t>(out.shape(1)),
        static_cast<uint64_t>(out.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(out.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(out.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(out.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(tma_block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(out.data<__nv_bfloat16>(),
                       FusedGdrTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrBf16TmaSwizzle(tma_block_dv));
}

inline CUtensorMap MakeFusedGdrOutputTmaDesc(core::Tensor& out, int block_dv)
{
    return MakeFusedGdrOutputTmaDesc<kChunkSize>(out, block_dv);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrResolventTmaDesc(const core::Tensor& resolvent)
{
    return MakeChunkedKktResolventTmaDesc<ChunkSize>(resolvent);
}

inline CUtensorMap MakeFusedGdrResolventTmaDesc(const core::Tensor& resolvent)
{
    return MakeFusedGdrResolventTmaDesc<kChunkSize>(resolvent);
}

template<int ChunkSize>
CUtensorMap MakeFusedGdrHResolventTmaDesc(const core::Tensor& resolvent)
{
    return MakeFusedGdrResolventTmaDesc<ChunkSize>(resolvent);
}

inline CUtensorMap MakeFusedGdrHResolventTmaDesc(const core::Tensor& resolvent)
{
    return MakeFusedGdrHResolventTmaDesc<kChunkSize>(resolvent);
}

template<class StateT>
CUtensorMap MakeContextParallelStateTmaDesc(StateT* ptr, int total_segments, int hv, int block_dv)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(StateT)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(StateT)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(StateT),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(block_dv),
        static_cast<uint32_t>(kHeadDim),
        1u,
        1u,
    };
    return MakeTmaDesc(ptr,
                       FusedGdrTmaDataType<StateT>(),
                       4,
                       global_dim,
                       global_stride,
                       box_dim,
                       ContextParallelStateTmaSwizzle<StateT>(block_dv));
}

template<class T>
CUtensorMap MakeCorrectInitialStatesSegmentMatrixTmaDesc(T* ptr, int total_segments, int hv)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(T)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(T)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(T),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(kCorrectInitialStatesMRowsPerTma),
        static_cast<uint32_t>(kHeadDim),
        1u,
        1u,
    };
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<T>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
}

template<class T>
CUtensorMap MakeFusedGdrHSegmentMatrixTmaDesc(T* ptr, int total_segments, int hv, int block_kk)
{
    const uint64_t global_dim[4] = {
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(kHeadDim),
        static_cast<uint64_t>(hv),
        static_cast<uint64_t>(total_segments),
    };
    const uint64_t global_stride[3] = {
        static_cast<uint64_t>(kHeadDim * sizeof(T)),
        static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(T)),
        static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(T),
    };
    const uint32_t box_dim[4] = {
        static_cast<uint32_t>(block_kk),
        static_cast<uint32_t>(kHeadDim),
        1u,
        1u,
    };
    return MakeTmaDesc(
        ptr, FusedGdrTmaDataType<T>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<int KernelChunkSize = kChunkSize>
struct Sm90GdrTmaDescPrepare {
    static constexpr int kDescriptorChunkSize = KernelChunkSize;

    static __device__ __forceinline__ int
    ContextParallelSegmentCount(int sequence_begin, int sequence_end, int segment_tokens)
    {
        const int sequence_len = sequence_end - sequence_begin;
        return sequence_len > 0 ? (sequence_len + segment_tokens - 1) / segment_tokens : 0;
    }

    static __device__ __forceinline__ int
    ContextParallelSequenceSegmentStart(const int32_t* q_offsets, int sequence_id, int segment_tokens)
    {
        int segment_start = 0;
        for (int seq = 0; seq < sequence_id; ++seq) {
            segment_start += ContextParallelSegmentCount(q_offsets[seq], q_offsets[seq + 1], segment_tokens);
        }
        return segment_start;
    }

    static __device__ __forceinline__ void BuildContextParallelSequenceMetadata(const int32_t* q_offsets,
                                                                                int32_t*       cp_sequence_starts,
                                                                                int32_t*       cp_q_offsets,
                                                                                int            sequence_id,
                                                                                int            sequence_num,
                                                                                int            total_segments,
                                                                                int            segment_tokens,
                                                                                int            tid)
    {
        if (tid == 0) {
            cp_sequence_starts[sequence_id] =
                ContextParallelSequenceSegmentStart(q_offsets, sequence_id, segment_tokens);
            if (sequence_id == sequence_num - 1) {
                cp_sequence_starts[sequence_num] = total_segments;
                cp_q_offsets[total_segments]     = q_offsets[sequence_num];
            }
        }
    }

    template<int Dim>
    static __device__ __forceinline__ void
    FusedGdrReplaceTmaAddressAndDim(CUtensorMap* desc, const void* global_address, int dim)
    {
        const uint32_t smem_ptr = cast_smem_ptr_to_uint(desc);
        uint64_t       smem_ptr64;
        asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                     :
                     : "r"(smem_ptr), "l"(global_address));
        asm volatile("cvt.u64.u32 %0, %1;" : "=l"(smem_ptr64) : "r"(smem_ptr));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;"
                     :
                     : "l"(smem_ptr64), "n"(Dim), "r"(static_cast<uint32_t>(dim)));
    }

    template<int TokenAxis, class T>
    static __device__ __forceinline__ void RebaseSequenceDescriptor(CUtensorMap*         output,
                                                                    CUtensorMap*         scratch,
                                                                    const CUtensorMap&   base_descriptor,
                                                                    StridedTensorBase<T> tensor,
                                                                    int                  physical_batch,
                                                                    int                  local_token,
                                                                    int                  sequence_len,
                                                                    int                  lane)
    {
        CopyTmaDescriptor(scratch, &base_descriptor, lane, 32);
        __syncwarp();
        if (lane == 0) {
            const int64_t element_offset = static_cast<int64_t>(physical_batch) * tensor.batch_stride
                                           + static_cast<int64_t>(local_token) * tensor.token_stride;
            FusedGdrReplaceTmaAddressAndDim<TokenAxis>(scratch, tensor.ptr + element_offset, sequence_len);
        }
        __syncwarp();
        PublishTmaDescriptor(output, scratch);
        __syncwarp();
    }

    template<int ChunkSize, class T>
    static __device__ __forceinline__ void
    FusedGdrBuildSequenceDataTmaDescriptors(CUtensorMap*               gmem_desc,
                                            CUtensorMap*               smem_desc,
                                            const CUtensorMap&         q_tma_desc,
                                            const CUtensorMap&         k_tma_desc,
                                            const CUtensorMap&         v_tma_desc,
                                            const CUtensorMap&         resolvent_tma_desc,
                                            const CUtensorMap&         out_tma_desc,
                                            StridedTensorBase<const T> q,
                                            StridedTensorBase<const T> k,
                                            StridedTensorBase<const T> v,
                                            StridedTensorBase<const T> resolvent,
                                            StridedTensorBase<T>       out,
                                            int                        tid,
                                            int                        local_seq_start,
                                            int                        physical_batch,
                                            int                        seq_len)
    {
        static_assert(kSupportedGdrChunkSize<ChunkSize>);
        const int lane_id = tid & 31;
        if (tid < 32) {
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrQDesc],
                                        &smem_desc[kFusedGdrQDesc],
                                        q_tma_desc,
                                        q,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrKDesc],
                                        &smem_desc[kFusedGdrKDesc],
                                        k_tma_desc,
                                        k,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrVDesc],
                                        &smem_desc[kFusedGdrVDesc],
                                        v_tma_desc,
                                        v,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrResolventDesc],
                                        &smem_desc[kFusedGdrResolventDesc],
                                        resolvent_tma_desc,
                                        resolvent,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrOutDesc],
                                        &smem_desc[kFusedGdrOutDesc],
                                        out_tma_desc,
                                        out,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
        }
        __syncwarp();
    }

    template<class T>
    static __device__ __forceinline__ void
    FusedGdrBuildSequenceDataTmaDescriptors(CUtensorMap*               gmem_desc,
                                            CUtensorMap*               smem_desc,
                                            const CUtensorMap&         q_tma_desc,
                                            const CUtensorMap&         k_tma_desc,
                                            const CUtensorMap&         v_tma_desc,
                                            const CUtensorMap&         resolvent_tma_desc,
                                            const CUtensorMap&         out_tma_desc,
                                            StridedTensorBase<const T> q,
                                            StridedTensorBase<const T> k,
                                            StridedTensorBase<const T> v,
                                            StridedTensorBase<const T> resolvent,
                                            StridedTensorBase<T>       out,
                                            int                        tid,
                                            int                        local_seq_start,
                                            int                        physical_batch,
                                            int                        seq_len)
    {
        FusedGdrBuildSequenceDataTmaDescriptors<kChunkSize>(gmem_desc,
                                                            smem_desc,
                                                            q_tma_desc,
                                                            k_tma_desc,
                                                            v_tma_desc,
                                                            resolvent_tma_desc,
                                                            out_tma_desc,
                                                            q,
                                                            k,
                                                            v,
                                                            resolvent,
                                                            out,
                                                            tid,
                                                            local_seq_start,
                                                            physical_batch,
                                                            seq_len);
    }

    template<int ChunkSize, class K>
    static __device__ __forceinline__ void ChunkedKktBuildTmaDescriptors(CUtensorMap*               gmem_desc,
                                                                         CUtensorMap*               smem_desc,
                                                                         const CUtensorMap&         k_tma_desc,
                                                                         const CUtensorMap&         resolvent_tma_desc,
                                                                         StridedTensorBase<const K> k,
                                                                         StridedTensorBase<K>       resolvent,
                                                                         int                        tid,
                                                                         int                        local_seq_start,
                                                                         int                        physical_batch,
                                                                         int                        seq_len)
    {
        static_assert(kSupportedGdrChunkSize<ChunkSize>);
        const int lane_id = tid & 31;
        if (tid < 32) {
            RebaseSequenceDescriptor<3>(&gmem_desc[kChunkedKktKDesc],
                                        &smem_desc[kChunkedKktKDesc],
                                        k_tma_desc,
                                        k,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kChunkedKktResolventDesc],
                                        &smem_desc[kChunkedKktResolventDesc],
                                        resolvent_tma_desc,
                                        resolvent,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
        }
        __syncwarp();
    }

    template<class K>
    static __device__ __forceinline__ void ChunkedKktBuildTmaDescriptors(CUtensorMap*               gmem_desc,
                                                                         CUtensorMap*               smem_desc,
                                                                         const CUtensorMap&         k_tma_desc,
                                                                         const CUtensorMap&         resolvent_tma_desc,
                                                                         StridedTensorBase<const K> k,
                                                                         StridedTensorBase<K>       resolvent,
                                                                         int                        tid,
                                                                         int                        local_seq_start,
                                                                         int                        physical_batch,
                                                                         int                        seq_len)
    {
        ChunkedKktBuildTmaDescriptors<kChunkSize>(gmem_desc,
                                                  smem_desc,
                                                  k_tma_desc,
                                                  resolvent_tma_desc,
                                                  k,
                                                  resolvent,
                                                  tid,
                                                  local_seq_start,
                                                  physical_batch,
                                                  seq_len);
    }

    template<int ChunkSize, class T>
    static __device__ __forceinline__ void
    FusedGdrHBuildSequenceDataTmaDescriptors(CUtensorMap*                   gmem_desc,
                                             CUtensorMap*                   smem_desc,
                                             const CUtensorMap&             k_tma_desc,
                                             const CUtensorMap&             v_tma_desc,
                                             const CUtensorMap&             g_tma_desc,
                                             const CUtensorMap&             resolvent_tma_desc,
                                             StridedTensorBase<const T>     k,
                                             StridedTensorBase<const T>     v,
                                             StridedTensorBase<const float> g_cumsum,
                                             StridedTensorBase<const T>     resolvent,
                                             int                            tid,
                                             int                            local_sequence_begin,
                                             int                            physical_batch,
                                             int                            sequence_len)
    {
        static_assert(kSupportedGdrChunkSize<ChunkSize>);
        const int lane_id = tid & 31;
        if (tid < 32) {
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrHKDesc],
                                        &smem_desc[kFusedGdrHKDesc],
                                        k_tma_desc,
                                        k,
                                        physical_batch,
                                        local_sequence_begin,
                                        sequence_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrHVDesc],
                                        &smem_desc[kFusedGdrHVDesc],
                                        v_tma_desc,
                                        v,
                                        physical_batch,
                                        local_sequence_begin,
                                        sequence_len,
                                        lane_id);
            RebaseSequenceDescriptor<1>(&gmem_desc[kFusedGdrHGDesc],
                                        &smem_desc[kFusedGdrHGDesc],
                                        g_tma_desc,
                                        g_cumsum,
                                        physical_batch,
                                        local_sequence_begin,
                                        sequence_len,
                                        lane_id);
            RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrHResolventDesc],
                                        &smem_desc[kFusedGdrHResolventDesc],
                                        resolvent_tma_desc,
                                        resolvent,
                                        physical_batch,
                                        local_sequence_begin,
                                        sequence_len,
                                        lane_id);
        }
        __syncwarp();
    }

    template<class T>
    static __device__ __forceinline__ void
    FusedGdrHBuildSequenceDataTmaDescriptors(CUtensorMap*                   gmem_desc,
                                             CUtensorMap*                   smem_desc,
                                             const CUtensorMap&             k_tma_desc,
                                             const CUtensorMap&             v_tma_desc,
                                             const CUtensorMap&             g_tma_desc,
                                             const CUtensorMap&             resolvent_tma_desc,
                                             StridedTensorBase<const T>     k,
                                             StridedTensorBase<const T>     v,
                                             StridedTensorBase<const float> g_cumsum,
                                             StridedTensorBase<const T>     resolvent,
                                             int                            tid,
                                             int                            local_sequence_begin,
                                             int                            physical_batch,
                                             int                            sequence_len)
    {
        FusedGdrHBuildSequenceDataTmaDescriptors<kChunkSize>(gmem_desc,
                                                             smem_desc,
                                                             k_tma_desc,
                                                             v_tma_desc,
                                                             g_tma_desc,
                                                             resolvent_tma_desc,
                                                             k,
                                                             v,
                                                             g_cumsum,
                                                             resolvent,
                                                             tid,
                                                             local_sequence_begin,
                                                             physical_batch,
                                                             sequence_len);
    }

    static __device__ __forceinline__ void FusedGdrHBuildTmaDescriptors(CUtensorMap*       gmem_desc,
                                                                        CUtensorMap*       smem_desc,
                                                                        const CUtensorMap& segment_state_tma_desc,
                                                                        const CUtensorMap& segment_m_tma_desc,
                                                                        int                tid)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            CopyTmaDescriptor(
                &smem_desc[kFusedGdrHSegmentStateDesc - kFusedGdrHDataDescCount], &segment_state_tma_desc, lane_id, 32);
            CopyTmaDescriptor(
                &smem_desc[kFusedGdrHSegmentMDesc - kFusedGdrHDataDescCount], &segment_m_tma_desc, lane_id, 32);
            __syncwarp();

            for (int idx = 0; idx < kFusedGdrHTensorDescCount; ++idx) {
                PublishTmaDescriptor(&gmem_desc[idx], &smem_desc[idx]);
            }
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void
    CorrectInitialStatesBuildTmaDescriptors(CUtensorMap*       gmem_desc,
                                            CUtensorMap*       smem_desc,
                                            const CUtensorMap& cp_state_tma_desc,
                                            const CUtensorMap& segment_state_tma_desc,
                                            const CUtensorMap& segment_m_tma_desc,
                                            int                tid)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesCpStateDesc], &cp_state_tma_desc, lane_id, 32);
            CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesSegmentStateDesc], &segment_state_tma_desc, lane_id, 32);
            CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesSegmentMDesc], &segment_m_tma_desc, lane_id, 32);
            __syncwarp();

            for (int idx = 0; idx < kCorrectInitialStatesExternalStateDesc; ++idx) {
                PublishTmaDescriptor(&gmem_desc[idx], &smem_desc[idx]);
            }
        }
        __syncwarp();
    }

    static constexpr int kSetupWarps               = 4;
    static constexpr int kSetupThreads             = kSetupWarps * 32;
    static constexpr int kSetupScanWarps           = 2;
    static constexpr int kSetupScanThreads         = kSetupScanWarps * 32;
    static constexpr int kSetupDescriptorWarps     = kSetupWarps - kSetupScanWarps;
    static constexpr int kSetupHeadsPerScan        = 4;
    static constexpr int kSetupDescriptorsPerBlock = kSetupDescriptorWarps;
    static constexpr int kSetupScanBarrier         = 0;
    static constexpr int kSetupMinBlocks           = 1;

    CUTE_HOST_DEVICE static constexpr int DescriptorTaskCount(bool context_parallel, int sequence_num)
    {
        return context_parallel ? ContextParallelFusedTaskBegin(sequence_num) + sequence_num :
                                  DirectFusedTaskBegin(sequence_num) + sequence_num;
    }

    CUTE_HOST_DEVICE static constexpr int DirectFusedTaskBegin(int sequence_num)
    {
        return sequence_num;
    }

    CUTE_HOST_DEVICE static constexpr int FusedGdrHTaskBegin(int sequence_num)
    {
        return sequence_num;
    }

    CUTE_HOST_DEVICE static constexpr int FusedGdrHTensorTaskBegin(int sequence_num)
    {
        return 2 * sequence_num;
    }

    CUTE_HOST_DEVICE static constexpr int ContextParallelFusedTaskBegin(int sequence_num)
    {
        return FusedGdrHTensorTaskBegin(sequence_num) + 2;
    }

    static_assert(sizeof(CUtensorMap) == 128);
    static_assert(kFusedGdrDataDescCount == 5);
    static_assert(kSetupScanBarrier + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

    struct alignas(128) SetupSharedStorage {
        alignas(16) float lower_total[kSetupHeadsPerScan];
        alignas(128) CUtensorMap desc[kSetupDescriptorWarps][kFusedGdrDataDescCount];
    };

    static_assert(sizeof(SetupSharedStorage) == 1408);

    static __device__ __forceinline__ void ScanGateHeads(const float* __restrict__ g,
                                                         float* __restrict__ g_cumsum,
                                                         int                 flat_token0,
                                                         int                 valid_token_end,
                                                         int                 token_num,
                                                         int                 head0,
                                                         int                 hv,
                                                         int64_t             input_gate_stride,
                                                         int64_t             input_gate_batch_stride,
                                                         int64_t             output_gate_stride,
                                                         int64_t             output_gate_batch_stride,
                                                         SetupSharedStorage& smem)
    {
        const int  scan_tid    = static_cast<int>(threadIdx.x);
        const int  warp        = scan_tid / 32;
        const int  lane        = scan_tid & 31;
        const int  flat_token  = flat_token0 + scan_tid;
        const bool valid_token = flat_token < valid_token_end;

        float   prefix[kSetupHeadsPerScan]{};
        int64_t input_gate_offset  = 0;
        int64_t output_gate_offset = 0;
        if (valid_token) {
            const int physical_batch = flat_token / token_num;
            const int physical_token = flat_token - physical_batch * token_num;
            input_gate_offset        = static_cast<int64_t>(physical_batch) * input_gate_batch_stride
                                + static_cast<int64_t>(physical_token) * input_gate_stride + head0;
            output_gate_offset = static_cast<int64_t>(physical_batch) * output_gate_batch_stride
                                 + static_cast<int64_t>(physical_token) * output_gate_stride + head0;
            CUTE_UNROLL
            for (int i = 0; i < kSetupHeadsPerScan; ++i) {
                if (head0 + i < hv) {
                    prefix[i] = g[input_gate_offset + i];
                }
            }
        }

        CUTE_UNROLL
        for (int step = 1; step < 32; step <<= 1) {
            CUTE_UNROLL
            for (int i = 0; i < kSetupHeadsPerScan; ++i) {
                const float addend = __shfl_up_sync(0xffffffffu, prefix[i], step);
                if (lane >= step) {
                    prefix[i] += addend;
                }
            }
        }

        if (warp == 0 && lane == 31) {
            CUTE_UNROLL
            for (int i = 0; i < kSetupHeadsPerScan; ++i) {
                smem.lower_total[i] = prefix[i];
            }
        }
        cutlass::arch::NamedBarrier::sync(kSetupScanThreads, kSetupScanBarrier);

        if (warp == 1) {
            CUTE_UNROLL
            for (int i = 0; i < kSetupHeadsPerScan; ++i) {
                prefix[i] += smem.lower_total[i];
            }
        }
        if (valid_token) {
            CUTE_UNROLL
            for (int i = 0; i < kSetupHeadsPerScan; ++i) {
                if (head0 + i < hv) {
                    g_cumsum[output_gate_offset + i] = prefix[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void RunSetup(bool               context_parallel,
                                                    Sm90GdrTmaLayout   layout,
                                                    const CUtensorMap* kkt_k_desc_ptr,
                                                    const CUtensorMap* kkt_resolvent_desc_ptr,
                                                    const CUtensorMap* fused_gdr_h_g_desc_ptr,
                                                    const CUtensorMap* fused_q_desc_ptr,
                                                    const CUtensorMap* fused_k_desc_ptr,
                                                    const CUtensorMap* fused_v_desc_ptr,
                                                    const CUtensorMap* fused_resolvent_desc_ptr,
                                                    const CUtensorMap* fused_out_desc_ptr,
                                                    const CUtensorMap* fused_gdr_h_v_desc_ptr,
                                                    const CUtensorMap* fused_gdr_h_resolvent_desc_ptr,
                                                    const CUtensorMap* context_parallel_segment_state_desc_ptr,
                                                    const CUtensorMap* context_parallel_segment_m_desc_ptr,
                                                    const CUtensorMap* correct_initial_states_cp_state_desc_ptr,
                                                    const CUtensorMap* correct_initial_states_segment_state_desc_ptr,
                                                    const CUtensorMap* correct_initial_states_segment_m_desc_ptr,
                                                    StridedTensorBase<const __nv_bfloat16> q,
                                                    StridedTensorBase<const __nv_bfloat16> k,
                                                    StridedTensorBase<const __nv_bfloat16> v,
                                                    StridedTensorBase<__nv_bfloat16>       resolvent,
                                                    StridedTensorBase<__nv_bfloat16>       out,
                                                    const float* __restrict__ g,
                                                    float* __restrict__ g_cumsum,
                                                    const int32_t* __restrict__ q_offsets,
                                                    const bool* __restrict__ finished,
                                                    void* __restrict__ workspace,
                                                    int                 total_chunks,
                                                    int                 sequence_num,
                                                    int                 token_num,
                                                    int                 hv,
                                                    int                 total_segments,
                                                    int                 segment_chunks,
                                                    int                 segment_tokens,
                                                    int64_t             input_gate_stride,
                                                    int64_t             input_gate_batch_stride,
                                                    int64_t             output_gate_stride,
                                                    int64_t             output_gate_batch_stride,
                                                    SetupSharedStorage& smem)
    {
        static_assert(kDescriptorChunkSize == 64, "the unified setup kernel is specialized for chunk64");

        const int  warp         = static_cast<int>(threadIdx.x) / 32;
        const int  lane         = static_cast<int>(threadIdx.x) & 31;
        const int  linear_task  = static_cast<int>(blockIdx.x);
        const int  head_quads   = (hv + kSetupHeadsPerScan - 1) / kSetupHeadsPerScan;
        const int  cumsum_tasks = total_chunks * head_quads;
        const bool has_cumsum   = linear_task < cumsum_tasks;

        if (warp < kSetupScanWarps) {
            if (!has_cumsum) {
                return;
            }

            const int global_chunk           = linear_task / head_quads;
            const int head0                  = kSetupHeadsPerScan * (linear_task % head_quads);
            int       sequence_id            = 0;
            int       local_chunk_id         = global_chunk;
            int       sequence_segment_start = 0;
            if (lane == 0) {
                for (int sequence = 0; sequence < sequence_num; ++sequence) {
                    const int sequence_begin = q_offsets[sequence];
                    const int sequence_end   = q_offsets[sequence + 1];
                    const int sequence_chunks =
                        (sequence_end - sequence_begin + kDescriptorChunkSize - 1) / kDescriptorChunkSize;
                    if (local_chunk_id < sequence_chunks) {
                        sequence_id = sequence;
                        break;
                    }
                    local_chunk_id -= sequence_chunks;
                    if (context_parallel) {
                        sequence_segment_start +=
                            ContextParallelSegmentCount(sequence_begin, sequence_end, segment_tokens);
                    }
                }
            }
            sequence_id            = __shfl_sync(0xffffffffu, sequence_id, 0);
            local_chunk_id         = __shfl_sync(0xffffffffu, local_chunk_id, 0);
            sequence_segment_start = __shfl_sync(0xffffffffu, sequence_segment_start, 0);

            const int sequence_begin = q_offsets[sequence_id];
            const int sequence_end   = q_offsets[sequence_id + 1];
            const int chunk_begin    = sequence_begin + local_chunk_id * kDescriptorChunkSize;
            ScanGateHeads(g,
                          g_cumsum,
                          chunk_begin,
                          sequence_end,
                          token_num,
                          head0,
                          hv,
                          input_gate_stride,
                          input_gate_batch_stride,
                          output_gate_stride,
                          output_gate_batch_stride,
                          smem);

            bool owns_segment_metadata = false;
            if (context_parallel) {
                owns_segment_metadata = local_chunk_id % segment_chunks == 0;
            }
            if (!owns_segment_metadata) {
                return;
            }

            const int local_segment = local_chunk_id / segment_chunks;
            const int segment_id    = sequence_segment_start + local_segment;
            const int segment_begin = sequence_begin + local_segment * segment_tokens;

            if (head0 == 0 && static_cast<int>(threadIdx.x) == 0) {
                auto* base                    = static_cast<char*>(workspace);
                auto* cp_state                = reinterpret_cast<float*>(base + layout.cp_state_offset);
                auto* cp_state_ptrs           = reinterpret_cast<int64_t*>(base + layout.cp_state_ptrs_offset);
                auto* cp_q_offsets            = reinterpret_cast<int32_t*>(base + layout.cp_q_offsets_offset);
                auto* cp_source_indices       = reinterpret_cast<int32_t*>(base + layout.cp_source_indices_offset);
                auto* cp_finished             = reinterpret_cast<bool*>(base + layout.cp_finished_offset);
                cp_q_offsets[segment_id]      = segment_begin;
                cp_source_indices[segment_id] = sequence_id;
                cp_state_ptrs[segment_id]     = static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                    cp_state + static_cast<int64_t>(segment_id) * hv * kHeadDim * kHeadDim));
                cp_finished[segment_id]       = finished[sequence_id];
            }
            return;
        }

        const int descriptor_warp  = warp - kSetupScanWarps;
        const int descriptor_task  = linear_task * kSetupDescriptorsPerBlock + descriptor_warp;
        const int descriptor_tasks = DescriptorTaskCount(context_parallel, sequence_num);
        if (descriptor_task >= descriptor_tasks) {
            return;
        }

        auto*        workspace_base = static_cast<char*>(workspace);
        auto*        kkt_desc       = reinterpret_cast<CUtensorMap*>(workspace_base + layout.kkt_desc_offset);
        CUtensorMap* scratch        = smem.desc[descriptor_warp];
        const StridedTensorBase<const __nv_bfloat16> resolvent_read{
            resolvent.ptr, resolvent.batch_stride, resolvent.token_stride};
        if (descriptor_task < sequence_num) {
            if (context_parallel) {
                auto* cp_sequence_starts =
                    reinterpret_cast<int32_t*>(workspace_base + layout.cp_sequence_starts_offset);
                auto* cp_q_offsets = reinterpret_cast<int32_t*>(workspace_base + layout.cp_q_offsets_offset);
                BuildContextParallelSequenceMetadata(q_offsets,
                                                     cp_sequence_starts,
                                                     cp_q_offsets,
                                                     descriptor_task,
                                                     sequence_num,
                                                     total_segments,
                                                     segment_tokens,
                                                     lane);
            }
            const int sequence_begin = q_offsets[descriptor_task];
            const int sequence_end   = q_offsets[descriptor_task + 1];
            const int sequence_len   = sequence_end - sequence_begin;
            if (sequence_len > 0) {
                const int physical_batch       = sequence_begin / token_num;
                const int local_sequence_begin = sequence_begin - physical_batch * token_num;
                ChunkedKktBuildTmaDescriptors<kDescriptorChunkSize>(&kkt_desc[descriptor_task * kKktTmaDescCount],
                                                                    scratch,
                                                                    *kkt_k_desc_ptr,
                                                                    *kkt_resolvent_desc_ptr,
                                                                    k,
                                                                    resolvent,
                                                                    lane,
                                                                    local_sequence_begin,
                                                                    physical_batch,
                                                                    sequence_len);
            }
            return;
        }

        if (!context_parallel) {
            const int sequence       = descriptor_task - DirectFusedTaskBegin(sequence_num);
            const int sequence_begin = q_offsets[sequence];
            const int sequence_end   = q_offsets[sequence + 1];
            const int sequence_len   = sequence_end - sequence_begin;
            if (sequence_len > 0) {
                const int physical_batch       = sequence_begin / token_num;
                const int local_sequence_begin = sequence_begin - physical_batch * token_num;
                auto*     direct_fused_desc =
                    reinterpret_cast<CUtensorMap*>(workspace_base + layout.direct_fused_desc_offset);
                const auto direct_slices = MakeFusedGdrTmaDescriptorSlices(direct_fused_desc, sequence_num);
                FusedGdrBuildSequenceDataTmaDescriptors<kDescriptorChunkSize>(
                    &direct_slices.data[sequence * kFusedGdrDataDescCount],
                    scratch,
                    *fused_q_desc_ptr,
                    *fused_k_desc_ptr,
                    *fused_v_desc_ptr,
                    *fused_resolvent_desc_ptr,
                    *fused_out_desc_ptr,
                    q,
                    k,
                    v,
                    resolvent_read,
                    out,
                    lane,
                    local_sequence_begin,
                    physical_batch,
                    sequence_len);
            }
            return;
        }

        auto* fused_gdr_h_desc = reinterpret_cast<CUtensorMap*>(workspace_base + layout.fused_gdr_h_desc_offset);
        auto* correct_initial_states_desc =
            reinterpret_cast<CUtensorMap*>(workspace_base + layout.correct_initial_states_desc_offset);
        auto* context_parallel_fused_gdr_desc =
            reinterpret_cast<CUtensorMap*>(workspace_base + layout.context_parallel_fused_gdr_desc_offset);
        const auto fused_gdr_h_slices = MakeFusedGdrHTmaDescriptorSlices(fused_gdr_h_desc, sequence_num);
        const auto correct_initial_states_slices =
            MakeCorrectInitialStatesTmaDescriptorSlices(correct_initial_states_desc);
        const auto context_parallel_fused_gdr_slices =
            MakeContextParallelFusedGdrTmaDescriptorSlices(context_parallel_fused_gdr_desc, sequence_num);
        const StridedTensorBase<const float> g_cumsum_read{g_cumsum, output_gate_batch_stride, output_gate_stride};

        const int fused_gdr_h_tensor_task_begin = FusedGdrHTensorTaskBegin(sequence_num);
        if (descriptor_task < fused_gdr_h_tensor_task_begin) {
            const int sequence       = descriptor_task - FusedGdrHTaskBegin(sequence_num);
            const int sequence_begin = q_offsets[sequence];
            const int sequence_end   = q_offsets[sequence + 1];
            const int sequence_len   = sequence_end - sequence_begin;
            if (sequence_len > 0) {
                const int physical_batch       = sequence_begin / token_num;
                const int local_sequence_begin = sequence_begin - physical_batch * token_num;
                FusedGdrHBuildSequenceDataTmaDescriptors<kDescriptorChunkSize>(
                    &fused_gdr_h_slices.data[sequence * kFusedGdrHDataDescCount],
                    scratch,
                    *fused_k_desc_ptr,
                    *fused_gdr_h_v_desc_ptr,
                    *fused_gdr_h_g_desc_ptr,
                    *fused_gdr_h_resolvent_desc_ptr,
                    k,
                    v,
                    g_cumsum_read,
                    resolvent_read,
                    lane,
                    local_sequence_begin,
                    physical_batch,
                    sequence_len);
            }
            return;
        }

        if (descriptor_task == fused_gdr_h_tensor_task_begin) {
            FusedGdrHBuildTmaDescriptors(fused_gdr_h_slices.segment_state,
                                         scratch,
                                         *context_parallel_segment_state_desc_ptr,
                                         *context_parallel_segment_m_desc_ptr,
                                         lane);
            return;
        }

        if (descriptor_task == fused_gdr_h_tensor_task_begin + 1) {
            CorrectInitialStatesBuildTmaDescriptors(correct_initial_states_slices.cp_state,
                                                    scratch,
                                                    *correct_initial_states_cp_state_desc_ptr,
                                                    *correct_initial_states_segment_state_desc_ptr,
                                                    *correct_initial_states_segment_m_desc_ptr,
                                                    lane);
            return;
        }

        const int sequence       = descriptor_task - ContextParallelFusedTaskBegin(sequence_num);
        const int sequence_begin = q_offsets[sequence];
        const int sequence_end   = q_offsets[sequence + 1];
        const int sequence_len   = sequence_end - sequence_begin;
        if (sequence_len > 0) {
            const int physical_batch       = sequence_begin / token_num;
            const int local_sequence_begin = sequence_begin - physical_batch * token_num;
            FusedGdrBuildSequenceDataTmaDescriptors<kDescriptorChunkSize>(
                &context_parallel_fused_gdr_slices.data[sequence * kFusedGdrDataDescCount],
                scratch,
                *fused_q_desc_ptr,
                *fused_k_desc_ptr,
                *fused_v_desc_ptr,
                *fused_resolvent_desc_ptr,
                *fused_out_desc_ptr,
                q,
                k,
                v,
                resolvent_read,
                out,
                lane,
                local_sequence_begin,
                physical_batch,
                sequence_len);
        }
    }
};

template<int ChunkSize = kChunkSize>
__global__ __launch_bounds__(
    Sm90GdrTmaDescPrepare<ChunkSize>::kSetupThreads,
    Sm90GdrTmaDescPrepare<ChunkSize>::
        kSetupMinBlocks) void Sm90GdrPrepareAndCumsumKernel(bool                                context_parallel,
                                                            Sm90GdrTmaLayout                    layout,
                                                            const __grid_constant__ CUtensorMap kkt_k_desc,
                                                            const __grid_constant__ CUtensorMap kkt_resolvent_desc,
                                                            const __grid_constant__ CUtensorMap fused_gdr_h_g_desc,
                                                            const __grid_constant__ CUtensorMap fused_q_desc,
                                                            const __grid_constant__ CUtensorMap fused_k_desc,
                                                            const __grid_constant__ CUtensorMap fused_v_desc,
                                                            const __grid_constant__ CUtensorMap fused_resolvent_desc,
                                                            const __grid_constant__ CUtensorMap fused_out_desc,
                                                            const __grid_constant__ CUtensorMap fused_gdr_h_v_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                fused_gdr_h_resolvent_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                context_parallel_segment_state_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                context_parallel_segment_m_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                correct_initial_states_cp_state_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                correct_initial_states_segment_state_desc,
                                                            const __grid_constant__ CUtensorMap
                                                                correct_initial_states_segment_m_desc,
                                                            StridedTensorBase<const __nv_bfloat16> q,
                                                            StridedTensorBase<const __nv_bfloat16> k,
                                                            StridedTensorBase<const __nv_bfloat16> v,
                                                            StridedTensorBase<__nv_bfloat16>       resolvent,
                                                            StridedTensorBase<__nv_bfloat16>       out,
                                                            const float* __restrict__ g,
                                                            float* __restrict__ g_cumsum,
                                                            const int32_t* __restrict__ q_offsets,
                                                            const bool* __restrict__ finished,
                                                            void* __restrict__ workspace,
                                                            int     total_chunks,
                                                            int     sequence_num,
                                                            int     token_num,
                                                            int     hv,
                                                            int     total_segments,
                                                            int     segment_chunks,
                                                            int     segment_tokens,
                                                            int64_t input_gate_stride,
                                                            int64_t input_gate_batch_stride,
                                                            int64_t output_gate_stride,
                                                            int64_t output_gate_batch_stride)
{
    using Kernel = Sm90GdrTmaDescPrepare<ChunkSize>;
    __shared__ typename Kernel::SetupSharedStorage smem;
    Kernel::RunSetup(context_parallel,
                     layout,
                     &kkt_k_desc,
                     &kkt_resolvent_desc,
                     &fused_gdr_h_g_desc,
                     &fused_q_desc,
                     &fused_k_desc,
                     &fused_v_desc,
                     &fused_resolvent_desc,
                     &fused_out_desc,
                     &fused_gdr_h_v_desc,
                     &fused_gdr_h_resolvent_desc,
                     &context_parallel_segment_state_desc,
                     &context_parallel_segment_m_desc,
                     &correct_initial_states_cp_state_desc,
                     &correct_initial_states_segment_state_desc,
                     &correct_initial_states_segment_m_desc,
                     q,
                     k,
                     v,
                     resolvent,
                     out,
                     g,
                     g_cumsum,
                     q_offsets,
                     finished,
                     workspace,
                     total_chunks,
                     sequence_num,
                     token_num,
                     hv,
                     total_segments,
                     segment_chunks,
                     segment_tokens,
                     input_gate_stride,
                     input_gate_batch_stride,
                     output_gate_stride,
                     output_gate_batch_stride,
                     smem);
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
