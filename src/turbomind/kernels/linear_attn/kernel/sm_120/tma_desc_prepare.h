#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class StateT, int ChunkSize>
struct Sm120GdrTmaDescPrepare {
    static_assert(ChunkSize == kChunk32Size);
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                  "chunked descriptor prep StateT must be float or bfloat16");

    static constexpr int kThreads                                    = 32;
    static constexpr int kMinBlocks                                  = 1;
    static constexpr int kKktTmaDescCount                            = 2;
    static constexpr int kFusedGdrTmaDescCount                       = 9;
    static constexpr int kFusedGdrDataDescCount                      = 8;
    static constexpr int kFusedGdrStateDescCount                     = 1;
    static constexpr int kFusedGdrHDataDescCount                     = 4;
    static constexpr int kFusedGdrHTensorDescCount                   = 2;
    static constexpr int kCorrectInitialStatesExternalDescCount      = 1;
    static constexpr int kCorrectInitialStatesF32BlockDv             = 4;
    static constexpr int kCorrectInitialStatesBf16BlockDv            = kCorrectInitialStatesF32BlockDv;
    static constexpr int kCorrectInitialStatesBf16ExternalTmaBlockDv = 8;
    static constexpr int kCorrectInitialStatesMRowsPerTma            = 32;

    enum FusedTmaDescIndex : int
    {
        kFusedGdrQDesc = 0,
        kFusedGdrQHiDesc,
        kFusedGdrKDesc,
        kFusedGdrKHiDesc,
        kFusedGdrVDesc,
        kFusedGdrGDesc,
        kFusedGdrResolventDesc,
        kFusedGdrOutDesc,
        kFusedGdrStateDesc,
    };

    enum FusedHTmaDescIndex : int
    {
        kFusedGdrHKDesc = 0,
        kFusedGdrHVDesc,
        kFusedGdrHGDesc,
        kFusedGdrHResolventDesc,
        kFusedGdrHSegmentStateDesc,
        kFusedGdrHSegmentMDesc,
    };

    enum CorrectInitialStatesTmaDescIndex : int
    {
        kCorrectInitialStatesCpStateDesc = 0,
        kCorrectInitialStatesSegmentStateDesc,
        kCorrectInitialStatesSegmentMDesc,
        kCorrectInitialStatesExternalStateDesc,
    };

    static_assert(kFusedGdrStateDesc == kFusedGdrDataDescCount);
    static_assert(kFusedGdrHSegmentStateDesc == kFusedGdrHDataDescCount);
    static_assert(kFusedGdrHSegmentMDesc - kFusedGdrHDataDescCount == 1);
    static_assert(kCorrectInitialStatesExternalStateDesc == 3);

    template<class TensorMapPtr>
    struct FusedGdrTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr state{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE FusedGdrTmaDescriptorSlices<TensorMapPtr> MakeFusedGdrTmaDescriptorSlices(TensorMapPtr base,
                                                                                                      int sequence_num)
    {
        return {base, base + sequence_num * kFusedGdrDataDescCount};
    }

    template<class TensorMapPtr>
    struct ContextParallelFusedGdrTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr cp_state{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE ContextParallelFusedGdrTmaDescriptorSlices<TensorMapPtr>
                            MakeContextParallelFusedGdrTmaDescriptorSlices(TensorMapPtr base, int sequence_num)
    {
        return {base, base + sequence_num * kFusedGdrDataDescCount};
    }

    template<class TensorMapPtr>
    struct FusedGdrHTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr segment_state{};
        TensorMapPtr segment_m{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE FusedGdrHTmaDescriptorSlices<TensorMapPtr>
                            MakeFusedGdrHTmaDescriptorSlices(TensorMapPtr base, int sequence_num)
    {
        auto* segment_state = base + sequence_num * kFusedGdrHDataDescCount;
        return {base, segment_state, segment_state + 1};
    }

    template<class TensorMapPtr>
    struct CorrectInitialStatesTmaDescriptorSlices {
        TensorMapPtr cp_state{};
        TensorMapPtr segment_state{};
        TensorMapPtr segment_m{};
        TensorMapPtr external_state{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE CorrectInitialStatesTmaDescriptorSlices<TensorMapPtr>
                            MakeCorrectInitialStatesTmaDescriptorSlices(TensorMapPtr base)
    {
        return {base + kCorrectInitialStatesCpStateDesc,
                base + kCorrectInitialStatesSegmentStateDesc,
                base + kCorrectInitialStatesSegmentMDesc,
                base + kCorrectInitialStatesExternalStateDesc};
    }

    template<class T>
    struct StridedTensorBase {
        T*      ptr{};
        int64_t batch_stride{};
        int64_t token_stride{};
    };

    template<class T>
    static StridedTensorBase<T> MakeStridedTensorBase(core::Tensor& tensor)
    {
        return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
    }

    template<class T>
    static StridedTensorBase<const T> MakeStridedTensorBase(const core::Tensor& tensor)
    {
        return {tensor.data<T>(), tensor.stride(0), tensor.stride(1)};
    }

    template<class T>
    static constexpr CUtensorMapDataType TmaDataType()
    {
        if constexpr (std::is_same_v<T, float>) {
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        }
        else {
            static_assert(std::is_same_v<T, __nv_bfloat16>);
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        }
    }

    static CUtensorMapSwizzle Bf16TmaSwizzle(int block_dv)
    {
        return block_dv == kContextParallelGdrBlockDv ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    }

    static constexpr CUtensorMapSwizzle SquareTmaSwizzle()
    {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    }

    template<class T>
    static CUtensorMap
    MakeQkTmaDesc(const core::Tensor& tensor, const uint32_t (&box_dims)[5], CUtensorMapSwizzle swizzle)
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
            const_cast<T*>(tensor.data<T>()), TmaDataType<T>(), 5, global_dims, global_strides, box_dims, swizzle);
    }

    static CUtensorMap MakeFusedGdrQkTmaDesc(const core::Tensor& tensor)
    {
        const uint32_t box_dims[5] = {64u, 1u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
        return MakeQkTmaDesc<__nv_bfloat16>(tensor, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
    }

    static CUtensorMap MakeFusedGdrValueTmaDesc(const core::Tensor& tensor, int block_dv)
    {
        const uint64_t global_dims[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(tensor.shape(2)),
            static_cast<uint64_t>(tensor.shape(1)),
            static_cast<uint64_t>(tensor.shape(0)),
        };
        const uint64_t global_strides[3] = {
            static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
        };
        const uint32_t box_dims[4] = {static_cast<uint32_t>(block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
        return MakeTmaDesc(const_cast<__nv_bfloat16*>(tensor.data<__nv_bfloat16>()),
                           TmaDataType<__nv_bfloat16>(),
                           4,
                           global_dims,
                           global_strides,
                           box_dims,
                           Bf16TmaSwizzle(block_dv));
    }

    static CUtensorMap MakeFusedGdrOutputTmaDesc(core::Tensor& tensor, int block_dv)
    {
        const uint64_t global_dims[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(tensor.shape(2)),
            static_cast<uint64_t>(tensor.shape(1)),
            static_cast<uint64_t>(tensor.shape(0)),
        };
        const uint64_t global_strides[3] = {
            static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
        };
        const uint32_t box_dims[4] = {static_cast<uint32_t>(block_dv), 1u, static_cast<uint32_t>(ChunkSize), 1u};
        return MakeTmaDesc(tensor.data<__nv_bfloat16>(),
                           TmaDataType<__nv_bfloat16>(),
                           4,
                           global_dims,
                           global_strides,
                           box_dims,
                           Bf16TmaSwizzle(block_dv));
    }

    static CUtensorMap MakeFusedGdrResolventTmaDesc(const core::Tensor& tensor)
    {
        const uint64_t global_dims[4] = {
            static_cast<uint64_t>(ChunkSize),
            static_cast<uint64_t>(tensor.shape(2)),
            static_cast<uint64_t>(tensor.shape(1)),
            static_cast<uint64_t>(tensor.shape(0)),
        };
        const uint64_t global_strides[3] = {
            static_cast<uint64_t>(tensor.stride(2)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(1)) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(tensor.stride(0)) * sizeof(__nv_bfloat16),
        };
        const uint32_t box_dims[4] = {static_cast<uint32_t>(ChunkSize), 1u, static_cast<uint32_t>(ChunkSize), 1u};
        return MakeTmaDesc(const_cast<__nv_bfloat16*>(tensor.data<__nv_bfloat16>()),
                           TmaDataType<__nv_bfloat16>(),
                           4,
                           global_dims,
                           global_strides,
                           box_dims,
                           SquareTmaSwizzle());
    }

    static CUtensorMap MakeFusedGdrGateTmaDesc(const core::Tensor& tensor)
    {
        const uint64_t global_dims[3] = {
            static_cast<uint64_t>(tensor.stride(1)),
            static_cast<uint64_t>(tensor.shape(1)),
            static_cast<uint64_t>(tensor.shape(0)),
        };
        const uint64_t global_strides[2] = {
            static_cast<uint64_t>(tensor.stride(1)) * sizeof(float),
            static_cast<uint64_t>(tensor.stride(0)) * sizeof(float),
        };
        const uint32_t box_dims[3] = {4u, static_cast<uint32_t>(ChunkSize), 1u};
        return MakeTmaDesc(const_cast<float*>(tensor.data<float>()),
                           TmaDataType<float>(),
                           3,
                           global_dims,
                           global_strides,
                           box_dims,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template<class Element>
    static CUtensorMap MakeContextParallelStateTmaDesc(Element* ptr, int total_segments, int hv, int block_dv)
    {
        const uint64_t global_dim[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(hv),
            static_cast<uint64_t>(total_segments),
        };
        const uint64_t global_stride[3] = {
            static_cast<uint64_t>(kHeadDim * sizeof(Element)),
            static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(Element)),
            static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(Element),
        };
        const uint32_t box_dim[4] = {static_cast<uint32_t>(block_dv), static_cast<uint32_t>(kHeadDim), 1u, 1u};
        return MakeTmaDesc(
            ptr, TmaDataType<Element>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    static CUtensorMap MakeCorrectInitialStatesSegmentMatrixTmaDesc(float* ptr, int total_segments, int hv)
    {
        const uint64_t global_dim[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(hv),
            static_cast<uint64_t>(total_segments),
        };
        const uint64_t global_stride[3] = {
            static_cast<uint64_t>(kHeadDim * sizeof(float)),
            static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(float)),
            static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(float),
        };
        const uint32_t box_dim[4] = {
            static_cast<uint32_t>(kHeadDim), static_cast<uint32_t>(kCorrectInitialStatesMRowsPerTma), 1u, 1u};
        return MakeTmaDesc(
            ptr, TmaDataType<float>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    static CUtensorMap MakeFusedGdrHSegmentMatrixTmaDesc(float* ptr, int total_segments, int hv, int block_kk)
    {
        const uint64_t global_dim[4] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(hv),
            static_cast<uint64_t>(total_segments),
        };
        const uint64_t global_stride[3] = {
            static_cast<uint64_t>(kHeadDim * sizeof(float)),
            static_cast<uint64_t>(kHeadDim * kHeadDim * sizeof(float)),
            static_cast<uint64_t>(hv) * kHeadDim * kHeadDim * sizeof(float),
        };
        const uint32_t box_dim[4] = {static_cast<uint32_t>(block_kk), static_cast<uint32_t>(kHeadDim), 1u, 1u};
        return MakeTmaDesc(
            ptr, TmaDataType<float>(), 4, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template<class Element>
    static CUtensorMap MakeFusedGdrStateHeadTmaDesc(Element* ptr, int block_dv)
    {
        const uint64_t global_dim[2]    = {static_cast<uint64_t>(kHeadDim), static_cast<uint64_t>(kHeadDim)};
        const uint64_t global_stride[1] = {static_cast<uint64_t>(kHeadDim * sizeof(Element))};
        const uint32_t box_dim[2]       = {static_cast<uint32_t>(block_dv), static_cast<uint32_t>(kHeadDim)};
        return MakeTmaDesc(
            ptr, TmaDataType<Element>(), 2, global_dim, global_stride, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template<int Dim>
    static __device__ __forceinline__ void
    ReplaceTmaAddressAndDim(CUtensorMap* desc, const void* global_address, int dim)
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
            ReplaceTmaAddressAndDim<TokenAxis>(scratch, tensor.ptr + element_offset, sequence_len);
        }
        __syncwarp();
        PublishTmaDescriptor(output, scratch);
        __syncwarp();
    }

    template<class T>
    static __device__ __forceinline__ void BuildSequenceDataTmaDescriptors(CUtensorMap*       gmem_desc,
                                                                           CUtensorMap*       smem_desc,
                                                                           const CUtensorMap& q_tma_desc,
                                                                           const CUtensorMap& q_hi_tma_desc,
                                                                           const CUtensorMap& k_tma_desc,
                                                                           const CUtensorMap& k_hi_tma_desc,
                                                                           const CUtensorMap& v_tma_desc,
                                                                           const CUtensorMap& g_tma_desc,
                                                                           const CUtensorMap& resolvent_tma_desc,
                                                                           const CUtensorMap& out_tma_desc,
                                                                           StridedTensorBase<const T>     q,
                                                                           StridedTensorBase<const T>     k,
                                                                           StridedTensorBase<const T>     v,
                                                                           StridedTensorBase<const float> g_cumsum,
                                                                           StridedTensorBase<const T>     resolvent,
                                                                           StridedTensorBase<T>           out,
                                                                           int                            tid,
                                                                           int local_seq_start,
                                                                           int physical_batch,
                                                                           int seq_len)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            const StridedTensorBase<const T> q_hi{q.ptr + 64, q.batch_stride, q.token_stride};
            const StridedTensorBase<const T> k_hi{k.ptr + 64, k.batch_stride, k.token_stride};
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrQDesc],
                                        &smem_desc[kFusedGdrQDesc],
                                        q_tma_desc,
                                        q,
                                        physical_batch,
                                        local_seq_start,
                                        seq_len,
                                        lane_id);
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrQHiDesc],
                                        &smem_desc[kFusedGdrQHiDesc],
                                        q_hi_tma_desc,
                                        q_hi,
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
            RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrKHiDesc],
                                        &smem_desc[kFusedGdrKHiDesc],
                                        k_hi_tma_desc,
                                        k_hi,
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
            RebaseSequenceDescriptor<1>(&gmem_desc[kFusedGdrGDesc],
                                        &smem_desc[kFusedGdrGDesc],
                                        g_tma_desc,
                                        g_cumsum,
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
        __syncthreads();
    }

    static __device__ __forceinline__ void BuildStateTmaDescriptor(
        CUtensorMap* gmem_desc, CUtensorMap* smem_desc, const CUtensorMap& state_tma_desc, const StateT* state, int tid)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            CopyTmaDescriptor(smem_desc, &state_tma_desc, lane_id, 32);
            __syncwarp();
            if (lane_id == 0) {
                ReplaceTmaAddress(smem_desc, state);
            }
            __syncwarp();
            PublishTmaDescriptor(gmem_desc, smem_desc);
        }
        __syncthreads();
    }

    static __device__ __forceinline__ StateT* GroupedStateBase(const int64_t* state_ptrs,
                                                               int            sequence,
                                                               int            value_head,
                                                               int            num_head_groups,
                                                               int            heads_per_block,
                                                               int64_t        state_layer_offset)
    {
        const int     head_group = value_head / heads_per_block;
        const int     local_head = value_head % heads_per_block;
        const int64_t address    = state_ptrs[sequence * num_head_groups + head_group];
        return reinterpret_cast<StateT*>(static_cast<uintptr_t>(address)) + state_layer_offset
               + static_cast<int64_t>(local_head) * kHeadDim * kHeadDim;
    }

    struct ChunkedKktTma {
        static constexpr int kTileDim = kHeadDim / 2;

        enum TmaDescIndex : int
        {
            kKDesc = 0,
            kResolventDesc,
        };

        static CUtensorMap MakeKDesc(const core::Tensor& k)
        {
            const uint32_t box_dims[5] = {
                static_cast<uint32_t>(kTileDim), 1u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
            return MakeQkTmaDesc<__nv_bfloat16>(k, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
        }

        static CUtensorMap MakeResolventDesc(const core::Tensor& resolvent)
        {
            return MakeFusedGdrResolventTmaDesc(resolvent);
        }

        template<class K>
        static __device__ __forceinline__ void Build(CUtensorMap*               gmem_desc,
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
            static_assert(std::is_same_v<K, __nv_bfloat16>);
            const int lane_id = tid & 31;
            if (tid < 32) {
                RebaseSequenceDescriptor<3>(&gmem_desc[kKDesc],
                                            &smem_desc[kKDesc],
                                            k_tma_desc,
                                            k,
                                            physical_batch,
                                            local_seq_start,
                                            seq_len,
                                            lane_id);
                RebaseSequenceDescriptor<2>(&gmem_desc[kResolventDesc],
                                            &smem_desc[kResolventDesc],
                                            resolvent_tma_desc,
                                            resolvent,
                                            physical_batch,
                                            local_seq_start,
                                            seq_len,
                                            lane_id);
            }
            __syncthreads();
        }
    };

    static __device__ void BuildContextParallelMetadata(const int32_t* q_offsets,
                                                        const bool*    finished,
                                                        float*         cp_state,
                                                        int64_t*       cp_state_ptrs,
                                                        int32_t*       cp_q_offsets,
                                                        int32_t*       cp_source_indices,
                                                        int32_t*       cp_sequence_starts,
                                                        bool*          cp_finished,
                                                        int            sequence_num,
                                                        int            hv,
                                                        int            segment_tokens)
    {
        int segment_id        = 0;
        cp_sequence_starts[0] = 0;
        for (int sequence_id = 0; sequence_id < sequence_num; ++sequence_id) {
            const int sequence_begin        = q_offsets[sequence_id];
            const int sequence_end          = q_offsets[sequence_id + 1];
            cp_sequence_starts[sequence_id] = segment_id;
            for (int segment_begin = sequence_begin; segment_begin < sequence_end; segment_begin += segment_tokens) {
                const int segment_limit       = segment_begin + segment_tokens;
                const int segment_end         = segment_limit < sequence_end ? segment_limit : sequence_end;
                cp_q_offsets[segment_id]      = segment_begin;
                cp_q_offsets[segment_id + 1]  = segment_end;
                cp_source_indices[segment_id] = sequence_id;
                cp_state_ptrs[segment_id]     = static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                    cp_state + static_cast<int64_t>(segment_id) * hv * kHeadDim * kHeadDim));
                cp_finished[segment_id]       = finished[sequence_id];
                ++segment_id;
            }
            cp_sequence_starts[sequence_id + 1] = segment_id;
        }
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
        __syncthreads();
    }

    static __device__ __forceinline__ void
    CopySingleTmaDescriptor(CUtensorMap* gmem_desc, CUtensorMap* smem_desc, const CUtensorMap& src_desc, int tid)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            CopyTmaDescriptor(smem_desc, &src_desc, lane_id, 32);
            __syncwarp();
            PublishTmaDescriptor(gmem_desc, smem_desc);
        }
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
    }

    static __device__ __forceinline__ void
    BuildCorrectInitialStateTmaDescriptor(CUtensorMap*       gmem_desc,
                                          CUtensorMap*       smem_desc,
                                          const CUtensorMap& external_state_tma_desc,
                                          const int64_t*     state_ptrs,
                                          int                tid,
                                          int                sequence_id,
                                          int                value_head,
                                          int                num_head_groups,
                                          int                heads_per_block,
                                          int64_t            state_layer_offset)
    {
        const int lane_id = tid & 31;
        if (tid < 32) {
            CopyTmaDescriptor(smem_desc, &external_state_tma_desc, lane_id, 32);
            __syncwarp();

            if (lane_id == 0) {
                auto* state_base = GroupedStateBase(
                    state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset);
                ReplaceTmaAddress(smem_desc, state_base);
            }
            __syncwarp();

            PublishTmaDescriptor(gmem_desc, smem_desc);
        }
        __syncthreads();
    }

    static __device__ __forceinline__ void Run(Sm120GdrTmaMode    mode,
                                               Sm120GdrTmaLayout  layout,
                                               const CUtensorMap& kkt_k_desc,
                                               const CUtensorMap& kkt_resolvent_desc,
                                               const CUtensorMap& fused_q_desc,
                                               const CUtensorMap& fused_q_hi_desc,
                                               const CUtensorMap& fused_k_desc,
                                               const CUtensorMap& fused_k_hi_desc,
                                               const CUtensorMap& fused_gdr_h_k_desc,
                                               const CUtensorMap& fused_v_desc,
                                               const CUtensorMap& fused_g_desc,
                                               const CUtensorMap& fused_resolvent_desc,
                                               const CUtensorMap& fused_state_desc,
                                               const CUtensorMap& fused_out_desc,
                                               const CUtensorMap& fused_gdr_h_v_desc,
                                               const CUtensorMap& context_parallel_segment_state_desc,
                                               const CUtensorMap& context_parallel_segment_m_desc,
                                               const CUtensorMap& correct_initial_states_cp_state_desc,
                                               const CUtensorMap& correct_initial_states_segment_state_desc,
                                               const CUtensorMap& correct_initial_states_segment_m_desc,
                                               const CUtensorMap& correct_initial_states_external_state_desc,
                                               const CUtensorMap& context_parallel_fused_gdr_state_desc,
                                               StridedTensorBase<const __nv_bfloat16> q,
                                               StridedTensorBase<const __nv_bfloat16> k,
                                               StridedTensorBase<const __nv_bfloat16> v,
                                               StridedTensorBase<const float>         g_cumsum,
                                               StridedTensorBase<__nv_bfloat16>       resolvent,
                                               StridedTensorBase<__nv_bfloat16>       out,
                                               const int64_t* __restrict__ state_ptrs,
                                               const int32_t* __restrict__ q_offsets,
                                               const bool* __restrict__ finished,
                                               void* __restrict__ workspace,
                                               int          sequence_num,
                                               int          hq,
                                               int          hv,
                                               int          num_head_groups,
                                               int          heads_per_block,
                                               int          token_num,
                                               int          total_segments,
                                               int          segment_tokens,
                                               int64_t      gate_stride,
                                               int64_t      gate_batch_stride,
                                               int64_t      state_layer_offset,
                                               CUtensorMap* smem_desc)
    {
        auto* base              = static_cast<char*>(workspace);
        auto* kkt_desc          = reinterpret_cast<CUtensorMap*>(base + layout.kkt_desc_offset);
        auto* direct_fused_desc = reinterpret_cast<CUtensorMap*>(base + layout.direct_fused_desc_offset);
        auto* fused_gdr_h_desc  = reinterpret_cast<CUtensorMap*>(base + layout.fused_gdr_h_desc_offset);
        auto* correct_initial_states_desc =
            reinterpret_cast<CUtensorMap*>(base + layout.correct_initial_states_desc_offset);
        auto* context_parallel_fused_gdr_desc =
            reinterpret_cast<CUtensorMap*>(base + layout.context_parallel_fused_gdr_desc_offset);
        auto* cp_q_offsets       = reinterpret_cast<int32_t*>(base + layout.cp_q_offsets_offset);
        auto* cp_source_indices  = reinterpret_cast<int32_t*>(base + layout.cp_source_indices_offset);
        auto* cp_sequence_starts = reinterpret_cast<int32_t*>(base + layout.cp_sequence_starts_offset);
        auto* cp_state_ptrs      = reinterpret_cast<int64_t*>(base + layout.cp_state_ptrs_offset);
        auto* cp_finished        = reinterpret_cast<bool*>(base + layout.cp_finished_offset);
        auto* cp_state           = reinterpret_cast<float*>(base + layout.cp_state_offset);
        const StridedTensorBase<const __nv_bfloat16> resolvent_read{
            resolvent.ptr, resolvent.batch_stride, resolvent.token_stride};

        const int tid  = static_cast<int>(threadIdx.x);
        const int task = static_cast<int>(blockIdx.x);

        if (mode == Sm120GdrTmaMode::kAllContextParallel && task == 0 && tid == 0) {
            BuildContextParallelMetadata(q_offsets,
                                         finished,
                                         cp_state,
                                         cp_state_ptrs,
                                         cp_q_offsets,
                                         cp_source_indices,
                                         cp_sequence_starts,
                                         cp_finished,
                                         sequence_num,
                                         hv,
                                         segment_tokens);
        }

        const bool needs_kkt_desc = mode == Sm120GdrTmaMode::kSolveKkt || mode == Sm120GdrTmaMode::kAllDirectFused
                                    || mode == Sm120GdrTmaMode::kAllContextParallel;
        const int kkt_task_count = needs_kkt_desc ? sequence_num : 0;
        if (needs_kkt_desc && task < kkt_task_count) {
            const int seq_start = q_offsets[task];
            const int seq_end   = q_offsets[task + 1];
            const int seq_len   = seq_end - seq_start;
            if (seq_len <= 0) {
                return;
            }
            const int physical_batch  = seq_start / token_num;
            const int local_seq_start = seq_start - physical_batch * token_num;

            ChunkedKktTma::Build(&kkt_desc[task * kKktTmaDescCount],
                                 smem_desc,
                                 kkt_k_desc,
                                 kkt_resolvent_desc,
                                 k,
                                 resolvent,
                                 tid,
                                 local_seq_start,
                                 physical_batch,
                                 seq_len);
            return;
        }

        const int direct_task_base         = kkt_task_count;
        auto      direct_slices            = MakeFusedGdrTmaDescriptorSlices(direct_fused_desc, sequence_num);
        auto      fused_gdr_h_slices       = MakeFusedGdrHTmaDescriptorSlices(fused_gdr_h_desc, sequence_num);
        auto correct_initial_states_slices = MakeCorrectInitialStatesTmaDescriptorSlices(correct_initial_states_desc);
        auto context_parallel_fused_gdr_slices =
            MakeContextParallelFusedGdrTmaDescriptorSlices(context_parallel_fused_gdr_desc, sequence_num);

        const bool needs_direct_desc = mode == Sm120GdrTmaMode::kAllDirectFused || mode == Sm120GdrTmaMode::kFusedOnly;
        const int  direct_data_desc_count  = needs_direct_desc ? sequence_num : 0;
        const int  direct_state_desc_count = needs_direct_desc ? sequence_num * hv : 0;
        const int  direct_desc_tasks       = direct_data_desc_count + direct_state_desc_count;
        if (needs_direct_desc && task >= direct_task_base && task < direct_task_base + direct_data_desc_count) {
            const int local     = task - direct_task_base;
            const int sequence  = local;
            const int seq_start = q_offsets[sequence];
            const int seq_end   = q_offsets[sequence + 1];
            const int seq_len   = seq_end - seq_start;
            if (seq_len <= 0) {
                return;
            }
            const int physical_batch  = seq_start / token_num;
            const int local_seq_start = seq_start - physical_batch * token_num;

            BuildSequenceDataTmaDescriptors(&direct_slices.data[sequence * kFusedGdrDataDescCount],
                                            smem_desc,
                                            fused_q_desc,
                                            fused_q_hi_desc,
                                            fused_k_desc,
                                            fused_k_hi_desc,
                                            fused_v_desc,
                                            fused_g_desc,
                                            fused_resolvent_desc,
                                            fused_out_desc,
                                            q,
                                            k,
                                            v,
                                            g_cumsum,
                                            resolvent_read,
                                            out,
                                            tid,
                                            local_seq_start,
                                            physical_batch,
                                            seq_len);
            return;
        }
        if (needs_direct_desc && task >= direct_task_base && task < direct_task_base + direct_desc_tasks) {
            const int local       = task - direct_task_base;
            const int state_local = local - direct_data_desc_count;
            const int sequence    = state_local / hv;
            const int value_head  = state_local - sequence * hv;
            const int seq_start   = q_offsets[sequence];
            const int seq_end     = q_offsets[sequence + 1];
            const int seq_len     = seq_end - seq_start;
            if (seq_len <= 0) {
                return;
            }

            const auto* state_ptr = GroupedStateBase(
                state_ptrs, sequence, value_head, num_head_groups, heads_per_block, state_layer_offset);
            BuildStateTmaDescriptor(&direct_slices.state[state_local * kFusedGdrStateDescCount],
                                    &smem_desc[kFusedGdrStateDesc],
                                    fused_state_desc,
                                    state_ptr,
                                    tid);
            return;
        }
        const int fused_gdr_h_data_tasks   = mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num : 0;
        const int fused_gdr_h_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
        const int fused_gdr_h_desc_tasks   = fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks;
        const int fused_gdr_h_task_base    = direct_task_base + direct_desc_tasks;
        if (task >= fused_gdr_h_task_base && task < fused_gdr_h_task_base + fused_gdr_h_data_tasks) {
            const int sequence_id    = task - fused_gdr_h_task_base;
            const int sequence_begin = q_offsets[sequence_id];
            const int sequence_end   = q_offsets[sequence_id + 1];
            const int sequence_len   = sequence_end - sequence_begin;
            if (sequence_len <= 0) {
                return;
            }
            const int physical_batch       = sequence_begin / token_num;
            const int local_sequence_begin = sequence_begin - physical_batch * token_num;

            FusedGdrHBuildSequenceDataTmaDescriptors(&fused_gdr_h_slices.data[sequence_id * kFusedGdrHDataDescCount],
                                                     smem_desc,
                                                     fused_gdr_h_k_desc,
                                                     fused_gdr_h_v_desc,
                                                     fused_g_desc,
                                                     fused_resolvent_desc,
                                                     k,
                                                     v,
                                                     g_cumsum,
                                                     resolvent_read,
                                                     tid,
                                                     local_sequence_begin,
                                                     physical_batch,
                                                     sequence_len);
            return;
        }
        if (task >= fused_gdr_h_task_base + fused_gdr_h_data_tasks
            && task < fused_gdr_h_task_base + fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks) {
            FusedGdrHBuildTmaDescriptors(fused_gdr_h_slices.segment_state,
                                         smem_desc,
                                         context_parallel_segment_state_desc,
                                         context_parallel_segment_m_desc,
                                         tid);
            return;
        }

        const int correct_initial_states_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
        const int correct_initial_states_external_tasks =
            mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num * hv : 0;
        const int correct_initial_states_desc_tasks =
            correct_initial_states_tensor_tasks + correct_initial_states_external_tasks;
        const int correct_initial_states_task_base = fused_gdr_h_task_base + fused_gdr_h_desc_tasks;
        if (task >= correct_initial_states_task_base
            && task < correct_initial_states_task_base + correct_initial_states_tensor_tasks) {
            CorrectInitialStatesBuildTmaDescriptors(correct_initial_states_slices.cp_state,
                                                    smem_desc,
                                                    correct_initial_states_cp_state_desc,
                                                    correct_initial_states_segment_state_desc,
                                                    correct_initial_states_segment_m_desc,
                                                    tid);
            return;
        }
        if (task >= correct_initial_states_task_base + correct_initial_states_tensor_tasks
            && task < correct_initial_states_task_base + correct_initial_states_tensor_tasks
                          + correct_initial_states_external_tasks) {
            const int local          = task - correct_initial_states_task_base;
            const int external_local = local - correct_initial_states_tensor_tasks;
            const int sequence_id    = external_local / hv;
            const int value_head     = external_local - sequence_id * hv;
            BuildCorrectInitialStateTmaDescriptor(
                &correct_initial_states_slices.external_state[external_local * kCorrectInitialStatesExternalDescCount],
                smem_desc,
                correct_initial_states_external_state_desc,
                state_ptrs,
                tid,
                sequence_id,
                value_head,
                num_head_groups,
                heads_per_block,
                state_layer_offset);
            return;
        }

        const int context_parallel_fused_gdr_data_tasks =
            mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num : 0;
        const int context_parallel_fused_gdr_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
        const int context_parallel_fused_gdr_task_base =
            correct_initial_states_task_base + correct_initial_states_desc_tasks;
        if (task >= context_parallel_fused_gdr_task_base
            && task < context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks) {
            const int sequence_id    = task - context_parallel_fused_gdr_task_base;
            const int sequence_begin = q_offsets[sequence_id];
            const int sequence_end   = q_offsets[sequence_id + 1];
            const int sequence_len   = sequence_end - sequence_begin;
            if (sequence_len <= 0) {
                return;
            }
            const int physical_batch       = sequence_begin / token_num;
            const int local_sequence_begin = sequence_begin - physical_batch * token_num;

            BuildSequenceDataTmaDescriptors(
                &context_parallel_fused_gdr_slices.data[sequence_id * kFusedGdrDataDescCount],
                smem_desc,
                fused_q_desc,
                fused_q_hi_desc,
                fused_k_desc,
                fused_k_hi_desc,
                fused_v_desc,
                fused_g_desc,
                fused_resolvent_desc,
                fused_out_desc,
                q,
                k,
                v,
                g_cumsum,
                resolvent_read,
                out,
                tid,
                local_sequence_begin,
                physical_batch,
                sequence_len);
            return;
        }
        if (task >= context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks
            && task < context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks
                          + context_parallel_fused_gdr_tensor_tasks) {
            CopySingleTmaDescriptor(context_parallel_fused_gdr_slices.cp_state,
                                    &smem_desc[kFusedGdrStateDesc],
                                    context_parallel_fused_gdr_state_desc,
                                    tid);
            return;
        }

        static_cast<void>(total_segments);
        static_cast<void>(hq);
        static_cast<void>(gate_stride);
        static_cast<void>(gate_batch_stride);
    }
};

template<class StateT, int ChunkSize>
__global__ __launch_bounds__(
    Sm120GdrTmaDescPrepare<StateT, ChunkSize>::kThreads,
    Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
        kMinBlocks) void Sm120GdrTmaDescPrepareKernel(Sm120GdrTmaMode                     mode,
                                                      Sm120GdrTmaLayout                   layout,
                                                      const __grid_constant__ CUtensorMap kkt_k_desc,
                                                      const __grid_constant__ CUtensorMap kkt_resolvent_desc,
                                                      const __grid_constant__ CUtensorMap fused_q_desc,
                                                      const __grid_constant__ CUtensorMap fused_q_hi_desc,
                                                      const __grid_constant__ CUtensorMap fused_k_desc,
                                                      const __grid_constant__ CUtensorMap fused_k_hi_desc,
                                                      const __grid_constant__ CUtensorMap fused_gdr_h_k_desc,
                                                      const __grid_constant__ CUtensorMap fused_v_desc,
                                                      const __grid_constant__ CUtensorMap fused_g_desc,
                                                      const __grid_constant__ CUtensorMap fused_resolvent_desc,
                                                      const __grid_constant__ CUtensorMap fused_state_desc,
                                                      const __grid_constant__ CUtensorMap fused_out_desc,
                                                      const __grid_constant__ CUtensorMap fused_gdr_h_v_desc,
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
                                                      const __grid_constant__ CUtensorMap
                                                          correct_initial_states_external_state_desc,
                                                      const __grid_constant__ CUtensorMap
                                                          context_parallel_fused_gdr_state_desc,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<const __nv_bfloat16> q,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<const __nv_bfloat16> k,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<const __nv_bfloat16> v,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<const float> g_cumsum,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<__nv_bfloat16> resolvent,
                                                      typename Sm120GdrTmaDescPrepare<StateT, ChunkSize>::
                                                          template StridedTensorBase<__nv_bfloat16> out,
                                                      const int64_t* __restrict__ state_ptrs,
                                                      const int32_t* __restrict__ q_offsets,
                                                      const bool* __restrict__ finished,
                                                      void* __restrict__ workspace,
                                                      int     sequence_num,
                                                      int     hq,
                                                      int     hv,
                                                      int     num_head_groups,
                                                      int     heads_per_block,
                                                      int     token_num,
                                                      int     total_segments,
                                                      int     segment_tokens,
                                                      int64_t gate_stride,
                                                      int64_t gate_batch_stride,
                                                      int64_t state_layer_offset)
{
    using Kernel = Sm120GdrTmaDescPrepare<StateT, ChunkSize>;
    __shared__ __align__(128) CUtensorMap smem_desc[Kernel::kFusedGdrTmaDescCount];
    Kernel::Run(mode,
                layout,
                kkt_k_desc,
                kkt_resolvent_desc,
                fused_q_desc,
                fused_q_hi_desc,
                fused_k_desc,
                fused_k_hi_desc,
                fused_gdr_h_k_desc,
                fused_v_desc,
                fused_g_desc,
                fused_resolvent_desc,
                fused_state_desc,
                fused_out_desc,
                fused_gdr_h_v_desc,
                context_parallel_segment_state_desc,
                context_parallel_segment_m_desc,
                correct_initial_states_cp_state_desc,
                correct_initial_states_segment_state_desc,
                correct_initial_states_segment_m_desc,
                correct_initial_states_external_state_desc,
                context_parallel_fused_gdr_state_desc,
                q,
                k,
                v,
                g_cumsum,
                resolvent,
                out,
                state_ptrs,
                q_offsets,
                finished,
                workspace,
                sequence_num,
                hq,
                hv,
                num_head_groups,
                heads_per_block,
                token_num,
                total_segments,
                segment_tokens,
                gate_stride,
                gate_batch_stride,
                state_layer_offset,
                smem_desc);
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
