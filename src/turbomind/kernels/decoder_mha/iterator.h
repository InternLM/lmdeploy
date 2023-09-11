#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"

namespace turbomind {

struct BlockIterator {
    const void** ptrs_;
    const void*  prefetch_;

    BlockIterator() = default;

    __device__ BlockIterator(const void** block_ptrs): ptrs_{block_ptrs}
    {
        // prefetch first ptr
        prefetch_ = *ptrs_++;
    }

    __device__ const void* Next()
    {
        // return prefetched ptr
        const void* ret = prefetch_;
        // prefetch next ptr
        prefetch_ = *ptrs_++;

        return ret;
    }
};

template<typename T, typename ThreadMap, int BlockLen, int Stages, bool kUseBlockIter>
struct Iterator {

    using ElementType = T;
    using AccessType  = Array<T, ThreadMap::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);

    static constexpr int kSizePerTile  = ThreadMap::kS * ThreadMap::kC;
    static constexpr int kSmemByteSize = kElementSize * Stages * kSizePerTile;

    BlockIterator block_iterator_;

    static constexpr int kIterCount = ThreadMap::kIterS * ThreadMap::kIterC;

    static constexpr int kStepC = ThreadMap::kDeltaC;
    static constexpr int kStepS = ThreadMap::kDeltaS * ThreadMap::kC - ThreadMap::kIterC * kStepC;
    static constexpr int kStepK =
        ThreadMap::kS * ThreadMap::kC - ThreadMap::kIterS * ThreadMap::kDeltaS * ThreadMap::kC;

    // (C, S, K) = (64, 384, 1536)

    // initial offset, used to reset src_offset when switching to a new block
    int init_offset_;

    int src_offset_;
    int dst_offset_;

    int iter_c_;
    int iter_b_;

    int  seq_len_;
    int  offset_s_;
    bool is_valid_s_;

    int block_size_;
    int block_k_;

    int head_idx_;

    const T* src_;
    T*       smem_;

    int smem_read_offset_;

    struct __align__(sizeof(AccessType)) SharedStorage
    {
        T smem_[Stages][kSizePerTile];
    };

    Iterator() = default;

    __device__ Iterator(T* src, T* smem, int step, int seq_len, int warp_id, int lane_id)
    {
        src_  = src;
        smem_ = smem;

        int2 init_offset_cs = ThreadMap::get_offset(warp_id, lane_id);

        init_offset_ = init_offset_cs.x + init_offset_cs.y * ThreadMap::kC;

        src_offset_       = init_offset_ + step * ThreadMap::kC;
        dst_offset_       = init_offset_;
        smem_read_offset_ = init_offset_;

        iter_c_ = 0;
        iter_b_ = 0;

        seq_len_    = seq_len;
        offset_s_   = init_offset_cs.y + step;
        is_valid_s_ = offset_s_ < seq_len;
    }

    __device__ Iterator(
        const void** block_ptrs, int block_size, int head_idx, T* smem, int step, int seqlen, int warp_id, int lane_id)
    {
        // src_  = src;
        int block_index = step / block_size;
        block_size_     = block_size;
        block_k_        = (block_index + 1) * block_size - step;  // offset to next block
        head_idx_       = head_idx;

        block_iterator_ = BlockIterator(block_ptrs + block_index);

        src_ = (const T*)block_iterator_.Next() + head_idx_ * block_size_ * ThreadMap::kC;

        smem_ = smem;

        int2 init_offset_cs = ThreadMap::get_offset(warp_id, lane_id);

        init_offset_ = init_offset_cs.x + init_offset_cs.y * ThreadMap::kC;

        src_offset_       = init_offset_ + (step - block_index * block_size) * ThreadMap::kC;
        dst_offset_       = init_offset_;
        smem_read_offset_ = init_offset_;

        iter_c_ = 0;
        iter_b_ = 0;

        seq_len_    = seqlen;
        offset_s_   = init_offset_cs.y + step;
        is_valid_s_ = offset_s_ < seqlen;
    }

    __device__ void PrefetchStage()
    {
        PRAGMA_UNROLL
        for (int i = 0; i < kIterCount; ++i) {
            Prefetch(is_valid_s_);
            ++(*this);
        }
        AdvancePrefetchStage();
    }

    __device__ void PrefetchBatch(int batch_idx, int batch_size)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < kIterCount) {
                Prefetch(is_valid_s_);
                ++(*this);
            }
        }
    }

    __device__ Iterator& operator++()
    {
        src_offset_ += kStepC;
        dst_offset_ += kStepC;
        ++iter_c_;
        if (iter_c_ < ThreadMap::kIterC) {
            return *this;
        }

        iter_c_ = 0;
        src_offset_ += kStepS;
        dst_offset_ += kStepS;

        offset_s_ += ThreadMap::kDeltaS;
        is_valid_s_ = offset_s_ < seq_len_;

        return *this;
    }

    __device__ void AdvancePrefetchStage()
    {
        src_offset_ += kStepK;
        dst_offset_ += kStepK;

        offset_s_ += ThreadMap::kS - ThreadMap::kIterS * ThreadMap::kDeltaS;

        is_valid_s_ = offset_s_ < seq_len_;

        if constexpr (kUseBlockIter) {
            if (is_valid_s_) {
                block_k_ -= ThreadMap::kS;
                if (block_k_ == 0) {
                    src_        = (const T*)block_iterator_.Next() + head_idx_ * block_size_ * ThreadMap::kC;
                    block_k_    = block_size_;
                    src_offset_ = init_offset_;
                }
            }
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("%d %d %d\n", offset_s_, src_offset_ / ThreadMap::kC, block_k_);
            // }
        }

        // if (init_offset_ / ThreadMap::kC == 0) {
        //     int k = dst_offset_ / (ThreadMap::kS * ThreadMap::kC);
        //     int s = dst_offset_ % (ThreadMap::kS * ThreadMap::kC) / ThreadMap::kC;
        //     int c = dst_offset_ % ThreadMap::kC;
        //     printf("tid=%d, k=%d, s=%d, c=%d, offset_s=%d, valid_s=%d, init_s=%d\n",
        //            threadIdx.x,
        //            k,
        //            s,
        //            c,
        //            offset_s_,
        //            (int)is_valid_s_,
        //            init_offset_ / ThreadMap::kC);
        // }

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("next stage %d\n", offset_s_);
        // }

        if (dst_offset_ >= Stages * kSizePerTile) {
            dst_offset_ -= Stages * kSizePerTile;
        }

        // if constexpr (Chained) {
        //     bool is_last_stage = *signal_iterator_;

        //     ++signal_iterator_;

        //     if (is_last_stage) {
        //         AdvancePrefetchSlice();
        //     }
        // }
    }

#if 0
    __device__ void AdvancePrefetchSlice()
    {
        src_        = (const T*)block_iterator_.Next();
        src_offset_ = init_offset_;

        ++iter_b_;
        offset_s_   = iter_b_ / 2 * BlockLen + init_offset_ / ThreadMap::kC;
        is_valid_s_ = offset_s_ < seq_len_;
    }
#endif

    static __device__ void CpAsync(T* dst, const T* src, bool mask)
    {
        const int     smem_int_ptr = cast_smem_ptr_to_uint(dst);
        constexpr int cp_size      = sizeof(AccessType);
        static_assert(cp_size == 16);
        // cp.async.cg.shared.global.L2::256B
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(smem_int_ptr),
                     "l"(src),
                     "n"(cp_size));
    }

    static __device__ void Copy(T* dst, const T* src, bool mask)
    {
        if (mask) {
            Ldg(*(AccessType*)dst, src);
        }
    }

    __device__ void Prefetch(bool mask)
    {
        // if (blockIdx.x == 0 && threadIdx.x == 0 && mask) {
        //     int  c    = src_offset_ % ThreadMap::kC;
        //     int  s    = src_offset_ / ThreadMap::kC;
        //     bool fuck = src_offset_ >= 128 * 4096;
        //     printf("%d %d %d %d %s\n", (int)threadIdx.x, c, s, offset_s_, fuck ? "FUCK" : "");
        // }

        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     int  c    = dst_offset_ % ThreadMap::kC;
        //     int  s    = dst_offset_ / ThreadMap::kC;
        //     bool fuck = (dst_offset_ >= Stages * kSizePerTile);
        //     printf("%d %d %d %s\n", c, s, dst_offset_, fuck ? "FUCK" : "");
        // }

        // if (init_offset_ / ThreadMap::kC == 0) {
        //     int k = dst_offset_ / (ThreadMap::kS * ThreadMap::kC);
        //     int s = dst_offset_ % (ThreadMap::kS * ThreadMap::kC) / ThreadMap::kC;
        //     int c = dst_offset_ % ThreadMap::kC;
        //     printf("tid=%d, k=%d, s=%d, c=%d, offset_s=%d, valid_s=%d, init_s=%d, mask=%d\n",
        //            threadIdx.x,
        //            k,
        //            s,
        //            c,
        //            offset_s_,
        //            (int)is_valid_s_,
        //            init_offset_ / ThreadMap::kC,
        //            (int)mask);
        // }

        CpAsync(smem_ + dst_offset_, src_ + src_offset_, mask);
        // Copy(smem_ + dst_offset_, src_ + src_offset_, mask);
    }

    __device__ void Load(AccessType (&frag)[ThreadMap::kIterC])
    {

        // if (init_offset_ / ThreadMap::kC == 0) {
        //     int k = smem_read_offset_ / (ThreadMap::kS * ThreadMap::kC);
        //     int s = smem_read_offset_ % (ThreadMap::kS * ThreadMap::kC) / ThreadMap::kC;
        //     int c = smem_read_offset_ % ThreadMap::kC;
        //     printf("tid=%d, k=%d, s=%d, c=%d, init_s=%d\n", threadIdx.x, k, s, c, init_offset_ / ThreadMap::kC);
        // }

        for (int vi = 0; vi < ThreadMap::kIterC; ++vi) {

            // int offset = smem_read_offset_ + vi * ThreadMap::kDeltaC;
            // if (offset >= Stages * kSizePerTile || offset % sizeof(AccessType)) {
            //     int c = offset % ThreadMap::kC;
            //     int s = offset / ThreadMap::kC;
            //     printf("%d %d %d\n", c, s, offset);
            // }

            Lds(frag[vi], smem_ + smem_read_offset_ + vi * ThreadMap::kDeltaC);
        }

        smem_read_offset_ += ThreadMap::kDeltaS * ThreadMap::kC;
    }

    __device__ void AdvanceComputeStage()
    {
        smem_read_offset_ += kStepK;

        if (smem_read_offset_ >= Stages * kSizePerTile) {
            smem_read_offset_ -= Stages * kSizePerTile;
        }
    }
};

}  // namespace turbomind