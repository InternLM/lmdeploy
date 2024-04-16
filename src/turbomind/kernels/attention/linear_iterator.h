// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<class T, int CTA_S, int HeadDim>
struct LinearIterator {

    const T* kv_cache_;
    int      key_to_val_;

    const T* key_ptr_{};
    int      tile_id_{};

    __device__ LinearIterator(const T* kv_cache, int key_to_val): kv_cache_{kv_cache}, key_to_val_{key_to_val} {}

    __device__ void SetTile(int tile_id)
    {
        key_ptr_ = kv_cache_ + tile_id * CTA_S * HeadDim;
        tile_id_ = tile_id;
    }

    __device__ void Advance()
    {
        --tile_id_;
        if (tile_id_ >= 0) {
            key_ptr_ -= CTA_S * HeadDim;
        }
    }

    template<int Index>
    __device__ const T* OffsetPtr(int offset) const
    {
        if constexpr (Index == 0) {
            return key_ptr_ + offset;
        }
        else if constexpr (Index == 1) {
            return key_ptr_ + offset + key_to_val_;
        }
        else {
            static_assert(Index != Index, "invalid index");
        }
    }
};

template<class Tkv_, int CTA_S, int HeadDim>
struct LinearIteratorFactory {
    using Tkv = Tkv_;

    const Tkv* kv_cache_;
    const int* cu_ctx_len_;
    int        stride_h_;
    int        key_to_val_;

    __device__ auto Create(int batch_idx, int head_idx)
    {
        int seq_ti = cu_ctx_len_[batch_idx] - cu_ctx_len_[0];

        const Tkv* kv_cache = kv_cache_ + head_idx * stride_h_ + seq_ti * HeadDim;

        return LinearIterator<Tkv, CTA_S, HeadDim>{kv_cache, key_to_val_};
    }
};

}  // namespace turbomind
