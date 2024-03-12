
#include "block.h"

#pragma once

namespace turbomind {

template<class BlockHead, int CTA_S>
struct BlockIterator {

    BlockHead block_head_;
    char**    block_ptrs_;

    char* block_{};
    int   block_id_{};
    int   block_ti_{};

    __device__ BlockIterator(BlockHead block_head, char** block_ptrs): block_head_{block_head}, block_ptrs_{block_ptrs}
    {
    }

    __device__ void SetTile(int iter)
    {
        block_head_.get_block_coord(iter * CTA_S, block_id_, block_ti_);
        block_ = block_ptrs_[block_id_];
    }

    __device__ void Advance()
    {
        block_ti_ -= CTA_S;
        if (block_ti_ < 0) {
            block_ti_ += block_head_.block_len();
            block_id_ -= 1;
        }
        if (block_id_ >= 0) {
            block_ = block_ptrs_[block_id_];
        }
    }

    template<int Index>
    __device__ auto OffsetPtr(int offset) const
    {
        if constexpr (Index == 0) {
            return block_head_.k_data(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 1) {
            return block_head_.v_data(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 2) {
            return block_head_.k_param(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 3) {
            return block_head_.v_param(block_, block_ti_) + offset;
        }
        else {
            static_assert(Index != Index, "invalid index");
        }
    }
};

template<class T, class Tkv, class BlockLayout_, int CTA_S>
struct BlockIteratorFactory {
    using BlockLayout = BlockLayout_;
    
    BlockLayout_ block_layout_;
    char**       block_ptrs_;
    const int*   cu_block_nums_;
    int          layer_idx_;

    __device__ auto Create(int batch_idx, int head_idx)
    {
        block::Head<T, Tkv, BlockLayout> head{block_layout_, layer_idx_, head_idx};

        char** block_ptrs = block_ptrs_ + cu_block_nums_[batch_idx];

        return BlockIterator<block::Head<T, Tkv, BlockLayout>, CTA_S>{head, block_ptrs};
    }
};

}  // namespace turbomind