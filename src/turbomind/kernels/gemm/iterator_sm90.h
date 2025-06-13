#pragma once

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

namespace turbomind::gemm {

template<int multicast>
struct GmemIteratorSm90 {

    const CUtensorMap* desc_ptr_;
    int2               offset_;
    int2               step_;

    __device__ GmemIteratorSm90(const CUtensorMap* desc_ptr, int2 offset, int2 step)
    {
        desc_ptr_ = desc_ptr;
        offset_   = offset;
        step_     = step;
    }

    __device__ void Step(uint64_t* mbar_ptr, void* smem_ptr, uint16_t mask, uint64_t cache_hint = 0)
    {
        if constexpr (multicast > 1) {
            cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
                desc_ptr_, mbar_ptr, mask, cache_hint, smem_ptr, offset_.x, offset_.y);
        }
        else {
            cute::SM90_TMA_LOAD_2D::copy(desc_ptr_, mbar_ptr, cache_hint, smem_ptr, offset_.x, offset_.y);
        }
        offset_.x += step_.x;
        offset_.y += step_.y;
    }
};

}  // namespace turbomind::gemm
