// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<class T, int CTA_K>
struct TileIterator {
    const T* ptr_A_;
    const T* ptr_B_;
    int      ki_;
    int      ke_;
    bool     mask_{true};

    template<class Param>
    __device__ TileIterator(const Param& param, int mi, int ni, int ki, int ke)
    {
        ptr_A_ = param.A + mi * param.k + ki;
        ptr_B_ = param.B + ni * param.k + ki;
        ki_    = ki;
        ke_    = ke;
    }

    template<int Idx>
    __device__ void Offset(int offset)
    {
        if constexpr (Idx == 0) {
            ptr_A_ += offset;
        }
        else if constexpr (Idx == 1) {
            ptr_B_ += offset;
        }
        else {
            static_assert(Idx != Idx, "invalid index");
        }
    }

    __device__ TileIterator& operator++()
    {
        ptr_A_ += CTA_K;
        ptr_B_ += CTA_K;

        ki_ += CTA_K;
        if (ki_ >= ke_) {
            mask_ = false;
        }
        return *this;
    }

    __device__ void set_mask(bool mask)
    {
        mask_ = mask;
    }

    __device__ explicit operator bool() const
    {
        return mask_;
    }

    template<int Idx>
    __device__ const T* OffsetPtr(int offset) const
    {
        if constexpr (Idx == 0) {
            return ptr_A_ + offset;
        }
        else if constexpr (Idx == 1) {
            return ptr_B_ + offset;
        }
        else {
            static_assert(Idx != Idx, "invalid index");
        }
    }
};

}  // namespace turbomind