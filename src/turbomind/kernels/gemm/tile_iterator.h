// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<class T, int CTA_M, int CTA_N, int CTA_K, int Flag>
struct TileIterator {
    const char* ptr_A_;
    const char* ptr_B_;
    int         ki_;
    int         ke_;
    bool        mask_{true};

    int cta_cnt_n_;
    int cta_cnt_k_;

    template<class Param>
    __device__ TileIterator(const Param& param, int mi, int ni, int ki, int ke)
    {
        ptr_A_ = reinterpret_cast<const char*>(param.A + mi * param.k + ki);

        if constexpr (!Flag) {
            ptr_B_ = reinterpret_cast<const char*>(param.B + ni * param.k + ki);
        }
        else {
            // ptr_B_     = param.B + ni / CTA_N * (CTA_N * CTA_K);
            // cta_cnt_n_ = (param.n + CTA_N - 1) / CTA_N;
            // cta_cnt_k_ = (param.k + CTA_K - 1) / CTA_K;
            // ptr_B_     = param.B + ni / CTA_N * cta_cnt_k_ * CTA_N * CTA_K;
            ptr_B_ = reinterpret_cast<const char*>(param.B + (ni / 16) * (param.k / 16) * 32 * 8);
        }

        ki_ = ki;
        ke_ = ke;
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

    __device__ int step_A() const
    {
        return sizeof(T) * CTA_K;
    }

    __device__ int step_B() const
    {
        if constexpr (!Flag) {
            return sizeof(T) * CTA_K;
        }
        else {
            return sizeof(T) * (CTA_K / 16) * 32 * 8;
        }
    }

    template<int Idx>
    __device__ int step() const
    {
        if constexpr (Idx == 0) {
            return step_A();
        }
        else {
            return step_B();
        }
    }

    __device__ TileIterator& operator++()
    {
        ptr_A_ += step_A();
        ptr_B_ += step_B();

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
            return (const T*)ptr_A_ + offset;
        }
        else if constexpr (Idx == 1) {
            return (const T*)ptr_B_ + offset;
        }
        else {
            static_assert(Idx != Idx, "invalid index");
        }
    }
};

}  // namespace turbomind