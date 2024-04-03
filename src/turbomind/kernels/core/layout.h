// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
namespace turbomind {

template<int Bits, int Base, int Shift>
struct Swizzle {

    using bit_mask = std::integral_constant<int, (1 << Bits) - 1>;
    using yyy_mask = std::integral_constant<int, bit_mask{} << (Base + Shift)>;
    using shift    = std::integral_constant<int, Shift>;

    template<class Offset>
    __host__ __device__ constexpr static auto apply(Offset offset)
    {
        return offset ^ ((offset & yyy_mask{}) >> shift{});
    }

    template<class Offset>
    __host__ __device__ constexpr auto operator()(Offset offset)
    {
        return apply(offset);
    }
};

struct Identity {

    template<class Offset>
    __device__ constexpr static auto apply(Offset offset)
    {
        return offset;
    }

    template<class Offset>
    __device__ Offset operator()(Offset offset)
    {
        return apply(offset);
    }

    template<int D>
    __device__ int AdvanceS(int offset, int s0, int s1)
    {
        return offset;
    }
};

template<int S_, int C_, int S0_, int C0_, class Swizzle_>
struct SmemLayoutV2 {

    // (C0,S0),(   C1,       S1)
    // ( 1,C0),(C0*S0, C0*S0*C1)

    static constexpr int S = S_;
    static constexpr int C = C_;

    static constexpr int S0 = S0_;
    static constexpr int C0 = C0_;

    static_assert(S % S0 == 0);
    static_assert(C % C0 == 0);

    static constexpr int S1 = S / S0;
    static constexpr int C1 = C / C0;

    static constexpr int kSize = S * C;

    static constexpr int kSize0 = S0 * C0;
    static constexpr int kSize1 = S1 * C1;

    using Swizzle = Swizzle_;

    __forceinline__ __device__ static int apply(int s, int c, int offset = 0)
    {
        int s1 = s / S0;
        int s0 = s % S0;
        int c1 = c / C0;
        int c0 = c % C0;
        //            variable             | uniform |         constant
        // return Swizzle::apply(s0 * C0 + c0) + offset + (s1 * C1 + c1) * kSize0;

        // return offset + Swizzle::apply(s0 * C0 + c0) + (s1 * C1 + c1) * kSize0;

        return Swizzle::apply(s0 * C0 + c0) + (s1 * C1 + c1) * kSize0 + offset;
    }

    __forceinline__ __device__ int operator()(int s, int c, int offset = 0)
    {
        return apply(s, c, offset);
    }
};

struct Offset {
    __device__ explicit Offset(int value): value_{value} {};
    __device__ int& operator()()
    {
        return value_;
    }
    __device__ const int& operator()() const
    {
        return value_;
    }
    int value_;
};

template<class T, class Layout>
struct SmemAccessor {
    using Pointer = get_pointer_type<T>;
    Pointer ptr_;
    Layout  layout_;

    __device__ SmemAccessor(Pointer ptr): ptr_{ptr} {}

    __device__ T& operator()(int s, int c)
    {
        return ptr_[layout_(s, c)];
    }

    __device__ T& operator()(int s, int c, int offset)
    {
        return ptr_[layout_(s, c, offset)];
    }

    // __device__ T& operator()(int s, int c, int offset)
    // {
    //     // return *((T*)((char*)ptr_ + offset) + layout_(s, c));
    //     return *(T*)((char*)(ptr_ + layout_(s, c)) + offset);
    // }
};

}  // namespace turbomind
