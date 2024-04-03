// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"

namespace turbomind {

template<typename T, int N>
struct Array {
    using value_type      = T;
    using size_type       = int;
    using difference_type = int;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    static_assert(N > 0);

    T __a[N];

    TM_HOST_DEVICE constexpr reference operator[](size_type i) noexcept
    {
        return __a[i];
    }

    TM_HOST_DEVICE constexpr const_reference operator[](size_type i) const noexcept
    {
        return __a[i];
    }

    TM_HOST_DEVICE constexpr reference front() noexcept
    {
        return *begin();
    }

    TM_HOST_DEVICE constexpr const_reference front() const noexcept
    {
        return *begin();
    }

    TM_HOST_DEVICE constexpr reference back() noexcept
    {
        return *(end() - 1);
    }

    TM_HOST_DEVICE constexpr const_reference back() const noexcept
    {
        return *(end() - 1);
    }

    TM_HOST_DEVICE constexpr pointer data() noexcept
    {
        return &__a[0];
    }

    TM_HOST_DEVICE constexpr const_pointer data() const noexcept
    {
        return &__a[0];
    }

    TM_HOST_DEVICE constexpr iterator begin() noexcept
    {
        return data();
    }

    TM_HOST_DEVICE constexpr const_iterator begin() const noexcept
    {
        return data();
    }

    TM_HOST_DEVICE constexpr iterator end() noexcept
    {
        return data() + N;
    }

    TM_HOST_DEVICE constexpr const_iterator end() const noexcept
    {
        return data() + N;
    }

    TM_HOST_DEVICE constexpr std::integral_constant<int, N> size() const noexcept
    {
        return {};
    }

    TM_HOST_DEVICE constexpr std::false_type empty() const noexcept
    {
        return {};
    }
};

template<int N>
struct Array<uint4_t, N> {
    using value_type      = detail::__uint4_t;
    using size_type       = int;
    using difference_type = int;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = SubBytePtr<uint4_t>;
    using const_pointer   = SubBytePtr<const uint4_t>;

    static_assert(N % 8 == 0);

    detail::__uint4_t __a[N / 8];

    TM_HOST_DEVICE constexpr reference operator[](size_type i) noexcept
    {
        return __a[i / 8];
    }

    TM_HOST_DEVICE constexpr const_reference operator[](size_type i) const noexcept
    {
        return __a[i / 8];
    }

    TM_HOST_DEVICE constexpr std::integral_constant<int, N> size() const noexcept
    {
        return {};
    }

    TM_HOST_DEVICE constexpr std::false_type empty() const noexcept
    {
        return {};
    }

    TM_HOST_DEVICE constexpr pointer data() noexcept
    {
        return {(char*)&__a[0]};
    }
};

static_assert(sizeof(Array<uint4_t, 8>) == 4);
static_assert(sizeof(Array<uint4_t, 16>) == 8);
static_assert(sizeof(Array<uint4_t, 24>) == 12);
static_assert(sizeof(Array<uint4_t, 32>) == 16);

}  // namespace turbomind