// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<class T>
struct basic_type {
    using type = T;
};

template<class T>
constexpr basic_type<T> type_c{};

template<auto v>
struct constant {
    using type       = constant;
    using value_type = decltype(v);

    static constexpr value_type value = v;

    constexpr value_type operator()() const noexcept
    {
        return v;
    }
    constexpr operator value_type() const noexcept
    {
        return v;
    }
};

template<auto u, auto v>
struct pair {
};

template<auto u, auto v>
constexpr auto first(pair<u, v>)
{
    return u;
}

template<auto u, auto v>
constexpr auto second(pair<u, v>)
{
    return v;
}

template<auto u, auto v, auto w>
struct triplet {
};

}  // namespace turbomind
