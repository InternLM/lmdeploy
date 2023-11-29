// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <utility>

namespace turbomind {

namespace detail {

template<int X>
inline constexpr std::integral_constant<int, X> _Int{};

template<class F, class P, class G, int... Xs, std::size_t... Is>
bool dispatch_impl(F&& f, P&& p, G g, std::integer_sequence<int, Xs...>, std::index_sequence<Is...>)
{
    constexpr int N = sizeof...(Xs);
    return (((((P&&)p)(_Int<Xs>) || (g && Is == N - 1)) && (((F&&)f)(_Int<Xs>), 1)) || ...);
}

}  // namespace detail

template<class F, class P, int... Is, class G = std::true_type>
bool dispatch(std::integer_sequence<int, Is...> seq, P&& p, F&& f, G g = {})
{
    return detail::dispatch_impl((F&&)f, (P&&)p, g, seq, std::make_index_sequence<sizeof...(Is)>{});
}

template<class F, int... Is, class G = std::true_type>
bool dispatch(std::integer_sequence<int, Is...> seq, F&& f)
{
    return (((F&&)f)(detail::_Int<Is>) || ...);
}

}  // namespace turbomind
