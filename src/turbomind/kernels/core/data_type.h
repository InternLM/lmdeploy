// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind {

struct uint1_t {
};
struct uint2_t {
};
struct uint3_t {
};
struct uint4_t {
};
struct uint5_t {
};
struct uint6_t {
};

template<class T>
struct bitsof_t: std::integral_constant<int, sizeof(T) * 8> {
};

template<>
struct bitsof_t<uint1_t>: std::integral_constant<int, 1> {
};

template<>
struct bitsof_t<uint2_t>: std::integral_constant<int, 2> {
};

template<>
struct bitsof_t<uint3_t>: std::integral_constant<int, 3> {
};  // 2 + 1

template<>
struct bitsof_t<uint4_t>: std::integral_constant<int, 4> {
};

template<>
struct bitsof_t<uint5_t>: std::integral_constant<int, 5> {
};  // 4 + 1

template<>
struct bitsof_t<uint6_t>: std::integral_constant<int, 6> {
};  // 4 + 2

template<class T>
inline constexpr bitsof_t<T> bitsof{};

namespace detail {

struct __uint4_t {
    uint32_t x;
};

}  // namespace detail

template<class T>
struct SubBytePtr {

    __device__ T& operator[](int i)
    {
        return *reinterpret_cast<T*>(ptr_ + i * bitsof<T> / bitsof<char>);
    }

    friend __device__ SubBytePtr operator+(const SubBytePtr a, int n)
    {
        return SubBytePtr{a.ptr_ + n * bitsof<T> / bitsof<char>};
    }

    friend __device__ SubBytePtr operator+(int n, const SubBytePtr a)
    {
        return a + n;
    }

    __device__ explicit operator T*() const
    {
        return (T*)ptr_;
    }

    char* ptr_;
};

template<class T, class SFINAE = void>
struct get_pointer_type_t {
    using type = T*;
};

template<class T>
struct get_pointer_type_t<T, std::enable_if_t<bitsof<T> % 8 != 0>> {
    using type = SubBytePtr<T>;
};

template<class T>
using get_pointer_type = typename get_pointer_type_t<T>::type;

}  // namespace turbomind
