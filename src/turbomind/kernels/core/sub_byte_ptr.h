// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"

namespace turbomind {

template<class T>
struct SubBytePtr {

    constexpr SubBytePtr() = default;

    constexpr __host__ __device__ explicit SubBytePtr(T* ptr): ptr_((char*)ptr) {}

    constexpr __host__ __device__ SubBytePtr(char* ptr): ptr_(ptr) {}

    __host__ __device__ T& operator[](int i)
    {
        return *reinterpret_cast<T*>(ptr_ + i * bitsof<T> / bitsof<char>);
    }

    friend __host__ __device__ SubBytePtr operator+(const SubBytePtr a, int n)
    {
        return SubBytePtr{a.ptr_ + n * bitsof<T> / bitsof<char>};
    }

    friend __host__ __device__ SubBytePtr operator+(int n, const SubBytePtr a)
    {
        return a + n;
    }

    friend __host__ __device__ bool operator==(const SubBytePtr& a, const SubBytePtr& b)
    {
        return a.ptr_ == b.ptr_;
    }

    __host__ __device__ explicit operator T*() const
    {
        return (T*)ptr_;
    }

    char* ptr_;
};

template<class T>
struct get_pointer_type_t<T, std::enable_if_t<bitsof<T> % 8 != 0>> {
    using type = SubBytePtr<T>;
};

}  // namespace turbomind
