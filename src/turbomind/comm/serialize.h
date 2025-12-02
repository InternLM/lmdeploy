// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/request.h"

namespace turbomind::comm {

template<typename T>
using TrivialType = std::enable_if_t<std::is_trivially_copyable_v<T>, char*>;

template<typename T>
using NonTrivialType = std::enable_if_t<!std::is_trivially_copyable_v<T>, char*>;

template<typename T>
inline NonTrivialType<T> serialize(char* data, size_t& size, const T& v)
{
    throw std::invalid_argument("not implemented");
    return {};
}

template<typename T>
inline NonTrivialType<T> deserialize(T& v, char* data)
{
    throw std::invalid_argument("not implemented");
    return {};
}

template<typename T>
inline char* serialize(char* data, size_t& size, const T* v, int n)
{
    for (int i = 0; i < n; ++i) {
        data = serialize(data, size, v[i]);
    }
    return data;
}

template<typename T>
inline char* deserialize(T* v, int n, char* data)
{
    for (int i = 0; i < n; ++i) {
        data = deserialize(v[i], data);
    }
    return data;
}

template<typename T>
inline TrivialType<T> serialize(char* data, size_t& size, const T& v)
{
    int n = sizeof(v);
    if (data) {
        std::memcpy(data, (char*)&v, n);
    }
    size += n;
    return data ? data + n : data;
}

template<typename T>
inline TrivialType<T> deserialize(T& v, char* data)
{
    int n = sizeof(v);
    std::memcpy((char*)&v, data, n);
    return data + n;
}

template<typename T>
inline TrivialType<T> serialize(char* data, size_t& size, const std::vector<T>& vec)
{
    data  = serialize(data, size, (int)vec.size());
    int n = sizeof(T) * vec.size();
    if (data) {
        std::memcpy(data, (char*)vec.data(), n);
    }
    size += n;
    return data ? data + n : data;
}

template<typename T>
inline TrivialType<T> deserialize(std::vector<T>& vec, char* data)
{
    int vsz;
    data = deserialize(vsz, data);
    vec.resize(vsz);
    int n = sizeof(T) * vec.size();
    std::memcpy((char*)vec.data(), data, n);
    return data + n;
}

char* serialize(char* data, size_t& size, const std::string& s);

char* deserialize(std::string& s, char* data);

char* serialize(char* data, size_t& size, const GenerationConfig& gen);

char* deserialize(GenerationConfig& gen, char* data);

char* serialize(char* data, size_t& size, const SessionParam& sess);

char* deserialize(SessionParam& sess, char* data);

char* serialize(char* data, size_t& size, const Layout& layout);

char* deserialize(Layout& layout, char* data);

char* serialize(char* data, size_t& size, const Buffer& buffer);

char* deserialize(Buffer& buffer, char* data);

char* serialize(char* data, size_t& size, const Tensor& tensor);

char* deserialize(Tensor& tensor, char* data);

char* serialize(char* data, size_t& size, const TensorMap& map);

char* deserialize(TensorMap& map, char* data);

char* serialize(char* data, size_t& size, const Request& req);

char* deserialize(Request& req, char* data);

}  // namespace turbomind::comm
