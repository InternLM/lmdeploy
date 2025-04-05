// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind::comm {

std::vector<char> streambuf_to_vector(std::streambuf* sb);

template<typename T>
inline void serialize(const T*, int n, std::vector<char>&)
{
    throw std::invalid_argument("not implemented");
}

template<typename T>
inline void deserialize(T*, int n, const std::vector<char>&)
{
    throw std::invalid_argument("not implemented");
}

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline void serialize(std::ostream& os, const T& v)
{
    os.write((char*)&v, sizeof(v));
}

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline void deserialize(std::istream& is, T& v)
{
    is.read((char*)&v, sizeof(v));
}

void serialize(std::ostream& os, const std::string& s);

void deserialize(std::istream& is, std::string& s);

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline void serialize(std::ostream& os, const std::vector<T>& vec)
{
    int size = vec.size();
    os.write((char*)&size, sizeof(int));
    os.write((char*)vec.data(), sizeof(T) * size);
}

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline void deserialize(std::istream& is, std::vector<T>& vec)
{
    int size;
    is.read((char*)&size, sizeof(int));
    vec.resize(size);
    is.read((char*)vec.data(), sizeof(T) * size);
}

void serialize(std::ostream& os, const GenerationConfig& gen);

void deserialize(std::istream& is, GenerationConfig& gen);

void serialize(std::ostream& os, const SessionParam& sess);

void deserialize(std::istream& is, SessionParam& sess);

void serialize(std::ostream& os, const Tensor& tensor);

void deserialize(std::istream& is, ManagedTensor& holder);

void serialize(std::ostream& os, const TensorMap& map);

void deserialize(std::istream& is, TensorMap& map, Request::TensorMap_& map_);

void serialize(std::ostream& os, const Request& req);

void deserialize(std::istream& is, Request& req);

}  // namespace turbomind::comm
