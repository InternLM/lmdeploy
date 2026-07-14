/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>   // std::make_unique
#include <sstream>  // std::stringstream
#include <string>
#include <vector>

namespace turbomind {

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{
    // This function came from a code snippet in stackoverflow under cc-by-1.0
    //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    // Disable format-security warning in this function.
#if defined(_MSC_VER)  // for visual studio
#pragma warning(push)
#pragma warning(warning(disable : 4996))
#elif defined(__GNUC__) || defined(__clang__)  // for gcc or clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf  = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

template<typename T>
inline std::string vec2str(std::vector<T> vec)
{
    std::stringstream ss;
    ss << "(";
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            ss << vec[i] << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

template<typename T>
inline std::string arr2str(T* arr, size_t size)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i] << ", ";
    }
    if (size > 0) {
        ss << arr[size - 1];
    }
    ss << ")";
    return ss.str();
}
}  // namespace turbomind
