/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>
#include <vector>

#pragma once

namespace fastertransformer {

template<typename T>
struct FtWeight {

public:
    std::string         name_;
    std::vector<size_t> shape_;
    size_t              size_ = 0;
    T*                  ptr_  = nullptr;

    FtWeight() {}
    FtWeight(const std::string name, const std::vector<size_t> shape, T* ptr): name_(name), shape_(shape), ptr_(ptr)
    {
        size_ = 1;
        for (uint i = 0; i < shape_.size(); i++) {
            size_ *= shape_[i];
        }
    }

    ~FtWeight()
    {
        size_ = 0;
        ptr_  = nullptr;
    }
};

}  // namespace fastertransformer