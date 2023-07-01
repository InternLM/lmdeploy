/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "cuda_fp16.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace turbomind {

template<typename T>
class GPUBuf {
public:
    GPUBuf(size_t size, bool random_init = true): size(size), ptr(nullptr)
    {
        deviceMalloc(&ptr, size, random_init);
    }
    template<typename T2>
    GPUBuf(const GPUBuf<T2>& buf_src): size(buf_src.size), ptr(nullptr)
    {
        deviceMalloc(&ptr, size, false);
        set(buf_src);
    }

    template<typename T2>
    void set(const GPUBuf<T2>& buf_src)
    {
        if (std::is_same<T, T2>::value) {
            cudaD2Dcpy(ptr, reinterpret_cast<T*>(buf_src.ptr), size);
        }
        else {
            invokeCudaCast(ptr, buf_src.ptr, size, 0);
        }
    }

    void set(const T* h_ptr)
    {
        cudaH2Dcpy(ptr, h_ptr, size);
    }

    void to_host(T* h_ptr) const
    {
        cudaD2Hcpy(h_ptr, ptr, size);
    }

    std::vector<T> to_host_vec() const
    {
        std::vector<T> host_vec(size);
        cudaD2Hcpy(host_vec.data(), ptr, size);
        return host_vec;
    }

    void zero()
    {
        deviceMemSetZero(ptr, size);
    }

    ~GPUBuf()
    {
        if (ptr != nullptr)
            cudaFree(ptr);
    }

    size_t size;
    T*     ptr;
};

}  // namespace turbomind
