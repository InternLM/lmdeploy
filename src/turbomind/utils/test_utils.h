/*
 * Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
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

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace turbomind {

#define TIMEIT(print, n, stream, fn, ...)                                                                              \
    ({                                                                                                                 \
        cudaEvent_t _macro_event_start, _macro_event_stop;                                                             \
        cudaEventCreate(&_macro_event_start);                                                                          \
        cudaEventCreate(&_macro_event_stop);                                                                           \
        cudaEventRecord(_macro_event_start, stream);                                                                   \
        for (int i = 0; i < n; i++) {                                                                                  \
            fn(__VA_ARGS__);                                                                                           \
        }                                                                                                              \
        cudaEventRecord(_macro_event_stop, stream);                                                                    \
        cudaStreamSynchronize(stream);                                                                                 \
        float ms = 0.0f;                                                                                               \
        cudaEventElapsedTime(&ms, _macro_event_start, _macro_event_stop);                                              \
        ms /= n;                                                                                                       \
        if (print)                                                                                                     \
            printf("[TIMEIT] " #fn ": %.2fµs\n", ms * 1000);                                                           \
        ms;                                                                                                            \
    })

template<typename T>
struct rel_abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs == 0 ? 0 : static_cast<T>(fabs(lhs - rhs) / fabs(lhs));
    }
};

template<typename T>
struct abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return static_cast<T>(fabs(lhs - rhs));
    }
};

}  // namespace turbomind
