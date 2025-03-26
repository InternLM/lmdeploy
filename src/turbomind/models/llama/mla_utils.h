// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void invokeMLACopyQKV(T*           qkv,
                      const T*     q,
                      const T*     kv_a,
                      const T*     kv_b,
                      int          token_num,
                      int          head_num,
                      int          nope_dim,
                      int          rope_dim,
                      int          kv_lora_rank,
                      int          v_head_dim,
                      cudaStream_t stream);

template<class T>
void dispatchMLACopyQKV(T*           qkv,
                        const T*     q,
                        const T*     kv_a,
                        const T*     kv_b,
                        int          token_num,
                        int          head_num,
                        int          nope_dim,
                        int          rope_dim,
                        int          kv_lora_rank,
                        int          v_head_dim,
                        cudaStream_t stream)
{
    auto invoke = [&](auto x) {
        using type = decltype(x);
        invokeMLACopyQKV((type*)qkv,
                         (const type*)q,
                         (const type*)kv_a,
                         (const type*)kv_b,
                         token_num,
                         head_num,
                         nope_dim,
                         rope_dim,
                         kv_lora_rank,
                         v_head_dim,
                         stream);
    };
    if constexpr (sizeof(T) == 2) {
        return invoke(uint16_t{});
    }
    FT_CHECK(0);
}

}  // namespace turbomind
