// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/fast_math.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind::verification_attention {

struct Arguments {
    void*       out{};
    const void* q{};
    const void* q_bias{};
    int64_t     q_stride{};

    char* const* block_ptrs{};
    const int*   block_ptr_offsets{};
    const int*   q_offsets{};
    const int*   k_offsets{};
    const bool*  finished{};

    int request_count{};
    int query_count{};
    int query_offset{};
    int max_query_length{};
    int max_key_length{};

    int query_head_count{};
    int kv_head_count{};
    int query_group_size{};
    cutlass::FastDivmod query_group_size_divmod{};
    int head_dim{};
    int block_len{};
    cutlass::FastDivmod block_len_divmod{};
    int cache_block_offset{};
    int window_size{};

    float qk_scale_log2{};
    RopeKernelParam rope{};

    int    split_count{};
    float* partial_o{};
    float* partial_ml{};

    DataType     data_type{};
    cudaStream_t stream{};
};

struct Capability {
    int      arch{};
    DataType data_type{};
    int      head_dim{};
    int      max_query_length{};
    int      quant_policy{};
    int      cp_size{};
    bool     is_mla{};
    bool     has_attention_sinks{};
    int      max_dynamic_smem_bytes{};
};

bool supports(const Capability& capability);

int choose_split_count(int query_count,
                       int base_cta_count,
                       int max_key_length,
                       int key_tile,
                       int partial_capacity,
                       int requested_max_splits,
                       int sm_count);

inline bool UsesWgmma(const Arguments& arguments)
{
    return arguments.data_type == DataType::kBfloat16
           && arguments.head_dim == 256
           && arguments.block_len == 64
           && arguments.max_query_length <= 16
           && arguments.max_query_length * arguments.query_group_size <= 128;
}

inline int WgmmaM(const Arguments& arguments)
{
    const int m = arguments.max_query_length * arguments.query_group_size;
    return m <= 32 ? 32 : m <= 64 ? 64 : 128;
}

inline bool UsesM32(const Arguments& arguments)
{
    return UsesWgmma(arguments) && WgmmaM(arguments) == 32;
}

inline bool UsesM64(const Arguments& arguments)
{
    return UsesWgmma(arguments) && WgmmaM(arguments) == 64;
}

inline bool UsesM128(const Arguments& arguments)
{
    return UsesWgmma(arguments) && WgmmaM(arguments) == 128;
}

inline bool UsesRuntimeWgmmaM(const Arguments& arguments)
{
    return UsesM32(arguments) || UsesM64(arguments) || UsesM128(arguments);
}

inline int CtaM(const Arguments& arguments)
{
    const int m = arguments.max_query_length * arguments.query_group_size;
    return m <= 64 ? 64 : 128;
}

inline int KeyTile(const Arguments& arguments)
{
    return arguments.head_dim == 128 || UsesRuntimeWgmmaM(arguments) ? 64 : 32;
}

void run(const Arguments& arguments);

}  // namespace turbomind::verification_attention
