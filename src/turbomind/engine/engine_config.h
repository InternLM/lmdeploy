// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <string>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/module.h"

namespace turbomind {

struct EngineConfig {
#define ENGINE_FIELDS(X)                                                                                               \
    X(DataType, data_type)                                                                                             \
    X(int, cache_block_seq_len, 0)                                                                                     \
    X(int, quant_policy, 0)                                                                                            \
    X(int, tune_layer_num, 1)                                                                                          \
    X(int, max_batch_size, 0)                                                                                          \
    X(int, max_prefill_token_num, 0)                                                                                   \
    X(int, max_context_token_num, 0)                                                                                   \
    X(int, session_len, 0)                                                                                             \
    X(float, cache_max_block_count, 0)                                                                                 \
    X(int, cache_chunk_size, 0)                                                                                        \
    X(bool, enable_prefix_caching, false)                                                                              \
    X(bool, enable_metrics, false)                                                                                     \
    X(int, num_tokens_per_iter, 0)                                                                                     \
    X(int, max_prefill_iters, 1)                                                                                       \
    X(int, async_, 0)                                                                                                  \
    X(int, outer_dp_size)                                                                                              \
    X(int, attn_dp_size)                                                                                               \
    X(int, attn_tp_size)                                                                                               \
    X(int, attn_cp_size)                                                                                               \
    X(int, mlp_tp_size)                                                                                                \
    X(std::vector<int>, devices)                                                                                       \
    X(int, nnodes)                                                                                                     \
    X(int, node_rank)                                                                                                  \
    X(std::string, communicator)

    ENGINE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(EngineConfig, ENGINE_FIELDS)

#undef ENGINE_FIELDS
};

}  // namespace turbomind
