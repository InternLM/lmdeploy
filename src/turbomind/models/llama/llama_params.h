// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>

#include "src/turbomind/engine/engine_config.h"

namespace turbomind {

struct EngineParam: EngineConfig {
    // Runtime-derived fields (set in CreateContext)
    int outer_dp_rank = 0;
    int attn_dp_rank  = 0;
    int attn_tp_rank  = 0;
    int attn_cp_rank  = 0;
    int mlp_tp_rank   = 0;
    int model_tp_rank = 0;  // rank(d_tp_group), in [0, attn_tp_size × attn_cp_size)

    // Derived field (set in Impl ctor)
    int max_forward_token_num = 0;
};

}  // namespace turbomind
