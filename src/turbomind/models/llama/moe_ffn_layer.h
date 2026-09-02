// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/moe_weight.h"

namespace turbomind {

class MoeFfnLayerImpl;

class MoeFfnLayer {
public:
    MoeFfnLayer(const EngineParam& engine, const Context& ctx, const MoeWeight* weights);

    ~MoeFfnLayer();

    struct ForwardParam {
        Tensor           input;
        Tensor           output;
        std::vector<int> local_token_num;
        const MoeWeight* weights;
        float            scale;
        int              layer_id;
        const bool*      token_mask;  // [tokens]; invalid tokens route nowhere
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

    Tensor GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums);

private:
    std::unique_ptr<MoeFfnLayerImpl> impl_;
};

}  // namespace turbomind
