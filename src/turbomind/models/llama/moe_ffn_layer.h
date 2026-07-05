// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/moe_weight.h"

namespace turbomind {

class MoeFfnLayerImpl;

class MoeFfnLayer {
public:
    MoeFfnLayer(const EngineParam& engine, const Context& ctx);

    ~MoeFfnLayer();

    struct ForwardParam {
        Tensor           input;
        Tensor           output;
        const MoeWeight* weights;
        float            scale;
        int              layer_id;

        // EP allgather-reducescatter path only.
        Tensor                     global_hidden_states;
        const std::vector<size_t>* ep_elem_counts{};
        Tensor                     shared_output;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    std::unique_ptr<MoeFfnLayerImpl> impl_;
};

}  // namespace turbomind
