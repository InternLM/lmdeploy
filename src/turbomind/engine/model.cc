// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/model.h"

#include <tuple>

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/speculative_model.h"

namespace turbomind {

Model::Model(std::unique_ptr<LanguageModel>    model,
             std::unique_ptr<VisionModel>      vision_model,
             std::unique_ptr<SpeculativeModel> spec_model,
             const EngineParam&                param,
             Context&                          context,
             int                               phases):
    target{std::move(model)},
    vision{std::move(vision_model)},
    spec{std::move(spec_model)},
    status{param.max_batch_size, phases},
    input_processor{param,
                    target->weights().hidden_units,
                    target->weights().data_type,
                    param.async_ ? 2 : 1,
                    spec != nullptr,
                    spec ? spec->policy().max_proposals() + 1 : 1,
                    spec ? spec->requires_successor_input_embeddings() : false},
    generation{kFloat32,
               param.max_batch_size,
               param.session_len,
               target->weights().vocab_size,
               target->weights().output->output_dim * context.comm.h_tp_group->n_ranks(),
               target->weights().hidden_units,
               target->weights().data_type,
               context.comm.h_tp_group,
               param.async_ ? 2 : 1,
               spec ? &spec->policy() : nullptr,
               param.enable_metrics},
    output_processor{*target, context.comm.h_tp_group->rank(), param.async_ ? 2 : 1}
{
}

Model::~Model() = default;

void Model::Run(BatchOp op, int phase, TensorMap& env)
{
    std::apply(
        [&](auto&&... modules) {
            auto run = [&](auto* module) {
                if (module) {
                    module->Run(op, phase, env);
                }
            };
            (run(modules), ...);
        },
        OrderedComponents());
}

}  // namespace turbomind
