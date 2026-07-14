// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

DecoderLayerWeight::DecoderLayerWeight(const core::ModuleConfig&) {}

DecoderLayerWeight::~DecoderLayerWeight() = default;

bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    // At least one of attention or linear_attn must exist
    if (!attention && !linear_attn) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    // At least one of feed_forward or moe_ffn must exist
    if (!feed_forward && !moe_ffn) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    // attention_norm must exist
    if (!attention_norm) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}

TM_MODULE_REGISTER(DecoderLayerWeight, core::ModuleConfig);

TM_MODULE_METHODS(DecoderLayerWeight, DECODER_LAYER_WEIGHT_CHILDREN, DECODER_LAYER_WEIGHT_PARAMS)

}  // namespace turbomind
