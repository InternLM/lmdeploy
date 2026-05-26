// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"

namespace turbomind {

ModelWeight::ModelWeight(const core::ModelWeightConfig& cfg):
    tp_size(cfg.tp_size), tp_rank(cfg.tp_rank), data_type(cfg.data_type), hidden_units(cfg.hidden_units)
{
}

void ModelWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child)
            child->prepare();
    });

    auto* l0 = layer(0);
    TM_CHECK(l0);
    // Find first full-attention layer (linear-attn layers have no attention child)
    DecoderLayerWeight* attn_layer = nullptr;
    for (int i = 0; i < (int)layers->size(); ++i) {
        if (layer(i)->attention) {
            attn_layer = layer(i);
            break;
        }
    }
    TM_CHECK(attn_layer) << "No full-attention layer found";
    head_dim    = attn_layer->attention->head_dim;
    kv_head_num = attn_layer->attention->kv_head_num;

    vocab_size        = tok_embeddings.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = TM_CHECK_NOTNULL(output)->output_dim * tp_size;

    layer_types.resize(num_layer);
    for (int i = 0; i < num_layer; ++i) {
        layer_types[i] = layer(i)->linear_attn ? 1 : 0;
    }

    EnsureFloatDtype(tok_embeddings, data_type);
}

DecoderLayerWeight* ModelWeight::layer(int i) const
{
    if (!layers) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}

std::vector<DecoderLayerWeight*> ModelWeight::layers_list() const
{
    if (!layers_cache_.empty()) {
        return layers_cache_;
    }
    if (!layers) {
        return {};
    }
    layers_cache_.resize(layers->size());
    for (int i = 0; i < layers->size(); ++i) {
        layers_cache_[i] = static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
    }
    return layers_cache_;
}

bool ModelWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!tok_embeddings) {
        missing.push_back(full_path() + ": missing tok_embeddings");
    }
    if (!norm) {
        missing.push_back(full_path() + ": missing norm");
    }
    return missing.empty();
}

TM_MODULE_REGISTER(ModelWeight, core::ModelWeightConfig);

TM_MODULE_METHODS(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)

}  // namespace turbomind
