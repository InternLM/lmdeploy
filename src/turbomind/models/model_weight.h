// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/utils/memory_utils.h"

#include <vector>

namespace turbomind::core {

struct ModelWeightConfig: ModuleConfig {
    ModelWeightConfig(): ModuleConfig{"ModelWeight"} {}

#define MODEL_WEIGHT_FIELDS(X)                                                                                         \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)                                                                                                    \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_units)

    MODEL_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(ModelWeightConfig, MODEL_WEIGHT_FIELDS)

#undef MODEL_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class DecoderLayerWeight;

/// Root weight module for a model. Owns the full weight tree.
class ModelWeight: public core::Module {
public:
    const char* type() const override
    {
        return "ModelWeight";
    }

    ModelWeight() = default;

    explicit ModelWeight(const core::ModelWeightConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define MODEL_WEIGHT_CHILDREN(X)                                                                                       \
    X(LinearWeight, output)                                                                                            \
    X(NormWeight, norm)                                                                                                \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X) X(tok_embeddings)

    TM_MODULE_DECLARE(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)

    // --- Accessors ---
    DecoderLayerWeight*              layer(int i) const;
    std::vector<DecoderLayerWeight*> layers_list() const;

    // --- Derived in prepare() from children -- public for direct access ---
    DataType         data_type{};
    int              hidden_units{};
    int              vocab_size{};
    int              vocab_size_padded{};
    int              embedding_size{};
    int              num_layer{};
    int              head_dim{};
    int              kv_head_num{};
    std::vector<int> layer_types;

    // --- From ModelWeightConfig at construction ---
    int tp_size{};
    int tp_rank{};

private:
    mutable std::vector<DecoderLayerWeight*> layers_cache_;
};

}  // namespace turbomind
