// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/norm/norm.h"
#include "src/turbomind/models/vision_model_weight.h"

#include <vector>

namespace turbomind::core {

struct InternVitConfig: ModuleConfig {
    InternVitConfig(): ModuleConfig{"InternVitWeight"} {}

    DataType data_type{};
    int      hidden_dim{0};
    int      depth{0};
    int      patch_in_dim{0};
    int      in_channels{0};
    int      image_height{0};
    int      image_width{0};
    int      patch_height{0};
    int      patch_width{0};
    int      num_patches{0};
    int      image_seq_length{0};
    NormType norm_type{NormType::kRMSNorm};

#define INTERNVIT_FIELDS(X)                                                                                            \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, depth)                                                                                                      \
    X(int, patch_in_dim)                                                                                               \
    X(int, in_channels)                                                                                                \
    X(int, image_height)                                                                                               \
    X(int, image_width)                                                                                                \
    X(int, patch_height)                                                                                               \
    X(int, patch_width)                                                                                                \
    X(int, num_patches)                                                                                                \
    X(int, image_seq_length)                                                                                           \
    X(NormType, norm_type, NormType::kRMSNorm)

    TM_FOR_EACH(InternVitConfig, INTERNVIT_FIELDS)

#undef INTERNVIT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class InternVitBlockWeight;
class LayerNormWeight;
class LinearWeight;

class InternVitWeight: public VisionModelWeight {
public:
    const char* type() const override
    {
        return "InternVitWeight";
    }

    InternVitWeight() = default;
    explicit InternVitWeight(const core::InternVitConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

#define INTERNVIT_WEIGHT_CHILDREN(X)                                                                                   \
    X(LinearWeight, patch_embed)                                                                                       \
    X(core::ModuleList, blocks)                                                                                        \
    X(LayerNormWeight, projector_norm)                                                                                 \
    X(LinearWeight, projector_fc1)                                                                                     \
    X(LinearWeight, projector_fc2)

#define INTERNVIT_WEIGHT_PARAMS(X)                                                                                     \
    X(cls_token)                                                                                                       \
    X(position_embeddings)

    TM_MODULE_DECLARE(InternVitWeight, INTERNVIT_WEIGHT_CHILDREN, INTERNVIT_WEIGHT_PARAMS)

    const core::InternVitConfig& config() const noexcept
    {
        return config_;
    }

    InternVitBlockWeight* block(int i) const;

private:
    core::InternVitConfig config_{};
};

}  // namespace turbomind
