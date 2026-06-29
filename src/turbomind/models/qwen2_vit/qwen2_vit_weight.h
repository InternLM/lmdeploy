// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/norm/norm.h"
#include "src/turbomind/models/vision_model_weight.h"

#include <vector>

namespace turbomind::core {

/// Root config for Qwen2-VL / Qwen2.5-VL ViT.
struct Qwen2VitConfig: ModuleConfig {
    Qwen2VitConfig(): ModuleConfig{"Qwen2VitWeight"} {}

    DataType         data_type{};
    int              hidden_dim{0};
    int              out_hidden_dim{0};
    int              depth{0};
    int              head_num{0};
    int              intermediate_size{0};
    int              patch_in_dim{0};
    int              in_channels{0};
    int              patch_size{0};
    int              temporal_patch_size{0};
    int              spatial_merge_size{0};
    int              window_size{0};
    bool             gated_mlp{false};
    bool             use_window_attention{false};
    NormType         norm_type{NormType::kLayerNorm};
    float            norm_eps{1e-6f};
    std::vector<int> fullatt_block_indexes;

#define QWEN2VIT_FIELDS(X)                                                                                             \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, out_hidden_dim)                                                                                             \
    X(int, depth)                                                                                                      \
    X(int, head_num)                                                                                                   \
    X(int, intermediate_size)                                                                                          \
    X(int, patch_in_dim)                                                                                               \
    X(int, in_channels)                                                                                                \
    X(int, patch_size)                                                                                                 \
    X(int, temporal_patch_size)                                                                                        \
    X(int, spatial_merge_size)                                                                                         \
    X(int, window_size)                                                                                                \
    X(bool, gated_mlp, false)                                                                                          \
    X(bool, use_window_attention, false)                                                                               \
    X(NormType, norm_type, NormType::kLayerNorm)                                                                       \
    X(std::vector<int>, fullatt_block_indexes)                                                                         \
    X(float, norm_eps, 1e-6f)

    TM_FOR_EACH(Qwen2VitConfig, QWEN2VIT_FIELDS)

#undef QWEN2VIT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

// Forward decls
class LayerNormWeight;
class LinearWeight;
class Qwen2VitBlockWeight;

/// Concrete Qwen2 ViT weight tree.
///
/// Tree:
///   patch_embed   LinearWeight (Conv3d-as-Linear; in_dim = C·T·patch²)
///   blocks        ModuleList   of Qwen2VitBlockWeight × depth
///   merger_fc1    LinearWeight (in: hidden·spatial_merge², out: 4·hidden)
///   merger_fc2    LinearWeight (in: 4·hidden,              out: out_hidden)
///   merger_norm   LayerNormWeight or NormWeight (over hidden_dim)
///
/// We expose ``merger_*`` as direct children rather than a sub-module to
/// keep the weight tree shallow — the merger has only three pieces.
class Qwen2VitWeight: public VisionModelWeight {
public:
    const char* type() const override
    {
        return "Qwen2VitWeight";
    }

    Qwen2VitWeight() = default;
    explicit Qwen2VitWeight(const core::Qwen2VitConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define QWEN2VIT_WEIGHT_CHILDREN(X)                                                                                    \
    X(LinearWeight, patch_embed)                                                                                       \
    X(core::ModuleList, blocks)                                                                                        \
    X(LinearWeight, merger_fc1)                                                                                        \
    X(LinearWeight, merger_fc2)                                                                                        \
    X(core::Module, merger_norm)

#define QWEN2VIT_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(Qwen2VitWeight, QWEN2VIT_WEIGHT_CHILDREN, QWEN2VIT_WEIGHT_PARAMS)

    // --- Accessors ---
    const core::Qwen2VitConfig& config() const noexcept
    {
        return config_;
    }

    Qwen2VitBlockWeight* block(int i) const;

private:
    core::Qwen2VitConfig config_{};
};

}  // namespace turbomind
