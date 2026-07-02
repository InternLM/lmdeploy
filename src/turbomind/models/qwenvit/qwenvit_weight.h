// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/norm/norm.h"
#include "src/turbomind/models/vision_model_weight.h"

#include <vector>

namespace turbomind::core {

/// Root config for the Qwen ViT family (Qwen2-VL / Qwen2.5-VL / Qwen3.5).
///
/// Carries every structural scalar the C++ runtime needs to allocate kernels.
/// The feature set of the two model families is the union of orthogonal toggles:
///   - window attention       (Qwen2.5):  use_window_attention / window_size / fullatt_block_indexes
///   - learned pos embedding   (Qwen3.5):  num_position_embeddings > 0 (+ pos_embed weight)
///   - gated SiLU MLP          (Qwen2.5):  gated_mlp
///   - tanh-approx GELU MLP    (Qwen3.5):  gelu_tanh
///   - RMSNorm vs LayerNorm    (Qwen2):    norm_type
///
/// Each field is visited by the X-macro below so pybind11 exposes it as a
/// read/write attribute on the Python ``QwenVitConfig``.
struct QwenVitConfig: ModuleConfig {
    QwenVitConfig(): ModuleConfig{"QwenVitWeight"} {}

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
    int              num_position_embeddings{0};
    int              window_size{0};
    bool             gated_mlp{false};
    bool             use_window_attention{false};
    bool             gelu_tanh{false};
    NormType         norm_type{NormType::kLayerNorm};
    float            norm_eps{1e-6f};
    std::vector<int> fullatt_block_indexes;

#define QWENVIT_FIELDS(X)                                                                                              \
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
    X(int, num_position_embeddings, 0)                                                                                 \
    X(int, window_size, 0)                                                                                             \
    X(bool, gated_mlp, false)                                                                                          \
    X(bool, use_window_attention, false)                                                                               \
    X(bool, gelu_tanh, false)                                                                                          \
    X(NormType, norm_type, NormType::kLayerNorm)                                                                       \
    X(std::vector<int>, fullatt_block_indexes)                                                                         \
    X(float, norm_eps, 1e-6f)

    TM_FOR_EACH(QwenVitConfig, QWENVIT_FIELDS)

#undef QWENVIT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

// Forward decls
class LayerNormWeight;
class LinearWeight;
class QwenVitBlockWeight;

/// Unified Qwen ViT weight tree (Qwen2-VL / Qwen2.5-VL / Qwen3.5).
///
/// Tree:
///   patch_embed   LinearWeight (Conv3d-as-Linear; in_dim = C·T·patch²)
///   pos_embed     raw tensor   (num_position_embeddings × hidden_dim) — Qwen3.5 only, optional
///   blocks        ModuleList   of QwenVitBlockWeight × depth
///   merger_fc1    LinearWeight (in: hidden·spatial_merge², out: 4·hidden)
///   merger_fc2    LinearWeight (in: 4·hidden,              out: out_hidden)
///   merger_norm   LayerNormWeight or NormWeight (over hidden_dim)
///
/// We expose ``merger_*`` as direct children rather than a sub-module to
/// keep the weight tree shallow — the merger has only three pieces.
class QwenVitWeight: public VisionModelWeight {
public:
    const char* type() const override
    {
        return "QwenVitWeight";
    }

    QwenVitWeight() = default;
    explicit QwenVitWeight(const core::QwenVitConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define QWENVIT_WEIGHT_CHILDREN(X)                                                                                     \
    X(LinearWeight, patch_embed)                                                                                       \
    X(core::ModuleList, blocks)                                                                                        \
    X(LinearWeight, merger_fc1)                                                                                        \
    X(LinearWeight, merger_fc2)                                                                                        \
    X(core::Module, merger_norm)

#define QWENVIT_WEIGHT_PARAMS(X) X(pos_embed)

    TM_MODULE_DECLARE(QwenVitWeight, QWENVIT_WEIGHT_CHILDREN, QWENVIT_WEIGHT_PARAMS)

    // --- Accessors ---
    const core::QwenVitConfig& config() const noexcept
    {
        return config_;
    }

    QwenVitBlockWeight* block(int i) const;

private:
    core::QwenVitConfig config_{};
};

}  // namespace turbomind
