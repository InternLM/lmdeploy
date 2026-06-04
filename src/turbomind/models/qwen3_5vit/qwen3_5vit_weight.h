// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/vision_model_weight.h"

#include <vector>

namespace turbomind::core {

/// Root config for Qwen3.5 ViT. Carries every structural scalar the
/// C++ runtime needs to allocate kernels later (depth, head_num,
/// patcher dims, …). Each field is visited by ``for_each`` via the
/// X-macro below — pybind11 then exposes every field as a read/write
/// attribute on the Python ``Qwen3_5VitConfig``.
struct Qwen3_5VitConfig: ModuleConfig {
    Qwen3_5VitConfig(): ModuleConfig{"Qwen3_5VitWeight"} {}

    DataType data_type{};
    int      hidden_dim{0};
    int      out_hidden_dim{0};
    int      depth{0};
    int      head_num{0};
    int      intermediate_size{0};
    int      patch_in_dim{0};
    int      in_channels{0};
    int      patch_size{0};
    int      temporal_patch_size{0};
    int      num_position_embeddings{0};
    int      spatial_merge_size{0};
    float    norm_eps{1e-6f};

#define QWEN3_5VIT_FIELDS(X)                                                                                           \
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
    X(int, num_position_embeddings)                                                                                    \
    X(int, spatial_merge_size)                                                                                         \
    X(float, norm_eps, 1e-6f)

    TM_FOR_EACH(Qwen3_5VitConfig, QWEN3_5VIT_FIELDS)

#undef QWEN3_5VIT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

// Forward decls
class LayerNormWeight;
class LinearWeight;
class Qwen3_5VitBlockWeight;

/// Concrete Qwen3.5 ViT weight tree.
///
/// Tree:
///   patch_embed   LinearWeight (Conv3d-as-Linear; in_dim = C·T·patch²)
///   pos_embed     raw tensor   (num_position_embeddings × hidden_dim)
///   blocks        ModuleList   of Qwen3_5VitBlockWeight × depth
///   merger_fc1    LinearWeight (in: hidden·spatial_merge², out: 4·hidden)
///   merger_fc2    LinearWeight (in: 4·hidden,              out: out_hidden)
///   merger_norm   LayerNormWeight (over hidden_dim)
///
/// We expose ``merger_*`` as direct children rather than a sub-module to
/// keep the weight tree shallow — the merger has only three pieces.
class Qwen3_5VitWeight: public VisionModelWeight {
public:
    const char* type() const override
    {
        return "Qwen3_5VitWeight";
    }

    Qwen3_5VitWeight() = default;
    explicit Qwen3_5VitWeight(const core::Qwen3_5VitConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

    std::unique_ptr<VisionModel> make_model(const EngineParam& engine, const Context& ctx, int phases) const override;

    // --- X-macro field lists ---
#define QWEN3_5VIT_WEIGHT_CHILDREN(X)                                                                                  \
    X(LinearWeight, patch_embed)                                                                                       \
    X(core::ModuleList, blocks)                                                                                        \
    X(LinearWeight, merger_fc1)                                                                                        \
    X(LinearWeight, merger_fc2)                                                                                        \
    X(LayerNormWeight, merger_norm)

#define QWEN3_5VIT_WEIGHT_PARAMS(X) X(pos_embed)

    TM_MODULE_DECLARE(Qwen3_5VitWeight, QWEN3_5VIT_WEIGHT_CHILDREN, QWEN3_5VIT_WEIGHT_PARAMS)

    // --- Accessors ---
    const core::Qwen3_5VitConfig& config() const noexcept
    {
        return config_;
    }

    Qwen3_5VitBlockWeight* block(int i) const;

private:
    core::Qwen3_5VitConfig config_{};
};

}  // namespace turbomind
