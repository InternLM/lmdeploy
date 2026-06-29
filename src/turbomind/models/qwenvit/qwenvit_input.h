// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/multimodal_input.h"

#include <array>
#include <utility>
#include <vector>

namespace turbomind {
namespace multimodal {

// Unified multimodal input for the Qwen2-VL / Qwen2.5-VL / Qwen3.5 ViT encoders.
struct QwenVitItem {
    Modality           modality;
    Tensor             data;
    int                token_begin;
    int                token_end;
    std::array<int, 3> grid_thw;

    QwenVitItem() = default;

    QwenVitItem(Modality modality, Tensor data, int token_begin, int token_end, std::array<int, 3> grid_thw):
        modality{modality}, data{std::move(data)}, token_begin{token_begin}, token_end{token_end}, grid_thw{grid_thw}
    {
    }
};

struct QwenVitInput final: Input {
    std::vector<QwenVitItem> items;

    QwenVitInput() = default;

    explicit QwenVitInput(std::vector<QwenVitItem> items): items{std::move(items)} {}
};

}  // namespace multimodal
}  // namespace turbomind
