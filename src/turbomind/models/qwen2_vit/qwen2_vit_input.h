// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/multimodal_input.h"

#include <array>
#include <utility>
#include <vector>

namespace turbomind {
namespace multimodal {

struct Qwen2VitItem {
    Modality           modality;
    Tensor             data;
    int                token_begin;
    int                token_end;
    std::array<int, 3> grid_thw;

    Qwen2VitItem() = default;

    Qwen2VitItem(Modality modality, Tensor data, int token_begin, int token_end, std::array<int, 3> grid_thw):
        modality{modality}, data{std::move(data)}, token_begin{token_begin}, token_end{token_end}, grid_thw{grid_thw}
    {
    }
};

struct Qwen2VitInput final: Input {
    std::vector<Qwen2VitItem> items;

    Qwen2VitInput() = default;

    explicit Qwen2VitInput(std::vector<Qwen2VitItem> items): items{std::move(items)} {}
};

}  // namespace multimodal
}  // namespace turbomind
