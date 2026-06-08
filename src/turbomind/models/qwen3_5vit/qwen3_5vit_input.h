// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/multimodal_input.h"

#include <array>
#include <utility>
#include <vector>

namespace turbomind {
namespace multimodal {

struct Qwen3_5VitItem {
    Modality           modality;
    Tensor             data;
    int                token_begin;
    int                token_end;
    std::array<int, 3> grid_thw;

    Qwen3_5VitItem() = default;

    Qwen3_5VitItem(Modality modality, Tensor data, int token_begin, int token_end, std::array<int, 3> grid_thw):
        modality{modality}, data{std::move(data)}, token_begin{token_begin}, token_end{token_end}, grid_thw{grid_thw}
    {
    }
};

struct Qwen3_5VitInput final: Input {
    std::vector<Qwen3_5VitItem> items;

    Qwen3_5VitInput() = default;

    explicit Qwen3_5VitInput(std::vector<Qwen3_5VitItem> items): items{std::move(items)} {}
};

}  // namespace multimodal
}  // namespace turbomind
