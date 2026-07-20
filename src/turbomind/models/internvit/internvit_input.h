// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/fingerprint.h"
#include "src/turbomind/engine/multimodal_input.h"

#include <utility>
#include <vector>

namespace turbomind {
namespace multimodal {

struct InternVitItem {
    Modality    modality;
    Tensor      data;
    int         token_begin;
    int         token_end;
    Fingerprint fingerprint{};  // image content hash from the converter (empty if none supplied)

    InternVitItem() = default;

    InternVitItem(Modality modality, Tensor data, int token_begin, int token_end, Fingerprint fingerprint = {}):
        modality{modality},
        data{std::move(data)},
        token_begin{token_begin},
        token_end{token_end},
        fingerprint{fingerprint}
    {
    }
};

struct InternVitInput final: Input {
    std::vector<InternVitItem> items;

    InternVitInput() = default;

    explicit InternVitInput(std::vector<InternVitItem> items): items{std::move(items)} {}
};

}  // namespace multimodal
}  // namespace turbomind
