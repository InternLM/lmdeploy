// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/internvit/internvit_block_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

void InternVitBlockWeight::prepare()
{
    for_each_child([](const char* /*name*/, core::Module* child) {
        if (child) {
            child->prepare();
        }
    });

    if (lambda_1) {
        EnsureFloatDtype(lambda_1, data_type);
    }
    if (lambda_2) {
        EnsureFloatDtype(lambda_2, data_type);
    }
}

TM_MODULE_REGISTER(InternVitBlockWeight, core::InternVitBlockConfig);

TM_MODULE_METHODS(InternVitBlockWeight, INTERNVIT_BLOCK_CHILDREN, INTERNVIT_BLOCK_PARAMS)

}  // namespace turbomind
