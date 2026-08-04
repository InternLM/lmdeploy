// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

namespace turbomind {

class ModelWeight;
class MoeWeight;

bool ModelHasMetaMoe(const ModelWeight& model);
void PrepareMetaMoe(ModelWeight& model);
void alias_routed_moe(MoeWeight& dst, const MoeWeight& donor);

}  // namespace turbomind
