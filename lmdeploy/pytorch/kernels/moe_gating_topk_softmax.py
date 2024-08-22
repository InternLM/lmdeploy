# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

moe_gating_topk_softmax = FunctionDispatcher(
    'moe_gating_topk_softmax').make_caller()
