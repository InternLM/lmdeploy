# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

fused_rotary_emb = FunctionDispatcher('fused_rotary_emb').make_caller()
