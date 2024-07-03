# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

fused_moe = FunctionDispatcher('fused_moe').make_caller()
