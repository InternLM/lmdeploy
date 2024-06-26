# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

mbgmm_a = FunctionDispatcher('mbgmm_a').make_caller()
mbgmm_b = FunctionDispatcher('mbgmm_b').make_caller()
