# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

mbgmv_a = FunctionDispatcher('mbgmv_a').make_caller()

mbgmv_b = FunctionDispatcher('mbgmv_b').make_caller()
