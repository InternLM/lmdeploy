# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

rearange_all_gather = FunctionDispatcher('rearange_all_gather').make_caller()
