# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

fill_kv_cache = FunctionDispatcher('fill_kv_cache').make_caller()
