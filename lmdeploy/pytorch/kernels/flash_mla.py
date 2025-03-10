# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

flash_mla_fwd = FunctionDispatcher('flash_mla_fwd').make_caller()
