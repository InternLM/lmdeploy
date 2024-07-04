# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

rms_norm = FunctionDispatcher('rms_norm').make_caller()
