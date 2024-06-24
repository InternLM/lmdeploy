# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

multinomial_sampling = FunctionDispatcher('multinomial_sampling').make_caller()
