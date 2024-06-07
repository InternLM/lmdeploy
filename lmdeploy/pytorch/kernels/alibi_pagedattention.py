# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

alibi_paged_attention_fwd = FunctionDispatcher(
    'alibi_paged_attention_fwd').make_caller()
