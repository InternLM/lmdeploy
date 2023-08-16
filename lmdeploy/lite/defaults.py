# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

OFFLOAD_MOD = (nn.Linear, )
KV_CACHE_SIGNATURE = 'past_key_value'
