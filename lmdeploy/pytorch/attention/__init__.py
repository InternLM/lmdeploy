# Copyright (c) OpenMMLab. All rights reserved.
# attention module is modified from:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/
from .selector import get_attn_backend

__all__ = ['get_attn_backend']
