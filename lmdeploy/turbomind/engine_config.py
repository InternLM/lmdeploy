# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass
class EngineConfig:
    """TurboMind Engine config.

    Args:
        model_name (str): the name of the deployed
        model_format (str): the layout of the deployed model. It can be one of the following values [hf, llama, awq], `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by AWQ.
        tp (int): the number of GPU cards used in tensor parallelism, default to 1
        session_len (int): the max session length of a sequence, default to None
        max_batch_size (int): the max batch size during inference, default to 128
        cache_max_entry_count (float): the percentage of gpu memory occupied by the k/v cache, default to 0.5
        quant_policy (int): , default to 0. When k/v is quantized into 8 bit, set it to 4
        rope_scaling_factor (int): scaling factor used for dynamic ntk, default to 0. TurboMind follows the implementation of transformer LlamaAttention
        use_logn_attn (bool): whether or not to use log attn: default to False
    """  # noqa: E501

    model_name: Optional[str] = None
    model_format: Optional[str] = None
    tp: int = 1
    session_len: Optional[int] = None
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.5
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
