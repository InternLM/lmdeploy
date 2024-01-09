# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass
class EngineConfig:
    """TurboMind Engine config.

    Args:​
        model_name (str): the name of the deployed model​
        model_format (str): the layout of the deployed model. It can be one of the following values [hf, llama, awq], `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by AWQ.​
        group_size (int): the group size used when quantizing weights to 4bit, default to 128​
        tp (int): the number of GPU cards used in tensor parallelism, default to 1​
        session_len (int): the max session length of a sequence, default to None​
        max_batch_size (int): the max batch size during inference, default to 128​
        max_context_token_num (int): the max number of tokens to be processed in each forward pass, default to 1​
        cache_max_entry_count (float): the percentage of gpu memory occupied by the k/v cache, default to 0.5​
        cache_block_seq_len (int): the length of a sequence in a k/v block, default to 128​
        cache_chunk_size (int): the number of blocks each time TurboMind engine tries to realloc from gpu memory, default to -1. When it is -1, ​
        num_tokens_per_iter (int): number of tokens to be processed per iteration, default to 0
        max_prefill_iters (int): max prefill iters for a single request, default to 1
        use_context_fmha (int): whether or not to use fmha in context decoding, default to 1​
        quant_policy: (int): , default to 0. When k/v is quantized into 8 bit, set it to 4​
        rope_scaling_factor (int): scaling factor used for dynamic ntk, default to 0. TurboMind follows the implementation of transformer LlamaAttention​
        use_logn_attn (bool): whether or not to use log attn: default to False​
    """  # noqa: E501

    model_name: Optional[str] = None
    model_format: Optional[str] = None
    tp: int = 1
    session_len: Optional[int] = None
    max_batch_size: int = 128
    group_size: int = 0
    max_context_token_num: int = 1
    cache_max_entry_count: float = 0.5
    cache_block_seq_len: int = 128
    cache_chunk_size: int = -1
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    use_context_fmha: int = 1
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
