# Copyright (c) OpenMMLab. All rights reserved.
from pydantic.dataclasses import dataclass


@dataclass
class EngineConfig:
    model_name: str = None
    model_format: str = None
    tp: int = 1
    session_len: int = 2048
    max_batch_size: int = 64
    group_size: int = 128
    kv_sym: bool = False
    kv_bits: int = 8
    max_context_token_num: int = 1
    step_length: int = 1
    cache_max_entry_count: float = 0.5
    cache_block_seq_len: int = 128
    cache_chunk_size: int = -1
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    extra_tokens_per_iter: int = 0
    use_context_fmha: int = 1
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_dynamic_ntk: bool = False
    use_logn_attn: bool = False
