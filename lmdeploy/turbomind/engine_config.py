# Copyright (c) OpenMMLab. All rights reserved.
from pydantic.dataclasses import dataclass


@dataclass
class EngineConfig:
    """TurboMind Engine config.

    Please read https://github.com/InternLM/lmdeploy/blob/main/docs/en/inference/turbomind_config.md for more details. # noqa: E501

    Args:
        model_name (str): Name of the given model.
        model_format (str): Should select from [None, hf, awa, llama].
        tp (int): tensor parallelism.
        session_len (int): max session length.
        max_batch_szie (int): max batch size.
        group_size (int): quantization parameter for awq.
        kv_sym (bool): quantization parameter for awq.
        kv_bits (bool): quantization parameter for awq.
        max_context_token_num (int): max context token num.
        step_length (int): generate step interval to read next request.
        cache_max_entry_count (int): number of cache blocks.
        cache_block_seq_len (int): sequence len for each cache block.
        cache_chunk_size (int): cache chunk size.
        num_tokens_per_iter (int): num tokens per iter.
        max_prefill_iters (int): max perfill iters.
        extra_tokens_per_iter (int): extra tokens per iter.
        use_context_fmha (int): whether use flashattention.
        quant_policy: (int): whether use kv int8.
        rope_scaling_factor (float): rope scaling factor.
        use_dynamic_ntk (bool): whether use dynamic ntk.
        use_logn_attn (bool): whether use logn attention scaling.
    """

    model_name: str = None
    model_format: str = None
    tp: int = 1
    session_len: int = None
    max_batch_size: int = 128
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
