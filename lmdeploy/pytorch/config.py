# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict

import torch


def _get_torch_dtype(config: Any, default: str = 'float16'):
    """Get the torch dtype from the model config.

    Args:
        config: Config of the hf model.
        default (str): default device type.
    """
    torch_dtype = getattr(config, 'torch_dtype', default)
    # torch_dtype in config could be none
    torch_dtype = torch_dtype or default
    return eval(f'torch.{torch_dtype}')


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int = 512
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    max_active_adapters: int = 64
    max_prefill_token_num: int = 8192


@dataclass
class CacheConfig:
    """Config of key value cache."""

    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int
    window_size: int = -1
    cache_max_entry_count: float = 0.8
    max_prefill_token_num: int = 8192


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    bos_token_id: int
    eos_token_id: int
    head_dim: int
    sliding_window: int = -1
    dtype: torch.dtype = torch.float16
    multi_query_attention: bool = False
    vocab_size: int = 40000
    json_config: dict = field(default_factory=dict)
    hf_config: Any = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_head_size(self):
        """get head size."""
        return self.head_dim

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        trust_remote_code: bool = True):
        """build ModelConfig from model path or name."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return cls.from_hf_config(hf_config, pretrained_model_name_or_path)

    @classmethod
    def from_hf_config(cls, hf_config: Any, model_path: str = None):
        """from huggingface config."""
        if model_path is None:
            model_path = ''

        def __build_falcon():
            """build falcon."""
            num_attention_heads = hf_config.num_attention_heads
            if hf_config.new_decoder_architecture:
                # 40b-instruct, GQA
                kv_head = hf_config.num_kv_heads
            if hf_config.multi_query:
                # 7b-instruct, MQA
                kv_head = 1
            else:
                # rw-1b, MHA
                kv_head = num_attention_heads
            head_dim = hf_config.hidden_size // num_attention_heads
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=kv_head,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=head_dim,
                multi_query_attention=hf_config.multi_query,
                vocab_size=hf_config.vocab_size,
            )

        def __build_chatglm():
            """build chatglm."""
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            bos_token_id = hf_config.bos_token_id
            if bos_token_id is None:
                bos_token_id = hf_config.pad_token_id
            init_kwargs = dict(empty_init=False)
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.multi_query_group_num,
                bos_token_id=bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=head_dim,
                vocab_size=hf_config.padded_vocab_size,
                init_kwargs=init_kwargs)

        def __build_gemma():
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=hf_config.head_dim,
                vocab_size=hf_config.vocab_size)

        def __build_default():
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            num_attention_heads = hf_config.num_attention_heads
            num_key_value_heads = getattr(hf_config, 'num_key_value_heads',
                                          num_attention_heads)
            use_sliding_window = getattr(hf_config, 'use_sliding_window', True)
            sliding_window = -1
            if use_sliding_window:
                sliding_window = getattr(hf_config, 'sliding_window',
                                         sliding_window) or -1
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                sliding_window=sliding_window,
                head_dim=head_dim,
                vocab_size=hf_config.vocab_size)

        if 'falcon' in model_path:
            model_config = __build_falcon()
        elif 'chatglm' in model_path:
            model_config = __build_chatglm()
        elif hf_config.model_type == 'gemma':
            model_config = __build_gemma()
        else:
            model_config = __build_default()

        model_config.dtype = _get_torch_dtype(hf_config)
        model_config.hf_config = hf_config
        model_config.json_config = hf_config.to_dict()
        return model_config
