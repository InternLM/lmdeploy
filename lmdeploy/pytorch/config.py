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
    return eval(f'torch.{torch_dtype}')


@dataclass
class EngineConfig:
    """PyTorch Engine Config.

    Args:
        model_name (str): name of the given model.
        tp (int): Tensor Parallelism. default 1.
        session_len (int): Max session length. Default None.
        max_batch_size: (int): Max batch size. Default 128.
        eviction_type (str): What action to perform when kv cache
            is full, ['recompute', 'copy'], Default 'recompute'.
        prefill_interval (int): Interval to perform prefill,
            Default 16.
        block_size (int): paging cache block size, default 64.
        num_cpu_blocks (int): Num cpu blocks. If num is 0, cache
            would be allocate according to current environment.
        num_gpu_blocks (int): Num gpu blocks. If num is 0, cache
            would be allocate according to current environment.
    """
    model_name: str = ''
    tp: int = 1
    session_len: int = None
    max_batch_size: int = 128
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0
    adapters: Dict[str, str] = None


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int = 512
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    max_active_adapters: int = 64
    max_prefill_seq_len: int = 64


@dataclass
class CacheConfig:
    """Config of key value cache."""

    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_heads: int
    bos_token_id: int
    eos_token_id: int
    dtype: torch.dtype = torch.float16
    multi_query_attention: bool = False
    json_config: dict = field(default_factory=dict)
    hf_config: Any = None

    def get_head_size(self):
        """get head size."""
        return self.hidden_size // self.num_heads

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
            if hf_config.new_decoder_architecture:
                # 40b-instruct, GQA
                kv_dim = hf_config.hidden_size // hf_config.num_attention_heads
                kv_dim *= hf_config.num_kv_heads
                kv_head = hf_config.num_kv_heads
            if hf_config.multi_query:
                # 7b-instruct, MQA
                kv_dim = hf_config.hidden_size // hf_config.num_attention_heads
                kv_head = 1
            else:
                # rw-1b, MHA
                kv_dim = hf_config.hidden_size
                kv_head = hf_config.num_attention_heads
            return ModelConfig(
                kv_dim,
                hf_config.num_hidden_layers,
                kv_head,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                multi_query_attention=hf_config.multi_query,
            )

        def __build_chatglm():
            """build chatglm."""
            return ModelConfig(hf_config.hidden_size //
                               hf_config.num_attention_heads *
                               hf_config.multi_query_group_num,
                               hf_config.num_layers,
                               hf_config.multi_query_group_num,
                               bos_token_id=hf_config.bos_token_id,
                               eos_token_id=hf_config.eos_token_id)

        def __build_default():
            return ModelConfig(hf_config.hidden_size,
                               hf_config.num_hidden_layers,
                               hf_config.num_attention_heads,
                               bos_token_id=hf_config.bos_token_id,
                               eos_token_id=hf_config.eos_token_id)

        if 'falcon' in model_path:
            model_config = __build_falcon()
        elif 'chatglm' in model_path:
            model_config = __build_chatglm()
        else:
            model_config = __build_default()

        model_config.dtype = _get_torch_dtype(hf_config)
        model_config.hf_config = hf_config
        model_config.json_config = hf_config.to_dict()
        return model_config
