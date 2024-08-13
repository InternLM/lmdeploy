# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM


def _update_torch_dtype(config: 'ModelConfig', default: str = 'float16'):
    """Update the torch dtype from the model config.

    Args:
        config (ModelConfig): The input model config.
        default (str): default device type.
    """
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    quantization_config = getattr(config.hf_config, 'quantization_config',
                                  dict())
    quant_method = quantization_config.get('quant_method', None)
    if quant_method == 'awq':
        logger.debug('set torch_dtype to float16 for awq.')
        config.hf_config.torch_dtype = 'float16'
        config.dtype = torch.float16
        return config

    torch_dtype = getattr(config.hf_config, 'torch_dtype', None)
    if torch_dtype is None:
        logger.warning('Model config does not have `torch_dtype`,'
                       f' use default: {default}')
        torch_dtype = default
        # update hf_config as well
        setattr(config.hf_config, 'torch_dtype', torch_dtype)

    config.dtype = eval(f'torch.{torch_dtype}')
    return config


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int = 512
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    max_active_adapters: int = 64


@dataclass
class CacheConfig:
    """Config of key value cache."""

    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int
    window_size: int = -1
    cache_max_entry_count: float = 0.8
    max_prefill_token_num: int = 4096
    enable_prefix_caching: bool = False

    def __post_init__(self):
        """post init."""
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        if self.window_size > 1 and self.enable_prefix_caching:
            logger.warning(
                'Prefix caching is not available for window attention.')
            self.enable_prefix_caching = False


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    bos_token_id: int
    eos_token_id: List[int]
    head_dim: int
    k_head_dim: int = None
    v_head_dim: int = None
    sliding_window: int = -1
    dtype: torch.dtype = torch.float16
    multi_query_attention: bool = False
    vocab_size: int = 40000
    hf_config: Any = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_arch: str = None
    unused_modules: List[str] = None
    auto_model_cls: Any = AutoModelForCausalLM
    cogvlm_style: bool = False

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
        from lmdeploy.pytorch.configurations import AutoModelConfigBuilder

        model_config = AutoModelConfigBuilder.build(hf_config, model_path)

        if model_config.k_head_dim is None:
            assert model_config.head_dim is not None
            model_config.k_head_dim = model_config.head_dim
        if model_config.v_head_dim is None:
            assert model_config.head_dim is not None
            model_config.v_head_dim = model_config.head_dim

        model_arch = model_config.hf_config.architectures[0]
        model_config.model_arch = model_arch
        # should after setting `hf_config` and `model_arch` attributes
        model_config = _update_torch_dtype(model_config)

        # update eos_token_id to list
        if isinstance(model_config.eos_token_id, int):
            model_config.eos_token_id = [model_config.eos_token_id]

        return model_config
