# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any

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
    max_prefill_token_num: int = 16384


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

        def __build_internlm2():
            """build internlm2."""
            num_key_value_groups = hf_config.num_attention_heads \
                // hf_config.num_key_value_heads
            return ModelConfig(hf_config.hidden_size // num_key_value_groups,
                               hf_config.num_hidden_layers,
                               hf_config.num_attention_heads //
                               num_key_value_groups,
                               bos_token_id=hf_config.bos_token_id,
                               eos_token_id=hf_config.eos_token_id)

        def __build_default():
            return ModelConfig(hf_config.hidden_size,
                               hf_config.num_hidden_layers,
                               hf_config.num_attention_heads,
                               bos_token_id=hf_config.bos_token_id,
                               eos_token_id=hf_config.eos_token_id)

        arch = getattr(hf_config, 'architectures', ['Unknown'])[0]
        auto_map = getattr(hf_config, 'auto_map', dict())
        causallm_name = auto_map.get('AutoModelForCausalLM', 'Unknown')

        if 'falcon' in model_path:
            model_config = __build_falcon()
        elif 'chatglm' in model_path:
            model_config = __build_chatglm()
        elif (arch == 'InternLM2ForCausalLM'
              or causallm_name == 'modeling_internlm2.InternLM2ForCausalLM'):
            model_config = __build_internlm2()
        else:
            model_config = __build_default()

        model_config.dtype = _get_torch_dtype(hf_config)
        model_config.hf_config = hf_config
        model_config.json_config = hf_config.to_dict()
        return model_config
