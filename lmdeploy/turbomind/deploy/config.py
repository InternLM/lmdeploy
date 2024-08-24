# Copyright (c) OpenMMLab. All rights reserved.
# from dataclasses import dataclass
import inspect
import json
from typing import TypeVar

# use pydantic.dataclasses.dataclass to check data type
from pydantic.dataclasses import dataclass


def init_config_from_dict(cls, env, allow_none=False):
    params = inspect.signature(cls).parameters
    used = {k: v for k, v in env.items() if k in params and v is not None}
    if not allow_none:
        return cls(**used)
    else:
        default = {
            k: None
            for k in params.keys() if params[k].default is inspect._empty
        }
        default.update(used)
        return cls(**default)


@dataclass
class ModelConfig:
    model_name: str = ''
    chat_template: str = ''
    model_arch: str = None
    head_num: int = None
    kv_head_num: int = None
    hidden_units: int = None
    vocab_size: int = None
    num_layer: int = None
    inter_size: int = None
    norm_eps: float = None
    attn_bias: int = None
    start_id: int = None
    end_id: int = None
    size_per_head: int = 128
    group_size: int = 0
    weight_type: str = None
    quant_policy: int = 0


@dataclass
class AttentionConfig:
    rotary_embedding: int = 128
    rope_theta: float = 10000.0
    max_position_embeddings: int = 0
    original_max_position_embeddings: int = 0
    rope_scaling_type: str = ''
    rope_scaling_factor: float = 0.0
    use_dynamic_ntk: int = 0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 1.0
    use_logn_attn: int = 0
    cache_block_seq_len: int = 64


@dataclass
class LoraConfig:
    lora_policy: str = ''
    lora_r: int = 0
    lora_scale: float = 0.0
    lora_max_wo_r: int = 0
    lora_rank_pattern: str = ''
    lora_scale_pattern: str = ''


@dataclass
class InternalEngineConfig:
    tensor_para_size: int = None
    session_len: int = None
    max_batch_size: int = 64
    max_prefill_token_num: int = 8192
    max_context_token_num: int = 1
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    enable_prefix_caching: bool = False
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1


@dataclass
class TurbomindModelConfig:
    """Config for turbomind model."""
    model_config: ModelConfig = None
    attention_config: AttentionConfig = None
    engine_config: InternalEngineConfig = None
    lora_config: LoraConfig = None

    @property
    def session_len(self):
        return self.engine_config.session_len

    def update_from_engine_config(self,
                                  config: TypeVar('TurbomindModelConfig')):
        """Update the attributes of this instance with the attributes from
        TurbomindEngineConfig.

        Args:
            config (TurbomindEngineConfig): The turbomind engine config
        """
        if config is None:
            return
        # Iterate over the fields of 'self.engine_config'
        for field_name, _ in self.engine_config.__dataclass_fields__.items():
            # If the field value in 'other' is not None,
            # update the corresponding field in 'self.engine_config'
            if hasattr(config, field_name) and getattr(config,
                                                       field_name) is not None:
                setattr(self.engine_config, field_name,
                        getattr(config, field_name))

        self.engine_config.tensor_para_size = config.tp
        assert self.session_len is not None
        if config.max_prefill_token_num is not None and \
                config.num_tokens_per_iter == 0:
            self.engine_config.num_tokens_per_iter = \
                config.max_prefill_token_num
            self.engine_config.max_prefill_iters = (
                self.session_len + config.max_prefill_token_num -
                1) // config.max_prefill_token_num

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)

    # @property
    # def valid(self):
    #     """Check if cfg is valid."""
    #     for config in [
    #             self.model_config, self.attn_config, self.lora_config,
    #             self.engine_config
    #     ]:
    #         for _, v in config.__dict__.items():
    #             if v is None:
    #                 return False
    #     return True
