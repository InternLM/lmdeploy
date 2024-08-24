# Copyright (c) OpenMMLab. All rights reserved.
# from dataclasses import dataclass
import inspect
import json
from dataclasses import asdict
from typing import Optional

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
    model_format: Optional[str] = None
    tensor_para_size: int = 1
    session_len: int = 0
    max_batch_size: int = 64
    max_prefill_token_num: int = 8192
    max_context_token_num: int = 1
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    enable_prefix_caching: bool = False
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0


@dataclass
class TurbomindModelConfig:
    """Config for turbomind model."""
    model_config: ModelConfig = None
    attention_config: AttentionConfig = None
    engine_config: InternalEngineConfig = None
    lora_config: LoraConfig = None

    def to_dict(self):
        # TODO(lvhan) make the sequence of dict is the same as the config attrs
        return dict(model_config=asdict(self.model_config),
                    attention_config=asdict(self.attention_config),
                    lora_config=asdict(self.lora_config),
                    engine_config=asdict(self.engine_config))

    @property
    def session_len(self):
        return self.engine_config.session_len

    @property
    def tensor_para_size(self):
        return self.engine_config.tensor_para_size

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
