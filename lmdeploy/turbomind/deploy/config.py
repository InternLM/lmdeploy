# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import json
from dataclasses import asdict, fields

# use pydantic.dataclasses.dataclass to check data type
from pydantic.dataclasses import dataclass

from lmdeploy.messages import TurbomindEngineConfig


def config_from_dict(cls, env):
    """initiate an instance of a config class from a dict."""
    params = inspect.signature(cls).parameters
    used = {k: v for k, v in env.items() if k in params and v is not None}
    return cls(**used)


def config_to_dict(config):
    """export config to a dict."""
    if not config:
        return dict()
    assert isinstance(config, (ModelConfig, AttentionConfig, LoraConfig)), \
        f'A dataclass is expected, but got {type(config)}'

    return asdict(config)


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
    session_len: int = None
    tp: int = 1
    model_format: str = 'hf'


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
class TurbomindModelConfig:
    """Config for turbomind model."""
    model_config: ModelConfig = None
    attention_config: AttentionConfig = None
    lora_config: LoraConfig = None

    def update_from_engine_config(self, config: TurbomindEngineConfig):
        """Update the attributes of this instance with the attributes from
        TurbomindEngineConfig.

        Args:
            config (TurbomindEngineConfig): The turbomind engine config
        """
        if config is None:
            return
        for key, value in asdict(config).items():
            if not value:
                continue

            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            if hasattr(self.attention_config, key):
                setattr(self.attention_config, key, value)

    @classmethod
    def from_dict(cls, config: dict = {}):
        """construct TurbomindModelConfig instance from config in a dict."""
        _cfg = {
            field.name: config.get(field.name, {})
            for field in fields(TurbomindModelConfig)
        }

        return TurbomindModelConfig(
            model_config=config_from_dict(ModelConfig, _cfg['model_config']),
            attention_config=config_from_dict(AttentionConfig,
                                              _cfg['attention_config']),
            lora_config=config_from_dict(LoraConfig, _cfg['lora_config']))

    def to_dict(self):
        """export to a dict."""
        return dict(model_config=config_to_dict(self.model_config),
                    attention_config=config_to_dict(self.attention_config),
                    lora_config=config_to_dict(self.lora_config))

    @property
    def session_len(self):
        return self.model_config.session_len

    @property
    def tensor_para_size(self):
        return self.model_config.tp

    @property
    def weight_type(self):
        return self.model_config.weight_type

    @property
    def group_size(self):
        return self.model_config.group_size

    @property
    def vocab_size(self):
        return self.model_config.vocab_size

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)
