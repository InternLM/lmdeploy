# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import json
from dataclasses import asdict, field, fields
from typing import List

# use pydantic.dataclasses.dataclass to check data type
from pydantic.dataclasses import dataclass

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def config_from_dict(cls, env):
    """Initiate an instance of a config class from a dict."""
    params = inspect.signature(cls).parameters
    used = {k: v for k, v in env.items() if k in params and v is not None}

    def _remove_none(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = _remove_none(v)
        return {k: v for k, v in d.items() if v is not None}

    used = _remove_none(used)
    return cls(**used)


def config_to_dict(config):
    """Export config to a dict."""
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
    # Turbomind used to assume token_embedding and lm_head has the same size
    # at vocab dim, i.e. `vocab_size`
    # But in molmo, embedding.shape is [vocab_size + 128, hidden_units]
    # while lm_head shape is [hidden_units, vocab_size].
    # Therefore, we add a new attr "embedding_size" to represent the vocab dim
    # of token_embedding
    embedding_size: int = 0
    num_layer: int = None
    inter_size: List[int] = None
    norm_eps: float = None
    attn_bias: int = 0
    mlp_bias: bool = False
    window_size: List[int] = field(default_factory=list)
    attn_sink: bool = False
    qk_norm: bool = False
    size_per_head: int = 128
    group_size: int = 64
    data_type: str = None
    weight_type: str = None
    expert_weight_type: str = None
    session_len: int = None
    attn_tp_size: int = 1
    mlp_tp_size: int = 1
    model_format: str = 'hf'
    expert_num: List[int] = ()
    expert_router_bias: bool = False
    expert_inter_size: int = 0
    experts_per_token: int = 0
    activation_type: str = ''
    moe_shared_gate: bool = False
    norm_topk_prob: bool = False
    routed_scale: float = 1.0
    topk_group: int = 1
    topk_method: str = 'greedy'
    moe_group_num: int = 1
    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    v_head_dim: int = 0
    # tuning
    tune_layer_num: int = 1

    def verify(self):
        invalid = {}
        for k, v in self.__dict__.items():
            if v is None:
                invalid[k] = v
        assert not invalid, f'incomplete model config: {invalid}'


@dataclass
class RopeParam:
    type: str
    base: float
    dim: int
    factor: float = 1.0
    max_position_embeddings: int = None
    attention_factor: float = 1.0
    beta_fast: float = 32
    beta_slow: float = 1
    low_freq_factor: float = None
    high_freq_factor: float = None
    original_max_position_embeddings: int = None
    mrope_section: List[int] = None


@dataclass
class AttentionConfig:
    softmax_scale: float = 0
    cache_block_seq_len: int = 64
    use_logn_attn: int = 0
    max_position_embeddings: int = 0
    rope_param: RopeParam = None


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

        # update from hf_overrides
        if hasattr(config, 'hf_overrides') and config.hf_overrides:
            hf_overrides = config.hf_overrides

            if hf_overrides.get('rope_scaling'):
                override_params = hf_overrides.get('rope_scaling')

                rope_param = self.attention_config.rope_param or RopeParam(type='', base=0, dim=0)
                rope_param.type = override_params.get('rope_type', '')
                if rope_param.type == 'yarn' and 'original_max_position_embeddings' in override_params:
                    rope_param.factor = self.attention_config.max_position_embeddings / override_params[
                        'original_max_position_embeddings']
                    rope_param.max_position_embeddings = override_params['original_max_position_embeddings']
                else:
                    rope_param.factor = override_params.get('factor', 1.0)
                    rope_param.max_position_embeddings = override_params.get('original_max_position_embeddings', None)

                self.attention_config.rope_param = rope_param
            logger.warning(f'Overriding HF config with {hf_overrides}')

        # use dynamic ntk
        if config.rope_scaling_factor:
            # some ut will create empty RopeParam, will check base/dim in src code
            rope_param = self.attention_config.rope_param or RopeParam(type='', base=0, dim=0)
            rope_param.type = 'dynamic'
            rope_param.factor = config.rope_scaling_factor
            rope_param.max_position_embeddings = self.attention_config.max_position_embeddings

            self.attention_config.rope_param = rope_param
            logger.warning(
                '`--rope-scaling-factor` will be removed in a future release. Please instead use `--hf-overrides`.')

    @classmethod
    def from_dict(cls, config: dict = {}):
        """Construct TurbomindModelConfig instance from config in a dict."""
        _cfg = {field.name: config.get(field.name, {}) for field in fields(TurbomindModelConfig)}

        return TurbomindModelConfig(model_config=config_from_dict(ModelConfig, _cfg['model_config']),
                                    attention_config=config_from_dict(AttentionConfig, _cfg['attention_config']),
                                    lora_config=config_from_dict(LoraConfig, _cfg['lora_config']))

    def to_dict(self):
        """Export to a dict."""
        return dict(model_config=config_to_dict(self.model_config),
                    attention_config=config_to_dict(self.attention_config),
                    lora_config=config_to_dict(self.lora_config))

    @property
    def session_len(self):
        return self.model_config.session_len

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
