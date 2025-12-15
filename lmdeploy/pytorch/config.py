# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.pytorch.utils import maybe_register_config_serialize_by_value


def _update_torch_dtype(config: 'ModelConfig', dtype: str):
    """Update the torch dtype from the model config.

    Args:
        config (ModelConfig): The input model config.
        dtype (str): user specified data type. Refer to
            `PyTorchEngineConfig.dtype` for detailed info
    """
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    quantization_config = getattr(config.hf_config, 'quantization_config', dict())
    quant_method = quantization_config.get('quant_method', None)
    if quant_method == 'awq':
        logger.debug('set torch_dtype to float16 for awq.')
        config.hf_config.torch_dtype = 'float16'
        config.dtype = torch.float16
        return config

    torch_dtype = getattr(config.hf_config, 'dtype', None)
    if torch_dtype is None:
        torch_dtype = getattr(config.hf_config, 'torch_dtype', None)

    # deal with case when torch_dtype is not string but torch.dtype
    if isinstance(torch_dtype, torch.dtype):
        torch_dtype = str(torch_dtype).split('.')[1]

    if torch_dtype is None:
        _dtype = 'float16' if dtype == 'auto' else dtype
        logger.warning('Model config does not have `torch_dtype`,'
                       f' use: {_dtype}')
        torch_dtype = _dtype
        # update hf_config as well
        setattr(config.hf_config, 'torch_dtype', torch_dtype)
    else:
        # change to user specified data type if it is not 'auto'
        if dtype == 'auto':
            torch_dtype = torch_dtype if torch_dtype in ['float16', 'bfloat16'] else 'float16'
        else:
            torch_dtype = dtype
    config.dtype = eval(f'torch.{torch_dtype}')
    return config


@dataclass
class BackendConfig:
    """Backend config."""
    eager_mode: bool = True
    device_type: str = 'cuda'


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

    max_batches: int
    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int
    window_size: int = -1
    cache_max_entry_count: float = 0.8
    max_prefill_token_num: int = 4096
    enable_prefix_caching: bool = False
    quant_policy: Literal[0, 4, 8] = 0
    device_type: str = 'cuda'
    num_state_caches: int = None
    states_shapes: List[Tuple] = field(default_factory=list)

    # reserved blocks for dummy inputs, init to 0 for unit test.
    num_reserved_gpu_blocks: int = 0

    # For PD Disaggregation
    role: EngineRole = EngineRole.Hybrid
    migration_backend: MigrationBackend = MigrationBackend.DLSlime

    def __post_init__(self):
        """Post init."""
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        if self.window_size > 1 and self.enable_prefix_caching:
            logger.warning('Prefix caching is not available for window attention.')
            self.enable_prefix_caching = False


class TPMode(enum.Enum):
    """TP Mode."""
    DEFAULT = enum.auto()
    DP_TP = enum.auto()


@dataclass
class DistConfig:
    dp: int = 1
    ep: int = 1
    dp_rank: int = 0
    enable_microbatch: bool = False
    enable_eplb: bool = False
    world_size: int = 1

    # tp
    tp: int = 1  # default tp, equal to attn_tp
    attn_tp: int = None  # tp for attention
    mlp_tp: int = None  # tp for mlp
    moe_tp: int = None  # tp for moe

    # tp mode
    mlp_tp_mode: TPMode = TPMode.DEFAULT
    moe_tp_mode: TPMode = TPMode.DEFAULT

    def __post_init__(self):
        """Post init."""
        assert self.dp_rank < self.dp
        assert self.dp >= 1

        dp = self.dp
        tp = self.tp
        ep = self.ep

        # ignore layer to for dp==1
        if dp == 1:
            self.mlp_tp = None
            self.attn_tp = None
            self.moe_tp = None

        # mlp and moe tp
        self.mlp_tp = self.mlp_tp or tp
        self.moe_tp = self.moe_tp or (1 if ep > 1 else self.mlp_tp)

        # world_size
        world_size = ep if ep > 1 else max(self.mlp_tp, self.moe_tp)
        self.world_size = world_size
        assert (world_size >= dp and world_size % dp == 0), (f'world_size {world_size}, dp {dp}')
        assert (world_size >= ep and world_size % ep == 0), (f'world_size {world_size}, ep {ep}')
        assert (world_size >= self.mlp_tp
                and world_size % self.mlp_tp == 0), (f'world_size {world_size}, mlp_tp {self.mlp_tp}')
        assert (world_size >= self.moe_tp
                and world_size % self.moe_tp == 0), (f'world_size {world_size}, moe_tp {self.moe_tp}')

        # attn tp
        self.attn_tp = self.attn_tp or self.world_size // dp
        self.tp = self.attn_tp
        if self.mlp_tp > 1:
            assert (self.mlp_tp >= self.attn_tp
                    and self.mlp_tp % self.attn_tp == 0), (f'mlp_tp {self.mlp_tp}, attn_tp {self.attn_tp}')
        if self.moe_tp > 1:
            assert (self.moe_tp >= self.attn_tp
                    and self.moe_tp % self.attn_tp == 0), (f'moe_tp {self.moe_tp}, attn_tp {self.attn_tp}')
        assert (world_size >= self.attn_tp
                and world_size % self.attn_tp == 0), (f'world_size {world_size}, attn_tp {self.attn_tp}')

        # tp mode
        self.mlp_tp_mode = TPMode.DEFAULT if (self.mlp_tp in [1, self.attn_tp]) else TPMode.DP_TP
        self.moe_tp_mode = TPMode.DEFAULT if (self.moe_tp in [1, self.attn_tp]) else TPMode.DP_TP

    def get_tp_by_layer(self, layer_type: str):
        """Get tp by layer type."""
        if layer_type == 'attn':
            return self.attn_tp, TPMode.DEFAULT
        elif layer_type == 'mlp':
            return self.mlp_tp, self.mlp_tp_mode
        elif layer_type == 'moe':
            return self.moe_tp, self.moe_tp_mode
        elif layer_type is None:
            # for some layer that we don't need tp
            return 1, TPMode.DEFAULT
        else:
            raise ValueError(f'Unknown layer type: {layer_type}')

    @classmethod
    def from_engine_config(cls, engine_config: PytorchEngineConfig):
        """From engine config."""
        dist_config = cls(
            dp=engine_config.dp,
            ep=engine_config.ep,
            dp_rank=engine_config.dp_rank,
            enable_microbatch=engine_config.enable_microbatch,
            enable_eplb=engine_config.enable_eplb,
            tp=engine_config.tp,
            attn_tp=engine_config.attn_tp_size,
            mlp_tp=engine_config.mlp_tp_size,
            moe_tp=engine_config.moe_tp_size,
        )
        return dist_config


def _override_hf_config_dict(hf_config: dict, key: str, hf_overrides):
    """Override hf_config dict."""
    from transformers import PretrainedConfig
    if key not in hf_config:
        # copy if key not in hf_config
        hf_config[key] = hf_overrides
        return

    hf_config_val = hf_config[key]
    is_dict = isinstance(hf_config_val, dict)
    is_cfg = isinstance(hf_config_val, PretrainedConfig)
    if not isinstance(hf_overrides, dict) or not (is_dict or is_cfg):
        # if one of them is not dict, just override
        hf_config[key] = hf_overrides
        return

    for key, value in hf_overrides.items():
        _override_hf_config(hf_config_val, key, value)


def _overide_hf_config_cfg(hf_config: list, key: str, hf_overrides):
    """Override hf_config config."""
    from transformers import PretrainedConfig
    if getattr(hf_config, key, None) is None:
        hf_config.update({key: hf_overrides})

    hf_config_val = getattr(hf_config, key)
    is_dict = isinstance(hf_config_val, dict)
    is_cfg = isinstance(hf_config_val, PretrainedConfig)
    if not isinstance(hf_overrides, dict) or not (is_dict or is_cfg):
        # if one of them is not list, just override
        hf_config.update({key: hf_overrides})
        return

    for key, value in hf_overrides.items():
        _override_hf_config(hf_config_val, key, value)


def _override_hf_config(hf_config: Any, key: str, hf_overrides):
    """Override HF config."""
    if isinstance(hf_config, dict):
        _override_hf_config_dict(hf_config, key, hf_overrides)
    else:
        _overide_hf_config_cfg(hf_config, key, hf_overrides)


def override_hf_config(hf_config: Any, hf_overrides: Dict[str, Any]):
    """Override HF config."""
    for k, v in hf_overrides.items():
        _override_hf_config(hf_config, k, v)


def _default_check_env(device: str):
    pass


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
    vocab_size: int = 40000
    hf_config: Any = None
    llm_config: Any = None
    cogvlm_style: bool = False
    custom_module_map: Dict[str, setattr] = None

    # flash mla
    use_flash_mla: bool = False
    use_mla_fp8_cache: bool = False
    mla_index_topk: Optional[int] = None

    # dllm
    model_paradigm: str = 'ar'
    dllm_mask_token: int = 0
    dllm_block_length: int = None

    # Added for deepseekv3.2 nsa index
    # caches would be added after kv cache
    cache_shapes: List[Tuple[List[int], torch.dtype]] = field(default_factory=list)
    # added for qwen3_next
    # could used for any SSM model.
    states_shapes: List[Tuple[Tuple[int], torch.dtype]] = field(default_factory=list)

    # check env for model-device combination
    check_env_func: Callable = _default_check_env

    def get_head_size(self):
        """Get head size."""
        return self.head_dim

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        dist_config: DistConfig = None,
        hf_overrides: Dict[str, Any] = None,
        is_draft_model: bool = False,
        spec_method: str = None,
    ):
        """Instantiate one of the configuration classes of the library from a
        pretrained model configuration.

        Args:
            pretrained_model_name_or_path (str): the pretrained model path
            trust_remote_code (bool):  Whether or not to allow for custom
                models defined on the Hub in their own modeling files.
            dtype (str): user specified data type for model weights and
                activations. Refer to `PyTorchEngineConfig` for details
            hf_overrides (Dict[str, Any]): overrides for the HF config.
        """
        from transformers import AutoConfig

        from lmdeploy.pytorch.transformers import config_from_pretrained
        from lmdeploy.utils import get_logger
        hf_config = config_from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        if getattr(hf_config, 'model_type', None) in ['phi3']:
            # phi3 + trust_remote_code leads to error when tp.
            hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        model_config = cls.from_hf_config(
            hf_config,
            pretrained_model_name_or_path,
            dtype=dtype,
            dist_config=dist_config,
            is_draft_model=is_draft_model,
            spec_method=spec_method,
        )

        if hf_overrides is not None:
            logger = get_logger('lmdeploy')
            logger.warning(f'Overriding HF config with {hf_overrides}')
            override_hf_config(model_config.hf_config, hf_overrides)

        # for serialization of transformers modules
        maybe_register_config_serialize_by_value(trust_remote_code)

        return model_config

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        model_path: str = None,
        dtype: str = 'auto',
        dist_config: DistConfig = None,
        is_draft_model: bool = False,
        spec_method: str = None,
    ):
        """From huggingface config."""
        from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
        if dist_config is None:
            dist_config = DistConfig()
        tp = dist_config.attn_tp

        model_config = AutoModelConfigBuilder.build(hf_config,
                                                    model_path,
                                                    tp=tp,
                                                    is_draft_model=is_draft_model,
                                                    spec_method=spec_method)

        if model_config.k_head_dim is None:
            assert model_config.head_dim is not None
            model_config.k_head_dim = model_config.head_dim
        if model_config.v_head_dim is None:
            assert model_config.head_dim is not None
            model_config.v_head_dim = model_config.head_dim

        # check for tp
        assert model_config.num_attention_heads % tp == 0
        if model_config.num_key_value_heads >= tp:
            assert model_config.num_key_value_heads % tp == 0
        else:
            assert tp % model_config.num_key_value_heads == 0

        # should after setting `hf_config` and `model_arch` attributes
        model_config = _update_torch_dtype(model_config, dtype)

        # update eos_token_id to list
        if isinstance(model_config.eos_token_id, int):
            model_config.eos_token_id = [model_config.eos_token_id]

        return model_config


class UnmaskingStrategy(enum.Enum):
    """Unmasking Strategy."""

    # unmasking from left to right
    SEQUENTIAL = enum.auto()
    # unmasking with confidence threshold
    LOW_CONFIDENCE_DYNAMIC = enum.auto()
    # unmasking with topk in a block
    LOW_CONFIDENCE_STATIC = enum.auto()

    @classmethod
    def from_str(cls, strategy: str):
        """From string."""
        strategy = strategy.lower()
        if strategy == 'sequential':
            return cls.SEQUENTIAL
        elif strategy == 'low_confidence_dynamic':
            return cls.LOW_CONFIDENCE_DYNAMIC
        elif strategy == 'low_confidence_static':
            return cls.LOW_CONFIDENCE_STATIC
        else:
            raise ValueError(f'Unknown unmasking strategy: {strategy}')


@dataclass
class DLLMConfig:
    block_length: int = 1
    unmasking_strategy: UnmaskingStrategy = UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC
    denoising_steps: int = None
    confidence_threshold: float = 0.85


@dataclass
class MiscConfig:
    prefill_interval: int = 16
    custom_module_map: str = None
    empty_init: bool = False
    model_format: str = None
    hf_overrides: Dict[str, Any] = None
    disable_vision_encoder: bool = False
    logprobs_mode: str = None
    dllm_config: DLLMConfig = None
    enable_return_routed_experts: bool = False

    @classmethod
    def from_engine_config(cls, engine_config: PytorchEngineConfig):
        """From engine config."""
        dllm_unmasking_strategy = UnmaskingStrategy.from_str(engine_config.dllm_unmasking_strategy)
        dllm_config = DLLMConfig(block_length=engine_config.dllm_block_length,
                                 unmasking_strategy=dllm_unmasking_strategy,
                                 denoising_steps=engine_config.dllm_denoising_steps,
                                 confidence_threshold=engine_config.dllm_confidence_threshold)
        misc_config = cls(
            custom_module_map=engine_config.custom_module_map,
            empty_init=engine_config.empty_init,
            prefill_interval=engine_config.prefill_interval,
            model_format=engine_config.model_format,
            hf_overrides=engine_config.hf_overrides,
            disable_vision_encoder=engine_config.disable_vision_encoder,
            logprobs_mode=engine_config.logprobs_mode,
            dllm_config=dllm_config,
            enable_return_routed_experts=engine_config.enable_return_routed_experts,
        )
        return misc_config


@dataclass
class SpecDecodeConfig:
    model: str
    method: str
    cache_config: CacheConfig = None
    num_speculative_tokens: int = 1
    model_config: ModelConfig = None

    @classmethod
    def from_config(
        cls,
        method: str,
        num_speculative_tokens: int,
        model: str,
        target_cache_cfg: CacheConfig,
        target_model: str = None,
        dtype: str = 'auto',
    ):
        model = model or target_model
        model_config = ModelConfig.from_pretrained(model,
                                                   trust_remote_code=True,
                                                   dtype=dtype,
                                                   is_draft_model=True,
                                                   spec_method=method)
        cache_config = None
        # include medusa
        no_caches = ['medusa']
        if method not in no_caches:
            cache_config = CacheConfig(max_batches=target_cache_cfg.max_batches,
                                       block_size=target_cache_cfg.block_size,
                                       num_cpu_blocks=target_cache_cfg.num_cpu_blocks,
                                       num_gpu_blocks=target_cache_cfg.num_gpu_blocks,
                                       cache_max_entry_count=target_cache_cfg.cache_max_entry_count,
                                       max_prefill_token_num=target_cache_cfg.max_prefill_token_num,
                                       device_type=target_cache_cfg.device_type,
                                       migration_backend=target_cache_cfg.migration_backend)
        obj = cls(
            model=model,
            method=method,
            cache_config=cache_config,
            model_config=model_config,
            num_speculative_tokens=num_speculative_tokens,
        )
        return obj
