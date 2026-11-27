# Copyright (c) OpenMMLab. All rights reserved.
import enum
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Literal

import torch

from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig, QuantPolicy
from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.pytorch.utils import maybe_register_config_serialize_by_value
from lmdeploy.utils import get_logger, is_bf16_supported

logger = get_logger('lmdeploy')


def validate_replayssm_config(buffer_len: int, num_spec_tokens: int) -> None:
    """Validate history capacity against the configured verification window."""
    if buffer_len not in (8, 16, 32):
        raise ValueError('replayssm_buffer_len must be one of 8, 16, or 32.')
    if num_spec_tokens < 0:
        raise ValueError('num_spec_tokens must be nonnegative.')
    tmax = num_spec_tokens + 1
    if num_spec_tokens > 0 and buffer_len < 2 * tmax:
        raise ValueError('MTP ReplaySSM requires replayssm_buffer_len >= '
                         '2 * (num_speculative_tokens + 1), got '
                         f'L={buffer_len}, Tmax={tmax}.')


def _resolve_ct_fp8_layout(weights: Mapping[str, Any]) -> tuple[int, int | None, bool, str]:
    """Resolve a compressed-tensors float-quantized FP8 weight layout.

    Returns ``(bits, group_size_or_None, blocked, ct_format)``. Two layouts
    are served: per-channel (``strategy == 'channel'`` with no group) and
    the canonical llm-compressor blocked output (``strategy == 'block'``
    with ``block_structure == [128, 128]`` and a null ``group_size``; the
    ground truth is e.g. RedHatAI/Qwen3-0.6B-FP8-BLOCK, whose
    ``.weight_scale`` is bf16 ``[ceil(N/128), ceil(K/128)]`` N-first and
    ``.weight`` is e4m3 ``[N, K]``). The legacy ``group``/``tensor_group``
    128 spellings keep working. Shared by the PT engine parser and the
    TurboMind converter so the two cannot drift.
    """
    ct_format = 'float-quantized'
    if weights.get('num_bits') != 8:
        raise ValueError(f'Unsupported float-quantized `num_bits`: expected 8, got {weights.get("num_bits")!r}.')
    strategy = weights.get('strategy')
    group_size = weights.get('group_size')
    if strategy == 'channel':
        if group_size is not None:
            raise ValueError(f'Unsupported float-quantized layout: strategy `channel` requires no '
                             f'group_size, got {group_size!r}.')
        return 8, None, False, ct_format
    if strategy == 'block':
        block_structure = weights.get('block_structure')
        if block_structure != [128, 128]:
            raise ValueError(f'Unsupported compressed-tensors FP8 block structure: {block_structure!r}. '
                             'Only [128, 128] is supported.')
        return 8, None, True, ct_format
    if strategy in ('tensor_group', 'group') and group_size == 128:
        return 8, 128, True, ct_format
    raise ValueError(f'Unsupported float-quantized weights layout: strategy={strategy!r}, '
                     f'group_size={group_size!r}. Supported: channel (no group), block [128, 128], '
                     'or group/tensor_group 128.')


def _parse_compressed_tensors_config(quant_config: Mapping[str, Any]):
    """Parse the compressed-tensors metadata needed by the inference backend.

    Returns ``(bits, group_size, ignored_layers, ct_format)``. The
    ``pack-quantized`` branch keeps its historical W4A16 validation; the
    ``nvfp4-pack-quantized`` and ``float-quantized`` branches validate the
    weight-only float layouts served through the TurboMind channel.
    """
    ct_format = quant_config.get('format')
    if ct_format not in ('pack-quantized', 'nvfp4-pack-quantized', 'float-quantized'):
        raise ValueError(f'Unsupported compressed-tensors `format`: {ct_format!r}. '
                         'Supported formats: pack-quantized, nvfp4-pack-quantized, float-quantized.')
    if quant_config.get('quantization_status') != 'compressed':
        raise ValueError('Compressed-tensors checkpoint must have `compressed` status.')

    config_groups = quant_config.get('config_groups')
    if not isinstance(config_groups, Mapping) or len(config_groups) != 1:
        raise ValueError('Compressed-tensors config must contain exactly one `config_groups` entry.')
    # Group keys differ across tools: 'group_0' (compressed-tensors
    # reference) vs 'config_group_0' (e.g. Ornith). Take the only entry,
    # mirroring the TurboMind converter's fallback.
    group = config_groups.get('group_0') or config_groups[sorted(config_groups)[0]]
    if not isinstance(group, Mapping) or group.get('targets') != ['Linear']:
        raise ValueError('Compressed-tensors `group_0` must target `Linear`.')

    ignored_layers = quant_config.get('ignore', [])
    if not isinstance(ignored_layers, list) or not all(isinstance(rule, str) for rule in ignored_layers):
        raise ValueError('Compressed-tensors `ignore` must be a list of strings.')

    weights = group.get('weights')
    if not isinstance(weights, Mapping):
        raise ValueError('Compressed-tensors `group_0.weights` must be an object.')

    if ct_format == 'pack-quantized':
        _validate_pack_quantized_group(group, weights)
        return weights['num_bits'], weights['group_size'], ignored_layers, ct_format

    # Both float lanes are weight-only: activation quantization metadata may
    # describe dynamic input quantization (applied inside the kernel), but a
    # static input scheme or any output scheme has no serving path.
    input_activations = group.get('input_activations')
    if input_activations is not None:
        if input_activations.get('dynamic') is not True:
            raise ValueError('Compressed-tensors float formats support only dynamic (per-token) input '
                             'activation quantization; static per-tensor W8A8 input scales are '
                             'unsupported and rejected here.')
    if group.get('output_activations') is not None:
        raise ValueError('Compressed-tensors output activation quantization is not supported.')

    expected_common = {
        'dynamic': False,
        'symmetric': True,
        'type': 'float',
    }
    for key, expected in expected_common.items():
        if weights.get(key) != expected:
            raise ValueError(
                f'Unsupported compressed-tensors `{key}`: expected {expected!r}, got {weights.get(key)!r}')

    if ct_format == 'nvfp4-pack-quantized':
        expected_nvfp4 = {
            'num_bits': 4,
            'group_size': 16,
            'strategy': 'tensor_group',
        }
        for key, expected in expected_nvfp4.items():
            if weights.get(key) != expected:
                raise ValueError(f'Unsupported nvfp4-pack-quantized `{key}`: expected {expected!r}, '
                                 f'got {weights.get(key)!r}.')
        return 4, 16, ignored_layers, ct_format

    # float-quantized FP8 weights: per-channel, 128x128 blocked, or the
    # legacy group-128 spellings -- resolved by the shared layout helper.
    bits, group_size, _blocked, _ct = _resolve_ct_fp8_layout(weights)
    return bits, group_size, ignored_layers, ct_format


def _validate_pack_quantized_group(group: Mapping[str, Any], weights: Mapping[str, Any]):
    """Validate the pre-packed W4A16 group (historical behavior)."""
    if group.get('input_activations') is not None or group.get('output_activations') is not None:
        raise ValueError('Compressed-tensors activation quantization is not supported.')
    expected_weights = {
        'num_bits': 4,
        # Group 32 is the pre-existing packed ct layout (MoE experts);
        # group 128 dense linears normalize to the canonical AWQ layout at
        # load time.
        'group_size': 32,
        'strategy': 'group',
        'symmetric': True,
        'dynamic': False,
        'type': 'int',
    }
    for key, expected in expected_weights.items():
        if key == 'group_size':
            if weights.get(key) not in (32, 128):
                raise ValueError(f'Unsupported compressed-tensors `{key}`: expected 32 or 128, '
                                 f'got {weights.get(key)!r}.')
            continue
        if weights.get(key) != expected:
            raise ValueError(
                f'Unsupported compressed-tensors `{key}`: expected {expected!r}, got {weights.get(key)!r}.')


def _matches_compressed_tensors_ignore(rule: str, prefix: str) -> bool:
    if rule.startswith('re:'):
        return re.match(rule[3:], prefix) is not None
    return rule == prefix


def normalize_cudagraph_capture_batch_sizes(capture_sizes: list[int] | None, max_batches: int) -> list[int] | None:
    """Normalize configured cudagraph capture batch sizes."""
    if capture_sizes is None:
        return None

    assert len(capture_sizes) > 0, 'cudagraph_capture_batch_sizes should not be empty'
    assert all(isinstance(size, int) and size > 0 for size in capture_sizes), (
        'cudagraph_capture_batch_sizes should be positive integers')

    capture_sizes = sorted({size for size in capture_sizes if size <= max_batches})
    assert len(capture_sizes) > 0, (
        'cudagraph_capture_batch_sizes should contain at least one value '
        f'<= max_batch_size ({max_batches})')
    if capture_sizes[-1] != max_batches:
        capture_sizes.append(max_batches)
    return capture_sizes


def _update_torch_dtype(config: 'ModelConfig', dtype: str, device_type: str = 'auto'):
    """Update the torch dtype from the model config.

    Args:
        config (ModelConfig): The input model config.
        dtype (str): user specified data type. Refer to
            `PyTorchEngineConfig.dtype` for detailed info
        device_type (str): The device type. Refer to `PyTorchEngineConfig.device_type` for detailed info
    """
    quantization_config = getattr(config.hf_config, 'quantization_config', dict())
    quant_method = quantization_config.get('quant_method', None)
    if quant_method in ('awq', 'gptq'):
        # The W4A16 linear modules are fp16. A bf16 model dtype would make
        # the residual stream dtype-mismatch the quantized linears.
        logger.debug(f'set torch_dtype to float16 for {quant_method}.')
        config.hf_config.torch_dtype = 'float16'
        config.dtype = torch.float16
        return config
    if quant_method == 'compressed-tensors':
        # Guard the shape: this runs before _parse_compressed_tensors_config
        # gets its clean ValueError, so a truthy non-Mapping must not turn
        # into an AttributeError here.
        config_groups = quantization_config.get('config_groups')
        ct_weights = config_groups.get('group_0', {}).get('weights') if isinstance(config_groups, Mapping) else None
        if isinstance(ct_weights, Mapping) and ct_weights.get('group_size') == 128:
            # Group-128 dense linears normalize into the fp16 AWQ family at
            # load time. Group-32 checkpoints keep their dtype: the packed
            # ct MoE path supports bf16 activations (DeepEP requires them).
            logger.debug('set torch_dtype to float16 for compressed-tensors group-128.')
            config.hf_config.torch_dtype = 'float16'
            config.dtype = torch.float16
            return config

    torch_dtype = getattr(config.hf_config, 'dtype', None)
    if torch_dtype is None and hasattr(config.hf_config, 'text_config'):
        torch_dtype = getattr(config.hf_config.text_config, 'dtype', None)

    if torch_dtype is None:
        torch_dtype = getattr(config.hf_config, 'torch_dtype', None)

    # deal with case when torch_dtype is not string but torch.dtype
    if isinstance(torch_dtype, torch.dtype):
        torch_dtype = str(torch_dtype).split('.')[1]

    if torch_dtype is None:
        # Mirror the TurboMind converter's auto resolution (converter.py):
        # no publisher dtype at all means capability-based fallback —
        # bfloat16 when the device supports it, else float16, so both
        # engines resolve a dtype-less checkpoint identically. A present
        # non-16-bit publisher dtype still resolves differently (TM
        # capability-falls, PT honors it) — that path is pre-existing and
        # out of scope here.
        _dtype = ('bfloat16' if dtype == 'auto' and is_bf16_supported(device_type) else
                  'float16' if dtype == 'auto' else dtype)
        logger.warning('Model config does not have `torch_dtype`,'
                       f' use: {_dtype}')
        torch_dtype = _dtype
        # update hf_config as well
        setattr(config.hf_config, 'torch_dtype', torch_dtype)
    else:
        if torch_dtype == 'bfloat16' and not is_bf16_supported(device_type):
            torch_dtype = 'float16'
        # change to user specified data type if it is not 'auto'
        if dtype == 'auto':
            torch_dtype = torch_dtype if torch_dtype in ['float16', 'bfloat16'] else 'float16'
        else:
            torch_dtype = dtype

    resolved_dtype = getattr(torch, torch_dtype, None)
    if not isinstance(resolved_dtype, torch.dtype):
        raise ValueError(f'Invalid torch dtype "{torch_dtype}" resolved from model config; '
                         'expected a torch.dtype attribute on torch.')
    config.dtype = resolved_dtype

    return config


@dataclass
class BackendConfig:
    """Backend config."""
    eager_mode: bool = True
    device_type: str = 'cuda'
    piecewise_cudagraph_max_tokens: int | None = None
    enable_deterministic: bool = False


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
    block_size: int  # Physical tokens per rank; logical block size is block_size * dcp.
    num_cpu_blocks: int
    num_gpu_blocks: int
    kernel_block_size: int = -1
    dcp: int = 1
    window_size: int = -1
    cache_max_entry_count: float = 0.8
    max_prefill_token_num: int = 8192
    cudagraph_capture_batch_sizes: list[int] | None = None
    enable_prefix_caching: bool = False
    quant_policy: QuantPolicy = QuantPolicy.NONE
    device_type: str = 'cuda'
    num_state_caches: int = None
    # Internal geometry derived from the model/backend state layout at startup.
    compact_state_capacity: 'SSMStateCapacity | None' = None
    prefix_cache_state_budget: int = 0
    prefix_cache_decode_state_interval: int = 0
    prefix_cache_decode_save_first: bool = False
    prefill_save_alignment: int = 1024
    prefill_save_last_only: bool = False
    mooncake_state_save_slots: int = 8
    # Model-owned checkpoint validity constraint, independent of prefill chunking.
    prefix_cache_block_aligned: bool = False
    prefix_cache_token_lookahead: int = 0
    # Requested feature flag; validated before allocating model/cache resources.
    enable_replayssm: bool = False
    replayssm_buffer_len: int = 16
    states_shapes: list[tuple] = field(default_factory=list)

    # reserved blocks for dummy inputs, init to 0 for unit test.
    num_reserved_gpu_blocks: int = 0

    # For PD Disaggregation
    role: EngineRole = EngineRole.Hybrid
    migration_backend: MigrationBackend = MigrationBackend.DLSlime
    kv_transfer_config: KVTransferConfig | None = None

    def __post_init__(self):
        """Post init."""
        assert self.dcp >= 1, 'invalid dcp'
        assert self.prefix_cache_state_budget >= 0, 'invalid prefix_cache_state_budget'
        assert self.prefix_cache_decode_state_interval >= 0, 'invalid prefix_cache_decode_state_interval'
        assert self.prefill_save_alignment > 0, 'invalid prefill_save_alignment'
        assert self.mooncake_state_save_slots > 0, 'invalid mooncake_state_save_slots'
        if self.window_size > 1 and self.enable_prefix_caching:
            logger.warning('Prefix caching is not available for window attention.')
            self.enable_prefix_caching = False
        if self.kernel_block_size == -1:
            self.kernel_block_size = self.block_size
        if self.prefix_cache_decode_state_interval > 0:
            assert self.prefix_cache_decode_state_interval % self.block_size == 0, (
                'prefix_cache_decode_state_interval must be a multiple of block_size')
        self.cudagraph_capture_batch_sizes = normalize_cudagraph_capture_batch_sizes(
            self.cudagraph_capture_batch_sizes, self.max_batches)

    @property
    def use_mooncake_store(self) -> bool:
        """Whether this engine uses the Mooncake Store connector."""
        config = self.kv_transfer_config
        return (config is not None and config.is_kv_transfer_instance
                and config.kv_connector == 'MooncakeStoreConnector')

    @property
    def num_store_state_caches(self) -> int:
        """Reserved tail slots, excluded from runtime and local APC
        allocation."""
        if self.states_shapes and self.use_mooncake_store and self.kv_transfer_config.is_kv_producer:
            return self.mooncake_state_save_slots
        return 0

    @property
    def needs_prefix_cache_identity(self) -> bool:
        """Local APC and Mooncake Store need stable multimodal identities."""
        return self.enable_prefix_caching or self.use_mooncake_store

    @property
    def enable_decode_checkpoints(self) -> bool:
        """Whether an SSM cache owner can save accepted decode states."""
        return bool(self.states_shapes and (self.enable_prefix_caching or self.num_store_state_caches)
                    and (self.prefix_cache_decode_save_first or self.prefix_cache_decode_state_interval))

    @property
    def state_lookup_alignment(self) -> int:
        """Cover prefill, decode and retry boundaries; interval 0 keeps prefill
        alignment."""
        return gcd(self.prefill_save_alignment, self.prefix_cache_decode_state_interval)


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
    dcp: int = 1  # decode context parallelism inside attention tp
    attn_tp: int = None  # tp for attention
    mlp_tp: int = None  # tp for mlp
    moe_tp: int = None  # tp for moe

    # tp mode
    mlp_tp_mode: TPMode = TPMode.DEFAULT
    moe_tp_mode: TPMode = TPMode.DEFAULT

    # communication
    communication_backend: Literal['nccl', 'auto'] = 'nccl'

    def __post_init__(self):
        """Post init."""
        assert self.dp_rank < self.dp
        assert self.dp >= 1
        assert self.dcp >= 1
        if self.dcp > 1:
            assert self.dp == 1 and self.ep == 1, 'DCP currently requires dp=1 and ep=1'

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
        assert self.attn_tp % self.dcp == 0, (
            f'attn_tp {self.attn_tp} must be divisible by dcp {self.dcp}')
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
            communication_backend=engine_config.communication_backend,
            tp=engine_config.tp,
            dcp=engine_config.dcp,
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


def override_hf_config(hf_config: Any, hf_overrides: dict[str, Any]):
    """Override HF config."""
    for k, v in hf_overrides.items():
        _override_hf_config(hf_config, k, v)


def _default_check_env(device: str):
    pass


def _patch_quantization_config(hf_config: Any, model_format: str = None):
    """Patch quantization config."""
    if model_format is None:
        return hf_config

    # skip the quantized llm and vlm models
    if hasattr(hf_config, 'quantization_config') or \
        (hasattr(hf_config, 'llm_config') and hasattr(hf_config.llm_config, 'quantization_config')) \
            or (hasattr(hf_config, 'text_config') and hasattr(hf_config.text_config, 'quantization_config')):
        logger.warning('Can not perform weight quantization on quantized model.')
        return hf_config

    if model_format == 'fp8':
        logger.debug('Patch quantization config for fp8.')
        from lmdeploy.pytorch.envs import scale_fmt
        quantization_config = dict(quant_method='fp8',
                                   fmt='e4m3',
                                   weight_block_size=[128, 128],
                                   scale_fmt=scale_fmt,
                                   lmdeploy_patched=True)
    else:
        raise RuntimeError(f'Unsupported weight quantization method: {model_format}')

    hf_config.quantization_config = quantization_config
    # for vlm models
    if hasattr(hf_config, 'text_config'):
        hf_config.text_config.quantization_config = quantization_config
    elif hasattr(hf_config, 'llm_config'):
        hf_config.llm_config.quantization_config = quantization_config

    return hf_config


@dataclass
class MemDecodeConfig:
    """Configuration for MemDecode auxiliary memory model and fusion."""

    memory_model_path: str
    memory_model_config: 'ModelConfig'
    lambda_value: float = 1.0
    adaptive_router: bool = True
    router_path: str | None = None
    lambda_base_only_threshold: float = -1.0

    def __post_init__(self):
        self.lambda_value = float(self.lambda_value)
        self.lambda_base_only_threshold = float(self.lambda_base_only_threshold)
        if not 0.0 <= self.lambda_value <= 1.0:
            raise ValueError(f'lambda_value must be in [0, 1], got {self.lambda_value}')
        if self.adaptive_router and self.router_path is None:
            raise ValueError('router_path is required when adaptive_router is enabled.')


@dataclass(frozen=True)
class SSMStateCapacity:
    """Separate request capacity from private state-unit capacity.

    This is an internal geometry contract, not a reinterpretation of
    ``CacheConfig.num_state_caches``. ``request_slots`` includes reserved slot
    zero; checkpoint and Store budgets count shared units, not extra banks.
    """

    request_slots: int
    max_update_tokens: int
    checkpoint_budget: int = 0
    store_budget: int = 0
    runtime_is_auxiliary: bool = False
    # Store consumers need shared load destinations even when runtime has no unit bank.
    load_budget: int = 0

    def __post_init__(self):
        values = (self.request_slots, self.max_update_tokens,
                  self.checkpoint_budget, self.store_budget, self.load_budget)
        if self.load_budget and not self.runtime_is_auxiliary:
            raise ValueError('Separate Store load capacity is only needed for auxiliary runtime.')
        if type(self.runtime_is_auxiliary) is not bool:
            raise ValueError('runtime_is_auxiliary must be a bool.')
        if self.request_slots == 0 and self.runtime_is_auxiliary:
            raise ValueError('Stateless models cannot have auxiliary runtime.')
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError('Compact state capacities must be non-negative integers.')
        if self.request_slots == 0:
            if any(values[1:]):
                raise ValueError('Stateless models must have zero compact state capacities.')
        elif self.max_update_tokens == 0:
            raise ValueError('Stateful models need at least one update token.')

    @classmethod
    def from_cache_config(cls, cache: CacheConfig, max_update_tokens: int, *,
                          runtime_is_auxiliary: bool = False) -> 'SSMStateCapacity':
        """Normalize default state slots and derive capacity for a stateful
        target."""
        # CacheConfig.states_shapes is populated later than executor selection.
        store_budget = (cache.mooncake_state_save_slots
                        if cache.use_mooncake_store and cache.kv_transfer_config.is_kv_producer else 0)
        if cache.num_state_caches is None:
            cache.num_state_caches = cache.max_batches + 2 + cache.prefix_cache_state_budget + store_budget
        load_budget = (cache.num_state_caches - cache.prefix_cache_state_budget - store_budget - 1
                       if runtime_is_auxiliary and cache.use_mooncake_store
                       and cache.kv_transfer_config.is_kv_consumer else 0)
        return cls.from_state_slots(cache.num_state_caches, cache.prefix_cache_state_budget,
                                    store_budget, max_update_tokens, runtime_is_auxiliary=runtime_is_auxiliary,
                                    load_budget=load_budget)

    @classmethod
    def from_state_slots(cls, num_state_caches: int | None, checkpoint_budget: int,
                    store_budget: int, max_update_tokens: int, *,
                         runtime_is_auxiliary: bool = False, load_budget: int = 0) -> 'SSMStateCapacity':
        """Preserve A=M-B-S, R=A-1; absence of state allocates nothing."""
        if num_state_caches is None:
            return cls(0, 0)
        request_slots = num_state_caches - checkpoint_budget - store_budget
        if request_slots < 1:
            raise ValueError('State capacity must include reserved request slot zero.')
        return cls(request_slots, max_update_tokens, checkpoint_budget, store_budget, runtime_is_auxiliary, load_budget)

    @property
    def request_capacity(self) -> int:
        return max(0, self.request_slots - 1)

    @property
    def bank_width(self) -> int:
        """Native uses one destination per token; auxiliary Replay uses
        none."""
        return self.max_update_tokens if self.request_slots and not self.runtime_is_auxiliary else 0

    @property
    def num_units(self) -> int:
        if not self.request_slots:
            return 0
        return (1 + self.request_capacity * self.bank_width
                + self.checkpoint_budget + self.store_budget + self.load_budget)


@dataclass
class StateCacheSpec:
    """Spec for a named sequence-scoped state cache (e.g. compressor
    scratch)."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    layer_ids: list[int] | None = None
    alignment: int = 256
    store_cacheable: bool = True
    # False only when resuming at every supported prefix boundary needs no
    # prior contents. Runtime allocation and speculative rollback are unchanged.
    prefix_checkpoint: bool = True


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    bos_token_id: int
    eos_token_id: list[int]
    head_dim: int
    k_head_dim: int = None
    v_head_dim: int = None
    sliding_window: int = -1
    dtype: torch.dtype = torch.float16
    vocab_size: int = 40000
    hf_config: Any = None
    llm_config: Any = None
    dist_config: DistConfig = None
    cogvlm_style: bool = False
    custom_module_map: dict[str, setattr] = None

    # flash mla
    use_flash_mla: bool = False
    mla_kv_cache_dtype: str | None = None
    mla_index_topk: int | None = None
    # Custom indexers (e.g. GLM KPool) can use the packed MLA cache without NSA.
    supports_mla_fp8_cache: bool = False
    # Custom sparse indexers also need score memory without selecting NSA.
    reserve_dsa_score_workspace: bool = False

    # dllm
    model_paradigm: str = 'ar'
    dllm_mask_token: int = 0
    dllm_block_length: int = None

    # added for qwen3_next
    # could used for any SSM model.
    states_shapes: list[tuple[tuple[int], torch.dtype]] = field(default_factory=list)
    # flag to indicate that the model uses gated delta rule layers
    # and requires prepare_chunk_indices during prefill
    is_gated_delta: bool = False
    # Effective flag: a supported target builder has installed the Replay schema.
    enable_replayssm: bool = False
    replayssm_buffer_len: int = 16

    # Named state-cache specs for models that need layered sequence state.
    state_cache_specs: list[StateCacheSpec] = field(default_factory=list)
    # Model/backend storage contract, not a runtime feature toggle.
    state_cache_is_indexed: bool = False
    use_standard_kv_cache: bool = True

    # A model can reject generic prefix-cache reuse when its auxiliary state
    # cannot be restored at scheduler block boundaries.  ``None`` keeps the
    # default supported behavior; a non-empty reason is surfaced before model
    # weights are built.
    prefix_caching_unsupported_reason: str | None = None

    # check env for model-device combination
    check_env_func: Callable = _default_check_env

    # fp32 lm head
    fp32_lm_head: bool = False
    tie_word_embeddings: bool = False

    # quant config
    quant_config: 'QuantizationConfig' = None

    # flags mark if this model use mrope
    use_mrope: bool = False

    # update cache config
    update_cache_config_func: Any = None

    # Number of contiguous TP ranks that own the same logical KV-head shard.
    num_replicate_key_value_heads: int = 1

    @property
    def use_mla_fp8_cache(self):
        """Whether MLA uses the DeepSeek-V3.2 FP8 cache layout."""
        return self.mla_kv_cache_dtype == 'fp8_ds_mla'

    def get_head_size(self):
        """Get head size."""
        return self.head_dim

    def get_num_qkv_head_by_tp(self):
        """Get q and kv heads per TP rank."""
        dist_config = self.dist_config or DistConfig()
        tp = dist_config.attn_tp
        assert self.num_attention_heads % tp == 0
        if self.num_key_value_heads >= tp:
            assert self.num_key_value_heads % tp == 0
        else:
            assert tp % self.num_key_value_heads == 0
        num_q_heads = self.num_attention_heads // tp
        num_kv_heads = max(self.num_key_value_heads // tp, 1)
        return num_q_heads, num_kv_heads

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = False,
        dtype: str = 'auto',
        dist_config: DistConfig = None,
        hf_overrides: dict[str, Any] = None,
        is_draft_model: bool = False,
        spec_method: str = None,
        num_spec_tokens: int = 0,
        enable_replayssm: bool = False,
        replayssm_buffer_len: int = 16,
        model_format: str = None,
        device_type: str = 'auto',
        block_size: int = 64,
    ):
        """Instantiate one of the configuration classes of the library from a
        pretrained model configuration.

        Args:
            pretrained_model_name_or_path (str): the pretrained model path
            trust_remote_code (bool):  Whether or not to allow for custom
                models defined on the Hub in their own modeling files.
            dtype (str): user specified data type for model weights and
                activations. Refer to `PyTorchEngineConfig` for details
            hf_overrides (dict[str, Any]): overrides for the HF config.
            model_format (str): the quantization format of the model.
        """
        from transformers import AutoConfig

        from lmdeploy.pytorch.transformers import config_from_pretrained
        hf_config = config_from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        if getattr(hf_config, 'model_type', None) in ['phi3']:
            # phi3 + trust_remote_code leads to error when tp.
            hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # update quantization config
        hf_config = _patch_quantization_config(hf_config, model_format=model_format)

        model_config = cls.from_hf_config(
            hf_config,
            pretrained_model_name_or_path,
            dtype=dtype,
            dist_config=dist_config,
            is_draft_model=is_draft_model,
            spec_method=spec_method,
            num_spec_tokens=num_spec_tokens,
            enable_replayssm=enable_replayssm,
            replayssm_buffer_len=replayssm_buffer_len,
            device_type=device_type,
        )
        fp32_lm_head = False
        if hf_overrides is not None:
            logger.warning(f'Overriding HF config with {hf_overrides}')
            fp32_lm_head = hf_overrides.get('fp32_lm_head', False)
            override_hf_config(model_config.hf_config, hf_overrides)

        # for fp32 head
        model_config.fp32_lm_head = fp32_lm_head
        model_config.tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        # for serialization of transformers modules
        maybe_register_config_serialize_by_value(trust_remote_code)

        # add quant_config
        model_config.quant_config = QuantizationConfig.from_config(hf_config)
        model_config.block_size = block_size
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
        device_type: str = 'auto',
        num_spec_tokens: int = 0,
        enable_replayssm: bool = False,
        replayssm_buffer_len: int = 16,
    ):
        """From huggingface config."""
        from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
        if dist_config is None:
            dist_config = DistConfig()
        tp = dist_config.attn_tp

        model_config = AutoModelConfigBuilder.build(
            hf_config,
            model_path,
            tp=tp,
            is_draft_model=is_draft_model,
            spec_method=spec_method,
            num_spec_tokens=num_spec_tokens,
            enable_replayssm=enable_replayssm,
            replayssm_buffer_len=replayssm_buffer_len,
            device_type=device_type,
        )

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
        model_config.dist_config = dist_config

        if dist_config.dcp > 1:
            if not model_config.use_flash_mla:
                replicas = model_config.num_replicate_key_value_heads
                assert replicas % dist_config.dcp == 0, (
                    'GQA DCP groups must share replicated KV heads: '
                    f'KV replication factor {replicas} must be divisible by dcp {dist_config.dcp}')
            assert model_config.sliding_window < 0, 'DCP does not support sliding-window attention'
            if model_config.mla_index_topk is not None:
                from lmdeploy.pytorch import envs
                assert envs.sparse_mla_backend != 'tilelang', 'DCP does not support TileLang attention'

        # should after setting `hf_config` and `model_arch` attributes
        model_config = _update_torch_dtype(model_config, dtype, device_type=device_type)

        if spec_method == 'dflash':
            model_config.model_paradigm = 'ar_spec'

        if model_config.enable_replayssm:
            from lmdeploy.pytorch.consts import (
                GDN_REPLAY_BETA_CACHE_NAME,
                GDN_REPLAY_CURSOR_CACHE_NAME,
                GDN_REPLAY_RAW_K_CACHE_NAME,
                GDN_REPLAY_RAW_V_CACHE_NAME,
            )
            speculative = num_spec_tokens > 0
            if is_draft_model or num_spec_tokens < 0 or speculative != (model_config.model_paradigm == 'ar_spec'):
                raise ValueError('ReplaySSM target schema must match num_spec_tokens and model_paradigm.')
            names = {spec.name for spec in model_config.state_cache_specs}
            raw_names = {GDN_REPLAY_RAW_K_CACHE_NAME, GDN_REPLAY_RAW_V_CACHE_NAME, GDN_REPLAY_BETA_CACHE_NAME}
            if (GDN_REPLAY_CURSOR_CACHE_NAME not in names
                    or names & raw_names != (raw_names if speculative else set())):
                raise ValueError('ReplaySSM raw-history schema must match the configured speculative protocol.')

        # update eos_token_id to list
        if isinstance(model_config.eos_token_id, int):
            model_config.eos_token_id = [model_config.eos_token_id]

        return model_config


def validate_dcp_dllm_compatibility(model_config: ModelConfig, block_length: int) -> None:
    """Reject block-sparse dLLM attention when DCP is enabled.

    The DCP GQA implementation currently evaluates dense attention.  The effective dLLM block length must therefore be
    one before the attention backend is constructed.
    """
    dist_config = model_config.dist_config
    if (dist_config is None or dist_config.dcp <= 1 or model_config.model_paradigm != 'dllm'
            or block_length == 1):
        return

    raise ValueError(
        'DCP does not support SDAR/dLLM block-sparse attention with '
        f'dllm_block_length={block_length}. Use dcp=1 or set dllm_block_length=1.')


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
    hf_overrides: dict[str, Any] = None
    language_model_only: bool = False
    logprobs_mode: str = None
    dllm_config: DLLMConfig = None
    enable_return_routed_experts: bool = False
    enable_chunked_prefill: bool = False
    memdecode_config: MemDecodeConfig = None

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
            language_model_only=engine_config.language_model_only,
            logprobs_mode=engine_config.logprobs_mode,
            dllm_config=dllm_config,
            enable_return_routed_experts=engine_config.enable_return_routed_experts,
            enable_chunked_prefill=False,
        )
        return misc_config


@dataclass
class SpecDecodeConfig:
    model: str
    method: str
    cache_config: CacheConfig = None
    num_speculative_tokens: int = 1
    model_config: ModelConfig = None
    dist_config: DistConfig = field(default_factory=DistConfig)
    target_layer_ids: tuple[int, ...] | None = None
    mask_token_id: int | None = None

    @classmethod
    def from_config(
        cls,
        method: str,
        num_speculative_tokens: int,
        model: str,
        target_cache_cfg: CacheConfig,
        target_model: str = None,
        dtype: str = 'auto',
        trust_remote_code: bool = False,
        model_format: str = None,
        hf_overrides: dict[str, Any] = None,
        dist_config: DistConfig = None,
    ):
        draft_model = model or target_model
        dist_config = dist_config or DistConfig()
        model_config = ModelConfig.from_pretrained(draft_model,
                                                   trust_remote_code=trust_remote_code,
                                                   dtype=dtype,
                                                   dist_config=dist_config,
                                                   is_draft_model=True,
                                                   spec_method=method,
                                                   block_size=target_cache_cfg.block_size,
                                                   model_format=model_format,
                                                   hf_overrides=hf_overrides,
                                                   device_type=target_cache_cfg.device_type,
                                                   )
        target_layer_ids = None
        mask_token_id = None
        if method == 'dflash':
            from lmdeploy.pytorch.spec_decode.dflash_utils import (
                parse_dflash_config,
                validate_dflash_cache_config,
                validate_dflash_runtime_config,
            )
            validate_dflash_cache_config(target_cache_cfg)
            validate_dflash_runtime_config(cache_config=target_cache_cfg)
            if target_model is None:
                raise ValueError('DFlash requires an explicit target_model for checkpoint compatibility checks.')
            target_model_config = ModelConfig.from_pretrained(
                target_model,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                dist_config=dist_config,
                is_draft_model=False,
                spec_method=method,
                num_spec_tokens=num_speculative_tokens,
                model_format=model_format,
                hf_overrides=hf_overrides,
                device_type=target_cache_cfg.device_type,
                block_size=target_cache_cfg.block_size,
            )
            # Hybrid target configs use ``ModelConfig.num_layers`` for the
            # number of KV-cache attention layers, while DFlash layer ids
            # address every transformer layer. Prefer the underlying text
            # config depth and fall back to the generic ModelConfig field.
            target_num_layers = getattr(target_model_config.llm_config, 'num_hidden_layers',
                                        target_model_config.num_layers)
            target_layer_ids, mask_token_id = parse_dflash_config(
                model_config.hf_config,
                num_speculative_tokens,
                target_num_layers=target_num_layers,
            )
        cache_config = None
        # include medusa
        no_caches = ['medusa']
        if method not in no_caches:
            cache_config = CacheConfig(max_batches=target_cache_cfg.max_batches,
                                       dcp=dist_config.dcp,
                                       block_size=target_cache_cfg.block_size,
                                       kernel_block_size=target_cache_cfg.kernel_block_size,
                                       num_cpu_blocks=target_cache_cfg.num_cpu_blocks,
                                       num_gpu_blocks=target_cache_cfg.num_gpu_blocks,
                                       cache_max_entry_count=target_cache_cfg.cache_max_entry_count,
                                       max_prefill_token_num=target_cache_cfg.max_prefill_token_num,
                                       cudagraph_capture_batch_sizes=target_cache_cfg.cudagraph_capture_batch_sizes,
                                       device_type=target_cache_cfg.device_type,
                                       quant_policy=target_cache_cfg.quant_policy,
                                       migration_backend=target_cache_cfg.migration_backend)
        obj = cls(
            model=draft_model,
            method=method,
            cache_config=cache_config,
            model_config=model_config,
            dist_config=dist_config,
            num_speculative_tokens=num_speculative_tokens,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
        )
        return obj


@dataclass
class QuantizationConfig:
    quant_method: str = None
    quant_dtype: torch.dtype = None
    scale_fmt: str = None
    bits: int = None
    group_size: int = None
    weight_block_size: tuple[int] = None
    activation_scheme: str = None
    ignored_layers: list[str] = field(default_factory=list)
    fp8_quant_scope: str | None = None
    ct_format: str | None = None
    hf_quant_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate quantization scope."""
        if self.fp8_quant_scope not in {None, 'moe_only'}:
            raise ValueError(f'Unsupported fp8 quant scope: {self.fp8_quant_scope}')
        if self.fp8_quant_scope is not None and self.quant_method != 'fp8':
            raise ValueError('fp8_quant_scope is only supported for fp8 quantization.')

    @classmethod
    def from_config(cls, hf_config: Any):
        quant_sources = []
        quant_config = getattr(hf_config, 'quantization_config', None)
        if quant_config is not None:
            quant_sources.append(('quantization_config', quant_config))
        for config_name in ('llm_config', 'text_config'):
            nested_config = getattr(hf_config, config_name, None)
            nested_quant_config = getattr(nested_config, 'quantization_config', None)
            if nested_quant_config is not None:
                quant_sources.append((f'{config_name}.quantization_config', nested_quant_config))

        # no quant config found in hf config
        if not quant_sources:
            return cls()

        quant_config = quant_sources[0][1]
        compressed_sources = [(path, config) for path, config in quant_sources
                              if isinstance(config, Mapping) and config.get('quant_method') == 'compressed-tensors']
        if compressed_sources:
            if len(compressed_sources) != len(quant_sources):
                paths = [path for path, _ in quant_sources]
                raise ValueError(f'Conflicting quantization configs found at {paths}: compressed-tensors cannot be '
                                 'combined with another quantization method')
            quant_config = compressed_sources[0][1]
            for path, candidate in compressed_sources[1:]:
                if candidate != quant_config:
                    raise ValueError(f'Conflicting compressed-tensors config found at `{path}`')

        quant_method = quant_config['quant_method']
        quant_dtype = quant_config.get('quant_dtype', None)
        scale_fmt = quant_config.get('scale_fmt', None)
        weight_block_size = quant_config.get('weight_block_size', None)
        activation_scheme = quant_config.get('activation_scheme', None)
        fp8_quant_scope = quant_config.get('fp8_quant_scope', None)

        bits = None
        group_size = None
        ignored_layers = None
        ct_format = None

        if quant_method == 'awq':
            bits = quant_config.get('bits', 4)
            group_size = quant_config.get('group_size', 128)
            if quant_dtype is None:
                # awq does not need a quant dtype, this is just a placeholder
                quant_dtype = 'bfloat16'
        elif quant_method in ('smooth_quant', 'modelslim'):
            if quant_dtype is None:
                quant_dtype = 'int8'
        elif quant_method == 'fp8':
            fmt = quant_config.get('fmt', 'e4m3')
            if fmt == 'e4m3':
                quant_dtype = 'float8_e4m3fn'
            elif fmt == 'e5m2':
                quant_dtype = 'float8_e5m2'
            else:
                raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        elif quant_method == 'compressed-tensors':
            bits, group_size, ignored_layers, ct_format = _parse_compressed_tensors_config(quant_config)
            if ct_format == 'float-quantized':
                _ct_weights = quant_config['config_groups']
                _ct_weights = _ct_weights.get('group_0') or _ct_weights[sorted(_ct_weights)[0]]
                _, _, ct_fp8_blocked, _ = _resolve_ct_fp8_layout(_ct_weights['weights'])
                # The canonical blocked checkpoints carry a null group_size;
                # surface blocked-ness through weight_block_size so the PT
                # dispatch does not misroute them to the per-channel lane.
                if ct_fp8_blocked and weight_block_size is None:
                    weight_block_size = (128, 128)
        elif quant_method == 'gptq':
            # GPTQ dense linears load through the AWQ family: the load-time
            # repack absorbs pack axis, nibble order, and zero point offset
            # (see normalize_gptq_weights and gptq_repack).
            bits = quant_config.get('bits', 4)
            group_size = quant_config.get('group_size', 128)
            if bits != 4:
                raise ValueError(f'Unsupported gptq `bits`: expected 4, got {bits!r}.')
            if quant_config.get('desc_act'):
                raise ValueError('Unsupported gptq `desc_act`: act-order reorders groups '
                                 'and the load-time repack assumes identity group order.')
            if quant_config.get('checkpoint_format', 'gptq') not in ('gptq', 'gptq_v2'):
                raise ValueError(f'Unsupported gptq `checkpoint_format`: '
                                 f'{quant_config.get("checkpoint_format")!r}.')
            if quant_dtype is None:
                quant_dtype = 'bfloat16'
        elif quant_method == 'mxfp4':
            # MXFP4 packs e2m1 weight codes with one e8m0 byte scale per
            # group of 32 along the input dim. The checkpoint contract fixes
            # the group size; an explicit conflicting value is a checkpoint
            # error, not a knob.
            bits = 4
            group_size = quant_config.get('group_size', 32)
            if group_size != 32:
                raise ValueError(f'Unsupported mxfp4 `group_size`: expected 32, got {group_size!r}.')
            if quant_dtype is None:
                # mxfp4 computes in bf16; placeholder like awq/gptq.
                quant_dtype = 'bfloat16'
        else:
            raise TypeError(f'Unsupported quant method: {quant_method}')

        if quant_dtype is not None:
            resolved_quant_dtype = getattr(torch, quant_dtype, None)
            if not isinstance(resolved_quant_dtype, torch.dtype):
                raise ValueError(f'Invalid quant dtype "{quant_dtype}" resolved from model config; '
                                 'expected a torch.dtype attribute on torch.')
            quant_dtype = resolved_quant_dtype

        if ignored_layers is None:
            ignored_layers = quant_config.get('ignored_layers', [])
            if not ignored_layers:
                ignored_layers = quant_config.get('modules_to_not_convert', [])

        return cls(
            quant_method=quant_method,
            quant_dtype=quant_dtype,
            scale_fmt=scale_fmt,
            bits=bits,
            group_size=group_size,
            weight_block_size=weight_block_size,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            fp8_quant_scope=fp8_quant_scope,
            ct_format=ct_format,
            hf_quant_config=quant_config,
        )

    def get_quant_method(self, prefix: str = '', module_kind: str = 'linear'):
        """Get quant method for module."""
        if module_kind not in {'linear', 'moe', 'norm'}:
            raise ValueError(f'Unsupported quant module kind: {module_kind}')
        if self.quant_method == 'modelslim':
            from lmdeploy.pytorch.backends.ascend.modelslim import get_modelslim_quant_method
            return get_modelslim_quant_method(self, prefix, module_kind)
        if self.quant_method == 'gptq':
            if module_kind == 'norm':
                return None
            # GPTQ streams normalize to canonical AWQ at load time (dense
            # linears and expert tensors alike — the loader's suffix filter
            # covers expert prefixes), so both kinds load through the AWQ
            # family and its routes.
            if self.ignored_layers and not prefix:
                raise ValueError('gptq dispatch requires a non-empty canonical module '
                                 'prefix when the checkpoint carries ignore rules')
            if self.ignored_layers and any(prefix in layer for layer in self.ignored_layers):
                return None
            return 'awq'
        if self.quant_method == 'compressed-tensors':
            if module_kind == 'norm':
                return None
            is_ignored = bool(prefix) and any(_matches_compressed_tensors_ignore(rule, prefix)
                                              for rule in self.ignored_layers)
            if is_ignored:
                return None
            if not prefix and self.ignored_layers:
                raise ValueError('compressed-tensors dispatch requires a non-empty canonical module '
                                 'prefix when the checkpoint carries ignore rules')
            if module_kind == 'linear':
                # Group-128 dense linears load through the AWQ family: the
                # weight loader normalizes each ct module to the canonical
                # AWQ layout at load time (see model_weight_loader). Group 32
                # keeps the packed ct modules. Most model patches build
                # linears without a prefix, so ignore-rule matching is
                # skipped. Checkpoints with ignore rules must pass prefixes.
                if self.ct_format == 'nvfp4-pack-quantized':
                    return 'nvfp4'
                if self.ct_format == 'float-quantized':
                    return 'ct_fp8'
                if self.group_size == 128:
                    return 'awq'
                return self.quant_method
            if module_kind == 'moe' and self.ct_format in ('nvfp4-pack-quantized', 'float-quantized'):
                # Grouped expert routes consume the same float formats
                # through the TurboMind grouped channel.
                return 'nvfp4' if self.ct_format == 'nvfp4-pack-quantized' else 'ct_fp8'
            return self.quant_method
        if self.quant_method == 'mxfp4':
            # The mxfp4 checkpoint contract quantizes only the routed
            # experts; attention, router, embeddings and lm_head stay in the
            # checkpoint dtype, so linear/norm modules are never quantized
            # and ignore-rule matching is unnecessary.
            return self.quant_method if module_kind == 'moe' else None
        if self.quant_method == 'fp8' and self.fp8_quant_scope == 'moe_only' and module_kind != 'moe':
            quant_method = None
            return quant_method
        if not prefix or not self.ignored_layers:
            quant_method = self.quant_method
            return quant_method

        is_ignore = any([prefix in layer_name for layer_name in self.ignored_layers])
        quant_method = None if is_ignore else self.quant_method
        return quant_method

    def load_normalize(self) -> bool:
        """Whether weight streams need load-time quant repack.

        Group-128 packed compressed-tensors checkpoints and gptq checkpoints normalize to the canonical AWQ layout (see
        normalize_ct_weights and normalize_gptq_weights); group-32 packed ct keys (MoE experts) are consumed raw and
        must pass through untouched. The float lanes (nvfp4-pack-quantized, float-quantized) keep their checkpoint
        suffixes: the float modules register matching parameter names.
        """
        if self.quant_method == 'gptq':
            return True
        return (self.quant_method == 'compressed-tensors' and self.group_size == 128
                and self.ct_format in (None, 'pack-quantized'))

    def get(self, key, default=None):
        """Get extra key from hf quant config."""
        return self.hf_quant_config.get(key, default)
