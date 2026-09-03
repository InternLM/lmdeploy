# Copyright (c) OpenMMLab. All rights reserved.

import _turbomind as _tm
import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.pytorch.config import override_hf_config
from lmdeploy.utils import get_logger

from ..utils import _get_and_verify_max_len
from .builders import _torch_dtype_to_cpp
from .models.base import INPUT_MODELS
from .models.utils import source_model_config
from .supported_models import SUPPORTED_ARCHS
from .weight_format import (
    AWQFormat,
    CompressedTensorFormat,
    FP8Format,
    GPTQFormat,
    MXFP4Format,
    TrivialFormat,
    WeightFormat,
    WeightFormatResolver,
)

logger = get_logger('lmdeploy')


def _build_quantized_formats(
        model_format: str | None,
        group_size: int | None) -> list[WeightFormat]:
    formats: list[WeightFormat] = []
    if model_format in (None, 'hf'):
        pass
    elif model_format == 'awq':
        formats.append(AWQFormat(block_in=group_size))
    elif model_format == 'gptq':
        formats.append(GPTQFormat(block_in=group_size))
    elif model_format == 'compressed-tensors':
        formats.append(CompressedTensorFormat(block_in=group_size))
    elif model_format == 'fp8':
        formats.extend((FP8Format(block_out=128), FP8Format(block_out=1)))
    elif model_format == 'mxfp4':
        formats.append(MXFP4Format())
    else:
        raise ValueError(f'unknown model_format: {model_format!r}')
    return formats


def _get_executable_dtypes(
        formats: list[WeightFormat],
        device) -> set[str]:
    executable: set[str] = set()

    with torch.cuda.device(device):
        gemm = _tm.Gemm()
        for name in ('bfloat16', 'float16'):
            candidate = _torch_dtype_to_cpp(getattr(torch, name))
            candidate_formats = [
                *formats,
                TrivialFormat(weight_dtype=candidate),
            ]
            if all(
                    candidate in gemm.data_types(
                        fmt.make_data_format())
                    for fmt in candidate_formats):
                executable.add(name)

    return executable


def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
    """Recursively merge override into base, mutating base in-place."""
    for k, v in override.items():
        key_path = f'{path}.{k}' if path else k
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v, key_path)
        else:
            if k not in base:
                logger.warning(f'hf_overrides key "{key_path}" not found in config, applying anyway')
            base[k] = v
    return base


def _apply_hf_overrides(cfg, override: dict):
    """Apply hf_overrides to a Transformers config object or nested dict."""
    override_hf_config(cfg, override)
    return cfg


_DEFAULT_GROUP_SIZES = {
    'awq': 128,
    'gptq': 128,
    'compressed-tensors': 128,
    'fp8': 128,
    'mxfp4': 32,
}

_SUPPORTED_GROUP_SIZES = {
    'awq': frozenset({128}),
    'gptq': frozenset({128}),
    'compressed-tensors': frozenset({32, 128}),
    'fp8': frozenset({128}),
    'mxfp4': frozenset({32}),
}


def _validate_quant_group_size(model_format: str | None, group_size: int | None) -> int | None:
    """Normalize and validate quantized group sizes.

    The low-level int4 kernels can be shared across formats, but we only expose the format/group-size combinations that
    are verified end to end.
    """
    if group_size in (None, 0):
        group_size = _DEFAULT_GROUP_SIZES.get(model_format, group_size)

    supported_group_sizes = _SUPPORTED_GROUP_SIZES.get(model_format)
    if supported_group_sizes is not None and group_size not in supported_group_sizes:
        supported = ', '.join(map(str, sorted(supported_group_sizes)))
        raise ValueError(f'Unsupported group_size={group_size} for model_format="{model_format}". '
                         f'Supported group_size values: {supported}.')

    return group_size


def get_registered_name(model_path: str, arch: str = None):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        arch (str): optional architecture string, to avoid reloading config
    """
    if arch is None:
        arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    return register_name


def _resolve_dtype(requested: str, hf_model_cfg,
                   executable: set[str]) -> str:
    """Resolve the request against publisher metadata and executable dtypes."""
    dtype = requested
    if dtype == 'auto':
        if getattr(hf_model_cfg, 'text_config', None):
            hf_model_cfg = hf_model_cfg.text_config
        elif getattr(hf_model_cfg, 'llm_config', None):
            hf_model_cfg = hf_model_cfg.llm_config
        config_dtype = getattr(hf_model_cfg, 'dtype', None)
        if config_dtype is None:
            config_dtype = getattr(hf_model_cfg, 'torch_dtype', None)
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(config_dtype)
        if dtype not in executable:
            dtype = ('bfloat16' if 'bfloat16' in executable else 'float16')

    if dtype not in executable:
        if 'float16' not in executable:
            raise RuntimeError('no executable data type for this device')
        logger.warning('data type downgraded to float16')
        dtype = 'float16'
    return dtype


def get_tm_config(model_path,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None,
                  trust_remote_code: bool = False):
    """Resolve dtype/model_format/group_size/session_len, mutate engine_config
    in place, build the model.

    Returns:
        tuple: (model, model_path, data_type)
    """
    # 1. Load HF config once; reused for quant_config, dtype, and session_len.
    arch, hf_model_cfg = get_model_arch(model_path, trust_remote_code=trust_remote_code)

    # 2. Reconcile quant_config (unchanged logic from the prior flow).
    quant_config = search_nested_config(
        hf_model_cfg.to_dict(), 'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or engine_config.model_format == quant_method, (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (
            f'mismatched quant group size: user input "{group_size}" '
            f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', (
                'compressed-tensors only supports pack-quantized format, '
                f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', (
                'pack-quantized requires 4-bit int, '
                f'but got {_num_bits}-bit {_type}')
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    group_size = _validate_quant_group_size(engine_config.model_format, group_size)

    # 3. Resolve dtype and format overrides.
    requested_dtype = engine_config.dtype
    quantized_formats = _build_quantized_formats(
        engine_config.model_format,
        group_size,
    )
    executable = _get_executable_dtypes(
        quantized_formats,
        engine_config.devices[0],
    )
    dtype_name = _resolve_dtype(
        requested_dtype,
        hf_model_cfg,
        executable,
    )
    dtype = getattr(torch, dtype_name)
    data_type = _torch_dtype_to_cpp(dtype)
    resolver = WeightFormatResolver(formats=[
        *quantized_formats,
        TrivialFormat(weight_dtype=data_type),
    ])

    engine_config.dtype = dtype_name

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with remaining resolved values.
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build model.
    cfg = source_model_config(hf_model_cfg)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _apply_hf_overrides(cfg, engine_config.hf_overrides)
    registered_name = get_registered_name(model_path, arch=arch)
    model_cls = INPUT_MODELS.get(registered_name)

    # VL aggregate classes declare `_vision = True` to opt into the vision-
    # branch contract: receive `language_model_only` and a dedicated
    # `vision_resolver` (TrivialFormat only, vision-native dtype). Text-only
    # models leave the flag unset and keep the strict (cfg, *, resolver)
    # signature.
    init_kwargs = {}
    if getattr(model_cls, '_vision', False):
        init_kwargs['language_model_only'] = engine_config.language_model_only
        vision_config = getattr(hf_model_cfg, 'vision_config', None)
        if (not engine_config.language_model_only
                and vision_config is not None):
            vision_executable = _get_executable_dtypes(
                [],
                engine_config.devices[0],
            )
            vision_dtype_name = _resolve_dtype(
                requested_dtype,
                vision_config,
                vision_executable,
            )
            vision_dtype = getattr(torch, vision_dtype_name)
            vision_data_type = _torch_dtype_to_cpp(vision_dtype)
            vision_resolver = WeightFormatResolver(formats=[
                TrivialFormat(weight_dtype=vision_data_type),
            ])
            init_kwargs['vision_resolver'] = vision_resolver
            init_kwargs['vision_data_type'] = vision_data_type
    model = model_cls(cfg, resolver=resolver, **init_kwargs)

    return model, model_path, data_type
