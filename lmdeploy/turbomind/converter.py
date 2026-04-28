# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ..utils import _get_and_verify_max_len, is_bf16_supported
from .builders import _cpp_dtype
from .models.base import INPUT_MODELS
from .models.utils import load_model_config
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


def _build_resolver(model_format: str | None,
                    group_size: int | None,
                    data_type: '_tm.DataType') -> WeightFormatResolver:
    """Build the active resolver: quantized format (if any) + trivial fallback.

    Called after the int4 fp16 force but before the ``compressed-tensors →
    awq`` rename, so compressed-tensors models get ``CompressedTensorFormat``.
    """
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
        formats.append(FP8Format())
    elif model_format == 'mxfp4':
        formats.append(MXFP4Format())
    else:
        raise ValueError(f'unknown model_format: {model_format!r}')
    formats.append(TrivialFormat())
    return WeightFormatResolver(data_type=data_type, formats=formats)


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


def get_registered_name(model_path: str, model_format: str):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4']
    """
    arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    return register_name


def _resolve_dtype(requested: str, hf_model_cfg) -> str:
    """Resolve 'auto' dtype against the HF config and the current device.

    Prefers `dtype` over the deprecated `torch_dtype` key. Falls back to
    float16 on hardware that does not support bfloat16.
    """
    has_bf16 = is_bf16_supported()
    dtype = requested
    if dtype == 'auto':
        dtype = 'bfloat16' if has_bf16 else 'float16'
        torch_dtype = getattr(hf_model_cfg, 'dtype', None)
        if torch_dtype is None:
            torch_dtype = getattr(hf_model_cfg, 'torch_dtype', None)
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(torch_dtype, dtype)

    if dtype == 'bfloat16' and not has_bf16:
        logger.warning('data type fallback to float16 since '
                       'torch.cuda.is_bf16_supported is False')
        dtype = 'float16'
    return dtype


def get_tm_config(model_path,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Resolve dtype/model_format/group_size/session_len, mutate engine_config
    in place, build the text model.

    Returns:
        tuple: (text_model, model_path)
    """
    # 1. Load HF config once; reused for quant_config, dtype, and session_len.
    _, hf_model_cfg = get_model_arch(model_path)

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
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # 3. Resolve dtype and format overrides.
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
    engine_config.dtype = dtype

    # Build resolver after dtype is finalized but before the CT→AWQ rename,
    # so compressed-tensors models instantiate CompressedTensorFormat.
    resolver = _build_resolver(engine_config.model_format,
                               group_size, _cpp_dtype(dtype))

    # C++-side label rename (does not affect resolver).
    if engine_config.model_format == 'compressed-tensors':
        engine_config.model_format = 'awq'

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with remaining resolved values.
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build text model (hf_overrides handling unchanged).
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    registered_name = get_registered_name(model_path, engine_config.model_format)
    model_cls = INPUT_MODELS.get(registered_name)
    text_model = model_cls(hf_cfg, engine_config, resolver=resolver)

    return text_model, model_path
