# Copyright (c) OpenMMLab. All rights reserved.
"""Batch-invariant CUDA math policy helpers."""

import os

import torch

from lmdeploy.pytorch.config import BackendConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_CUBLAS_WORKSPACE_CONFIG = ':16:8'
_CUBLASLT_WORKSPACE_SIZE = '1'
_BATCH_INVARIANT_POLICY_ENABLED = False


def is_batch_invariant_policy_enabled() -> bool:
    """Return whether the CUDA batch-invariant policy has been applied."""
    return _BATCH_INVARIANT_POLICY_ENABLED


def _cuda_initialized() -> bool:
    """Return whether CUDA is already initialized without initializing it."""
    try:
        return torch.cuda.is_initialized()
    except Exception:
        return False


def _set_env_before_cuda_init(name: str, value: str):
    """Set an env var that must take effect before CUDA/cuBLAS init."""
    current = os.environ.get(name)
    if current == value:
        return
    if current is not None and _cuda_initialized():
        raise RuntimeError(
            f'enable_batch_invariant requires {name}={value}, but found {name}={current} '
            'after CUDA was already initialized. Set the environment before importing/running LMDeploy, '
            'or create the engine before any CUDA work in this process.')
    if current is not None:
        logger.warning(f'Override {name}={current} with {name}={value} for enable_batch_invariant.')
    os.environ[name] = value


def _set_torch_precision_policy():
    """Disable CUDA matmul precision shortcuts that can change reductions."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    cudnn_conv = getattr(torch.backends.cudnn, 'conv', None)
    if cudnn_conv is not None and hasattr(cudnn_conv, 'fp32_precision'):
        cudnn_conv.fp32_precision = 'ieee'
    cudnn_rnn = getattr(torch.backends.cudnn, 'rnn', None)
    if cudnn_rnn is not None and hasattr(cudnn_rnn, 'fp32_precision'):
        cudnn_rnn.fp32_precision = 'ieee'

    preferred_blas_library = getattr(torch.backends.cuda, 'preferred_blas_library', None)
    if preferred_blas_library is not None:
        preferred_blas_library(backend='cublaslt')

    matmul_backend = torch.backends.cuda.matmul
    for dtype_name in ('fp16', 'bf16'):
        attr_name = f'allow_{dtype_name}_reduced_precision_reduction'
        if hasattr(matmul_backend, attr_name):
            try:
                setattr(matmul_backend, attr_name, (False, False))
            except TypeError:
                setattr(matmul_backend, attr_name, False)
        split_k_attr_name = f'{attr_name}_split_k'
        if hasattr(matmul_backend, split_k_attr_name):
            try:
                setattr(matmul_backend, split_k_attr_name, False)
            except AttributeError:
                pass


def validate_batch_invariant_device(device: int | torch.device | None = None):
    """Validate that the current CUDA device is in the initial support
    scope."""
    if not torch.cuda.is_available():
        raise RuntimeError('enable_batch_invariant currently requires a CUDA Hopper GPU.')

    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        device_name = torch.cuda.get_device_name(device)
        raise RuntimeError(
            'enable_batch_invariant currently supports Hopper CUDA GPUs only, '
            f'but detected {device_name} with SM{major}.{minor}.')


def apply_batch_invariant_policy(config: BackendConfig,
                                 *,
                                 validate_device: bool = False):
    """Apply the opt-in Hopper batch-invariant policy.

    The environment settings must be applied before CUDA/cuBLAS/cuBLASLt choose algorithms. Call this once in the parent
    process before any CUDA query, and again in worker processes before they build models or initialize CUDA-heavy
    runtime state.
    """
    global _BATCH_INVARIANT_POLICY_ENABLED

    if not config.enable_batch_invariant:
        return

    if config.device_type != 'cuda':
        raise ValueError('enable_batch_invariant is currently supported only for CUDA backend.')

    _set_env_before_cuda_init('CUBLAS_WORKSPACE_CONFIG', _CUBLAS_WORKSPACE_CONFIG)
    _set_env_before_cuda_init('CUBLASLT_WORKSPACE_SIZE', _CUBLASLT_WORKSPACE_SIZE)
    _set_torch_precision_policy()

    if validate_device:
        validate_batch_invariant_device()

    _BATCH_INVARIANT_POLICY_ENABLED = True
