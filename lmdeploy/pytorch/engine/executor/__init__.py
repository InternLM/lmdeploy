# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from logging import Logger

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch import envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.utils import get_logger

from .base import ExecutorBase


def _finalize_sparse_mla_cache_policy(model_configs: Sequence[ModelConfig], cache_config: CacheConfig) -> None:
    """Translate the generic cache policy for sparse-MLA models.

    This runs before executors copy configs to workers or build model
    operators. Sparse MLA records its physical dtype on ``ModelConfig`` and
    must not expose that choice as generic KV quantization at runtime.
    """
    sparse_mla_configs = [config for config in model_configs if config.mla_index_topk is not None]
    if not sparse_mla_configs or cache_config.quant_policy == QuantPolicy.NONE:
        return
    if cache_config.quant_policy != QuantPolicy.FP8:
        raise ValueError(f'Sparse MLA does not support quant_policy={cache_config.quant_policy}. '
                         'Use none/0 for BF16 or fp8/16 for FP8.')

    for model_config in sparse_mla_configs:
        model_config.mla_kv_cache_dtype = 'fp8_ds_mla'
    cache_config.quant_policy = QuantPolicy.NONE


def _validate_dcp_config(model_config: ModelConfig, cache_config: CacheConfig,
                         dist_config: DistConfig,
                         device_type: str) -> None:
    """Validate the supported FlashMLA DCP surface before workers start."""
    dcp = dist_config.dcp
    if dcp == 1:
        return

    if device_type != 'cuda':
        raise ValueError('DCP is supported only by the CUDA PyTorch backend')
    if dist_config.dp != 1 or dist_config.ep != 1:
        raise ValueError('DCP currently requires dp=1 and ep=1')
    if not model_config.use_flash_mla:
        raise ValueError('DCP requires a FlashMLA-backed MLA model')
    if model_config.dtype != torch.bfloat16:
        raise ValueError('DCP requires a bfloat16 MLA model')
    is_sparse_mla = model_config.mla_index_topk is not None
    supported_cache_policies = ((QuantPolicy.NONE, QuantPolicy.FP8)
                                if is_sparse_mla else (QuantPolicy.NONE, ))
    if cache_config.quant_policy not in supported_cache_policies:
        raise ValueError(
            f'DCP MLA does not support quant_policy={cache_config.quant_policy}')
    replica_count = model_config.num_replicate_key_value_heads
    if dcp > replica_count or replica_count % dcp != 0:
        raise ValueError(
            f'dcp {dcp} must divide KV-head replica count {replica_count}')
    if (cache_config.role != EngineRole.Hybrid
            or cache_config.kv_transfer_config is not None):
        raise ValueError(
            'DCP does not support disaggregation or KV-cache connectors')


def get_distributed_executor_backend(world_size: int, dp: int, device_type: str, logger: Logger = None):
    """Get distributed executor backend."""
    from lmdeploy.pytorch.backends import get_backend

    def _log_info(message: str):
        if logger is not None:
            logger.info(message)

    def _log_and_set_backend(message: str, executor_backend: str):
        """Log and set backend."""
        message += f' distributed_executor_backend={executor_backend}.'
        _log_info(message)
        return executor_backend

    executor_backend = envs.executor_backend
    if executor_backend is not None:
        return _log_and_set_backend('found environment LMDEPLOY_EXECUTOR_BACKEND.', executor_backend)

    if world_size == 1:
        return 'uni'

    if dp > 1:
        executor_backend = 'ray'
        return _log_and_set_backend(f'dp={dp}.', 'ray')

    backend = get_backend(device_type)
    if not backend.support_ray():
        return _log_and_set_backend(f'device={device_type} does not support ray.', 'mp')
    else:
        return 'ray'

    # TODO: fix mp hanging, do not delete the comment.
    # device_count = backend.device_count()
    # if device_count is None:
    #     return _log_and_set_backend(f'device={device_type} can not get device_count.', 'mp')

    # if device_count < world_size:
    #     executor_backend = 'ray'
    #     return _log_and_set_backend(f'local device_count({device_count})<world_size({world_size}),', 'ray')
    # else:
    #     executor_backend = 'mp'
    #     return _log_and_set_backend(f'local device_count({device_count})>=world_size({world_size}),', 'mp')


def build_executor(
    model_path: str,
    cache_config: CacheConfig,
    backend_config: BackendConfig,
    dist_config: DistConfig,
    misc_config: MiscConfig,
    adapters: dict[str, str] = None,
    device_type: str = 'cuda',
    distributed_executor_backend: str = None,
    dtype: str = 'auto',
    specdecode_config: SpecDecodeConfig = None,
    trust_remote_code: bool = False,
) -> ExecutorBase:
    """Build model agent executor."""
    logger = get_logger('lmdeploy')
    dp = dist_config.dp
    world_size = dist_config.world_size

    model_config = ModelConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        hf_overrides=misc_config.hf_overrides,
        dist_config=dist_config,
        is_draft_model=False,
        spec_method=None if specdecode_config is None else specdecode_config.method,
        num_spec_tokens=0 if specdecode_config is None else specdecode_config.num_speculative_tokens,
        model_format=misc_config.model_format,
        device_type=device_type,
        block_size=cache_config.block_size,
    )

    _validate_dcp_config(model_config, cache_config, dist_config, device_type)

    # Finalize cache policy before any executor copies configs to workers or
    # builds backend operators. Target and memory models share CacheConfig.
    shared_cache_models = [model_config]
    if memdecode_config := misc_config.memdecode_config:
        shared_cache_models.append(memdecode_config.memory_model_config)
    _finalize_sparse_mla_cache_policy(shared_cache_models, cache_config)
    if specdecode_config is not None and specdecode_config.cache_config is not None:
        _finalize_sparse_mla_cache_policy([specdecode_config.model_config], specdecode_config.cache_config)

    if distributed_executor_backend is None:
        distributed_executor_backend = get_distributed_executor_backend(world_size, dp, device_type, logger)

    if dp > 1:
        assert distributed_executor_backend == 'ray', (
            'dp>1 requires distributed_executor_backend="ray", ',
            f'get distributed_executor_backend="{distributed_executor_backend}"')

    if misc_config.empty_init:
        assert distributed_executor_backend == 'ray', (
            'empty_init requires distributed_executor_backend="ray", ',
            f'get distributed_executor_backend="{distributed_executor_backend}"')

    if distributed_executor_backend is not None:
        logger.info(f'Build <{distributed_executor_backend}> executor.')
    if distributed_executor_backend == 'uni':
        assert world_size == 1, 'uni executor only support world_size==1.'
        from .uni_executor import UniExecutor
        return UniExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            specdecode_config=specdecode_config,
            trust_remote_code=trust_remote_code
        )
    elif distributed_executor_backend == 'mp':
        from .mp_executor import MPExecutor
        logger.warning('MPExecutor will be deprecated in future releases, please use RayExecutor instead.')
        return MPExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            specdecode_config=specdecode_config,
            trust_remote_code=trust_remote_code
        )
    elif distributed_executor_backend == 'ray':
        from .ray_executor import RayExecutor
        return RayExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            dtype=dtype,
            specdecode_config=specdecode_config,
            trust_remote_code=trust_remote_code
        )
    else:
        raise RuntimeError(f'Unsupported distributed_executor_backend: {distributed_executor_backend}.')
