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


def _validate_replayssm_preallocation(cache_config: CacheConfig,
                                      specdecode_config: SpecDecodeConfig = None) -> None:
    """Reject unsupported ReplaySSM combinations before model/cache
    allocation."""
    if not cache_config.enable_replayssm:
        return
    if cache_config.device_type != 'cuda':
        raise ValueError('ReplaySSM requires the PyTorch CUDA backend.')
    if cache_config.role != EngineRole.Hybrid:
        raise ValueError('ReplaySSM does not support P/D-disaggregated engine roles.')
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is not None and transfer_config.is_kv_transfer_instance:
        # External caches do not encode the Replay ring/checkpoint protocol in
        # their compatibility identity. Even producer-only use can collide with
        # native state, so reject every connector role rather than only loads.
        raise ValueError('ReplaySSM does not support external KV-cache connectors; '
                         'disable ReplaySSM or remove kv_transfer_config.')
    from lmdeploy.pytorch.config import validate_replayssm_config

    num_spec_tokens = 0 if specdecode_config is None else specdecode_config.num_speculative_tokens
    validate_replayssm_config(cache_config.replayssm_buffer_len, num_spec_tokens)
    if specdecode_config is not None and specdecode_config.method not in ('qwen3_5_mtp', 'deepseek_mtp'):
        raise ValueError(
            'ReplaySSM speculative decoding supports only qwen3_5_mtp or GLM-5.3 deepseek_mtp, got '
            f'{specdecode_config.method!r}.')


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
    _validate_replayssm_preallocation(cache_config, specdecode_config)

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
        enable_replayssm=cache_config.enable_replayssm,
        replayssm_buffer_len=cache_config.replayssm_buffer_len,
    )

    if cache_config.enable_replayssm:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9:
            capability = torch.cuda.get_device_capability()
            raise ValueError(
                'ReplaySSM is currently qualified only for SM90 CUDA GPUs, '
                f'got SM{capability[0]}{capability[1]}.')
        if not model_config.enable_replayssm:
            raise ValueError('ReplaySSM requires a supported Gated DeltaNet or GLM-5.3 KDA model.')
        if specdecode_config is not None:
            expected = ('deepseek_mtp' if model_config.hf_config.model_type == 'glm5_next'
                        else 'qwen3_5_mtp')
            if specdecode_config.method != expected:
                raise ValueError(f'ReplaySSM for this model requires {expected}.')
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
