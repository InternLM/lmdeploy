# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.config import (BackendConfig, CacheConfig, DistConfig, MiscConfig, SchedulerConfig,
                                     SpecDecodeConfig)
from lmdeploy.utils import get_logger, get_max_batch_size, get_model


class ConfigBuilder:

    @staticmethod
    def update_engine_config(engine_config: PytorchEngineConfig):
        """Update pytorch engine config."""
        logger = get_logger('lmdeploy')

        # make sure engine exits
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        else:
            engine_config = copy.deepcopy(engine_config)

        if engine_config.max_batch_size is None:
            engine_config.max_batch_size = get_max_batch_size(engine_config.device_type)

        if engine_config.dllm_block_length is not None:
            max_prefill_token_num = engine_config.max_prefill_token_num
            max_batch_size = engine_config.max_batch_size
            if max_batch_size * engine_config.dllm_block_length > max_prefill_token_num:
                engine_config.max_batch_size = max_prefill_token_num // engine_config.dllm_block_length
                logger.warning(f'Update max_batch_size to {engine_config.max_batch_size} '
                               f'since dllm_block_length({engine_config.dllm_block_length}) * max_batch_size '
                               f'({max_batch_size}) > max_prefill_token_num ({max_prefill_token_num}).')

        if engine_config.dp != 1:
            if engine_config.tp == 1 and engine_config.ep == 1:
                logger.warning('Data parallelism is enabled but tensor parallelism and '
                               'expert parallelism are not enabled. Setting dp=1.')
                engine_config.dp = 1
                engine_config.dp_rank = 0

        return engine_config

    @staticmethod
    def build_scheduler_config(engine_config: PytorchEngineConfig):
        """Build scheduler config."""
        scheduler_config = SchedulerConfig(max_batches=engine_config.max_batch_size,
                                           max_session_len=engine_config.session_len,
                                           prefill_interval=engine_config.prefill_interval)
        return scheduler_config

    @staticmethod
    def build_cache_config(engine_config: PytorchEngineConfig):
        """Build cache config."""
        cache_config = CacheConfig(
            max_batches=engine_config.max_batch_size,
            block_size=engine_config.block_size,
            num_cpu_blocks=engine_config.num_cpu_blocks,
            num_gpu_blocks=engine_config.num_gpu_blocks,
            cache_max_entry_count=engine_config.cache_max_entry_count,
            max_prefill_token_num=engine_config.max_prefill_token_num,
            enable_prefix_caching=engine_config.enable_prefix_caching,
            quant_policy=engine_config.quant_policy,
            device_type=engine_config.device_type,
            migration_backend=engine_config.migration_backend,
            role=engine_config.role,
            # reserve 1 blocks for dummy input and padding
            num_reserved_gpu_blocks=1)
        return cache_config

    @staticmethod
    def build_backend_config(engine_config: PytorchEngineConfig):
        """Build backend config."""
        backend_config = BackendConfig(
            eager_mode=engine_config.eager_mode,
            device_type=engine_config.device_type,
        )
        return backend_config

    @staticmethod
    def build_dist_config(engine_config: PytorchEngineConfig):
        """Build dist config."""
        dist_config = DistConfig.from_engine_config(engine_config=engine_config)
        return dist_config

    @staticmethod
    def build_misc_config(engine_config: PytorchEngineConfig):
        """Build misc config."""
        misc_config = MiscConfig.from_engine_config(engine_config)
        return misc_config

    @staticmethod
    def build_specdecode_config(target_model, speculative_config: SpeculativeConfig, engine_config: PytorchEngineConfig,
                                cache_config: CacheConfig):
        """Build spec decode config."""
        specdecode_config = None
        if speculative_config is not None:
            draft_model = speculative_config.model
            if draft_model and not os.path.exists(speculative_config.model):
                draft_model = get_model(draft_model, engine_config.download_dir, engine_config.revision)

            specdecode_config = SpecDecodeConfig.from_config(
                method=speculative_config.method,
                num_speculative_tokens=speculative_config.num_speculative_tokens,
                model=draft_model,
                target_model=target_model,
                target_cache_cfg=cache_config,
                dtype=engine_config.dtype,
            )
        return specdecode_config
