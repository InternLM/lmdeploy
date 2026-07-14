# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class AutoModelConfigBuilder(ABC):

    _sub_classes = list()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        AutoModelConfigBuilder.register_builder(cls)

    @classmethod
    def register_builder(cls, sub_cls):
        """Register builder."""
        if sub_cls not in AutoModelConfigBuilder._sub_classes:
            AutoModelConfigBuilder._sub_classes.append(sub_cls)

    @classmethod
    def condition(cls, hf_config):
        """config."""
        raise NotImplementedError(f'`condition` of {cls.__name__} not implemented.')

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        from .default import DefaultModelConfigBuilder

        if cls != AutoModelConfigBuilder:
            raise NotImplementedError(f'`build` of {cls.__name__} not implemented.')

        valid_builder = DefaultModelConfigBuilder
        for builder in cls._sub_classes:
            if builder == valid_builder:
                continue

            if builder.condition(hf_config):
                valid_builder = builder
                break

        logger.debug(f'build model config with {valid_builder.__name__}')

        cfg = valid_builder.build(hf_config, model_path, **kwargs)
        if cfg.hf_config is None:
            cfg.hf_config = hf_config
        if cfg.llm_config is None:
            cfg.llm_config = hf_config

        return cfg

    @classmethod
    def update_num_kv_heads(cls, hf_config, tp, num_key_value_heads):
        """Update num kv heads."""
        # update num_kv_heads for tp mode
        if tp > 1 and tp > num_key_value_heads:
            assert tp % num_key_value_heads == 0
            n_replicate = tp // num_key_value_heads
            hf_config.num_replicate_key_value_heads = n_replicate
            num_key_value_heads = tp

        hf_config.num_key_value_heads = num_key_value_heads
        return num_key_value_heads
