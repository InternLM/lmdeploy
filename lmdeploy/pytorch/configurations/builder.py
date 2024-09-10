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
        """register builder."""
        if sub_cls not in AutoModelConfigBuilder._sub_classes:
            AutoModelConfigBuilder._sub_classes.append(sub_cls)

    @classmethod
    def condition(cls, hf_config):
        """config."""
        raise NotImplementedError(
            f'`condition` of {cls.__name__} not implemented.')

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        from .default import DefaultModelConfigBuilder

        if cls != AutoModelConfigBuilder:
            raise NotImplementedError(
                f'`build` of {cls.__name__} not implemented.')

        valid_builder = DefaultModelConfigBuilder
        for builder in cls._sub_classes:
            if builder == valid_builder:
                continue

            if builder.condition(hf_config):
                valid_builder = builder
                break

        logger.debug(f'build model config with {valid_builder.__name__}')

        cfg = valid_builder.build(hf_config, model_path)
        if cfg.hf_config is None:
            cfg.hf_config = hf_config

        return cfg
