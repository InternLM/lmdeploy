# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.engine_checker import EngineChecker


def _build_checker(engine_config: PytorchEngineConfig):
    checker = object.__new__(EngineChecker)
    checker.engine_config = engine_config

    def log_and_raise(*args, **kwargs):
        raise RuntimeError(kwargs.get('message'))

    checker.log_and_exit = log_and_raise
    return checker


def test_engine_checker_rejects_split_kernel_blocks_for_pd_migration():
    engine_config = PytorchEngineConfig(max_batch_size=1,
                                        role=EngineRole.Prefill,
                                        block_size=64,
                                        kernel_block_size=32)
    checker = _build_checker(engine_config)

    with pytest.raises(RuntimeError, match='PD migration does not support block_size != kernel_block_size'):
        checker.check()


def test_engine_checker_allows_split_kernel_blocks_for_hybrid_engine():
    engine_config = PytorchEngineConfig(max_batch_size=1,
                                        role=EngineRole.Hybrid,
                                        block_size=64,
                                        kernel_block_size=32)
    checker = _build_checker(engine_config)

    checker.check()
