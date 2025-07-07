# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.messages import PytorchEngineConfig

from ..check_env.adapter import AdapterChecker
from ..check_env.base import BaseChecker
from ..check_env.dist import DistChecker
from ..check_env.model import ModelChecker
from ..check_env.torch import TorchChecker
from ..check_env.transformers import TransformersChecker


class EngineChecker(BaseChecker):
    """Check transformers is available."""

    def __init__(self,
                 model_path: str,
                 engine_config: PytorchEngineConfig,
                 trust_remote_code: bool = True,
                 logger=None):
        super().__init__(logger)
        logger = self.get_logger()

        self.engine_config = engine_config

        dtype = engine_config.dtype
        device_type = engine_config.device_type

        # pytorch
        torch_checker = TorchChecker(logger=logger)

        if device_type == 'cuda':
            # triton
            from ..check_env.cuda import CudaChecker
            from ..check_env.triton import TritonChecker
            cuda_checker = CudaChecker(model_format=engine_config.model_format, logger=logger)
            cuda_checker.register_required_checker(torch_checker)
            triton_checker = TritonChecker(logger=logger)
            triton_checker.register_required_checker(cuda_checker)
            self.register_required_checker(triton_checker)
        else:
            # deeplink
            from ..check_env.deeplink import DeeplinkChecker
            dl_checker = DeeplinkChecker(device_type, logger=logger)
            self.register_required_checker(dl_checker)
            self.register_required_checker(torch_checker)

        # transformers

        # model
        trans_checker = TransformersChecker()
        model_checker = ModelChecker(model_path=model_path,
                                     trust_remote_code=trust_remote_code,
                                     dtype=dtype,
                                     device_type=device_type,
                                     logger=logger)
        model_checker.register_required_checker(torch_checker)
        model_checker.register_required_checker(trans_checker)
        self.register_required_checker(model_checker)

        # adapters
        adapters = engine_config.adapters
        if adapters is not None:
            adapter_paths = list(adapters.values())
            for adapter in adapter_paths:
                adapter_checker = AdapterChecker(adapter, logger=logger)
                self.register_required_checker(adapter_checker)

        # dist
        dist_checker = DistChecker(engine_config.tp,
                                   engine_config.dp,
                                   engine_config.ep,
                                   engine_config.distributed_executor_backend,
                                   device_type=engine_config.device_type,
                                   logger=logger)
        self.register_required_checker(dist_checker)

    def check(self):
        """check."""
        engine_config = self.engine_config

        if engine_config.thread_safe:
            self.log_and_exit(
                mod_name='Engine',
                message='thread safe mode is no longer supported.\n'
                'Read https://github.com/InternLM/lmdeploy/blob/main/docs/en/advance/pytorch_multithread.md for more details.',  # noqa: E501
            )

        if engine_config.max_batch_size <= 0:
            self.log_and_exit(mod_name='Engine',
                              message='max_batch_size should be'
                              f' greater than 0, but got {engine_config.max_batch_size}')
