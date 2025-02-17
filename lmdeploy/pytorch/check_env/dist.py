# Copyright (c) OpenMMLab. All rights reserved.

from .base import BaseChecker


class DistChecker(BaseChecker):
    """check dist environment."""

    def __init__(self, tp: int, dp: int, distributed_executor_backend: str, device_type: str, logger=None):
        super().__init__(logger)
        self.world_size = tp * dp
        self.distributed_executor_backend = distributed_executor_backend
        self.device_type = device_type

    def check(self):
        """check."""
        distributed_executor_backend = self.distributed_executor_backend

        if distributed_executor_backend is None:
            from lmdeploy.pytorch.engine.executor import get_distributed_executor_backend
            distributed_executor_backend = get_distributed_executor_backend(self.world_size, self.device_type)

        if distributed_executor_backend not in [None, 'mp', 'ray']:
            self.log_and_exit(mod_name='Dist',
                              message=f'Unsupported distributed_executor_backend: {distributed_executor_backend}')

        if distributed_executor_backend == 'ray':
            try:
                import ray  # noqa: F401
            except BaseException:
                self.log_and_exit(mod_name='Dist', message='Multi-nodes support requires `ray`.')

            from lmdeploy.pytorch.backends import get_backend
            backend = get_backend(self.device_type)
            if not backend.support_ray():
                self.log_and_exit(mod_name='Dist', message=f'device={self.device_type} does not support ray.')
