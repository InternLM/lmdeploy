# Copyright (c) OpenMMLab. All rights reserved.

from .base import BaseChecker


class DistChecker(BaseChecker):
    """check dist environment."""

    def __init__(self, tp: int, dp: int, nproc_per_node: int, logger=None):
        super().__init__(logger)
        self.tp = tp
        self.dp = dp
        self.nproc_per_node = nproc_per_node

    def check(self):
        """check."""

        tp = self.tp
        dp = self.dp
        world_size = tp * dp
        nproc_per_node = self.nproc_per_node
        if nproc_per_node is None:
            nproc_per_node = world_size

        if world_size % nproc_per_node != 0:
            self.log_and_exit(
                mod_name='Dist',
                message=f'world_size={world_size} cannot be evenly divided by nproc_per_node={nproc_per_node}.')

        nnodes = world_size // nproc_per_node
        if nnodes > 1:
            try:
                import ray  # noqa: F401
            except BaseException:
                self.log_and_exit(mod_name='Dist', message='Multi-nodes support requires `ray`.')
