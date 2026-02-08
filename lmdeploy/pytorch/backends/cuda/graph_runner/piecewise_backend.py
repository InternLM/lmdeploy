# Copyright (c) OpenMMLab. All rights reserved.
"""Dlinfer Piecewise Backend for torch.compile.

Strategy: Separate attention operations (executed eagerly) from compute operations
(optimized with ACL Graph) using enable_graph_mode=True.
"""

from collections import defaultdict
from typing import Any, Callable, Tuple

import torch
import torch.fx as fx

from lmdeploy.utils import get_logger

from .cudagraph_wrapper import CudagraphPiecewiseWrapper
from .eager_wrapper import EagerExecutionWrapper
from .graph_splitter import split_graph

logger = get_logger('lmdeploy')

_global_graph_pool = None


def get_graph_pool() -> Any:
    """Get the global unique graph_pool instance."""
    global _global_graph_pool
    if _global_graph_pool is None:
        _global_graph_pool = torch.cuda.graph_pool_handle()
        logger.debug('Created global graph pool for shared use across instances')
    return _global_graph_pool


class LMDeployPiecewiseBackend:
    """Custom torch.compile backend for Dlinfer with piecewise graph
    optimization.

    Reference: vLLM VllmBackend (vllm/vllm/compilation/backends.py)

    Features:
    1. Receives dynamo-traced FX graphs
    2. Splits graphs by splitting operations
    3. Wraps non-attention parts with ACL Graph
    4. Returns executable split GraphModule
    """

    def __init__(self, graph_pool: Any, is_decoding: bool):
        self._compilation_count = 0
        self._graph_pool = graph_pool
        self._is_decoding = is_decoding

        self.splitting_ops = self.get_split_ops()
        self._key = None

    def get_split_ops(self) -> list[str]:
        """Get the list of splitting operations."""
        from lmdeploy.pytorch.compile_util import get_custom_op_manager

        custom_op_mgr = get_custom_op_manager()
        if self._is_decoding:
            splitting_ops = custom_op_mgr.get_split_decoding_ops()
        else:
            splitting_ops = custom_op_mgr.get_split_prefill_ops()

        splitting_ops = [op.replace('::', '.') for op in splitting_ops]
        splitting_ops = [f'{op}.default' for op in splitting_ops]

        return splitting_ops

    def __call__(
        self,
        gm: fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        fullgraph: bool = True,
        mode: str = 'reduce-overhead',
        dynamic: bool | None = None,
    ) -> Callable[..., Any]:
        """Backend entry point.

        Args:
            gm: Dynamo-traced FX graph
            example_inputs: Example inputs (fake tensors)

        Returns:
            split_gm: Executable split GraphModule
        """
        if not isinstance(gm, fx.GraphModule):
            raise TypeError('gm must be a torch.fx.GraphModule instance')
        if not example_inputs:
            raise ValueError('example_inputs cannot be empty')

        # Increment compilation count for debugging
        self._compilation_count += 1

        logger.info(f'LMDeployPiecewiseBackend called: compilation #{self._compilation_count}, '
                    f'example_inputs count: {len(example_inputs)}, '
                    f'is_decoding: {self._is_decoding}')

        input_buffers = defaultdict(list)
        try:
            logger.debug('Step 1: Splitting graph...')
            split_gm, split_items = split_graph(gm, self.splitting_ops)

            if not split_items:
                raise RuntimeError('Graph splitting produced no submodules')

            logger.debug(f'Graph split into {len(split_items)} submodules')

            # Debug Step 1: Analyze graph splitting results
            logger.debug('Step 2: Wrapping submodules...')
            for item in split_items:
                submod_name = item.submod_name
                original_submod = getattr(split_gm, submod_name)

                if item.is_splitting_graph:
                    wrapped = EagerExecutionWrapper(op_or_module=original_submod, op_name=f'eager_{submod_name}')
                else:
                    is_first = item.graph_id == 0
                    is_last = item.graph_id == len(split_items) - 1
                    wrapped = CudagraphPiecewiseWrapper(
                        runnable=original_submod,
                        is_first_graph=is_first,
                        is_last_graph=is_last,
                        pool=self._graph_pool,
                        input_buffers=input_buffers,
                        backend=self,
                    )

                    split_gm.__dict__[submod_name] = wrapped

            return split_gm

        except Exception as e:
            logger.error(f'Error in LMDeployPiecewiseBackend: {e}', exc_info=True)
            raise

    def set_key(self, key):
        self._key = key

    def get_key(self):
        return self._key

    def reset(self) -> None:
        """Reset state (for testing)."""

    def __del__(self):
        self.reset()


def create_backend(graph_pool, is_decoding: bool) -> LMDeployPiecewiseBackend:
    return LMDeployPiecewiseBackend(graph_pool, is_decoding)
