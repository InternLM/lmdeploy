# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.profiler import record_function

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def try_reuse_input_buffers(args: Tuple[List], kwargs: Dict[str, Any], input_buffers: List[torch.Tensor]) -> Any:
    """Try reuse input buffers for cudagraph capture.

    Note that this might not safe if there are overlaps between tensors.
    following tensors will not be reused:
    1. nn.Parameter and nn.Buffer
    2. non-tensor inputs
    3. tensors marked as cudagraph output
    """
    from lmdeploy.pytorch import envs
    if envs.disable_graph_buffer_reuse:
        return args, kwargs

    args = list(args)
    # make input dict
    input_args = {}
    for i, arg in enumerate(args):
        if isinstance(arg, (nn.Parameter, nn.Buffer)):
            # skip parameter and buffer
            continue
        if not isinstance(arg, torch.Tensor):
            # skip non-tensor
            continue
        input_args[i] = arg

    for k, arg in kwargs.items():
        if isinstance(arg, (nn.Parameter, nn.Buffer)):
            # skip parameter and buffer
            continue
        if not isinstance(arg, torch.Tensor):
            # skip non-tensor
            continue
        input_args[k] = arg

    buf_ptr_dict = {buf.data_ptr(): buf for buf in input_buffers}
    keys = list(input_args.keys())
    # filter out tensors that are already in buffers
    for key in keys:
        arg = input_args[key]
        if getattr(arg, 'is_cudagraph_output', False):
            # do not reuse cudagraph output
            input_args.pop(key)
        arg_ptr = arg.data_ptr()
        if arg_ptr in buf_ptr_dict:
            buf = buf_ptr_dict[arg_ptr]
            if arg.shape == buf.shape and arg.dtype == buf.dtype and arg.stride() == buf.stride():
                # arg already in buffer
                input_args.pop(key)
                buf_ptr_dict.pop(arg_ptr)

    # reuse tensor
    # we will only reuse tensors with same shape for now.
    buf_shape_dict = {(buf.shape, buf.stride(), buf.dtype): buf for buf in buf_ptr_dict.values()}
    keys = list(input_args.keys())
    for key in keys:
        arg = input_args[key]
        shape_key = (arg.shape, arg.stride(), arg.dtype)
        if shape_key in buf_shape_dict:
            buf = buf_shape_dict[shape_key]
            buf.copy_(arg)
            # reuse buffer
            if isinstance(key, int):
                args[key] = buf
            else:
                kwargs[key] = buf
            # remove used buffer
            buf_shape_dict.pop(shape_key)
            input_args.pop(key)

    # add remaining buffers to input_buffers
    for buf in input_args.values():
        input_buffers.append(buf)

    return tuple(args), kwargs


def _copy_buffer(src: torch.Tensor, dst: torch.Tensor):
    if isinstance(src, nn.Parameter):
        return
    if isinstance(src, nn.Buffer):
        return
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        if src.data_ptr() != dst.data_ptr():
            dst.copy_(src)


def _mark_is_cudagraph_output(tensor: torch.Tensor):
    """Mark tensor as cudagraph output."""
    tensor.is_cudagraph_output = True


def _mark_outputs_cudagraph(outputs: Any):
    if isinstance(outputs, torch.Tensor):
        _mark_is_cudagraph_output(outputs)
    elif isinstance(outputs, (List, Tuple)):
        for out in outputs:
            if isinstance(out, torch.Tensor):
                _mark_is_cudagraph_output(out)
    elif isinstance(outputs, Dict):
        for out in outputs.values():
            if isinstance(out, torch.Tensor):
                _mark_is_cudagraph_output(out)


class CudagraphPiecewiseWrapperImpl:
    """Wrapper to force eager execution.

    Features:
    - Wraps operations or modules to always execute in eager mode
    - Temporarily disables graph mode internally even when outer enable_graph_mode=True
    """

    def __init__(self, runnable: Callable, is_first_graph: bool, is_last_graph: bool, pool: Tuple[int, int],
                 input_buffers: List[torch.Tensor]):
        super().__init__()
        self.runnable = runnable
        self.is_first_graph = is_first_graph
        self.is_last_graph = is_last_graph

        self.input_args = []
        self.input_kwargs = {}
        self.outputs = None
        self.input_buffers = input_buffers

        # cudagraph
        self._pool = pool
        self._graph: torch.cuda.CUDAGraph = None

    @record_function('piecewise_graph_capture')
    def _capture(self, *args, **kwargs) -> Any:
        logger.debug('Capture graph')

        args, kwargs = try_reuse_input_buffers(args, kwargs, self.input_buffers)

        # warmup
        with record_function('piecewise_graph_warmup'):
            warmup_output = self.runnable(*args, **kwargs)

        self._graph = torch.cuda.CUDAGraph()
        current_stream = torch.cuda.current_stream()
        with torch.cuda.graph(self._graph, pool=self._pool, stream=current_stream, capture_error_mode='thread_local'):
            outputs = self.runnable(*args, **kwargs)

        self.input_args = args
        self.input_kwargs = kwargs
        self.outputs = outputs

        _mark_outputs_cudagraph(outputs)

        return warmup_output

    def _copy_inputs(self, *args, **kwargs):
        for arg, arg_buffer in zip(args, self.input_args):
            _copy_buffer(arg, arg_buffer)

        for k, arg in kwargs.items():
            if k not in self.input_kwargs:
                continue
            arg_buffer = self.input_kwargs[k]
            _copy_buffer(arg, arg_buffer)

    @record_function('piecewise_graph_forward')
    def forward(self, *args, **kwargs) -> Any:
        if self._graph is None:
            self._capture(*args, **kwargs)
            self._graph.replay()
            return self.outputs

        self._copy_inputs(*args, **kwargs)

        self._graph.replay()
        return self.outputs

    def __repr__(self):
        return f'CudagraphPiecewiseWrapper(is_first={self.is_first_graph}, is_last={self.is_last_graph})'

    def __del__(self):
        if self._graph is not None:
            del self._graph


class CudagraphPiecewiseWrapper(nn.Module):
    """Wrapper to force eager execution.

    Features:
    - Wraps operations or modules to always execute in eager mode
    - Temporarily disables graph mode internally even when outer enable_graph_mode=True
    """

    def __init__(self, runnable: Callable, is_first_graph: bool, is_last_graph: bool, pool: Tuple[int, int],
                 input_buffers: Dict[Any, List[torch.Tensor]], backend: Any):
        super().__init__()
        self.runnable = runnable
        self.is_first_graph = is_first_graph
        self.is_last_graph = is_last_graph

        self.input_args = []
        self.input_kwargs = {}
        self.outputs = None
        self.input_buffers = input_buffers
        self.backend = backend

        self._pool = pool

        self.impl_map: Dict[Any, CudagraphPiecewiseWrapperImpl] = dict()

    def forward(self, *args, **kwargs) -> Any:
        key = self.backend.get_key()
        if key not in self.impl_map:
            buffers = self.input_buffers[key]
            wrapper_impl = CudagraphPiecewiseWrapperImpl(
                self.runnable,
                self.is_first_graph,
                self.is_last_graph,
                pool=self._pool,
                input_buffers=buffers,
            )
            self.impl_map[key] = wrapper_impl

        wrapper_impl = self.impl_map[key]

        return wrapper_impl.forward(*args, **kwargs)
