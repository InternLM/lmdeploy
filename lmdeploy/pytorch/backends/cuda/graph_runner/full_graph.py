# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from packaging import version
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.envs import enable_decode_torch_compile, fake_capture
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.model_inputs import StepContextManager

logger = get_logger('lmdeploy')


@functools.lru_cache
def _configure_decode_torch_compile():
    """Configure the Dynamo policy used by decode compilation."""
    from torch._dynamo import config, trace_rules

    # Decode capture intentionally specializes fixed batch buckets. Keep those
    # specializations compiled instead of falling back to eager at the default
    # recompile limit.
    config.recompile_limit = 1024
    config.accumulated_recompile_limit = 1024

    # Dynamo otherwise enters user Triton launchers and may fail on mutable
    # autotune state or infer incorrect fake shapes. The outer CUDA graph still
    # captures these eager kernel launches with the compiled regions around them.
    trace_rules.add('lmdeploy.pytorch.kernels')
    trace_rules.add('lmdeploy.pytorch.third_party.flash_attn_interface')
    trace_rules.add('flash_attn_interface')
    trace_rules.add('triton')


def _get_decode_torch_compile_options(config) -> dict[str, bool]:
    """Build options supported by the installed Inductor version."""
    options = {
        'emulate_precision_casts': True,
        'triton.cudagraphs': False,
    }

    # Division-rounding emulation was added in PyTorch 2.10 and renamed in
    # PyTorch 2.11. Inductor rejects unknown options instead of ignoring them.
    if hasattr(config, 'emulate_divison_rounding'):
        options['emulate_divison_rounding'] = True
    elif hasattr(config, 'eager_numerics') and hasattr(config.eager_numerics, 'division_rounding'):
        options['eager_numerics.division_rounding'] = True

    return options


def build_decode_model_forward(model: torch.nn.Module) -> Callable[..., Any]:
    """Build the model callable used only while capturing decode graphs."""
    if not enable_decode_torch_compile:
        return model

    if version.parse(torch.__version__) < version.parse('2.8'):
        logger.warning(f'Decode torch.compile requires PyTorch >= 2.8, but found {torch.__version__}; '
                       'using the raw model.')
        return model

    logger.info('Enabling torch.compile for decode CUDA graph capture.')
    _configure_decode_torch_compile()
    from torch._inductor import config
    return torch.compile(
        model,
        fullgraph=False,
        dynamic=False,
        options=_get_decode_torch_compile_options(config),
    )


def _make_graph_meta(
    model_config: ModelConfig,
    ctx_mgr: StepContextManager,
    *,
    max_batches: int,
    max_tokens: int,
    num_blocks: int,
    is_decoding: bool,
    decode_query_len: int,
    device: torch.device,
) -> CudaGraphMeta:
    """Build the fixed metadata owned by one full CUDA graph."""
    step_meta_plan = ctx_mgr.backend_step_meta_plan
    if step_meta_plan is not None and not step_meta_plan.is_supported:
        step_meta_plan = None

    return CudaGraphMeta(
        max_batchs=max_batches,
        max_tokens=max_tokens,
        num_blocks=num_blocks,
        is_decoding=is_decoding,
        device=device,
        input_buffers=dict(),
        output_buffers=dict(),
        vocab_size=model_config.vocab_size,
        use_mla_fp8_cache=model_config.use_mla_fp8_cache,
        use_flash_mla=model_config.use_flash_mla,
        mla_index_topk=model_config.mla_index_topk,
        use_fa3_decoding=(model_config.model_paradigm == 'ar_spec' and not model_config.use_flash_mla),
        is_ssm=bool(model_config.states_shapes),
        use_mrope=model_config.use_mrope,
        block_size=model_config.block_size,
        decode_query_len=decode_query_len,
        step_meta_plan=step_meta_plan,
    )


class CUDASingleGraphRunner:
    """Own capture and replay state for one full CUDA graph."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        decode_query_len: int,
        pool: tuple[int, int],
        model_config: ModelConfig,
        device: torch.device,
        model_forward: Callable[..., Any] | None = None,
    ):
        self.model = model
        self._model_forward = model if model_forward is None else model_forward
        self._ctx_mgr = model.ctx_mgr
        self.meta = _make_graph_meta(
            model_config,
            self._ctx_mgr,
            max_batches=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            decode_query_len=decode_query_len,
            device=device,
        )
        self._pool = pool
        self._graph: torch.cuda.CUDAGraph | None = None
        self._use_graph = not fake_capture
        logger.info(f'Initialized CUDASingleGraphRunner with max_batches={max_batches}, max_tokens={max_tokens}, '
                    f'num_blocks={num_blocks}, is_decoding={is_decoding}, use_graph={self._use_graph}')

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Allocate stable buffers, warm up, and capture the model forward."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self._bind_inputs(**kwargs)
        capture_stream = torch.cuda.current_stream() if self._use_graph else None

        # warmup
        warmup_output = self._model_forward(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        if self._use_graph:
            assert capture_stream is not None
            output = self._capture_model(padded_kwargs, capture_stream)
        else:
            output = warmup_output

        self.meta.output_buffers = self.model.make_output_buffers(output)
        return self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """Refill stable buffers and replay the captured model forward."""
        padded_kwargs = self._bind_inputs(**kwargs)
        if self._use_graph:
            assert self._graph is not None
            self._graph.replay()
            output_buffers = self.meta.output_buffers
        else:
            output = self._model_forward(**padded_kwargs)
            output_buffers = self.model.make_output_buffers(output)
        return self.model.get_outputs_cudagraph(output_buffers, **kwargs)

    def _bind_inputs(self, **kwargs) -> dict[str, Any]:
        """Fill stable buffers and bind the current step context to them."""
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        self.model.update_context_cudagraph(self.meta, self._ctx_mgr.current_context())
        return padded_kwargs

    def _capture_model(self, padded_kwargs: dict[str, Any], capture_stream: torch.cuda.Stream):
        """Capture the prepared model call into this runner's full graph."""
        step_meta_plan = self.meta.step_meta_plan
        if step_meta_plan is not None:
            assert self.meta.step_meta_buffers is not None
            step_meta_plan.prepare_cudagraph_capture(
                self.meta,
                self.meta.input_buffers,
                self._ctx_mgr.current_context(),
                self.meta.step_meta_buffers,
                padded_kwargs['attn_metadata'],
            )

        self._graph = torch.cuda.CUDAGraph()
        # CUDA work in another thread must not invalidate this capture.
        with torch.cuda.graph(self._graph,
                              pool=self._pool,
                              stream=capture_stream,
                              capture_error_mode='thread_local'):
            return self._model_forward(**padded_kwargs)

    def __del__(self):
        """Release the captured graph before the remaining runner state."""
        self._graph = None
