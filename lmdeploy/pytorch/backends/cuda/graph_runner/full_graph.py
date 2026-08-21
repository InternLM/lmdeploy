# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch
from packaging import version
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.envs import enable_decode_torch_compile, fake_capture
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

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


class CUDASingleGraphRunner:
    """Cuda single graph runner."""

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
        self.model_forward = model if model_forward is None else model_forward
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config
        step_meta_plan = getattr(self.ctx_mgr, 'backend_step_meta_plan', None)
        if not getattr(step_meta_plan, 'is_supported', False):
            step_meta_plan = None

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
            use_mla_fp8_cache=getattr(self.model_config, 'use_mla_fp8_cache', False),
            use_flash_mla=getattr(self.model_config, 'use_flash_mla', False),
            mla_index_topk=getattr(self.model_config, 'mla_index_topk', None),
            use_fa3_decoding=(model_config.model_paradigm == 'ar_spec'
                              and not getattr(model_config, 'use_flash_mla', False)),
            is_ssm=len(model_config.states_shapes) > 0,
            use_mrope=model_config.use_mrope,
            block_size=model_config.block_size,
            decode_query_len=decode_query_len,
            step_meta_plan=step_meta_plan,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None
        self.USE_GRAPH = not fake_capture
        logger.info(f'Initialized CUDASingleGraphRunner with max_batches={max_batches}, max_tokens={max_tokens}, '
                    f'num_blocks={num_blocks}, is_decoding={is_decoding}, use_graph={self.USE_GRAPH}')

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        warmup_output = self.model_forward(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        if self.USE_GRAPH:
            step_meta_plan = self.meta.step_meta_plan
            if step_meta_plan is not None:
                step_ctx = self.ctx_mgr.current_context()
                assert self.meta.step_meta_buffers is not None
                step_meta_plan.prepare_cudagraph_capture(
                    self.meta,
                    self.meta.input_buffers,
                    step_ctx,
                    self.meta.step_meta_buffers,
                    padded_kwargs['attn_metadata'],
                )

            self._graph = torch.cuda.CUDAGraph()
            # unsafe kernel call in other thread might invalid the capture
            # so we set thread_safe capture mode here.
            with torch.cuda.graph(self._graph,
                                  pool=self.pool,
                                  stream=current_stream,
                                  capture_error_mode='thread_local'):
                output = self.model_forward(**padded_kwargs)
        else:
            output = warmup_output

        output_buffers = self.model.make_output_buffers(output)
        self.meta.output_buffers = output_buffers
        output = self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        if self.USE_GRAPH:
            assert self._graph is not None
            self._graph.replay()
            output_buffers = self.meta.output_buffers
        else:
            output = self.model_forward(**padded_kwargs)
            output_buffers = self.model.make_output_buffers(output)
        output = self.model.get_outputs_cudagraph(output_buffers, **kwargs)
        return output

    def __del__(self):
        """del."""
        del self._graph
