# Copyright (c) OpenMMLab. All rights reserved.
"""Utility functions for model agent.

Extracted from agent.py to reduce module size.
"""
import torch

from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.model_inputs import ModelInputs, step_ctx_manager


def msg_with_rank(rank: int, msg: str):
    """Return message with rank."""
    return f'rank[{rank}] - {msg}'


def cache_swapping(cache_engine: CacheEngine, swap_in_map: dict, swap_out_map: dict):
    """Perform cache swapping."""
    issued_cache_op = False
    swap_in_map = swap_in_map or dict()
    swap_out_map = swap_out_map or dict()
    if len(swap_in_map) > 0:
        cache_engine.swap_in(swap_in_map)
        issued_cache_op = True
    if len(swap_out_map) > 0:
        cache_engine.swap_out(swap_out_map)
        issued_cache_op = True

    if issued_cache_op:
        cache_engine.events.wait()


@torch.inference_mode()
def model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    cache_engine: CacheEngine,
    state_cache_engine: StateCacheEngine,
    stream: torch.cuda.Stream = None,
):
    """Perform model forward."""
    stream = stream or torch.cuda.current_stream()
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        ctx_mgr = model.ctx_mgr
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=cache_engine.model_config,
            cache_config=cache_engine.cache_config,
            kv_caches=cache_engine.gpu_cache,
            state_kv_caches=state_cache_engine.gpu_cache,
        )
        outputs = model.forward(inputs, context)
    return outputs