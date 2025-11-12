# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from .protocol import CompletionRequest

if TYPE_CHECKING:
    from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig


def check_request(request: CompletionRequest, engine_config: 'TurbomindEngineConfig | PytorchEngineConfig') -> str:
    if not isinstance(request, CompletionRequest):
        raise TypeError(f'Invalid request type, expected CompletionRequest, got {type(request)}')

    # Check logprobs settings
    try:
        logprobs_mode = engine_config.logprobs_mode
        logprobs = request.logprobs
        if logprobs_mode is None and logprobs:
            return f'logprobs({logprobs}) requested but not enabled logprobs_mode in engine configuration.'
    except AttributeError:
        pass

    # check sampling settings
    if request.n <= 0:
        return f'The n {request.n!r} must be a positive int.'
    if not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''
