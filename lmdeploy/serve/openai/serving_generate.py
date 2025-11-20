# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from .protocol import GenerateReqInput

if TYPE_CHECKING:
    from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig


def check_request(request: GenerateReqInput, engine_config: 'TurbomindEngineConfig | PytorchEngineConfig') -> str:
    if not isinstance(request, GenerateReqInput):
        raise TypeError(f'Invalid request type, expected GenerateReqInput, got {type(request)}')

    # Check logprobs settings
    try:
        logprobs_mode = engine_config.logprobs_mode
        return_logprob = request.return_logprob
        if logprobs_mode is None and return_logprob:
            return f'return_logprob({return_logprob}) requested but not enabled logprobs_mode in engine configuration.'
    except AttributeError:
        pass

    if (request.prompt is not None) ^ (request.input_ids is None):
        return 'You must specify exactly one of prompt or input_ids'

    # check sampling settings
    if not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''
