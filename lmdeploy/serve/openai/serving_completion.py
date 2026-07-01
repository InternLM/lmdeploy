# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.serve.core.generation_config import build_generation_config

from .protocol import CompletionRequest

if TYPE_CHECKING:
    from .api_server import VariableInterface


def _effective_sampling(request: CompletionRequest, server_context: 'VariableInterface') -> dict:
    gen = build_generation_config(request, server_context.default_gen_config)
    return {
        'temperature': gen.temperature,
        'top_p': gen.top_p,
        'top_k': gen.top_k,
    }


def check_request(request: CompletionRequest, server_context: 'VariableInterface') -> str:
    engine_config = server_context.get_engine_config()
    session_manager = server_context.get_session_manager()
    try:
        # Check logprobs settings
        logprobs_mode = engine_config.logprobs_mode
        logprobs = request.logprobs or 0
        if logprobs > 0 and logprobs_mode is None:
            return f'logprobs({logprobs}) requested but not enabled logprobs_mode in engine configuration.'
        if logprobs_mode is not None and logprobs < 0:
            return 'logprobs must be non-negative when logprobs_mode is enabled in engine configuration.'
    except AttributeError:
        pass

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    sampling = _effective_sampling(request, server_context)

    # check sampling settings
    if request.n <= 0:
        return f'The n {request.n!r} must be a positive int.'
    if not (0 < sampling['top_p'] <= 1):
        return f'The top_p {sampling["top_p"]!r} must be in (0, 1].'
    if sampling['top_k'] < 0:
        return f'The top_k {sampling["top_k"]!r} cannot be a negative integer.'
    if not (0 <= sampling['temperature'] <= 2):
        return f'The temperature {sampling["temperature"]!r} must be in [0, 2]'

    return ''
