# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole

from .protocol import GenerateReqInput

if TYPE_CHECKING:
    from .api_server import VariableInterface


def check_request(request: GenerateReqInput, server_context: 'VariableInterface') -> str:
    engine_config = server_context.get_engine_config()
    session_manager = server_context.get_session_manager()
    logprobs_mode = getattr(engine_config, 'logprobs_mode', None)
    return_logprob = request.return_logprob
    top_logprobs_num = request.top_logprobs_num or 0
    if top_logprobs_num < 0:
        return f'The top_logprobs_num {request.top_logprobs_num!r} cannot be a negative integer.'
    if top_logprobs_num > 0 and return_logprob is not True:
        return 'top_logprobs_num requires return_logprob=True.'
    if hasattr(engine_config, 'logprobs_mode') and logprobs_mode is None and return_logprob:
        return f'return_logprob({return_logprob}) requested but not enabled logprobs_mode in engine configuration.'

    if (request.prompt is not None) ^ (request.input_ids is None):
        return 'You must specify exactly one of prompt or input_ids'

    if request.prompt is not None and request.prompt == '':
        return 'The prompt must not be an empty string'

    if request.input_ids is not None and len(request.input_ids) == 0:
        return 'The input_ids must not be an empty list'

    if request.max_tokens is not None and request.max_tokens < 0:
        return f'The max_tokens {request.max_tokens!r} must be non-negative.'

    # Check for input-logprob specific requirements
    if request.logprob_start_len >= 0:
        if not isinstance(engine_config, PytorchEngineConfig) or engine_config.role != EngineRole.Hybrid:
            return 'input logprobs are supported only by the PyTorch hybrid engine.'
        if getattr(getattr(server_context, 'async_engine', None), 'speculative_config', None) is not None:
            return 'logprob_start_len is not supported with speculative decoding.'
        if return_logprob is not True:
            return 'logprob_start_len requires return_logprob=True.'
        if logprobs_mode not in ('raw_logits', 'raw_logprobs'):
            return 'logprob_start_len requires raw_logits or raw_logprobs mode.'
        if request.max_tokens != 0:
            return 'logprob_start_len requires max_tokens=0.'
    elif request.max_tokens is not None and request.max_tokens == 0:
        return f'The max_tokens {request.max_tokens!r} must be a positive integer.'
    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k is not None and request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''
