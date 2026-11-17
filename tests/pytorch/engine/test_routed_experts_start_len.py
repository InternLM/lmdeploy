# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from lmdeploy.messages import GenerationConfig, ResponseType
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, GenerateReqInput


@pytest.mark.parametrize('start', [0, 3, 8, 11])
@pytest.mark.parametrize('finish', [ResponseType.FINISH, ResponseType.CANCEL])
@pytest.mark.parametrize('shared_store', [False, True])
def test_crop_only_returned_routes(start, finish, shared_store):
    routes = np.arange(44, dtype=np.uint16).reshape(11, 2, 2)
    original = routes.copy()
    resp = SimpleNamespace(type=finish, data={'routed_experts': routes})
    instance = SimpleNamespace(_enable_transfer_obj_ref=shared_store)
    store = MagicMock()
    store.put.remote.return_value = 'object-ref'
    with patch('lmdeploy.pytorch.engine.engine_instance._SHARED_STORE', store, create=True), \
            patch('ray.get', return_value='key'):
        out = EngineInstance._get_extra_outputs(instance, resp, 12, start)
    if shared_store:
        np.testing.assert_array_equal(store.put.remote.call_args.args[0], routes[start:])
        assert out['routed_experts'] == 'key'
    else:
        np.testing.assert_array_equal(out['routed_experts'], routes[start:])
    # Scheduler / BlockTrie full history is not mutated by the response slice.
    np.testing.assert_array_equal(resp.data['routed_experts'], original)


@pytest.mark.parametrize('kind', [ResponseType.SUCCESS, ResponseType.FINISH, ResponseType.CANCEL])
def test_missing_routes_are_not_fabricated(kind):
    resp = SimpleNamespace(type=kind, data={'routed_experts': None})
    assert EngineInstance._get_extra_outputs(SimpleNamespace(), resp, 8, 5) == {'routed_experts': None}


@pytest.mark.parametrize('bad', [-1, 1.5, True, '2'])
def test_python_config_rejects_invalid_start(bad):
    with pytest.raises(ValueError, match='routed_experts_start_len'):
        GenerationConfig(routed_experts_start_len=bad)


@pytest.mark.parametrize('request_type', [GenerateReqInput, ChatCompletionRequest])
@pytest.mark.parametrize('start', [0, 4])
def test_api_to_generation_config(request_type, start):
    request = request_type(model='test', messages=[], input_ids=list(range(8)), routed_experts_start_len=start)
    config = build_generation_config(request, {}, max_new_tokens=4)
    assert config.routed_experts_start_len == start


@pytest.mark.parametrize('request_type', [GenerateReqInput, ChatCompletionRequest])
@pytest.mark.parametrize('bad', [-1, 1.5, True, '2'])
def test_api_rejects_invalid_start(request_type, bad):
    with pytest.raises(ValidationError):
        request_type(model='test', messages=[], input_ids=[1, 2], routed_experts_start_len=bad)


def test_chat_extra_body_start():
    request = ChatCompletionRequest(model='test', messages=[], input_ids=[1, 2],
                                    extra_body={'routed_experts_start_len': 2})
    assert build_generation_config(request, {}, max_new_tokens=1).routed_experts_start_len == 2


def test_direct_engine_rejects_out_of_prompt_before_submission():
    instance = SimpleNamespace(max_input_len=32)
    iterator = EngineInstance.async_stream_infer(instance, 1, [1, 2],
                                                GenerationConfig(routed_experts_start_len=3))
    with pytest.raises(ValueError, match='prompt length'):
        asyncio.run(anext(iterator))


@pytest.mark.parametrize('speculative', [False, True])
def test_crop_after_valid_target_route_selection(speculative):
    from lmdeploy.pytorch.messages import HistoryRouterExperts, SchedulerSequence
    from lmdeploy.pytorch.strategies.ar_spec.sequence import SchedulerSequenceARSpec

    routes = np.arange(56, dtype=np.uint16).reshape(14, 2, 2)
    history = HistoryRouterExperts()
    history.append(routes)
    sequence = SimpleNamespace(return_routed_experts=True, all_routed_experts=history, num_valid_ids=9)
    sequence_type = SchedulerSequenceARSpec if speculative else SchedulerSequence
    target_routes = sequence_type.routed_experts.fget(sequence)
    assert len(target_routes) == 8  # Draft / unaccepted tail rows are already excluded.
    resp = SimpleNamespace(type=ResponseType.FINISH, data={'routed_experts': target_routes})
    out = EngineInstance._get_extra_outputs(SimpleNamespace(_enable_transfer_obj_ref=False), resp, 9, 5)
    np.testing.assert_array_equal(out['routed_experts'], routes[5:8])
    np.testing.assert_array_equal(history.get_real(), routes)
