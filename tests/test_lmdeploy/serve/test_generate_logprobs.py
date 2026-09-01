# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from pydantic import ValidationError

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.openai.endpoints.generate import (
    _create_token_logprobs,
    _create_top_logprobs,
    check_request,
    register,
)
from lmdeploy.serve.openai.protocol import GenerateReqInput


def _make_generate(context):
    router = APIRouter()
    register(router, context)
    for route in router.routes:
        if getattr(route, 'path', None) == '/generate':
            return route.endpoint
    raise AssertionError('/generate route not registered')


class _SessionManager:

    def has(self, session_id):
        return False

    def get(self, *args, **kwargs):
        raise AssertionError('empty scoring response must not create a session')

    def remove(self, session):
        pass


class _ServerContext:

    def __init__(self, config, speculative_config=None, async_engine=None):
        self._session_manager = _SessionManager()
        self.async_engine = async_engine or SimpleNamespace(
            backend_config=config,
            session_mgr=self._session_manager,
            speculative_config=speculative_config)
        self.default_gen_config = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self._session_manager

    def create_session(self, session_id):
        return SimpleNamespace()


def test_generate_input_logprob_validation():
    config = PytorchEngineConfig(logprobs_mode='raw_logprobs',
                                 role=EngineRole.Hybrid)
    context = _ServerContext(config)
    valid = GenerateReqInput(input_ids=[1, 2],
                             return_logprob=True,
                             logprob_start_len=0,
                             max_tokens=0)
    assert check_request(valid, context) == ''

    assert 'return_logprob=True' in check_request(
        GenerateReqInput(input_ids=[1, 2], logprob_start_len=0), context)
    assert check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         logprob_start_len=3,
                         max_tokens=0), context) == ''
    assert check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         logprob_start_len=1,
                         max_tokens=0), context) == ''
    assert 'max_tokens=0' in check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         logprob_start_len=0), context)
    assert 'max_tokens=0' in check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         logprob_start_len=0,
                         max_tokens=5), context)
    assert 'positive integer' in check_request(
        GenerateReqInput(input_ids=[1, 2], max_tokens=0), context)
    assert 'top_logprobs_num requires return_logprob=True' in check_request(
        GenerateReqInput(input_ids=[1, 2], top_logprobs_num=2), context)
    assert 'negative integer' in check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         top_logprobs_num=-1), context)
    assert check_request(
        GenerateReqInput(prompt='hi',
                         return_logprob=True,
                         logprob_start_len=0,
                         max_tokens=0), context) == ''
    assert check_request(
        GenerateReqInput(input_ids=[1, 2],
                         image_data='https://example.com/image.png',
                         return_logprob=True,
                         logprob_start_len=0,
                         max_tokens=0), context) == ''

    assert 'not enabled logprobs_mode' in check_request(
        GenerateReqInput(prompt='Paris is the capital of',
                         max_tokens=2,
                         return_logprob=True),
        _ServerContext(PytorchEngineConfig(role=EngineRole.Hybrid)))
    assert check_request(
        GenerateReqInput(prompt='Paris is the capital of',
                         max_tokens=2,
                         return_logprob=True),
        _ServerContext(TurbomindEngineConfig())) == ''
    assert 'PyTorch hybrid' in check_request(
        valid, _ServerContext(TurbomindEngineConfig()))
    assert 'PyTorch hybrid' in check_request(
        valid, _ServerContext(SimpleNamespace(logprobs_mode='raw_logprobs')))
    assert 'speculative decoding' in check_request(
        valid, _ServerContext(config, speculative_config=SimpleNamespace(method='qwen3_5_mtp')))
    assert 'raw_logits or raw_logprobs' in check_request(
        valid,
        _ServerContext(
            PytorchEngineConfig(logprobs_mode='processed_logprobs',
                                role=EngineRole.Hybrid)))
    with pytest.raises(ValidationError):
        GenerateReqInput(input_ids=[1], logprob_start_len=-2)


def test_generate_logprob_formatting_preserves_requested_empty():
    assert _create_token_logprobs(None, None) is None
    assert _create_token_logprobs([], []) is None
    assert _create_token_logprobs(
        [4], [{4: -0.75, 8: -2.0}]) == [(-0.75, 4)]
    with pytest.raises(ValueError):
        _create_token_logprobs([4], [])

    assert _create_token_logprobs(
        [5], [{5: -0.25}]) == [(-0.25, 5)]
    assert _create_top_logprobs(None, None, 2) is None
    assert _create_top_logprobs([], [], 2) is None
    assert _create_top_logprobs([4], [{4: -0.75}], 0) is None
    assert _create_top_logprobs(
        [4], [{4: -0.75, 8: -2.0, 9: -3.0}], 3) == [[(-0.75, 4), (-2.0, 8), (-3.0, 9)]]
    assert _create_top_logprobs(
        [4], [{4: -0.75, 8: -2.0, 9: -3.0, 10: -4.0}], 3) == [[(-2.0, 8), (-3.0, 9),
                                                                 (-4.0, 10)]]
    with pytest.raises(ValueError):
        _create_top_logprobs([4], [], 2)


class _RawRequest:

    async def is_disconnected(self):
        return False


class _NonemptyEngine:

    def __init__(self,
                 expect_image=False,
                 expected_logprobs=1,
                 logprob_token_ids=None,
                 logprobs=None):
        self.backend_config = PytorchEngineConfig(
            logprobs_mode='raw_logprobs', role=EngineRole.Hybrid)
        self.session_mgr = _SessionManager()
        self.epoch = 1
        self.expect_image = expect_image
        self.expected_logprobs = expected_logprobs
        self.logprob_token_ids = logprob_token_ids or [2, 3]
        self.logprobs = logprobs or [{2: -0.25, 8: -2.0}, {3: -0.5, 9: -3.0}]

    async def preprocess(self, **kwargs):
        if self.expect_image:
            assert kwargs['input_ids'] is None
            content = kwargs['messages'][0]['content']
            assert content[0] == {'type': 'text', 'text': [1, 2, 3]}
            assert content[1] == {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://example.com/image.png'
                }
            }
        else:
            assert kwargs['input_ids'] == [1, 2, 3]
        assert kwargs['gen_config'].max_new_tokens == 0
        assert kwargs['gen_config'].logprob_start_len == 0
        assert kwargs['gen_config'].logprobs == self.expected_logprobs
        return SimpleNamespace(input_token_len=len(self.logprob_token_ids) + 1)

    async def generate(self, prepared_request, stream_response=True):
        assert prepared_request.input_token_len == len(self.logprob_token_ids) + 1
        assert stream_response is True
        yield SimpleNamespace(
            response='',
            token_ids=[],
            routed_experts=None,
            logprob_token_ids=self.logprob_token_ids,
            logprobs=self.logprobs,
            finish_reason='length',
            input_token_len=len(self.logprob_token_ids) + 1,
            generate_token_len=0,
        )


async def _call_nonempty_route(stream,
                               image_data=None,
                               top_logprobs_num=None,
                               logprob_token_ids=None,
                               logprobs=None):
    engine = _NonemptyEngine(
        expect_image=image_data is not None,
        expected_logprobs=top_logprobs_num or 1,
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs)
    context = _ServerContext(engine.backend_config, async_engine=engine)
    generate = _make_generate(context)
    request = GenerateReqInput(input_ids=[1, 2, 3],
                               image_data=image_data,
                               return_logprob=True,
                               logprob_start_len=0,
                               max_tokens=0,
                               top_logprobs_num=top_logprobs_num,
                               stream=stream)
    result = await generate(request, raw_request=_RawRequest())
    if stream:
        return ''.join([chunk async for chunk in result.body_iterator])
    return result


class _DecodeEngine:

    def __init__(self, expected_logprobs=1):
        self.backend_config = PytorchEngineConfig(
            logprobs_mode='raw_logprobs',
            role=EngineRole.Hybrid,
            enable_return_routed_experts=True)
        self.session_mgr = _SessionManager()
        self.epoch = 1
        self.expected_logprobs = expected_logprobs

    async def preprocess(self, **kwargs):
        assert kwargs['input_ids'] == [1, 2]
        assert kwargs['gen_config'].max_new_tokens == 2
        assert kwargs['gen_config'].logprob_start_len == -1
        assert kwargs['gen_config'].logprobs == self.expected_logprobs
        assert kwargs['gen_config'].return_routed_experts is True
        return SimpleNamespace(input_token_len=2)

    async def generate(self, prepared_request, stream_response=True):
        assert prepared_request.input_token_len == 2
        assert stream_response is True
        yield SimpleNamespace(
            response='x',
            token_ids=[7],
            routed_experts=[[[1, 2]]],
            logprobs=[{7: -0.125, 8: -2.0}],
            finish_reason='length',
            input_token_len=2,
            generate_token_len=1,
        )


def test_generate_keeps_decode_logprobs_and_routed_experts():
    engine = _DecodeEngine()
    context = _ServerContext(engine.backend_config, async_engine=engine)
    generate = _make_generate(context)
    request = GenerateReqInput(input_ids=[1, 2],
                               max_tokens=2,
                               return_logprob=True,
                               return_routed_experts=True)
    result = asyncio.run(generate(request, raw_request=_RawRequest()))

    assert result.text == 'x'
    assert result.output_ids == [7]
    assert result.meta_info.output_token_logprobs == [(-0.125, 7)]
    assert result.meta_info.output_top_logprobs is None
    assert result.meta_info.input_token_logprobs is None
    assert result.meta_info.routed_experts == [[[1, 2]]]


def test_generate_keeps_decode_top_logprobs():
    engine = _DecodeEngine(expected_logprobs=2)
    context = _ServerContext(engine.backend_config, async_engine=engine)
    generate = _make_generate(context)
    request = GenerateReqInput(input_ids=[1, 2],
                               max_tokens=2,
                               return_logprob=True,
                               top_logprobs_num=2,
                               return_routed_experts=True)
    result = asyncio.run(generate(request, raw_request=_RawRequest()))

    assert result.meta_info.output_token_logprobs == [(-0.125, 7)]
    assert result.meta_info.output_top_logprobs == [[(-0.125, 7), (-2.0, 8)]]
    assert result.meta_info.input_token_logprobs is None
    assert result.meta_info.input_top_logprobs is None


def test_generate_nonempty_input_logprobs_nonstream():
    result = asyncio.run(_call_nonempty_route(stream=False))
    assert result.text == ''
    assert result.output_ids == []
    assert result.meta_info.input_token_logprobs == [(-0.25, 2), (-0.5, 3)]
    assert result.meta_info.output_token_logprobs is None


def test_generate_input_logprobs_allow_image_with_input_ids():
    result = asyncio.run(
        _call_nonempty_route(stream=False,
                             image_data='https://example.com/image.png'))
    assert result.meta_info.input_token_logprobs == [(-0.25, 2), (-0.5, 3)]
    assert result.meta_info.output_token_logprobs is None


def test_generate_input_logprobs_use_processed_input_ids():
    result = asyncio.run(
        _call_nonempty_route(stream=False,
                             image_data='https://example.com/image.png',
                             logprob_token_ids=[8, 9, 2, 3],
                             logprobs=[{
                                 8: -0.1
                             }, {
                                 9: -0.2
                             }, {
                                 2: -0.3
                             }, {
                                 3: -0.4
                             }]))
    assert result.meta_info.prompt_tokens == 5
    assert result.meta_info.input_token_logprobs == [
        (-0.1, 8), (-0.2, 9), (-0.3, 2), (-0.4, 3)
    ]
    assert result.meta_info.output_token_logprobs is None


def test_generate_nonempty_input_top_logprobs_nonstream():
    result = asyncio.run(
        _call_nonempty_route(stream=False, top_logprobs_num=2))
    assert result.meta_info.input_token_logprobs == [(-0.25, 2), (-0.5, 3)]
    assert result.meta_info.input_top_logprobs == [[(-0.25, 2), (-2.0, 8)], [(-0.5, 3), (-3.0, 9)]]
    assert result.meta_info.output_token_logprobs is None
    assert result.meta_info.output_top_logprobs is None


def test_generate_nonempty_input_logprobs_stream():
    body = asyncio.run(_call_nonempty_route(stream=True))
    assert body.count('"input_token_logprobs":[[-0.25,2],[-0.5,3]]') == 1
    assert body.count('\ndata: ') == 1
    assert body.endswith('data: [DONE]\n\n')
