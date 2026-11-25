# Copyright (c) OpenMMLab. All rights reserved.
"""``top_logprobs_format='flat'`` on ``/generate``.

The engine is faked: it yields the LogprobArrays rows the PyTorch engine
produces, and the dicts the same rows give in the default format.
"""
import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from lmdeploy.messages import LogprobArrays, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.openai.endpoints.generate import (
    _create_flat_top_logprobs,
    _create_token_logprobs_from_arrays,
    _create_top_logprobs,
    check_request,
    register,
)
from lmdeploy.serve.openai.protocol import GenerateReqInput

FLAT_KEYS = ('output_top_logprobs_shape', 'output_top_logprobs_ids_b64', 'output_top_logprobs_b64')


def _rows(sampled, k, seed=0, vocab=64):
    """Rows as the sampler builds them: [sampled, top-k], from one
    distribution."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((len(sampled), vocab)).astype(np.float32)
    logprobs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    top = np.argsort(-logprobs, axis=-1, kind='stable')[:, :k]
    ids = np.concatenate([np.asarray(sampled)[:, None], top], axis=1).astype(np.int32)
    vals = np.take_along_axis(logprobs, ids, axis=1).astype(np.float32)
    return LogprobArrays(vals=vals, ids=ids)


def _dicts(rows):
    """What the default format carries for the same rows."""
    return [dict(zip(i, v)) for v, i in zip(rows.vals.tolist(), rows.ids.tolist())]


def _decode(meta):
    n, k = meta['output_top_logprobs_shape']
    ids = np.frombuffer(base64.b64decode(meta['output_top_logprobs_ids_b64']), dtype='<i4').reshape(n, k)
    vals = np.frombuffer(base64.b64decode(meta['output_top_logprobs_b64']), dtype='<f4').reshape(n, k)
    return ids, vals


class _SessionManager:

    def has(self, session_id):
        return False

    def remove(self, session):
        pass


class _Context:

    def __init__(self, config, async_engine=None):
        self.async_engine = async_engine or SimpleNamespace(backend_config=config, speculative_config=None)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    def create_session(self, session_id):
        return SimpleNamespace(session_id=1234)


class _RawRequest:

    async def is_disconnected(self):
        return False


class _Engine:
    """Yields ``chunks`` of (token_ids, rows); rows are arrays or dicts per
    the request."""

    def __init__(self, chunks, k):
        self.backend_config = PytorchEngineConfig(logprobs_mode='raw_logprobs', role=EngineRole.Hybrid)
        self.chunks = chunks
        self.k = k
        self.gen_config = None

    async def preprocess(self, **kwargs):
        self.gen_config = kwargs['gen_config']
        assert self.gen_config.logprobs == (self.k or 1)
        return SimpleNamespace(input_token_len=2)

    async def generate(self, prepared_request, stream_response=True):
        array = self.gen_config.logprobs_format == 'array'
        total = 0
        for i, (token_ids, rows) in enumerate(self.chunks):
            total += len(token_ids)
            if rows is not None and not array:
                rows = _dicts(rows)
            yield SimpleNamespace(response='x' * len(token_ids),
                                  token_ids=token_ids,
                                  routed_experts=None,
                                  logprobs=rows,
                                  finish_reason='length' if i == len(self.chunks) - 1 else None,
                                  input_token_len=2,
                                  generate_token_len=total)


def _call(chunks, k, fmt, stream, **request_kwargs):
    engine = _Engine(chunks, k)
    router = APIRouter()
    register(router, _Context(engine.backend_config, async_engine=engine))
    generate = next(route.endpoint for route in router.routes if route.path == '/generate')
    request = GenerateReqInput(input_ids=[1, 2],
                               max_tokens=8,
                               return_logprob=True,
                               top_logprobs_num=k,
                               top_logprobs_format=fmt,
                               stream=stream,
                               **request_kwargs)

    async def run():
        result = await generate(request, raw_request=_RawRequest())
        if stream:
            body = ''.join([chunk async for chunk in result.body_iterator])
            return body
        return result

    return asyncio.run(run()), engine


def _stream_metas(body):
    events = [line[len('data: '):] for line in body.split('\n') if line.startswith('data: ')]
    assert events[-1] == '[DONE]'
    return [json.loads(event)['meta_info'] for event in events[:-1]]


def test_request_validation():
    pt = _Context(PytorchEngineConfig(logprobs_mode='raw_logprobs', role=EngineRole.Hybrid))
    ok = dict(input_ids=[1, 2], return_logprob=True, top_logprobs_num=4)
    assert GenerateReqInput(**ok).top_logprobs_format == 'nested'
    assert check_request(GenerateReqInput(**ok, top_logprobs_format='flat'), pt) == ''
    assert check_request(GenerateReqInput(input_ids=[1, 2], return_logprob=True, top_logprobs_format='flat'),
                         pt) == ''
    assert 'PyTorch engine' in check_request(GenerateReqInput(**ok, top_logprobs_format='flat'),
                                             _Context(TurbomindEngineConfig()))
    assert 'logprob_start_len' in check_request(
        GenerateReqInput(**ok, top_logprobs_format='flat', logprob_start_len=0, max_tokens=0), pt)
    assert 'requires return_logprob' in check_request(
        GenerateReqInput(input_ids=[1, 2], top_logprobs_num=4, top_logprobs_format='flat'), pt)
    with pytest.raises(ValidationError):
        GenerateReqInput(**ok, top_logprobs_format='binary')


def test_flat_encoding_layout():
    rows = _rows([3, 5, 9], k=4)
    fields = _create_flat_top_logprobs(rows, 4)
    assert fields['output_top_logprobs_shape'] == [3, 4]
    raw_ids = base64.b64decode(fields['output_top_logprobs_ids_b64'])
    raw_vals = base64.b64decode(fields['output_top_logprobs_b64'])
    assert len(raw_ids) == len(raw_vals) == 3 * 4 * 4
    # Explicit little-endian decoding, independent of the host byte order.
    np.testing.assert_array_equal(np.frombuffer(raw_ids, dtype='<i4').reshape(3, 4), rows.ids[:, 1:])
    np.testing.assert_array_equal(np.frombuffer(raw_vals, dtype='<f4').reshape(3, 4), rows.vals[:, 1:])
    # Rows wider than the request's k keep the first k candidates.
    assert _create_flat_top_logprobs(rows, 2)['output_top_logprobs_shape'] == [3, 2]
    # Big-endian or float64 inputs are converted, not reinterpreted.
    other = LogprobArrays(vals=rows.vals.astype('>f8'), ids=rows.ids.astype('>i8'))
    assert _create_flat_top_logprobs(other, 4) == fields

    assert _create_flat_top_logprobs(None, 4) is None
    assert _create_flat_top_logprobs(rows[:0], 4) is None
    assert _create_flat_top_logprobs(rows, 0) is None

    assert _create_token_logprobs_from_arrays([3, 5, 9], rows) == list(zip(rows.vals[:, 0].tolist(), [3, 5, 9]))
    assert _create_token_logprobs_from_arrays([], rows[:0]) is None
    assert _create_token_logprobs_from_arrays([3], None) is None
    with pytest.raises(ValueError):
        _create_token_logprobs_from_arrays([3, 5], rows)


@pytest.mark.parametrize('k', [1, 3, 8])
def test_flat_and_nested_carry_the_same_numbers(k):
    # Token 0 of each row is the argmax (in top-k); the others are random.
    probe = _rows([0, 0, 0, 0], k=1, seed=k)
    sampled = [int(probe.ids[0, 1]), 1, 2, int(probe.ids[3, 1])]
    rows = _rows(sampled, k=k, seed=k)
    nested = _create_top_logprobs(sampled, _dicts(rows), k)
    ids, vals = _decode(_create_flat_top_logprobs(rows, k))
    for row_ids, row_vals, row in zip(ids, vals, nested):
        assert sorted(zip(row_vals.tolist(), row_ids.tolist()), reverse=True) == sorted(row, reverse=True)
        assert len(set(row_ids.tolist())) == k


def test_nonstream_flat_equals_nested_and_keeps_token_logprobs():
    rows = _rows([5, 6, 7, 8, 9], k=3)
    chunks = [([5, 6], rows[:2]), ([7], rows[2:3]), ([8, 9], rows[3:])]
    flat, engine = _call(chunks, 3, 'flat', stream=False)
    assert engine.gen_config.logprobs_format == 'array'
    nested, engine = _call(chunks, 3, 'nested', stream=False)
    assert engine.gen_config.logprobs_format == 'dict'

    assert flat.output_ids == nested.output_ids == [5, 6, 7, 8, 9]
    assert flat.meta_info.output_token_logprobs == nested.meta_info.output_token_logprobs
    assert flat.meta_info.output_top_logprobs is None
    meta = jsonable_encoder(flat)['meta_info']
    ids, vals = _decode(meta)
    assert ids.shape == (5, 3)
    for row_ids, row_vals, row in zip(ids, vals, nested.meta_info.output_top_logprobs):
        assert list(zip(row_vals.tolist(), row_ids.tolist())) == [tuple(item) for item in row]
    assert all(key not in jsonable_encoder(nested)['meta_info'] for key in FLAT_KEYS)


def test_stream_chunks_concatenate_to_the_full_array():
    rows = _rows([5, 6, 7, 8], k=4)
    chunks = [([5, 6], rows[:2]), ([], None), ([7], rows[2:3]), ([8], rows[3:])]
    body, _ = _call(chunks, 4, 'flat', stream=True)
    metas = _stream_metas(body)
    assert len(metas) == 4
    assert all(all(key in meta for key in FLAT_KEYS) for meta in metas)
    assert metas[1]['output_token_logprobs'] is None
    assert all(metas[1][key] is None for key in FLAT_KEYS)

    decoded = [_decode(meta) for meta in metas if meta['output_top_logprobs_shape'] is not None]
    assert [len(meta['output_token_logprobs']) for meta in metas if meta['output_token_logprobs']] == [
        len(ids) for ids, _ in decoded
    ]
    np.testing.assert_array_equal(np.concatenate([ids for ids, _ in decoded]), rows.ids[:, 1:])
    np.testing.assert_array_equal(np.concatenate([vals for _, vals in decoded]), rows.vals[:, 1:])
    assert sum((meta['output_token_logprobs'] or [] for meta in metas), []) == [
        [v, t] for v, t in zip(rows.vals[:, 0].tolist(), [5, 6, 7, 8])
    ]


def test_flat_without_top_k_returns_only_token_logprobs():
    rows = _rows([5, 6], k=1)
    result, _ = _call([([5, 6], rows)], 0, 'flat', stream=False)
    meta = jsonable_encoder(result)['meta_info']
    assert meta['output_token_logprobs'] == [[v, t] for v, t in zip(rows.vals[:, 0].tolist(), [5, 6])]
    assert all(meta[key] is None for key in FLAT_KEYS)


def test_empty_response():
    result, _ = _call([([], None)], 4, 'flat', stream=False)
    meta = jsonable_encoder(result)['meta_info']
    assert result.output_ids == []
    assert meta['output_token_logprobs'] is None
    assert all(meta[key] is None for key in FLAT_KEYS)


def test_nested_stream_bytes_are_unchanged():
    rows = LogprobArrays(vals=np.array([[-0.125, -0.125, -2.0]], dtype=np.float32),
                         ids=np.array([[7, 7, 8]], dtype=np.int32))
    body, _ = _call([([7], rows)], 2, 'nested', stream=True)
    assert body == ('data: {"text":"x","output_ids":[7],"meta_info":{"prompt_tokens":2,"completion_tokens":1,'
                    '"finish_reason":{"type":"length"},"output_token_logprobs":[[-0.125,7]],'
                    '"input_token_logprobs":null,"output_top_logprobs":[[[-0.125,7],[-2.0,8]]],'
                    '"input_top_logprobs":null,"routed_experts":null}}\n\n'
                    'data: [DONE]\n\n')
