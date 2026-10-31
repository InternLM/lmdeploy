# Copyright (c) OpenMMLab. All rights reserved.
"""CPU tests for the Intern-Decision task; no checkpoint or GPU required."""
import asyncio
import base64
import copy
import io
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lmdeploy.intern_decision_encoding import (
    DEFAULT_TEMPERATURE,
    DecisionOutputError,
    compile_row,
    decode_answers,
    scale_probabilities,
)
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.decision_protocol import DecisionRequest
from lmdeploy.serve.openai.endpoints.decisions import _score_request
from lmdeploy.serve.openai.protocol import GenerateReqInput
from lmdeploy.serve.parsers.decision_parser import DecisionContext, DecisionParser

QUESTIONS = {
    'action': {
        'type': 'choice',
        'instructions': 'Select an action.',
        'criteria': {
            'stop': 'Stop.',
            'go': 'Go.'
        }
    },
    'risk': {
        'type': 'score',
        'criteria': ['low', 'mid', 'high']
    },
    'red': {
        'type': 'noul'
    },
}


def _png(size=(4, 3), color=(255, 0, 0), fmt='PNG', **save_kwargs):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def _image_reference(payload, mime='image/png'):
    data = base64.b64encode(payload).decode()
    return {'url': f'data:{mime};base64,{data}'}


# ---------------------------------------------------------------- engine config


def test_logprob_extra_token_ids_requires_input_logprobs():
    GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0, logprob_extra_token_ids=[1, 2])
    with pytest.raises(ValueError, match='logprob_start_len'):
        GenerationConfig(logprob_extra_token_ids=[1])
    with pytest.raises(ValueError, match='non-negative'):
        GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0, logprob_extra_token_ids=[-1])


# ---------------------------------------------------------------- protocol


@pytest.mark.parametrize('change', [
    {'questions': {}},
    {'questions': {'q': {'type': 'rank'}}},
    {'questions': {'q': {'type': 'choice', 'criteria': ['a']}}},
    {'questions': {'q': {'type': 'score', 'criteria': {'x': 'not a number'}}}},
    {'questions': {f'q{i}': {'type': 'noul'} for i in range(17)}},
    {'state': float('nan')},
    {'unknown': 1},
    {'images': None},
    {'images': 123},
    {'images': [123]},
    {'images': Path('image.png')},
])
def test_invalid_request(change):
    with pytest.raises(ValueError):
        DecisionRequest.model_validate({'state': {}, 'questions': QUESTIONS, **change})


def test_training_metadata_is_dropped():
    request = DecisionRequest.model_validate({'state': 1, 'questions': QUESTIONS, 'targets': {'a': 'A'}, 'id': 3})
    assert 'targets' not in request.model_dump() and 'id' not in request.model_dump()


@pytest.mark.parametrize('images', [
    'https://example.com/image.png',
    {'url': 'https://example.com/image.png'},
    {'url': 'https://example.com/image.png', 'detail': 'high'},
    ['https://example.com/first.png', {'url': 'https://example.com/second.png'}],
    [],
    {},
])
def test_image_input_matches_generate_schema(images):
    decision = DecisionRequest.model_validate({'state': 1, 'questions': QUESTIONS, 'images': images})
    generate = GenerateReqInput(prompt='Inspect the images.', image_data=images)
    assert decision.model_dump()['images'] == generate.model_dump()['image_data']


@pytest.mark.parametrize('image', [
    {},
    {'url': 123},
    {'type': 'image/png', 'data': 'base64'},
    {'type': 'image_url', 'image_url': {'url': 'image.png'}},
])
def test_invalid_image_reference_is_rejected_by_shared_processor(image):
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    # Dict internals intentionally pass the request schema, just as in /generate.
    parser = DecisionParser({'state': 1, 'questions': QUESTIONS, 'images': image}, config())
    inputs = parser.prepare_input()
    with pytest.raises(ValueError):
        asyncio.run(MultimodalProcessor.async_parse_multimodal_item(inputs.messages))


def test_reserved_marker_rejected():
    with pytest.raises(ValueError, match='Reserved'):
        compile_row({'state': 'say <decision>', 'questions': QUESTIONS})


# ---------------------------------------------------------------- output


@pytest.mark.parametrize('temperature', [1.0, 2.747760550703])
def test_output_semantics_and_calibration(temperature):
    request = DecisionRequest.model_validate({'state': 1, 'questions': QUESTIONS})
    logits = [[2.0, 0.5], [0.1, 0.7, 1.9], [1.2, -0.3]]
    answers = decode_answers(request.questions, ('action', 'risk', 'red'), logits, temperature=temperature)

    def expected(values, row):
        probs = dict(zip(values, torch.softmax(torch.tensor(row), -1).tolist()))
        logs = {k: math.log(v) for k, v in probs.items()}
        top = max(logs.values())
        weights = {k: math.exp((v - top) / temperature) for k, v in logs.items()}
        return {k: w / sum(weights.values()) for k, w in weights.items()}

    action = expected(['stop', 'go'], logits[0])
    risk = expected(['0', '1', '2'], logits[1])
    red = expected(['no', 'yes'], logits[2])
    assert answers['action']['choice'] == answers['action']['decision'] == 'stop'
    assert answers['action']['probabilities'] == pytest.approx(action)
    assert answers['risk']['score'] == pytest.approx(sum(float(k) * v for k, v in risk.items()))
    assert answers['risk']['legend'] == {'0': 'low', '1': 'mid', '2': 'high'}
    assert answers['red']['noul'] == pytest.approx(red['yes'])
    assert answers['red']['confidence'] == pytest.approx(max(red.values()))


def test_temperature_flattens_but_keeps_argmax():
    probs = {'a': 0.9, 'b': 0.1}
    scaled = scale_probabilities(probs, 3.0)
    assert scaled['a'] > scaled['b'] and scaled['a'] < probs['a']
    assert scale_probabilities(probs, 1.0) == probs


@pytest.mark.parametrize('logits', [[[1.0, float('nan')]], [[1.0]], [[1.0, 2.0], [0.0, 1.0]]])
def test_output_rejects_bad_logits(logits):
    questions = {'q': {'type': 'noul'}}
    with pytest.raises(DecisionOutputError):
        decode_answers(questions, ('q', ), logits)


MARKER, IMAGE_PAD = 900, 800
SYMBOLS = {'A': 32, 'B': 33, 'C': 34}


def config(**kwargs):
    values = dict(marker_id=MARKER, symbol_ids=SYMBOLS, image_token_id=IMAGE_PAD,
                  temperature=DEFAULT_TEMPERATURE)
    return DecisionContext(**{**values, **kwargs})


class FakeSessionManager:

    def __init__(self):
        self.sessions = {}
        self.count = 0

    def get(self):
        self.count += 1
        session = SimpleNamespace(session_id=self.count, epoch=None)
        self.sessions[session.session_id] = session
        return session

    def remove(self, session):
        self.sessions.pop(session.session_id, None)


class FakeEngine:
    epoch = 0
    model_name = 'intern-decision'

    def __init__(self, *, expand=4, fail=None, abort=False, misplaced_marker=False):
        self.session_mgr = FakeSessionManager()
        self.requests = []
        self.expand, self.fail, self.abort = expand, fail, abort
        self.misplaced_marker = misplaced_marker

    async def preprocess(self, messages, session, *, gen_config, input_ids=None, **kwargs):
        assert input_ids is None and kwargs['do_preprocess']
        # Observe the actual parser->engine contract: image references and complete messages.
        content = messages[1]['content']
        images = sum(part['type'] == 'image_url' for part in content) if isinstance(content, list) else 0
        if images:
            assert all(not isinstance(part.get('image_url', {}).get('url'), Image.Image) for part in content)
        fields = messages[-1]['content'].count('<decision>')
        ids = [1] + [IMAGE_PAD] * (self.expand * images) + [2, 3]
        for _ in range(fields):
            ids.extend([7, MARKER, 8])
        if self.misplaced_marker:
            # A marker at position 0 has no earlier token to score against.
            ids = [MARKER] * fields
        prepared = SimpleNamespace(session=session, inputs={'input_ids': ids}, input_token_len=len(ids),
                                   gen_config=gen_config)
        self.requests.append(prepared)
        return prepared

    async def generate(self, request, stream_response=True):
        assert stream_response is False
        if self.fail:
            raise self.fail
        targets = request.inputs['input_ids'][request.gen_config.logprob_start_len + 1:]
        # Distinct marker scores and invalid non-marker scores catch row-selection mistakes.
        rows = []
        question_index = 0
        for target in targets:
            if target == MARKER:
                direction = 1 if question_index % 2 == 0 else -1
                rows.append({token: direction * float(token)
                             for token in request.gen_config.logprob_extra_token_ids})
                question_index += 1
            else:
                rows.append({token: float('nan') for token in request.gen_config.logprob_extra_token_ids})
        yield GenOut(response='', token_ids=[], input_token_len=request.input_token_len, generate_token_len=0,
                     logprobs=None if self.abort else rows,
                     finish_reason='abort' if self.abort else 'length')


async def score(engine, request, cfg=None):
    from lmdeploy.serve.openai.api_server import ServerContext
    context = ServerContext()
    context.async_engine = engine
    parser = DecisionParser(request, cfg or config())
    return await _score_request(context, parser)


@pytest.mark.parametrize('images', [0, 2])
def test_scoring_uses_final_marker_positions(images):
    engine = FakeEngine()
    request = {'state': 1, 'questions': QUESTIONS, 'images': [_image_reference(_png())] * images}
    result = asyncio.run(score(engine, request))
    prepared, = engine.requests
    assert prepared.inputs['input_ids'][prepared.gen_config.logprob_start_len + 1] == MARKER
    assert prepared.gen_config.logprob_extra_token_ids == [32, 33, 34]
    assert result['answers'] == decode_answers(
        QUESTIONS, tuple(QUESTIONS), [[32., 33.], [-32., -33., -34.], [32., 33.]])
    assert result['usage']['input_tokens'] == prepared.input_token_len
    assert result['usage']['generated_tokens'] == 0
    assert not engine.session_mgr.sessions


def test_image_blocks_reach_shared_loader_in_order(tmp_path):
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    path = tmp_path / 'red.png'
    path.write_bytes(_png())
    row = {
        'state': 1,
        'questions': QUESTIONS,
        'images': [
            str(path),
            {**_image_reference(_png(size=(5, 2), color=(0, 0, 255))), 'detail': 'high'},
        ],
    }
    original = copy.deepcopy(row)
    text_messages = compile_row({'state': 1, 'questions': QUESTIONS}).messages
    # The compiler must not mutate the caller's blocks or change the text prompt.
    compiled = compile_row(row)
    assert row == original
    inputs = DecisionParser(row, config()).prepare_input()
    assert row == original
    assert inputs.messages == compiled.messages
    assert inputs.messages[0] == text_messages[0]
    assert inputs.messages[2] == text_messages[2]
    expected_blocks = [
        {'type': 'image_url', 'image_url': {'url': str(path)}},
        {'type': 'image_url', 'image_url': original['images'][1]},
    ]
    assert inputs.messages[1]['content'] == [
        *expected_blocks, {'type': 'text', 'text': text_messages[1]['content']}
    ]

    async def run():
        parsed = await MultimodalProcessor.async_parse_multimodal_item(inputs.messages)
        images = [block['data'] for block in parsed[1]['content'][:-1]]
        try:
            first, second = images
            assert first.size == (4, 3) and first.getpixel((0, 0)) == (255, 0, 0)
            assert second.size == (5, 2) and second.getpixel((0, 0)) == (0, 0, 255)
            assert parsed[1]['content'][-1]['text'] == text_messages[1]['content']
            assert parsed[1]['content'][1]['detail'] == 'high'
            assert inputs.messages[1]['content'][:-1] == expected_blocks
            assert row == original
        finally:
            for image in images:
                image.close()

    asyncio.run(run())


def test_image_blocks_require_a_vision_encoder():
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    parser = DecisionParser({'state': 1, 'questions': QUESTIONS, 'images': [_image_reference(_png())]}, config())
    inputs = parser.prepare_input()
    processor = MultimodalProcessor(tokenizer=None, chat_template=None, vl_encoder=None)
    with pytest.raises(ValueError, match='Multimodal input is not supported'):
        asyncio.run(processor.get_prompt_input(inputs.messages, **inputs.preprocess_kwargs))


@pytest.mark.parametrize('as_dict', [False, True])
def test_single_image_reference_reaches_shared_loader(as_dict):
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    reference = _image_reference(_png())
    images = reference if as_dict else reference['url']
    parser = DecisionParser({'state': 1, 'questions': QUESTIONS, 'images': images}, config())
    inputs = parser.prepare_input()
    assert len(inputs.messages[1]['content']) == 2

    async def run():
        parsed = await MultimodalProcessor.async_parse_multimodal_item(inputs.messages)
        image = parsed[1]['content'][0]['data']
        try:
            assert image.size == (4, 3)
            assert image.getpixel((0, 0)) == (255, 0, 0)
        finally:
            image.close()

    asyncio.run(run())


@pytest.mark.parametrize('failure', ['engine', 'marker', 'abort'])
def test_failure_releases_session(failure):
    engine = FakeEngine(fail=RuntimeError('boom') if failure == 'engine' else None,
                        abort=failure == 'abort', misplaced_marker=failure == 'marker')
    with pytest.raises((ValueError, RuntimeError, RequestError)):
        asyncio.run(score(engine, {'state': 1, 'questions': QUESTIONS}))
    assert not engine.session_mgr.sessions


@pytest.mark.parametrize('case', ['missing', 'too_few_rows', 'too_many_rows', 'missing_candidate', 'nan'])
def test_invalid_scoring_output_is_internal_error(case):
    parser = DecisionParser({'state': 1, 'questions': {'q': {'type': 'noul'}}}, config())
    inputs = parser.prepare_input()
    parser.finalize_input(SimpleNamespace(inputs={'input_ids': [1, MARKER]}, gen_config=inputs.gen_config))
    output = SimpleNamespace(logprobs=[{32: 1., 33: 2.}], finish_reason='length')
    if case == 'missing':
        output.logprobs = None
    elif case == 'too_few_rows':
        output.logprobs = []
    elif case == 'too_many_rows':
        output.logprobs *= 2
    elif case == 'missing_candidate':
        output.logprobs = [{32: 1.}]
    else:
        output.logprobs = [{32: float('nan'), 33: 2.}]
    with pytest.raises(RequestError) as exc:
        parser.parse_output(output, model_name='intern-decision',
                            timing=dict(preprocess_ms=0., inference_ms=0., server_ms=0.))
    assert exc.value.code is ErrorCode.INTERNAL_ERROR
