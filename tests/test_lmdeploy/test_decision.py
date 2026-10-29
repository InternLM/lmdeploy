# Copyright (c) OpenMMLab. All rights reserved.
"""CPU tests for the Intern-Decision task; no checkpoint or GPU required."""
import asyncio
import base64
import io
import math
from pathlib import Path

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
from lmdeploy.serve.openai.decision_protocol import DecisionRequest
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
