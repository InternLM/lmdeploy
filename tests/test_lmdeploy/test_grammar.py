import json

import pytest
from transformers import AutoConfig, AutoTokenizer

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind.turbomind import TurboMind


@pytest.fixture(scope='module')
def tiny_model_id():
    return 'Qwen/Qwen2.5-0.5B'


@pytest.fixture(scope='module')
def tmp_workspace(tmp_path_factory):
    return tmp_path_factory.mktemp('tm_workspace')


guide = {
    'type': 'object',
    'properties': {
        'name': {
            'type': 'string'
        },
        'skills': {
            'type': 'array',
            'items': {
                'type': 'string',
                'maxLength': 10
            },
            'minItems': 3
        },
        'work history': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'company': {
                        'type': 'string'
                    },
                    'duration': {
                        'type': 'string'
                    }
                },
                'required': ['company']
            }
        }
    },
    'required': ['name', 'skills', 'work history']
}


def test_tm_grammar_json_schema(tiny_model_id, tmp_workspace):
    schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string'}}})

    tm = TurboMind(
        model_path=tiny_model_id,
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_id),
        engine_config=TurbomindEngineConfig(
            max_batch_size=1,
            session_len=512,
        ),
        decode_grammar=schema,
        decode_grammar_type='json_schema',
        decode_grammar_threads=2,
        decode_grammar_vocab_size=AutoConfig.from_pretrained(tiny_model_id).vocab_size,
    )
    assert hasattr(tm, 'grammar')


def test_tm_grammar_regex(tiny_model_id, tmp_workspace):
    regex = r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'

    tm = TurboMind(
        model_path=tiny_model_id,
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_id),
        engine_config=TurbomindEngineConfig(max_batch_size=1, session_len=512),
        decode_grammar=regex,
        decode_grammar_type='regex',
        decode_grammar_vocab_size=AutoConfig.from_pretrained(tiny_model_id).vocab_size,
    )
    assert hasattr(tm, 'grammar')


def test_tm_grammar_invalid_type(tiny_model_id):
    with pytest.raises(AssertionError, match='Decode grammar type .* should be in'):
        TurboMind(
            model_path=tiny_model_id,
            tokenizer=AutoTokenizer.from_pretrained(tiny_model_id),
            engine_config=TurbomindEngineConfig(max_batch_size=1, session_len=512),
            decode_grammar='dummy',
            decode_grammar_type='wrong',
            decode_grammar_vocab_size=AutoConfig.from_pretrained(tiny_model_id).vocab_size,
        )


def test_instance_set_grammar(tiny_model_id):
    schema = json.dumps({'type': 'string'})
    tm = TurboMind(
        model_path=tiny_model_id,
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_id),
        engine_config=TurbomindEngineConfig(max_batch_size=1, session_len=512),
        decode_grammar=schema,
        decode_grammar_vocab_size=AutoConfig.from_pretrained(tiny_model_id).vocab_size,
    )
    instance = tm.create_instance()
    assert instance is not None


def test_tm_no_grammar_by_default(tiny_model_id):
    tm = TurboMind(
        model_path=tiny_model_id,
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_id),
        engine_config=TurbomindEngineConfig(max_batch_size=1, session_len=512),
    )
    assert not hasattr(tm, 'grammar')


def test_tm_guided_pipeline(tiny_model_id):
    pipe = pipeline(tiny_model_id,
                    backend_config=TurbomindEngineConfig(max_batch_size=1, session_len=1024),
                    log_level='INFO')
    gen_config = GenerationConfig(response_format=dict(type='json_schema', json_schema=dict(name='test', schema=guide)))
    response = pipe(['Make a self introduction please.'], gen_config=gen_config)
    assert False, response
