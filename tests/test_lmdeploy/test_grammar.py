import json

import pytest
from jsonschema import validate

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig


@pytest.fixture(scope='module')
def tiny_model_id():
    return 'internlm/internlm2_5-1_8b'


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


def test_tm_guided_pipeline(tiny_model_id):
    pipe = pipeline(tiny_model_id,
                    backend_config=TurbomindEngineConfig(max_batch_size=1, session_len=1024),
                    log_level='INFO')
    gen_config = GenerationConfig(response_format=dict(type='json_schema', json_schema=dict(name='test', schema=guide)))
    response = pipe(['Make a self introduction please.'], gen_config=gen_config)
    validate(instance=json.loads(response[0].text), schema=guide)
