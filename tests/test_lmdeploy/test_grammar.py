import json
import re

import pytest
from jsonschema import validate

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig

MODEL_IDS = [
    'Qwen/Qwen3-0.6B',
    'OpenGVLab/InternVL3_5-1B',
]

BACKEND_FACTORIES = [
    ('tm', lambda: TurbomindEngineConfig(max_batch_size=2, session_len=1024)),
    ('pt', lambda: PytorchEngineConfig(max_batch_size=1, session_len=1024)),
]

SCHEMA_MAP = {
    'json_schema': {
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
                'minItems': 3,
                'maxItems': 10,
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
                        },
                    },
                    'required': ['company'],
                },
            },
        },
        'required': ['name', 'skills', 'work history'],
    },
    'regex_schema': 'call me [A-Za-z]{1,10}',
    'json_object': None,
}


@pytest.mark.parametrize('model_id', MODEL_IDS)
@pytest.mark.parametrize('backend_name,backend_factory', BACKEND_FACTORIES)
@pytest.mark.parametrize('schema_type', list(SCHEMA_MAP.keys()) + [None])
def test_guided_matrix(model_id, backend_name, backend_factory, schema_type):
    pipe = pipeline(
        model_id,
        backend_config=backend_factory(),
        log_level='INFO',
    )

    if schema_type is None:
        enable_guide = False
    else:
        enable_guide = True
        response_format = {'type': schema_type}
        schema = SCHEMA_MAP[schema_type]
        if schema_type == 'json_schema':
            response_format[schema_type] = dict(name='test', schema=schema)
        elif schema_type == 'regex_schema':
            response_format[schema_type] = schema

    try:
        if enable_guide:
            gen_config = GenerationConfig(response_format=response_format)
        else:
            gen_config = GenerationConfig()

        response = pipe(['Make a self introduction please.'] * 3, gen_config=gen_config)
        assert response and response[0].text

        if enable_guide:
            if schema_type == 'json_schema':
                validate(instance=json.loads(response[0].text), schema=schema)
            elif schema_type == 'json_object':
                validate(instance=json.loads(response[0].text), schema={'type': 'object', 'additionalProperties': True})
            elif schema_type == 'regex_schema':
                assert re.fullmatch(schema, response[0].text)
    finally:
        pipe.close()
