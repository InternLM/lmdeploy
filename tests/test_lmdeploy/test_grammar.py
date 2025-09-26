import json

import pytest
from jsonschema import validate

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig  # , PytorchEngineConfig

MODEL_IDS = [
    'Qwen/Qwen3-0.6B',
    'OpenGVLab/InternVL3_5-1B',
]

BACKEND_FACTORIES = [
    ('tm', lambda: TurbomindEngineConfig(max_batch_size=2, session_len=1024)),
    # ('pt', lambda: PytorchEngineConfig(max_batch_size=1, session_len=1024)),
]

GUIDE_SCHEMA = {
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
}


@pytest.mark.parametrize('model_id', MODEL_IDS)
@pytest.mark.parametrize('backend_name,backend_factory', BACKEND_FACTORIES)
@pytest.mark.parametrize('enable_guide', [True, False])
def test_guided_matrix(model_id, backend_name, backend_factory, enable_guide):
    pipe = pipeline(
        model_id,
        backend_config=backend_factory(),
        log_level='INFO',
    )

    try:
        if enable_guide:
            gen_config = GenerationConfig(response_format=dict(
                type='json_schema',
                json_schema=dict(name='test', schema=GUIDE_SCHEMA),
            ), )
        else:
            gen_config = GenerationConfig()

        response = pipe(['Make a self introduction please.'] * 3, gen_config=gen_config)
        assert response and response[0].text

        if enable_guide:
            validate(instance=json.loads(response[0].text), schema=GUIDE_SCHEMA)
    finally:
        pipe.close()
