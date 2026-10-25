"""ETE for ``POST /v1/responses`` (lmdeploy Text V1)."""

import pytest
import requests
from utils.config_utils import get_restful_chat_model_list
from utils.constant import BACKEND_LIST, BASE_URL, CAPPED_MAX_COMPLETION_TOKENS
from utils.restful_return_check import (
    assert_responses_batch_return,
    assert_responses_error,
    get_client_and_model,
)

_RESPONSES_URL = f'{BASE_URL}/v1/responses'
_MATH_PROMPT = 'What is 13 * 24?'
_QED_INSTRUCTION = 'End the final answer with QED.'
_MATH_PROMPT_QED = f'{_MATH_PROMPT} {_QED_INSTRUCTION}'
_MAX_OUTPUT_TOKENS = CAPPED_MAX_COMPLETION_TOKENS * 4


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


def _responses_json(model_name: str, **extra) -> dict:
    body = {
        'model': model_name,
        'input': _MATH_PROMPT,
        'max_output_tokens': CAPPED_MAX_COMPLETION_TOKENS,
        'temperature': 0.01,
    }
    body.update(extra)
    return body


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_chat_model_list())
class TestRestfulOpenAIResponses:

    @pytest.mark.pr_test
    def test_return_info(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            input=_MATH_PROMPT,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        )
        output = response.model_dump()
        assert_responses_batch_return(output, model_name)
        assert '312' in output['output_text']

    @pytest.mark.pr_test
    def test_return_info_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        stream = client.responses.create(
            model=model_name,
            input=_MATH_PROMPT,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
            stream=True,
        )
        events = list(stream)
        assert events[0].type == 'response.created'
        assert events[-1].type == 'response.completed'
        streaming_text = ''.join(event.delta for event in events if event.type == 'response.output_text.delta')
        completed = events[-1].response
        assert streaming_text == completed.output_text
        assert_responses_batch_return(completed.model_dump(), model_name)
        assert '312' in completed.output_text

    def test_instructions(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            instructions=_QED_INSTRUCTION,
            input=_MATH_PROMPT_QED,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        )
        output = response.model_dump()
        assert_responses_batch_return(output, model_name)
        assert output['instructions'] == _QED_INSTRUCTION
        assert '312' in output['output_text']
        assert 'QED' in output['output_text']

    def test_chat_history_input(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            input=[
                {'role': 'system', 'content': _QED_INSTRUCTION},
                {'role': 'user', 'content': 'What is 5 * 3?'},
                {'role': 'assistant', 'content': '15. QED.'},
                {'role': 'user', 'content': f'Multiply the result by 2. {_QED_INSTRUCTION}'},
            ],
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        )
        output = response.model_dump()
        assert_responses_batch_return(output, model_name)
        assert '30' in output['output_text']
        assert 'QED' in output['output_text']

    def test_input_text_content_parts(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            input=[{
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': _MATH_PROMPT}],
            }],
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        )
        output = response.model_dump()
        assert_responses_batch_return(output, model_name)
        assert '312' in output['output_text']

    def test_max_output_tokens_incomplete(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            input='Write a long story about a city.',
            max_output_tokens=5,
            temperature=0.01,
        )
        assert_responses_batch_return(response.model_dump(), model_name, status='incomplete')

    def test_unknown_model(self, backend, model_case):
        resp = requests.post(
            _RESPONSES_URL,
            json=_responses_json('definitely-not-a-deployed-model-name'),
            timeout=30,
        )
        assert_responses_error(resp, status_code=404, error_type='not_found_error', param='model')

    @pytest.mark.parametrize(
        'input_value',
        [
            pytest.param(None, id='missing'),
            pytest.param([], id='empty-list'),
        ],
    )
    def test_invalid_input(self, backend, model_case, openai_client_and_model, input_value):
        _, model_name = openai_client_and_model
        body = _responses_json(model_name)
        if input_value is None:
            body.pop('input')
        else:
            body['input'] = input_value
        resp = requests.post(_RESPONSES_URL, json=body, timeout=30)
        assert_responses_error(resp, status_code=400, error_type='invalid_request_error', param='input')

    @pytest.mark.parametrize(
        'field_name,value',
        [
            pytest.param('background', True, id='background'),
            pytest.param('previous_response_id', 'resp_not_supported', id='previous_response_id'),
            pytest.param('conversation', 'conv_not_supported', id='conversation'),
        ],
    )
    def test_text_v1_rejects_unsupported_stateful_fields(
            self, backend, model_case, openai_client_and_model, field_name, value):
        _, model_name = openai_client_and_model
        resp = requests.post(
            _RESPONSES_URL,
            json=_responses_json(model_name, **{field_name: value}),
            timeout=30,
        )
        assert_responses_error(
            resp,
            status_code=400,
            error_type='invalid_request_error',
            param=field_name,
            message_substr='not supported by Responses Text V1',
        )

    def test_invalid_temperature(self, backend, model_case, openai_client_and_model):
        _, model_name = openai_client_and_model
        resp = requests.post(
            _RESPONSES_URL,
            json=_responses_json(model_name, temperature=-0.1),
            timeout=30,
        )
        assert_responses_error(resp, status_code=400, error_type='invalid_request_error', param='temperature')

    def test_unsupported_input_item_type(self, backend, model_case, openai_client_and_model):
        _, model_name = openai_client_and_model
        resp = requests.post(
            _RESPONSES_URL,
            json=_responses_json(model_name, input=[{'type': 'computer_call', 'call_id': 'call_x'}]),
            timeout=30,
        )
        assert_responses_error(resp, status_code=400, error_type='invalid_request_error', param='input')

    def test_json_request_required(self, backend, model_case):
        resp = requests.post(
            _RESPONSES_URL,
            headers={'Content-Type': 'text/plain'},
            data='{}',
            timeout=30,
        )
        assert_responses_error(
            resp,
            status_code=400,
            error_type='invalid_request_error',
            message_substr='application/json',
        )
