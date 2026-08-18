# Copyright (c) OpenMMLab. All rights reserved.
import json

from lmdeploy.deepseek_v32_encoding import (
    bos_token,
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
)
from lmdeploy.model import MODELS, DeepseekV32ChatTemplate, get_chat_template
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager

WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string'
                }
            },
            'required': ['city'],
        },
    },
}


def test_deepseek_v32_minimal_chat_and_thinking_modes():
    messages = [{'role': 'user', 'content': 'Hello'}]

    assert encode_messages(messages, thinking_mode='chat') == (
        f'{bos_token}<｜User｜>Hello<｜Assistant｜></think>'
    )
    assert encode_messages(messages, thinking_mode='thinking') == (
        f'{bos_token}<｜User｜>Hello<｜Assistant｜><think>'
    )


def test_deepseek_v32_uses_function_call_block():
    messages = [
        {
            'role': 'system',
            'content': 'You may call tools.',
            'tools': [WEATHER_TOOL],
        },
        {
            'role': 'user',
            'content': 'Weather in Paris?',
        },
        {
            'role': 'assistant',
            'reasoning_content': 'I should call the weather tool.',
            'tool_calls': [{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': '{"city": "Paris"}',
                },
            }],
        },
    ]

    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=False)

    assert '## Tools' in prompt
    assert '"name": "get_weather"' in prompt
    assert '<｜DSML｜function_calls>' in prompt
    assert '</｜DSML｜function_calls>' in prompt
    assert '<｜DSML｜tool_calls>' not in prompt
    assert '<｜DSML｜parameter name="city" string="true">Paris' in prompt


def test_deepseek_v32_tool_results_reopen_thinking():
    messages = [
        {
            'role': 'user',
            'content': 'Weather in Paris?',
        },
        {
            'role': 'assistant',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': '{"city": "Paris"}',
                },
            }],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': 'Sunny',
        },
    ]

    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=False)

    assert '<function_results>\n<result>Sunny</result>\n</function_results>\n\n<think>' in prompt


def test_deepseek_v32_parse_completion_text():
    completion = (
        'I should call a tool.</think>\n\n'
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
        f'{eos_token}'
    )

    parsed = parse_message_from_completion_text(completion, thinking_mode='thinking')

    assert parsed['reasoning_content'] == 'I should call a tool.'
    assert parsed['content'] == ''
    assert parsed['tool_calls'][0]['function']['name'] == 'get_weather'
    assert json.loads(parsed['tool_calls'][0]['function']['arguments']) == {'city': 'Paris'}


def test_deepseek_v32_chat_template_uses_vllm_thinking_switches():
    model = MODELS.get('deepseek-v32')()
    assert model.messages2prompt([{'role': 'user', 'content': 'Hello'}]) == (
        f'{bos_token}<｜User｜>Hello<｜Assistant｜></think>'
    )
    assert model.messages2prompt([{'role': 'user', 'content': 'Hello'}], thinking=True) == (
        f'{bos_token}<｜User｜>Hello<｜Assistant｜><think>'
    )
    assert model.messages2prompt([{'role': 'user', 'content': 'Hello'}], enable_thinking=True) == (
        f'{bos_token}<｜User｜>Hello<｜Assistant｜><think>'
    )


def test_deepseek_v32_chat_template_normalizes_lmdeploy_tools_and_dict_arguments():
    model = MODELS.get('deepseek-v32')()
    prompt = model.messages2prompt(
        [
            {'role': 'user', 'content': 'List files'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'str_replace_editor',
                            'arguments': {
                                'command': 'view',
                                'path': '/testbed',
                            },
                        },
                    }
                ],
            },
        ],
        tools=[
            {
                'name': 'str_replace_editor',
                'description': 'Edit files',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'type': 'string'
                        },
                        'path': {
                            'type': 'string'
                        },
                    },
                    'required': ['command', 'path'],
                },
            }
        ],
        enable_thinking=True,
        drop_thinking=False,
    )
    assert '## Tools' in prompt
    assert '<｜DSML｜function_calls>' in prompt
    assert '<｜DSML｜tool_calls>' not in prompt
    assert '"name": "str_replace_editor"' in prompt
    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert 'parameter name="arguments"' not in prompt


def test_deepseek_v32_chat_template_match_minimal_config(tmp_path):
    (tmp_path / 'config.json').write_text(
        json.dumps({
            'model_type': 'deepseek_v32',
            'architectures': ['DeepseekV32ForCausalLM'],
        }),
        encoding='utf-8',
    )
    assert DeepseekV32ChatTemplate.match(str(tmp_path)) == 'deepseek-v32'
    assert isinstance(get_chat_template(str(tmp_path)), DeepseekV32ChatTemplate)


def _make_response_parser(thinking=True):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('deepseek-v32')
    cls.tool_parser_cls = ToolParserManager.get('deepseek-v32')
    request = ChatCompletionRequest(
        model='deepseek-ai/DeepSeek-V3.2',
        messages=[],
        stream=True,
        chat_template_kwargs={'thinking': thinking},
    )
    return cls(request=request)


def test_deepseek_v32_response_parser_complete_dsml_function_calls():
    completion = (
        'I should call a tool.</think>\n\n'
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
    )

    parser = _make_response_parser(thinking=True)
    content, tool_calls, reasoning_content = parser.parse_complete(completion)
    assert content is None
    assert reasoning_content == 'I should call a tool.'
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_weather'
    assert json.loads(tool_calls[0].function.arguments) == {'city': 'Paris'}
    assert parser.validate_complete(completion)


def test_deepseek_v32_response_parser_streaming_dsml_function_calls():
    text = (
        'need data</think>\n\n'
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="search">\n'
        '<｜DSML｜parameter name="query" string="true">DeepSeek V3.2</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
    )
    parser = _make_response_parser(thinking=True)

    deltas = parser.stream_chunk(delta_text=text, delta_token_ids=[])
    reasoning = ''.join(delta.reasoning_content or '' for delta, _ in deltas)
    tool_deltas = [tool_call for delta, _ in deltas for tool_call in (delta.tool_calls or [])]

    assert reasoning == 'need data'
    assert tool_deltas[0].function.name == 'search'
    assert json.loads(tool_deltas[1].function.arguments) == {'query': 'DeepSeek V3.2'}
