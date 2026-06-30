# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from pathlib import Path

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

TESTS_DIR = Path(__file__).parent / 'data' / 'deepseek_v32_encoding'


def _load_json(name: str):
    return json.loads((TESTS_DIR / name).read_text(encoding='utf-8'))


def _load_text(name: str):
    return (TESTS_DIR / name).read_text(encoding='utf-8').strip()


def _load_main_messages():
    td = _load_json('test_input.json')
    messages = td['messages']
    messages[0]['tools'] = td['tools']
    return messages


def test_official_deepseek_v32_main_fixture():
    messages = _load_main_messages()
    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=True, add_default_bos_token=True)
    assert prompt == _load_text('test_output.golden')

    tool_call_message = messages[4]
    tool_call_prompt = encode_messages([tool_call_message],
                                       context=messages[:4],
                                       thinking_mode='thinking',
                                       drop_thinking=True,
                                       add_default_bos_token=True)
    tool_call_message_wo_id = copy.deepcopy(tool_call_message)
    for tool_call in tool_call_message_wo_id['tool_calls']:
        tool_call.pop('id')
    parsed_tool_call_message = parse_message_from_completion_text(tool_call_prompt, thinking_mode='thinking')
    parsed_tool_call_message.pop('content')
    assert tool_call_message_wo_id == parsed_tool_call_message

    thinking_message = messages[-6]
    thinking_prompt = encode_messages([thinking_message],
                                      context=messages[:-6],
                                      thinking_mode='thinking',
                                      drop_thinking=True,
                                      add_default_bos_token=True)
    parsed_thinking_message = parse_message_from_completion_text(thinking_prompt, thinking_mode='thinking')
    parsed_thinking_message.pop('tool_calls')
    assert thinking_message == parsed_thinking_message


def test_official_deepseek_v32_search_fixtures():
    messages = _load_json('test_input_search_wo_date.json')['messages']
    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=True, add_default_bos_token=True)
    assert prompt == _load_text('test_output_search_wo_date.golden')

    messages = _load_json('test_input_search_w_date.json')['messages']
    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=True, add_default_bos_token=True)
    assert prompt == _load_text('test_output_search_w_date.golden')


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
    messages = _load_main_messages()
    prompt = encode_messages(messages, thinking_mode='thinking', drop_thinking=True, add_default_bos_token=True)

    start = prompt.find('<｜Assistant｜>') + len('<｜Assistant｜>')
    end = prompt.find(eos_token, start)
    completion = prompt[start:end]

    parser = _make_response_parser(thinking=True)
    content, tool_calls, reasoning_content = parser.parse_complete(completion)
    assert content is None
    assert reasoning_content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_datetime'
    assert json.loads(tool_calls[0].function.arguments) == {'timezone': 'Asia/Shanghai'}
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
