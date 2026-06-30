# Copyright (c) OpenMMLab. All rights reserved.
import json
from pathlib import Path

from lmdeploy.deepseek_v4_encoding import (
    REASONING_EFFORT_MAX,
    bos_token,
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
)
from lmdeploy.model import MODELS, DeepseekV4ChatTemplate, get_chat_template
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager

TESTS_DIR = Path(__file__).parent / 'data' / 'deepseek_v4_encoding'


def _load_json(name: str):
    return json.loads((TESTS_DIR / name).read_text(encoding='utf-8'))


def _load_text(name: str):
    return (TESTS_DIR / name).read_text(encoding='utf-8')


def test_case_1():
    """Thinking mode with tool calls (multi-turn, tool results merged into user)."""
    td = _load_json('test_input_1.json')
    messages = td['messages']
    messages[0]['tools'] = td['tools']
    gold = _load_text('test_output_1.golden')
    prompt = encode_messages(messages, thinking_mode='thinking')
    assert prompt == gold

    marker = '<｜Assistant｜><think>'
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find('<｜User｜>', first_start)
    parsed_tc = parse_message_from_completion_text(prompt[first_start:first_end], thinking_mode='thinking')
    assert parsed_tc['reasoning_content'] == (
        'The user wants to know the weather in Beijing. I should use the get_weather tool.'
    )
    assert parsed_tc['content'] == ''
    assert len(parsed_tc['tool_calls']) == 1
    assert parsed_tc['tool_calls'][0]['function']['name'] == 'get_weather'
    assert json.loads(parsed_tc['tool_calls'][0]['function']['arguments']) == {
        'location': 'Beijing',
        'unit': 'celsius',
    }

    last_start = prompt.rfind(marker) + len(marker)
    parsed_final = parse_message_from_completion_text(prompt[last_start:], thinking_mode='thinking')
    assert parsed_final['reasoning_content'] == 'Got the weather data. Let me format a nice response.'
    assert '22°C' in parsed_final['content']
    assert parsed_final['tool_calls'] == []


def test_case_2():
    """Thinking mode without tools (drop_thinking removes earlier reasoning)."""
    messages = _load_json('test_input_2.json')
    gold = _load_text('test_output_2.golden')
    prompt = encode_messages(messages, thinking_mode='thinking')
    assert prompt == gold

    marker = '<｜Assistant｜><think>'
    last_start = prompt.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[last_start:], thinking_mode='thinking')
    assert parsed['reasoning_content'] == 'The user asks about the capital of France. It is Paris.'
    assert parsed['content'] == 'The capital of France is Paris.'
    assert parsed['tool_calls'] == []

    assert 'The user said hello' not in prompt


def test_case_3():
    """Interleaved thinking + search (developer with tools, latest_reminder)."""
    messages = _load_json('test_input_3.json')
    gold = _load_text('test_output_3.golden')
    assert encode_messages(messages, thinking_mode='thinking') == gold


def test_case_4():
    """Quick instruction task with latest_reminder (chat mode, action task)."""
    messages = _load_json('test_input_4.json')
    gold = _load_text('test_output_4.golden')
    assert encode_messages(messages, thinking_mode='chat') == gold


def test_deepseek_v4_chat_template_normalizes_lmdeploy_tools():
    model = MODELS.get('deepseek-v4')()
    prompt = model.messages2prompt(
        [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': "What's the weather in Beijing?"},
        ],
        tools=[
            {
                'name': 'get_weather',
                'description': 'Get weather for a location.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string'
                        }
                    },
                    'required': ['location'],
                },
            }
        ],
        enable_thinking=True,
        reasoning_effort='max',
    )
    assert prompt.startswith(bos_token + REASONING_EFFORT_MAX)
    assert '## Tools' in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith('<｜Assistant｜><think>')


def test_deepseek_v4_reasoning_effort_does_not_enable_thinking():
    model = MODELS.get('deepseek-v4')()
    prompt = model.messages2prompt(
        [{'role': 'user', 'content': 'Hello'}],
        reasoning_effort='max',
    )
    assert REASONING_EFFORT_MAX not in prompt
    assert prompt == f'{bos_token}<｜User｜>Hello<｜Assistant｜></think>'


def test_deepseek_v4_chat_template_match_minimal_config(tmp_path):
    (tmp_path / 'config.json').write_text(
        json.dumps({
            'model_type': 'deepseek_v4',
            'architectures': ['DeepseekV4ForCausalLM'],
        }),
        encoding='utf-8',
    )
    assert DeepseekV4ChatTemplate.match(str(tmp_path)) == 'deepseek-v4'
    assert isinstance(get_chat_template(str(tmp_path)), DeepseekV4ChatTemplate)


def _make_response_parser(thinking=True):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('deepseek-v4')
    cls.tool_parser_cls = ToolParserManager.get('deepseek-v4')
    request = ChatCompletionRequest(
        model='deepseek-ai/DeepSeek-V4',
        messages=[],
        stream=True,
        chat_template_kwargs={'thinking': thinking},
    )
    return cls(request=request)


def test_deepseek_v4_response_parser_complete_dsml_tool_call():
    td = _load_json('test_input_1.json')
    messages = td['messages']
    messages[0]['tools'] = td['tools']
    prompt = encode_messages(messages, thinking_mode='thinking')

    marker = '<｜Assistant｜><think>'
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find('<｜User｜>', first_start)
    completion = prompt[first_start:first_end].removesuffix(eos_token)

    parser = _make_response_parser(thinking=True)
    content, tool_calls, reasoning_content = parser.parse_complete(completion)
    assert content is None
    assert reasoning_content == (
        'The user wants to know the weather in Beijing. I should use the get_weather tool.'
    )
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_weather'
    assert json.loads(tool_calls[0].function.arguments) == {
        'location': 'Beijing',
        'unit': 'celsius',
    }
    assert parser.validate_complete(completion)


def test_deepseek_v4_response_parser_streaming_dsml_tool_call():
    text = (
        'need a tool</think>\n\n'
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="search">\n'
        '<｜DSML｜parameter name="query" string="true">DeepSeek V4</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
    )
    parser = _make_response_parser(thinking=True)

    deltas = parser.stream_chunk(delta_text=text, delta_token_ids=[])
    reasoning = ''.join(delta.reasoning_content or '' for delta, _ in deltas)
    tool_deltas = [tool_call for delta, _ in deltas for tool_call in (delta.tool_calls or [])]

    assert reasoning == 'need a tool'
    assert tool_deltas[0].function.name == 'search'
    assert json.loads(tool_deltas[1].function.arguments) == {'query': 'DeepSeek V4'}


def test_deepseek_v4_response_parser_reasoning_effort_does_not_enable_thinking():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('deepseek-v4')
    cls.tool_parser_cls = None
    request = ChatCompletionRequest(
        model='deepseek-ai/DeepSeek-V4',
        messages=[],
        stream=True,
        reasoning_effort='max',
    )
    parser = cls(request=request)

    deltas = parser.stream_chunk(delta_text='hello', delta_token_ids=[])
    assert len(deltas) == 1
    delta, tool_emitted = deltas[0]
    assert tool_emitted is False
    assert delta.content == 'hello'
    assert delta.reasoning_content is None
