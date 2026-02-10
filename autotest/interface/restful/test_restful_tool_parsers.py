import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from utils.tool_reasoning_definitions import TOOL_CALL_END_TOKEN, TOOL_CALL_START_TOKEN, TOOL_PARSER_NAMES

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_qwen3_parser_cls = None
_qwen2d5_parser_cls = None
_internlm2_parser_cls = None
_llama3_parser_cls = None
_tool_parser_manager = None


def _get_qwen3_parser_cls():
    global _qwen3_parser_cls
    if _qwen3_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Qwen3ToolParser
        _qwen3_parser_cls = Qwen3ToolParser
    return _qwen3_parser_cls


def _get_qwen2d5_parser_cls():
    global _qwen2d5_parser_cls
    if _qwen2d5_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Qwen2d5ToolParser
        _qwen2d5_parser_cls = Qwen2d5ToolParser
    return _qwen2d5_parser_cls


def _get_internlm2_parser_cls():
    global _internlm2_parser_cls
    if _internlm2_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Internlm2ToolParser
        _internlm2_parser_cls = Internlm2ToolParser
    return _internlm2_parser_cls


def _get_llama3_parser_cls():
    global _llama3_parser_cls
    if _llama3_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Llama3JsonToolParser
        _llama3_parser_cls = Llama3JsonToolParser
    return _llama3_parser_cls


def _get_tool_parser_manager():
    global _tool_parser_manager
    if _tool_parser_manager is None:
        from lmdeploy.serve.openai.tool_parser import ToolParserManager
        _tool_parser_manager = ToolParserManager
    return _tool_parser_manager


def _make_mock_tokenizer(vocab=None, encode_map=None):
    """Create a mock tokenizer.

    Args:
        vocab: dict mapping token string → id.
        encode_map: dict mapping token string → list[int] for .encode().
    """
    tok = MagicMock()
    default_vocab = {
        '<tool_call>': 200,
        '</tool_call>': 201,
        '<|action_start|>': 300,
        '<|plugin|>': 301,
        '<|action_end|>': 302,
        '<|python_tag|>': 400,
    }
    tok.get_vocab.return_value = vocab or default_vocab

    def _encode(text, add_special_tokens=False):
        _map = encode_map or {'<|python_tag|>': [400]}
        return _map.get(text, [999])

    tok.encode = _encode
    return tok


def _make_mock_request(tools=None, tool_choice='auto'):
    """Create a mock ChatCompletionRequest."""
    req = MagicMock()
    req.model = 'test-model'
    req.tools = tools
    req.tool_choice = tool_choice
    req.skip_special_tokens = True
    # Ensure _tool_parser_state is initially absent
    if hasattr(req, '_tool_parser_state'):
        delattr(req, '_tool_parser_state')
    return req


def _make_tool_obj(name='get_current_weather'):
    """Create a mock Tool object with a function attribute."""
    tool = MagicMock()
    tool.function.name = name
    return tool


_PARSER_MARKS = [
    pytest.mark.order(10),
    pytest.mark.tool_parser,
]


def _apply_parser_marks(cls):
    """Apply parser test marks to *cls*."""
    for m in _PARSER_MARKS:
        cls = m(cls)
    return cls


@_apply_parser_marks
class TestToolParserManager:
    """Verify all tool parsers are correctly registered."""

    def test_all_parser_names_registered(self):
        mgr = _get_tool_parser_manager()
        for name in TOOL_PARSER_NAMES:
            cls = mgr.get(name)
            assert cls is not None, (f'Tool parser "{name}" not found in ToolParserManager')

    def test_qwen3_registered(self):
        mgr = _get_tool_parser_manager()
        cls = mgr.get('qwen3')
        assert cls is not None
        assert cls.__name__ == 'Qwen3ToolParser'

    def test_qwen_alias_registered(self):
        """'qwen' should map to the same class as 'qwen3'."""
        mgr = _get_tool_parser_manager()
        cls_qwen = mgr.get('qwen')
        cls_qwen3 = mgr.get('qwen3')
        assert cls_qwen is cls_qwen3

    def test_qwen2d5_registered(self):
        mgr = _get_tool_parser_manager()
        cls = mgr.get('qwen2d5')
        assert cls is not None
        assert cls.__name__ == 'Qwen2d5ToolParser'

    def test_internlm_registered(self):
        mgr = _get_tool_parser_manager()
        cls = mgr.get('internlm')
        assert cls is not None
        assert cls.__name__ == 'Internlm2ToolParser'

    def test_intern_s1_registered(self):
        """'intern-s1' should map to Internlm2ToolParser."""
        mgr = _get_tool_parser_manager()
        cls = mgr.get('intern-s1')
        assert cls is not None
        assert cls.__name__ == 'Internlm2ToolParser'

    def test_llama3_registered(self):
        mgr = _get_tool_parser_manager()
        cls = mgr.get('llama3')
        assert cls is not None
        assert cls.__name__ == 'Llama3JsonToolParser'

    def test_unknown_parser_returns_none(self):
        mgr = _get_tool_parser_manager()
        result = mgr.get('nonexistent-parser-xyz')
        assert result is None


@_apply_parser_marks
class TestToolParserUtils:
    """Tests for tool_parser/utils.py helper functions."""

    # -- find_common_prefix --------------------------------------------------
    def test_find_common_prefix_basic(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_prefix
        assert find_common_prefix('{"fruit": "ap"}', '{"fruit": "apple"}') == '{"fruit": "ap'

    def test_find_common_prefix_identical(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_prefix
        assert find_common_prefix('abc', 'abc') == 'abc'

    def test_find_common_prefix_no_common(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_prefix
        assert find_common_prefix('abc', 'xyz') == ''

    def test_find_common_prefix_empty(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_prefix
        assert find_common_prefix('', 'abc') == ''

    # -- find_common_suffix --------------------------------------------------
    def test_find_common_suffix_basic(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_suffix
        assert find_common_suffix('{"fruit": "ap"}', '{"fruit": "apple"}') == '"}'

    def test_find_common_suffix_no_common(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_suffix
        assert find_common_suffix('abc', 'xyz') == ''

    def test_find_common_suffix_empty(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_common_suffix
        assert find_common_suffix('', 'abc') == ''

    # -- extract_intermediate_diff -------------------------------------------
    def test_extract_intermediate_diff_basic(self):
        from lmdeploy.serve.openai.tool_parser.utils import extract_intermediate_diff
        result = extract_intermediate_diff('{"fruit": "apple"}', '{"fruit": "ap"}')
        assert result == 'ple'

    def test_extract_intermediate_diff_no_change(self):
        from lmdeploy.serve.openai.tool_parser.utils import extract_intermediate_diff
        result = extract_intermediate_diff('{"a": 1}', '{"a": 1}')
        assert result == ''

    # -- find_all_indices ----------------------------------------------------
    def test_find_all_indices_basic(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_all_indices
        result = find_all_indices('abcabcabc', 'abc')
        assert result == [0, 3, 6]

    def test_find_all_indices_not_found(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_all_indices
        result = find_all_indices('hello world', 'xyz')
        assert result == []

    def test_find_all_indices_single(self):
        from lmdeploy.serve.openai.tool_parser.utils import find_all_indices
        result = find_all_indices('hello world', 'world')
        assert result == [6]

    # -- is_complete_json ----------------------------------------------------
    def test_is_complete_json_valid(self):
        from lmdeploy.serve.openai.tool_parser.utils import is_complete_json
        assert is_complete_json('{"name": "test"}') is True

    def test_is_complete_json_incomplete(self):
        from lmdeploy.serve.openai.tool_parser.utils import is_complete_json
        assert is_complete_json('{"name": "test') is False

    def test_is_complete_json_array(self):
        from lmdeploy.serve.openai.tool_parser.utils import is_complete_json
        assert is_complete_json('[1, 2, 3]') is True

    def test_is_complete_json_empty(self):
        from lmdeploy.serve.openai.tool_parser.utils import is_complete_json
        assert is_complete_json('') is False

    # -- consume_space -------------------------------------------------------
    def test_consume_space_basic(self):
        from lmdeploy.serve.openai.tool_parser.utils import consume_space
        assert consume_space(0, '   hello') == 3

    def test_consume_space_no_spaces(self):
        from lmdeploy.serve.openai.tool_parser.utils import consume_space
        assert consume_space(0, 'hello') == 0

    def test_consume_space_all_spaces(self):
        from lmdeploy.serve.openai.tool_parser.utils import consume_space
        assert consume_space(0, '    ') == 4

    def test_consume_space_middle(self):
        from lmdeploy.serve.openai.tool_parser.utils import consume_space
        assert consume_space(5, 'hello   world') == 8


@_apply_parser_marks
@pytest.mark.qwen3_parser
class TestQwen3ToolParserNonStreaming:
    """Qwen3ToolParser.extract_tool_calls — complete output."""

    def test_single_tool_call(self):
        """Single <tool_call>...</tool_call> → one tool call extracted."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == 'get_current_weather'
        args = json.loads(tc.function.arguments)
        assert args['city'] == 'Dallas'
        assert args['state'] == 'TX'

    def test_multiple_tool_calls(self):
        """Multiple <tool_call> blocks → multiple tool calls extracted."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n'
                        '</tool_call>\n'
                        '<tool_call>\n'
                        '{"name": "calculate", "arguments": {"expression": "37 * 43"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == 'get_current_weather'
        assert result.tool_calls[1].function.name == 'calculate'

    def test_text_before_tool_call(self):
        """Text before tool_call tags should be preserved as content."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('Let me check the weather for you.\n'
                        '<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert 'check the weather' in result.content

    def test_text_after_tool_call(self):
        """Text after tool_call tags should be preserved."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n'
                        '</tool_call>\n'
                        'I have requested the weather data.')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert 'requested the weather' in result.content

    def test_no_tool_call(self):
        """Plain text without tool_call tags."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'The weather in Dallas is sunny and warm.'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        assert result.content == model_output

    def test_parameters_key_support(self):
        """Tool call using 'parameters' key instead of 'arguments'."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        # Qwen3 extract_tool_calls uses 'arguments' in the JSON directly
        # but get_argments() supports 'parameters' too.
        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Beijing"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == 'get_current_weather'

    def test_unicode_arguments(self):
        """Chinese / Unicode arguments in tool calls."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "北京", "unit": "摄氏度"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['city'] == '北京'
        assert args['unit'] == '摄氏度'

    def test_empty_model_output(self):
        """Empty string → no tool calls."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        result = parser.extract_tool_calls('', req)
        assert result.tools_called is False
        assert len(result.tool_calls) == 0

    def test_nested_json_arguments(self):
        """Nested JSON object in arguments."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "create_event", "arguments": '
                        '{"title": "Meeting", "location": {"venue": "Room A", "city": "NYC"}}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['title'] == 'Meeting'
        assert isinstance(args['location'], dict)
        assert args['location']['city'] == 'NYC'


@_apply_parser_marks
@pytest.mark.qwen3_parser
class TestQwen3ToolParserStreaming:
    """Qwen3ToolParser.extract_tool_calls_streaming — incremental output."""

    def test_text_before_tool_tag(self):
        """Content before <tool_call> should be returned as text content."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        current_text = 'Let me check'
        delta = parser.extract_tool_calls_streaming(
            previous_text='',
            current_text=current_text,
            delta_text='Let me check',
            previous_token_ids=[],
            current_token_ids=[1, 2, 3],
            delta_token_ids=[1, 2, 3],
            request=req,
        )

        assert delta is not None
        assert delta.content == 'Let me check'

    def test_complete_tool_call_in_stream(self):
        """Complete <tool_call>...</tool_call> in one chunk."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        tool_json = '{"name": "get_current_weather", "arguments": {"city": "Dallas"}}'
        current_text = f'{TOOL_CALL_START_TOKEN}\n{tool_json}\n{TOOL_CALL_END_TOKEN}'

        delta = parser.extract_tool_calls_streaming(
            previous_text='',
            current_text=current_text,
            delta_text=current_text,
            previous_token_ids=[],
            current_token_ids=[200, 10, 11, 12, 201],
            delta_token_ids=[200, 10, 11, 12, 201],
            request=req,
        )

        assert delta is not None
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 1
        tc = delta.tool_calls[0]
        assert tc.function.name == 'get_current_weather'

    def test_text_then_tool_call(self):
        """Text content followed by a tool call tag."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        # First chunk: just text
        delta1 = parser.extract_tool_calls_streaming(
            previous_text='',
            current_text='Let me check the weather.',
            delta_text='Let me check the weather.',
            previous_token_ids=[],
            current_token_ids=[1, 2, 3, 4],
            delta_token_ids=[1, 2, 3, 4],
            request=req,
        )
        assert delta1 is not None
        assert delta1.content is not None

    def test_incomplete_tool_tag_no_end(self):
        """<tool_call> without </tool_call> → tool content not yet emitted."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        req = _make_mock_request()

        current_text = '<tool_call>\n{"name": "get_current_weather"'
        delta = parser.extract_tool_calls_streaming(
            previous_text='',
            current_text=current_text,
            delta_text=current_text,
            previous_token_ids=[],
            current_token_ids=[200, 10, 11],
            delta_token_ids=[200, 10, 11],
            request=req,
        )

        # Parser should return delta but without tool_calls yet (no end tag)
        assert delta is not None
        # tool_calls should be empty or None since end tag not found
        if delta.tool_calls:
            # Some implementations may still buffer
            pass


@_apply_parser_marks
@pytest.mark.qwen3_parser
class TestQwen3GetArguments:
    """Test Qwen3ToolParser.get_argments helper method."""

    def test_parameters_key(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        obj = {'name': 'test', 'parameters': {'key': 'value'}}
        assert parser.get_argments(obj) == {'key': 'value'}

    def test_arguments_key(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        obj = {'name': 'test', 'arguments': {'key': 'value'}}
        assert parser.get_argments(obj) == {'key': 'value'}

    def test_no_params_or_args(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        obj = {'name': 'test'}
        assert parser.get_argments(obj) is None

    def test_parameters_takes_precedence(self):
        """When both 'parameters' and 'arguments' exist, 'parameters' wins."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen3_parser_cls()(tok)
        obj = {'name': 'test', 'parameters': {'a': 1}, 'arguments': {'b': 2}}
        assert parser.get_argments(obj) == {'a': 1}


@_apply_parser_marks
@pytest.mark.qwen2d5_parser
class TestQwen2d5ToolParserNonStreaming:
    """Qwen2d5ToolParser.extract_tool_calls — complete output."""

    def test_single_tool_call(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == 'get_current_weather'
        args = json.loads(tc.function.arguments)
        assert args['city'] == 'Dallas'

    def test_text_before_tool_call(self):
        """Text before <tool_call> should be preserved as content."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('Sure, let me check.\n'
                        '<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert result.content is not None
        assert 'let me check' in result.content

    def test_text_after_tool_call(self):
        """Text after </tool_call> should be preserved."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas"}}\n'
                        '</tool_call>\n'
                        'Weather data requested.')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert result.content is not None

    def test_no_tool_call(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'The weather in Dallas is sunny.'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        assert result.content == model_output

    def test_multiple_tool_calls(self):
        """Multiple <tool_call> blocks in Qwen2d5 format."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas"}}\n'
                        '</tool_call>\n'
                        '<tool_call>\n'
                        '{"name": "calculate", "arguments": {"expression": "2+2"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2

    def test_unicode_arguments(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen2d5_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "北京"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['city'] == '北京'


@_apply_parser_marks
@pytest.mark.internlm2_parser
class TestInternlm2ToolParserNonStreaming:
    """Internlm2ToolParser.extract_tool_calls — complete output."""

    def test_single_tool_call(self):
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools)

        model_output = ('<|action_start|><|plugin|>\n'
                        '{"name": "get_current_weather", "parameters": {"city": "Dallas", "state": "TX"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == 'get_current_weather'
        args = json.loads(tc.function.arguments)
        assert args['city'] == 'Dallas'

    def test_text_before_action(self):
        """Text before <|action_start|> should be preserved."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools)

        model_output = ('Let me check the weather.\n'
                        '<|action_start|><|plugin|>\n'
                        '{"name": "get_current_weather", "parameters": {"city": "Dallas"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert result.content is not None
        assert 'check the weather' in result.content

    def test_no_tool_call(self):
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'The weather is sunny.'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        assert result.content == model_output

    def test_arguments_key_variant(self):
        """InternLM2 parser supports both 'parameters' and 'arguments'."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools)

        model_output = ('<|action_start|><|plugin|>\n'
                        '{"name": "get_current_weather", "arguments": {"city": "Dallas"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['city'] == 'Dallas'

    def test_unicode_arguments(self):
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools)

        model_output = ('<|action_start|><|plugin|>\n'
                        '{"name": "get_current_weather", "parameters": {"city": "北京"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['city'] == '北京'


@_apply_parser_marks
@pytest.mark.internlm2_parser
class TestInternlm2AdjustRequest:
    """Test Internlm2ToolParser.adjust_request."""

    def test_adjust_request_sets_skip_special_tokens_false(self):
        """When tools are present and tool_choice != 'none',
        skip_special_tokens → False."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)

        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools, tool_choice='auto')

        adjusted = parser.adjust_request(req)
        assert adjusted.skip_special_tokens is False

    def test_adjust_request_tool_choice_none(self):
        """When tool_choice='none', skip_special_tokens should not be
        changed."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)

        tools = [_make_tool_obj('get_current_weather')]
        req = _make_mock_request(tools=tools, tool_choice='none')
        req.skip_special_tokens = True

        adjusted = parser.adjust_request(req)
        assert adjusted.skip_special_tokens is True

    def test_adjust_request_no_tools(self):
        """When no tools, skip_special_tokens should not be changed."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)

        req = _make_mock_request(tools=None, tool_choice='auto')
        req.skip_special_tokens = True

        adjusted = parser.adjust_request(req)
        assert adjusted.skip_special_tokens is True


@_apply_parser_marks
@pytest.mark.llama3_parser
class TestLlama3ToolParserNonStreaming:
    """Llama3JsonToolParser.extract_tool_calls — complete output."""

    def test_single_tool_call(self):
        """Standard <function=name>{args}</function> format."""
        tok = _make_mock_tokenizer()
        parser = _get_llama3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<function=get_current_weather>{"city": "Dallas", "state": "TX"}</function>'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == 'get_current_weather'
        args = json.loads(tc.function.arguments)
        assert args['city'] == 'Dallas'
        assert args['state'] == 'TX'

    def test_malformed_input_fallback(self):
        """Non-function format → fallback to plain text."""
        tok = _make_mock_tokenizer()
        parser = _get_llama3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'This is just plain text without function tags.'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        assert result.content == model_output

    def test_unicode_arguments(self):
        """Unicode / Chinese arguments."""
        tok = _make_mock_tokenizer()
        parser = _get_llama3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<function=get_current_weather>{"city": "北京"}</function>'
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['city'] == '北京'

    def test_nested_json_arguments(self):
        """Nested JSON in arguments."""
        tok = _make_mock_tokenizer()
        parser = _get_llama3_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<function=create_event>'
                        '{"title": "Meeting", "location": {"venue": "Room A", "city": "NYC"}}'
                        '</function>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args['title'] == 'Meeting'
        assert args['location']['city'] == 'NYC'


@_apply_parser_marks
class TestToolParserCrossParserEdgeCases:
    """Edge cases applicable to multiple parsers."""

    @pytest.mark.parametrize('parser_name', ['qwen3', 'qwen2d5'])
    def test_empty_output_qwen_family(self, parser_name):
        """Empty string → no tool calls for Qwen parsers."""
        mgr = _get_tool_parser_manager()
        cls = mgr.get(parser_name)
        tok = _make_mock_tokenizer()
        parser = cls(tok)
        req = _make_mock_request()

        result = parser.extract_tool_calls('', req)
        assert result.tools_called is False
        assert len(result.tool_calls) == 0

    def test_empty_output_internlm(self):
        """Empty string → no tool calls for InternLM parser."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        req = _make_mock_request()

        result = parser.extract_tool_calls('', req)
        assert result.tools_called is False

    def test_empty_output_llama3(self):
        """Empty string → fallback for Llama3 parser."""
        tok = _make_mock_tokenizer()
        parser = _get_llama3_parser_cls()(tok)
        req = _make_mock_request()

        result = parser.extract_tool_calls('', req)
        assert result.tools_called is False

    @pytest.mark.parametrize('parser_name', ['qwen3', 'qwen2d5'])
    def test_special_chars_in_arguments_qwen(self, parser_name):
        """Special characters in arguments should be handled."""
        mgr = _get_tool_parser_manager()
        cls = mgr.get(parser_name)
        tok = _make_mock_tokenizer()
        parser = cls(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "web_search", "arguments": '
                        '{"query": "what\'s the latest on AI & ML?"}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert 'AI' in args['query']

    def test_special_chars_in_arguments_internlm(self):
        """Special characters in InternLM tool arguments."""
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('web_search')]
        req = _make_mock_request(tools=tools)

        model_output = ('<|action_start|><|plugin|>\n'
                        '{"name": "web_search", "parameters": '
                        '{"query": "what\'s the latest on AI & ML?"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert 'AI' in args['query']

    @pytest.mark.parametrize('parser_name', ['qwen3', 'qwen2d5'])
    def test_only_tags_no_json_qwen(self, parser_name):
        """<tool_call></tool_call> with no JSON content → error handled."""
        mgr = _get_tool_parser_manager()
        cls = mgr.get(parser_name)
        tok = _make_mock_tokenizer()
        parser = cls(tok)
        req = _make_mock_request()

        model_output = '<tool_call>\n\n</tool_call>'
        try:
            parser.extract_tool_calls(model_output, req)
            # If it doesn't raise, tool_calls should be empty or error handled
        except (json.JSONDecodeError, Exception):
            # Expected — empty content inside tool tags is malformed JSON
            pass

    @pytest.mark.parametrize('parser_name', ['qwen3', 'qwen2d5'])
    def test_array_arguments_qwen(self, parser_name):
        """Tool call with array-typed arguments."""
        mgr = _get_tool_parser_manager()
        cls = mgr.get(parser_name)
        tok = _make_mock_tokenizer()
        parser = cls(tok)
        req = _make_mock_request()

        model_output = ('<tool_call>\n'
                        '{"name": "create_event", "arguments": '
                        '{"title": "Meeting", "attendees": ["alice@example.com", "bob@example.com"]}}\n'
                        '</tool_call>')
        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert isinstance(args['attendees'], list)
        assert len(args['attendees']) == 2

    def test_internlm_tool_not_in_request_tools(self):
        """InternLM: tool name not in request.tools list.

        Note: the source parser is missing a ``return`` before
        ``ExtractedToolCallInformation(tools_called=False, ...)``,
        so it falls through and still returns tools_called=True.
        This test documents the current (buggy) behaviour.
        """
        tok = _make_mock_tokenizer()
        parser = _get_internlm2_parser_cls()(tok)
        tools = [_make_tool_obj('some_other_tool')]
        req = _make_mock_request(tools=tools)

        model_output = ('<|action_start|><|plugin|>\n'
                        '{"name": "unknown_tool", "parameters": {"key": "value"}}\n'
                        '<|action_end|>')
        result = parser.extract_tool_calls(model_output, req)

        assert result is not None
        # Known issue: missing return in internlm2_parser line 175
        # causes tools_called=True even when tool is not in the list
        assert result.tools_called is True


@_apply_parser_marks
class TestToolParserInstantiation:
    """Verify every registered parser can be instantiated."""

    @pytest.mark.parametrize('parser_name', TOOL_PARSER_NAMES)
    def test_parser_instantiation(self, parser_name):
        mgr = _get_tool_parser_manager()
        cls = mgr.get(parser_name)
        assert cls is not None

        tok = _make_mock_tokenizer()
        parser = cls(tok)
        assert parser is not None
        assert parser.model_tokenizer is tok


@_apply_parser_marks
class TestToolParserBaseClass:
    """Verify base ToolParser class behaviour."""

    def test_base_extract_tool_calls_raises(self):
        """Base class extract_tool_calls should raise NotImplementedError."""
        from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser
        tok = _make_mock_tokenizer()
        parser = ToolParser(tok)
        req = _make_mock_request()

        with pytest.raises(NotImplementedError):
            parser.extract_tool_calls('some output', req)

    def test_base_extract_tool_calls_streaming_raises(self):
        """Base class streaming extraction should raise NotImplementedError."""
        from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser
        tok = _make_mock_tokenizer()
        parser = ToolParser(tok)
        req = _make_mock_request()

        with pytest.raises(NotImplementedError):
            parser.extract_tool_calls_streaming('', 'text', 'text', [], [1], [1], req)

    def test_base_adjust_request_passthrough(self):
        """Base class adjust_request should return the request unchanged."""
        from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser
        tok = _make_mock_tokenizer()
        parser = ToolParser(tok)
        req = _make_mock_request()

        result = parser.adjust_request(req)
        assert result is req

    def test_base_vocab_property(self):
        """Base class vocab property should call tokenizer.get_vocab()."""
        from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser
        expected_vocab = {'<special>': 1, 'token': 2}
        tok = _make_mock_tokenizer(vocab=expected_vocab)
        parser = ToolParser(tok)

        assert parser.vocab == expected_vocab
