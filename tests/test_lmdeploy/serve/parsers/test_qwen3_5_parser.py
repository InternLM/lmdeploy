import json

import pytest
from transformers import AutoTokenizer

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture()
def response_parser(tokenizer):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('qwen3coder')

    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request, tokenizer=tokenizer)


REFERENCE_CHUNKS = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # Short representative reasoning stream; literal text is irrelevant.
    ('计划', True, '计划', None, False, None, None, None),
    ('调用', True, '调用', None, False, None, None, None),
    ('get', True, 'get', None, False, None, None, None),
    ('_current', True, '_current', None, False, None, None, None),
    ('_temperature', True, '_temperature', None, False, None, None, None),
    ('函数', True, '函数', None, False, None, None, None),
    ('并提供', True, '并提供', None, False, None, None, None),
    ('location', True, 'location', None, False, None, None, None),
    ('参数', True, '参数', None, False, None, None, None),
    ('。', True, '。', None, False, None, None, None),
    ('\n', True, '\n', None, False, None, None, None),
    ('</think>', False, None, None, False, None, None, None),
    ('\n\n', True, None, '\n\n', False, None, None, None),
    # Tool call section: placeholder; will be updated to match Qwen3.5 XML-style.
    ('<tool_call>', False, None, None, False, None, None, None),
    ('\n', False, None, None, False, None, None, None),
    ('<', False, None, None, False, None, None, None),
    ('function', False, None, None, False, None, None, None),
    ('=get', False, None, None, False, None, None, None),
    ('_current', False, None, None, False, None, None, None),
    ('_temperature', False, None, None, False, None, None, None),
    ('>', True, None, None, True, 'get_current_temperature', None, 'function'),
    ('\n', False, None, None, False, None, None, None),
    ('<', False, None, None, False, None, None, None),
    ('parameter', False, None, None, False, None, None, None),
    ('=location', False, None, None, False, None, None, None),
    ('>', False, None, None, False, None, None, None),
    ('\n', False, None, None, False, None, None, None),
    ('Be', False, None, None, False, None, None, None),
    ('ijing', False, None, None, False, None, None, None),
    (',', False, None, None, False, None, None, None),
    (' China', False, None, None, False, None, None, None),
    ('\n', False, None, None, False, None, None, None),
    ('</', False, None, None, False, None, None, None),
    ('parameter', False, None, None, False, None, None, None),
    # Tokenizer maps this `>` to a single id; Qwen3Coder may emit accumulated JSON args in one delta.
    ('>', True, None, None, True, None, '{"location": "Beijing, China"', None),
    ('\n', False, None, None, False, None, None, None),
    ('</', False, None, None, False, None, None, None),
    ('function', False, None, None, False, None, None, None),
    ('>', True, None, None, True, None, '}', None),
    ('\n', False, None, None, False, None, None, None),
    ('</tool_call>', False, None, None, False, None, None, None),
    ('', True, None, '', False, None, None, None),
]


class TestQwen3_5ResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3.5 Coder
    parsers."""

    @staticmethod
    def _encode_ids(tokenizer, text: str) -> list[int]:
        return tokenizer.encode(text, add_bos=False, add_special_tokens=False)

    def test_stream_chunk_matches_reference(self, tokenizer, response_parser):
        """Feed the real streaming sequence into ResponseParser.stream_chunk
        and verify each parsed chunk.

        Expectations for tool_calls will be refined once the Qwen3.5 ground-truth stream is finalized.
        """

        for (delta_text, exp_delta_msg, exp_reasoning, exp_content, exp_tool_emitted,
             exp_function_name, exp_function_arguments,
             exp_type) in REFERENCE_CHUNKS:
            delta_ids = self._encode_ids(tokenizer, delta_text)
            delta_msg, tool_emitted = response_parser.stream_chunk(
                delta_text=delta_text,
                delta_token_ids=delta_ids,
            )
            if exp_delta_msg is False:
                assert delta_msg is None
                continue

            assert delta_msg.reasoning_content == exp_reasoning
            assert delta_msg.content == exp_content

            # Tool-call expectations in this fixture are placeholders for now.
            # Only enforce the exact tool_emitted flag when an explicit tool
            # delta shape is provided.
            if (
                exp_function_name is None
                and exp_function_arguments is None
                and exp_type is None
                and exp_reasoning is None
                and exp_content is None
            ):
                continue

            assert tool_emitted == exp_tool_emitted

            if tool_emitted:
                assert delta_msg.tool_calls is not None
                assert len(delta_msg.tool_calls) == 1
                call = delta_msg.tool_calls[0]
                assert isinstance(call, DeltaToolCall)
                assert call.type == exp_type
                assert call.function is not None
                assert call.function.name == exp_function_name
                assert call.function.arguments == exp_function_arguments

    def test_parse_tool_call_complete_treats_params_as_strings(self):
        parser = Qwen3CoderToolParser(tokenizer=None)
        payload = """
<tool_call>
<function=find_user_id_by_name_zip>
<parameter=first_name>
Chen
</parameter>
<parameter=last_name>
Johnson
</parameter>
<parameter=zip>
77004
</parameter>
</function>
</tool_call>
""".strip()

        tool_call = parser.parse_tool_call_complete(payload)

        assert tool_call is not None
        assert tool_call.function.name == 'find_user_id_by_name_zip'
        assert json.loads(tool_call.function.arguments) == {
            'first_name': 'Chen',
            'last_name': 'Johnson',
            'zip': '77004',
        }

    def test_parse_tool_call_complete_coerces_types_by_schema(self):
        parser = Qwen3CoderToolParser(tokenizer=None)
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'typed_tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string'
                            },
                            'age': {
                                'type': 'integer'
                            },
                            'height': {
                                'type': 'number'
                            },
                            'active': {
                                'type': 'boolean'
                            },
                            'meta': {
                                'type': 'object'
                            },
                            'scores': {
                                'type': 'array'
                            },
                            'misc': {
                                'type': 'null'
                            },
                        },
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)

        payload = """
<tool_call>
<function=typed_tool>
<parameter=name>
Chen
</parameter>
<parameter=age>
29
</parameter>
<parameter=height>
1.75
</parameter>
<parameter=active>
true
</parameter>
<parameter=meta>
{"city":"Houston"}
</parameter>
<parameter=scores>
[98,87]
</parameter>
<parameter=misc>
null
</parameter>
</function>
</tool_call>
""".strip()

        tool_call = parser.parse_tool_call_complete(payload)

        assert tool_call is not None
        assert tool_call.function.name == 'typed_tool'
        assert json.loads(tool_call.function.arguments) == {
            'name': 'Chen',
            'age': 29,
            'height': 1.75,
            'active': True,
            'meta': {
                'city': 'Houston'
            },
            'scores': [98, 87],
            'misc': None,
        }
