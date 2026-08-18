# Copyright (c) OpenMMLab. All rights reserved.
import json

import pytest

from lmdeploy._guided_decoding import compile_response_format
from lmdeploy.serve.parsers.tool_parser import ToolParser


class _Compiler:

    def __init__(self):
        self.calls = []

    def compile_json_schema(self, schema):
        self.calls.append(('json_schema', schema))
        return 'json-grammar'

    def compile_regex(self, regex):
        self.calls.append(('regex_schema', regex))
        return 'regex-grammar'

    def compile_structural_tag(self, structural_tag):
        self.calls.append(('structural_tag', structural_tag))
        return 'structural-grammar'


@pytest.mark.parametrize(
    ('response_format', 'expected_result', 'expected_kind'),
    [
        ({
            'type': 'json_schema',
            'json_schema': {
                'name': 'answer',
                'schema': {
                    'type': 'object',
                },
            },
        }, 'json-grammar', 'json_schema'),
        ({
            'type': 'regex_schema',
            'regex_schema': '[a-z]+',
        }, 'regex-grammar', 'regex_schema'),
        ({
            'type': 'json_object',
        }, 'json-grammar', 'json_schema'),
        ({
            'type': 'structural_tag',
            'format': {
                'type': 'const_string',
                'value': 'tool',
            },
        }, 'structural-grammar', 'structural_tag'),
    ],
)
def test_compile_response_format_dispatch(response_format, expected_result, expected_kind):
    compiler = _Compiler()

    result = compile_response_format(compiler, response_format)

    assert result == expected_result
    assert compiler.calls[0][0] == expected_kind
    if expected_kind in ('json_schema', 'structural_tag'):
        assert isinstance(compiler.calls[0][1], str)
        json.loads(compiler.calls[0][1])


def test_compile_response_format_rejects_unknown_type():
    with pytest.raises(ValueError, match='unsupported format type'):
        compile_response_format(_Compiler(), {'type': 'unknown'})


@pytest.mark.parametrize(
    'model_format',
    [
        'qwen_3',
        'qwen_3_5',
        'qwen_3_coder',
        'llama',
        'glm_4_7',
        'deepseek_v3_2',
        'deepseek_v4',
    ],
)
def test_python_xgrammar_compiles_required_structural_formats(model_format):
    import xgrammar as xgr

    class XGrammarToolParser(ToolParser):
        structural_tag_model = model_format

    tokenizer_info = xgr.TokenizerInfo(
        [bytes([token_id]) for token_id in range(256)],
        vocab_type=xgr.VocabType.RAW,
        vocab_size=256,
        stop_token_ids=[0],
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    response_format = XGrammarToolParser.build_required_tool_response_format(
        object(),
        [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                        },
                    },
                    'required': ['city'],
                },
            },
        }],
        reasoning=True,
    )

    compiled = compile_response_format(compiler, response_format)

    assert isinstance(compiled, xgr.CompiledGrammar)
