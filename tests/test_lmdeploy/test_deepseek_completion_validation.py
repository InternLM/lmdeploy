# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy import deepseek_v4_encoding, deepseek_v32_encoding


@pytest.fixture(params=[(deepseek_v32_encoding, 'function_calls'), (deepseek_v4_encoding, 'tool_calls')],
                ids=['v3.2', 'v4'])
def decoder(request):
    return request.param


@pytest.mark.parametrize(('mode', 'text'), [
    ('chat', 'truncated reply'),
    ('thinking', 'unfinished thought'),
    ('chat', 'first{eos}second'),
    ('chat', 'reply<think>{eos}'),
    ('chat', '\n\n<｜DSML｜{block}'),
    ('chat', '\n\n<｜DSML｜{block}>\n<｜DSML｜invoke missing-name>\n</｜DSML｜invoke>\n</｜DSML｜{block}>{eos}'),
    ('chat', '\n\n<｜DSML｜{block}>\n</｜DSML｜{block}>extra{eos}'),
])
def test_malformed_completion(decoder, mode, text):
    encoding, block = decoder
    with pytest.raises(ValueError):
        encoding.parse_message_from_completion_text(text.format(block=block, eos=encoding.eos_token), mode)


@pytest.mark.parametrize('include_eos', [True, False])
def test_completed_tool_block(decoder, include_eos):
    encoding, block = decoder
    text = f'\n\n<｜DSML｜{block}>\n<｜DSML｜invoke name="test">\n</｜DSML｜invoke>\n</｜DSML｜{block}>'
    if include_eos:
        text += encoding.eos_token
    parsed = encoding.parse_message_from_completion_text(text, 'chat')
    assert parsed['tool_calls'] == [{'type': 'function', 'function': {'name': 'test', 'arguments': '{}'}}]
