# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.serve.openai.chat_completions.logprobs import _create_chat_completion_logprobs


class _Tokenizer:

    def convert_ids_to_tokens(self, token_id):
        return f'tok{token_id}'


def _top(item):
    return [(top.token, top.logprob) for top in item.top_logprobs]


@pytest.mark.parametrize('top_logprobs', [1, 3])
def test_top_logprobs_include_selected_token_in_top_k(top_logprobs):
    # Engines return the model top-k; the selected token 1 is the most likely.
    tops = {1: -0.1, 2: -1.0, 3: -2.0}
    tops = {k: v for k, v in list(tops.items())[:top_logprobs]}
    result = _create_chat_completion_logprobs(_Tokenizer(), [1], [tops], top_logprobs)

    item = result.content[0]
    assert (item.token, item.logprob) == ('tok1', -0.1)
    assert _top(item) == [(f'tok{k}', v) for k, v in tops.items()]


def test_top_logprobs_drop_selected_token_outside_top_k():
    # The selected token 4 is appended after the model top-2.
    tops = {4: -3.0, 1: -0.1, 2: -1.0}
    result = _create_chat_completion_logprobs(_Tokenizer(), [4], [tops], 2)

    item = result.content[0]
    assert (item.token, item.logprob) == ('tok4', -3.0)
    assert _top(item) == [('tok1', -0.1), ('tok2', -1.0)]


def test_top_logprobs_zero_returns_empty_list():
    # logprobs=True without top_logprobs still asks the engine for one entry.
    tops = {4: -3.0, 1: -0.1}
    result = _create_chat_completion_logprobs(_Tokenizer(), [4], [tops], 0)

    item = result.content[0]
    assert (item.token, item.logprob) == ('tok4', -3.0)
    assert item.top_logprobs == []
