import re

INPUT_LENGTH_ERROR = 'internal error happened, status code ResponseType.INPUT_LENGTH_ERROR'


def get_chat_message_text(choice):
    msg = choice.get('message') or {}
    texts = []
    for key in ('reasoning_content', 'content'):
        value = msg.get(key)
        if isinstance(value, str):
            texts.append(value)
    return ''.join(texts)


def get_chat_delta_text(choice):
    delta = choice.get('delta') or {}
    texts = []
    for key in ('reasoning_content', 'content'):
        value = delta.get(key)
        if isinstance(value, str):
            texts.append(value)
    return ''.join(texts)


def assert_chat_message_error(choice, error_message=INPUT_LENGTH_ERROR):
    msg = choice.get('message') or {}
    assert msg.get('content') == error_message or msg.get('reasoning_content') == error_message


def assert_chat_delta_error(choice, error_message=INPUT_LENGTH_ERROR):
    delta = choice.get('delta') or {}
    assert delta.get('content') == error_message or delta.get('reasoning_content') == error_message


def assert_chat_message_empty(choice):
    assert not get_chat_message_text(choice)


def assert_chat_delta_empty(choice):
    assert not get_chat_delta_text(choice)


def assert_chat_completions_batch_return(output, model_name, check_logprobs: bool = False, logprobs_num: int = 5):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        msg = message.get('message') or {}
        content = msg.get('content')
        reasoning = msg.get('reasoning_content')
        assert (isinstance(content, str) and len(content) > 0) or (
            isinstance(reasoning, str) and len(reasoning) > 0)
        assert msg.get('role') == 'assistant'
        if check_logprobs:
            len(message.get('logprobs').get('content')) == output.get('usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_completions_batch_return(output, model_name, check_logprobs: bool = False, logprobs_num: int = 5):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None
    assert output.get('object') == 'text_completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        assert len(message.get('text')) > 0
        if check_logprobs:
            len(message.get('logprobs').get('content')) == output.get('usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_usage(usage):
    assert usage.get('prompt_tokens') > 0
    assert usage.get('total_tokens') > 0
    assert usage.get('completion_tokens') > 0
    assert usage.get('completion_tokens') + usage.get('prompt_tokens') == usage.get('total_tokens')


def assert_logprobs(logprobs, logprobs_num):
    assert_logprob_element(logprobs)
    assert len(logprobs.get('top_logprobs')) >= 0
    assert type(logprobs.get('top_logprobs')) is list
    assert len(logprobs.get('top_logprobs')) <= logprobs_num
    for logprob_element in logprobs.get('top_logprobs'):
        assert_logprob_element(logprob_element)


def assert_logprob_element(logprob):
    assert len(logprob.get('token')) > 0 and type(logprob.get('token')) is str
    assert len(logprob.get('bytes')) > 0 and type(logprob.get('bytes')) is list
    assert type(logprob.get('logprob')) is float


def assert_chat_completions_stream_return(output,
                                          model_name,
                                          is_last: bool = False,
                                          check_logprobs: bool = False,
                                          logprobs_num: int = 5):
    print(output)
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion.chunk'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('delta').get('role') == 'assistant'
        assert message.get('index') == 0
        delta = message.get('delta') or {}
        assert isinstance(delta.get('content'), str) or isinstance(delta.get('reasoning_content'), str)
        if not is_last:
            assert message.get('finish_reason') is None
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1)
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)
        if is_last is True:
            content = delta.get('content')
            reasoning = delta.get('reasoning_content')
            assert content is None or len(content) == 0 or 'error' in content
            assert reasoning is None or len(reasoning) == 0 or 'error' in reasoning
            assert message.get('finish_reason') in ['stop', 'length', 'error']
            if check_logprobs is True:
                assert message.get('logprobs') is None


def assert_completions_stream_return(output,
                                     model_name,
                                     is_last: bool = False,
                                     check_logprobs: bool = False,
                                     logprobs_num: int = 5):
    print(output)
    assert output.get('id') is not None
    assert output.get('object') == 'text_completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('index') == 0
        assert len(message.get('text')) >= 0
        if is_last is False:
            assert message.get('finish_reason') is None
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1)
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)

        if is_last is True:
            assert len(message.get('text')) == 0
            assert message.get('finish_reason') in ['stop', 'length']
            if check_logprobs is True:
                assert message.get('logprobs') is None


def has_repeated_fragment(text, repeat_count=5):
    pattern = r'(.+?)\1{' + str(repeat_count - 1) + ',}'
    match = re.search(pattern, text.replace('\n', ''))
    if match:
        repeated_fragment = match.group(1)
        start_pos = match.start()
        return True, {'repeated_fragment': repeated_fragment, 'position': start_pos}
    return False, f'{text} does not contain repeated fragments'
