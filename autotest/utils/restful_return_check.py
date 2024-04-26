def assert_chat_completions_batch_return(output, model_name):
    assert output.get('usage').get('prompt_tokens') > 0
    assert output.get('usage').get('total_tokens') > 0
    assert output.get('usage').get('completion_tokens') > 0
    assert output.get('usage').get('completion_tokens') + output.get(
        'usage').get('prompt_tokens') == output.get('usage').get(
            'total_tokens')
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        assert len(message.get('message').get('content')) > 0
        assert message.get('message').get('role') == 'assistant'


def assert_chat_completions_stream_return(output,
                                          model_name,
                                          is_first: bool = False,
                                          is_last: bool = False):
    assert output.get('id') is not None
    if is_first is False:
        assert output.get('object') == 'chat.completion.chunk'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('delta').get('role') == 'assistant'
        assert message.get('index') == 0
        if is_last is False:
            assert message.get('finish_reason') is None
        if is_first is False and is_last is False:
            assert len(message.get('delta').get('content')) >= 0
        if is_last is True:
            assert len(message.get('delta').get('content')) == 0
            assert message.get('finish_reason') in ['stop', 'length']


def assert_chat_interactive_batch_return(output):
    assert output.get('input_tokens') > 0
    assert output.get('tokens') > 0
    assert output.get('history_tokens') >= 0
    assert output.get('finish_reason') in ['stop', 'length']
    assert len(output.get('text')) > 0


def assert_chat_interactive_stream_return(output,
                                          is_last: bool = False,
                                          index: int = None):
    assert output.get('input_tokens') > 0
    if index is not None:
        assert output.get('tokens') >= index and output.get(
            'tokens') <= index + 6
    assert output.get('tokens') > 0
    assert output.get('history_tokens') >= 0
    if is_last:
        assert len(output.get('text')) >= 0
        assert output.get('finish_reason') in ['stop', 'length']
    else:
        assert len(output.get('text')) >= 0
        assert output.get('finish_reason') is None


def get_repeat_times(input, sub_input):
    time = input.count(sub_input)
    return time
