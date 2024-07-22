def assert_chat_completions_batch_return(output,
                                         model_name,
                                         check_logprobs: bool = False,
                                         logprobs_num: int = 5):
    assert_usage(output.get('usage'))
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
        if check_logprobs:
            len(message.get('logprobs').get('content')) == output.get(
                'usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_completions_batch_return(output,
                                    model_name,
                                    check_logprobs: bool = False,
                                    logprobs_num: int = 5):
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
            len(message.get('logprobs').get('content')) == output.get(
                'usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_usage(usage):
    assert usage.get('prompt_tokens') > 0
    assert usage.get('total_tokens') > 0
    assert usage.get('completion_tokens') > 0
    assert usage.get('completion_tokens') + usage.get(
        'prompt_tokens') == usage.get('total_tokens')


def assert_logprobs(logprobs, logprobs_num):
    assert_logprob_element(logprobs)
    assert len(logprobs.get('top_logprobs')) >= 0
    assert type(logprobs.get('top_logprobs')) == list
    assert len(logprobs.get('top_logprobs')) <= logprobs_num
    for logprob_element in logprobs.get('top_logprobs'):
        assert_logprob_element(logprob_element)


def assert_logprob_element(logprob):
    assert len(logprob.get('token')) > 0 and type(logprob.get('token')) == str
    assert len(logprob.get('bytes')) > 0 and type(logprob.get('bytes')) == list
    assert type(logprob.get('logprob')) == float


def assert_chat_completions_stream_return(output,
                                          model_name,
                                          is_last: bool = False,
                                          check_logprobs: bool = False,
                                          logprobs_num: int = 5):
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion.chunk'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('delta').get('role') == 'assistant'
        assert message.get('index') == 0
        assert len(message.get('delta').get('content')) >= 0
        if is_last is False:
            assert message.get('finish_reason') is None
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) == 1)
                assert_logprobs(
                    message.get('logprobs').get('content')[0], logprobs_num)

        if is_last is True:
            assert len(message.get('delta').get('content')) == 0
            assert message.get('finish_reason') in ['stop', 'length']
            if check_logprobs is True:
                assert message.get('logprobs') is None


def assert_completions_stream_return(output,
                                     model_name,
                                     is_last: bool = False,
                                     check_logprobs: bool = False,
                                     logprobs_num: int = 5):
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
                assert (len(message.get('logprobs').get('content')) == 1)
                assert_logprobs(
                    message.get('logprobs').get('content')[0], logprobs_num)

        if is_last is True:
            assert len(message.get('text')) == 0
            assert message.get('finish_reason') in ['stop', 'length']
            if check_logprobs is True:
                assert message.get('logprobs') is None


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
    print(index, output)
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
