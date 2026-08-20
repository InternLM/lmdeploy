import pytest
import requests
from openai import BadRequestError
from utils.constant import BACKEND_LIST, BASE_URL, DEFAULT_MAX_COMPLETION_TOKENS, RESTFUL_MODEL_LIST
from utils.restful_return_check import (
    CONTEXT_LENGTH_ERROR,
    assert_chat_completions_batch_return,
    assert_chat_completions_stream_return,
    assert_openai_invalid_request_error,
    encode_prompt,
    get_chat_delta_text,
    get_chat_message_text,
    get_client_and_model,
    has_repeated_fragment,
)

_OVERSIZE_CHAT_PROMPT = 'Hi, pls intro yourself' * 60000
_CHAT_COMPLETIONS_URL = f'{BASE_URL}/v1/chat/completions'
_CHAT_MESSAGES = [{'role': 'user', 'content': 'Hi, pls intro yourself'}]


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulOpenAI:

    @pytest.mark.pr_test
    def test_return_info(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 temperature=0.01)

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)

    @pytest.mark.pr_test
    def test_return_info_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_single_stopword(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stop=' is')

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in get_chat_message_text(output.get('choices')[0])
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_single_stopword_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=' is',
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is ' not in get_chat_delta_text(outputList[index].get('choices')[0])
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Shanghai is'
                },
            ],
            temperature=0.01,
            stop=[' is', '上海', ' to'],
        )

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in get_chat_message_text(output.get('choices')[0])
        assert ' 上海' not in get_chat_message_text(output.get('choices')[0])
        assert ' to' not in get_chat_message_text(output.get('choices')[0])
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=[' is', '上海', ' to'],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is' not in get_chat_delta_text(outputList[index].get('choices')[0])
            assert '上海' not in get_chat_delta_text(outputList[index].get('choices')[0])
            assert ' to ' not in get_chat_delta_text(outputList[index].get('choices')[0])
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_minimum_topp(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     temperature=0.01,
                                                     top_p=0.0000000001,
                                                     max_tokens=10)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        texts = [get_chat_message_text(output.get('choices')[0]) for output in outputList]
        assert texts[0] == texts[1]
        assert texts[1] == texts[2]

    def test_minimum_topp_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     top_p=0.0000000001,
                                                     max_tokens=10,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += get_chat_delta_text(outputList[index].get('choices')[0])
            responseList.append(response)
        assert responseList[0] == responseList[1] or responseList[1] == responseList[2]

    @pytest.mark.pr_test
    def test_mistake_modelname_return(self, backend, model_case, openai_client_and_model):
        client, _ = openai_client_and_model
        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(
                model='error',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Shanghai is'
                    },
                ],
                temperature=0.01,
                stop=[' is', '上海', ' to'],
            )

    def test_mistake_modelname_return_streaming(self, backend, model_case, openai_client_and_model):
        client, _ = openai_client_and_model

        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(model='error',
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': 'Hi, pls intro yourself'
                                               },
                                           ],
                                           max_tokens=5,
                                           temperature=0.01,
                                           stream=True)

    @pytest.mark.pr_test
    def test_mutilple_times_response_should_not_same(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     max_tokens=100)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        texts = [get_chat_message_text(output.get('choices')[0]) for output in outputList]
        assert texts[0] != texts[1] or texts[1] != texts[2]

    def test_mutilple_times_response_should_not_same_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     max_tokens=100,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += get_chat_delta_text(outputList[index].get('choices')[0])
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[1] == responseList[2]

    def test_longtext_input(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        with pytest.raises(BadRequestError) as ei:
            client.chat.completions.create(model=model_name,
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': _OVERSIZE_CHAT_PROMPT,
                                               },
                                           ],
                                           max_tokens=100)
        assert ei.value.status_code == 400
        assert_openai_invalid_request_error(ei.value.body, message_substr=CONTEXT_LENGTH_ERROR)

    @pytest.mark.pr_test
    def test_longtext_input_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        with pytest.raises(BadRequestError) as ei:
            client.chat.completions.create(model=model_name,
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': _OVERSIZE_CHAT_PROMPT,
                                               },
                                           ],
                                           max_tokens=100,
                                           stream=True)
        assert ei.value.status_code == 400
        assert_openai_invalid_request_error(ei.value.body, message_substr=CONTEXT_LENGTH_ERROR)

    @pytest.mark.pr_test
    def test_max_tokens(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    def test_max_tokens_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        _, length = encode_prompt(BASE_URL, response, add_bos=False)
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True, check_logprobs=True, logprobs_num=10)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name, check_logprobs=True, logprobs_num=10)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        _, length = encode_prompt(BASE_URL, response, add_bos=False)
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    def test_minimum_repetition_penalty(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Shanghai is'}],
            extra_body={'repetition_penalty': 0.0000001, 'min_new_tokens': 100},
            temperature=0.01,
            max_tokens=200,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        result, msg = has_repeated_fragment(get_chat_message_text(output.get('choices')[0]))
        assert result, msg

    def test_minimum_repetition_penalty_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            extra_body={'repetition_penalty': 0.0000001, 'min_new_tokens': 100},
            temperature=0.01,
            max_tokens=200,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        result, msg = has_repeated_fragment(response)
        assert result, msg

    def test_repetition_penalty_bigger_than_1(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Shanghai is'}],
            extra_body={'repetition_penalty': 1.2},
            temperature=0.01,
            max_tokens=200,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)

    def test_repetition_penalty_bigger_than_1_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            extra_body={'repetition_penalty': 1.2},
            temperature=0.01,
            max_tokens=200,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_ignore_eos(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, what is your name?'}],
            extra_body={'ignore_eos': True},
            max_tokens=100,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        completion_tokens = output.get('usage', {}).get('completion_tokens')
        assert completion_tokens == 101 or completion_tokens == 100
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, what is your name?'}],
            extra_body={'ignore_eos': True},
            max_tokens=100,
            temperature=0.01,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        _, length = encode_prompt(BASE_URL, response, add_bos=False)
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length >= 99 and length <= 101

    def test_max_tokens_default_cap_no_overshoot_followup(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        max_tokens = DEFAULT_MAX_COMPLETION_TOKENS
        overshoot_slack = 1
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Continue writing forever without stopping.'}],
            extra_body={'ignore_eos': True},
            max_tokens=max_tokens,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        completion_tokens = output.get('usage', {}).get('completion_tokens')
        assert completion_tokens is not None, 'Missing usage.completion_tokens'
        assert completion_tokens <= max_tokens + overshoot_slack, (
            f'Length cap overshoot: completion_tokens={completion_tokens} > '
            f'max_tokens={max_tokens}+{overshoot_slack}')
        followup = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Say hi in one word.'}],
            max_tokens=8,
            temperature=0.01,
        )
        assert_chat_completions_batch_return(followup.model_dump(), model_name)

    def test_max_completion_tokens(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            max_completion_tokens=5,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') in (5, 6)

    def test_max_completion_tokens_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            max_completion_tokens=5,
            temperature=0.01,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        _, length = encode_prompt(BASE_URL, response, add_bos=False)
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length in (5, 6)

    @pytest.mark.parametrize(
        'invalid_payload',
        [
            pytest.param({'max_tokens': 0}, id='max_tokens_zero'),
            pytest.param({'max_tokens': -1}, id='max_tokens_negative'),
            pytest.param({'temperature': True}, id='temperature_bool'),
        ],
    )
    def test_rejects_invalid_request_parameters(
            self, backend, model_case, openai_client_and_model, invalid_payload):
        """Invalid types/ranges must return HTTP 400 (raw JSON, not SDK)."""
        _, model_name = openai_client_and_model
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            json={
                'model': model_name,
                'messages': _CHAT_MESSAGES,
                **invalid_payload,
            },
            timeout=30,
        )
        assert_openai_invalid_request_error(resp)

    def test_input_validation(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        messages = [
            {
                'role': 'user',
                'content': 'Hi, pls intro yourself'
            },
        ],
        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=0)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=1.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p='test')

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n=0)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n='test')

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=-0.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=2.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature='test')

    def test_input_validation_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        messages = [
            {
                'role': 'user',
                'content': 'Hi, pls intro yourself'
            },
        ],
        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=0, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=1.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p='test', stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n=0, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n='test', stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=-0.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=2.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature='test', stream=True)
