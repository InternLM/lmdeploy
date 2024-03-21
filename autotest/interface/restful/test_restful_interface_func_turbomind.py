import pytest
from utils.restful_return_check import (assert_chat_completions_batch_return,
                                        assert_chat_completions_stream_return,
                                        assert_chat_interactive_batch_return,
                                        assert_chat_interactive_stream_return)

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2-chat-20b'
MODEL_NAME = 'internlm2-chat-20b'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatCompletions:

    def test_chat_completions_ignore_eos_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, what is your name?',
                ignore_eos=True,
                max_tokens=100,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert output.get('usage').get('completion_tokens') == 101
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_chat_completions_ignore_eos_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, what is your name?',
                ignore_eos=True,
                stream=True,
                max_tokens=100,
                temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert len(outputList) == 103

    def test_chat_completions_max_tokens_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                max_tokens=5,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6

    def test_chat_completions_max_tokens_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                max_tokens=5,
                temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert len(outputList) == 8

    def test_chat_completions_repetition_penalty_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        response = ''
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                repetition_penalty=0.1,
                temperature=0.01,
                max_tokens=200):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
            response += outputList[index].get('choices')[0].get('delta').get(
                'content')
        assert 'pls pls ' * 5 in response or \
            'Hi, pls intro yourself\n' * 5 in response

    def test_chat_completions_topp_min_batch(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME, messages='Shanghai is', top_p=0.1):
                outputList.append(output)
            assert_chat_completions_batch_return(output, MODEL_NAME)
        assert outputList[0].get('choices')[0].get('message').get(
            'content') == outputList[1].get('choices')[0].get('message').get(
                'content')
        assert outputList[1].get('choices')[0].get('message').get(
            'content') == outputList[2].get('choices')[0].get('message').get(
                'content')

    def test_chat_completions_topp_min_stream(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME,
                    messages='Hi, pls intro yourself',
                    stream=True,
                    top_p=0.1):
                outputList.append(output)
            assert_chat_completions_stream_return(outputList[0], MODEL_NAME,
                                                  True, False)
            assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                                  False, True)
            for index in range(1, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index],
                                                      MODEL_NAME)
                response += outputList[index].get('choices')[0].get(
                    'delta').get('content')
            responseList.append(response)
        assert responseList[0] == responseList[1]
        assert responseList[1] == responseList[2]

    def test_chat_completions_longinput_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself' * 10000,
                stream=True,
                temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert outputList[1].get('choices')[0].get('finish_reason') == 'length'
        assert outputList[1].get('choices')[0].get('delta').get(
            'content') == ''
        assert len(outputList) == 2


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatInteractive:

    def test_chat_interactive_ignore_eos_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, what is your name?',
                ignore_eos=True,
                request_output_len=100,
                temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert output.get('tokens') == 101
        assert output.get('finish_reason') == 'length'

    def test_chat_interactive_ignore_eos_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(
                prompt='Hi, what is your name?',
                ignore_eos=True,
                stream=True,
                request_output_len=100,
                temperature=0.01):
            outputList.append(output)
        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
        assert output.get('finish_reason') == 'length'
        assert len(outputList) == 102

    def test_chat_interactive_max_tokens_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself',
                request_output_len=5,
                temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert output.get('finish_reason') == 'length'
        assert output.get('tokens') == 6

    def test_chat_interactive_max_tokens_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself',
                stream=True,
                request_output_len=5,
                temperature=0.01):
            outputList.append(output)
        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
        assert output.get('finish_reason') == 'length'
        assert len(outputList) == 7

    def test_chat_interactive_topp_min_batch(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                         top_p=0.01):
                continue
            assert_chat_interactive_batch_return(output)
            outputList.append(output)
        assert outputList[0] == outputList[1]
        assert outputList[1] == outputList[2]

    def test_chat_interactive_topp_min_stream(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_interactive_v1(
                    model=MODEL_NAME,
                    prompt='Hi, pls intro yourself',
                    stream=True,
                    top_p=0.01):
                outputList.append(output)
            assert_chat_interactive_stream_return(outputList[-1],
                                                  True,
                                                  index=len(outputList) - 2)
            for index in range(0, len(outputList) - 1):
                assert_chat_interactive_stream_return(outputList[index],
                                                      index=index)
                response += outputList[index].get('text')
            responseList.append(response)
        assert responseList[0] == responseList[1]
        assert responseList[1] == responseList[2]
