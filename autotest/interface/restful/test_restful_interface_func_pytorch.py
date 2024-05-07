import pytest
from utils.restful_return_check import (assert_chat_completions_batch_return,
                                        assert_chat_completions_stream_return,
                                        assert_chat_interactive_batch_return,
                                        assert_chat_interactive_stream_return)

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2-chat-20b'
MODEL_NAME = 'internlm2'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(8)
@pytest.mark.pytorch
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatCompletions:

    def test_ignore_eos(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, what is your name?',
                ignore_eos=True,
                max_tokens=100,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert output.get('usage').get(
            'completion_tokens') == 101 or output.get('usage').get(
                'completion_tokens') == 100
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self):
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
        response = ''
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
            response += outputList[index].get('choices')[0].get('delta').get(
                'content')
        input_ids1, length = api_client.encode(response)
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert length == 101

    def test_max_tokens(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                max_tokens=5,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get(
            'usage').get('completion_tokens') == 5

    def test_max_tokens_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                max_tokens=5,
                temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
            response += outputList[index].get('choices')[0].get('delta').get(
                'content')
        input_ids1, length = api_client.encode(response)
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert length == 6


@pytest.mark.order(8)
@pytest.mark.pytorch
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatInteractive:

    def test_ignore_eos(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, what is your name?',
                ignore_eos=True,
                request_output_len=100,
                temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert output.get('tokens') == 100
        assert output.get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self):
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
        assert outputList[-1].get('tokens') == 100

    def test_max_tokens(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself',
                request_output_len=5,
                temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert output.get('finish_reason') == 'length'
        assert output.get('tokens') == 5

    def test_max_tokens_streaming(self):
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
        assert outputList[-1].get('tokens') == 5
