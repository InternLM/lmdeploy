from typing import Literal

import pytest
from utils.restful_return_check import (assert_chat_completions_batch_return, assert_chat_completions_stream_return,
                                        get_repeat_times)

from lmdeploy.serve.openai.api_client import APIClient, get_model_list

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2_5-20b-chat'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.pytorch
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceBase:

    def test_get_model(self, config):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        assert model_name == '/'.join([config.get('model_path'), MODEL]), api_client.available_models

        model_list = get_model_list(BASE_URL + '/v1/models')
        assert model_name in model_list, model_list

    def test_encode(self):
        api_client = APIClient(BASE_URL)
        input_ids1, length1 = api_client.encode('Hi, pls intro yourself')
        input_ids2, length2 = api_client.encode('Hi, pls intro yourself', add_bos=False)
        input_ids3, length3 = api_client.encode('Hi, pls intro yourself', do_preprocess=True)
        input_ids4, length4 = api_client.encode('Hi, pls intro yourself', do_preprocess=True, add_bos=False)
        input_ids5, length5 = api_client.encode('Hi, pls intro yourself' * 100, add_bos=False)

        assert len(input_ids1) == length1 and length1 > 0
        assert len(input_ids2) == length2 and length2 > 0
        assert len(input_ids3) == length3 and length3 > 0
        assert len(input_ids4) == length4 and length4 > 0
        assert len(input_ids5) == length5 and length5 > 0
        assert length1 == length2 + 1
        assert input_ids2 == input_ids1[1:]
        assert input_ids1[0] == 1 and input_ids3[0] == 1
        assert length5 == length2 * 100
        assert input_ids5 == input_ids2 * 100


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatCompletions:

    def test_return_info_with_prompt(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_return_info_with_messegae(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[{
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     }],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_return_info_with_prompt_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_return_info_with_messegae_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[{
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     }],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_single_stopword(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_single_stopword_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' to' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert ' 上海' not in output.get('choices')[0].get('message').get('content')
        assert ' to' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert '上海' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert ' to' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_special_words(self):
        message = '<|im_start|>system\n当开启工具以及代码时，根据需求选择合适的工具进行调用\n' + \
                '<|im_end|><|im_start|>system name=<|interpreter|>\n你现在已经' + \
                '能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。当你向 python ' + \
                '发送含有 Python >代码的消息时，它将在该环境中执行。这个工具适用于多种场景，' + \
                '如数据分析或处理（包括数据操作、统计分析、图表绘制），复杂的计算问题（解决数学和物理' + \
                '难题），编程示例（理解编程概念或特性），文本处理和分析（比如文本解析和自然语言处理），机器学习和数据科学（用于' + \
                '展示模型训练和数据可视化），以及文件操作和数据导入（处理CSV、JSON等格式的文件）。<|im_end|>\n' + \
                '<|im_start|>user\n设 $L$ 为圆周$x^2+y^2=2x$，计算曲线积分：$I=\\int_L' + \
                '{x\\mathrm{d}s}=$<|im_end|>\n<|im_start|>assistant'
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=message,
                                                     skip_special_tokens=False,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert '<|action_start|><|interpreter|>' in output.get('choices')[0].get('message').get('content')

        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=message,
                                                     skip_special_tokens=True,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert '<|action_start|><|interpreter|>' not in output.get('choices')[0].get('message').get('content')

    def test_minimum_repetition_penalty(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert ' is is' * 5 in output.get('choices')[0].get('message').get('content') or ' a a' * 5 in output.get(
            'choices')[0].get('message').get('content')

    def test_minimum_repetition_penalty_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        response = ''
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     stream=True,
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     max_tokens=200):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        assert get_repeat_times(response, 'pls intro yourself') > 5 or get_repeat_times(response, ' pls ') > 5

    def test_repetition_penalty_bigger_than_1(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Shanghai is',
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_repetition_penalty_bigger_than_1_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     stream=True,
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     max_tokens=200):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            continue

    def test_minimum_topp(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages='Shanghai is',
                                                         top_p=0.1,
                                                         max_tokens=10):
                outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') == outputList[1].get('choices')[0].get(
            'message').get('content')
        assert outputList[1].get('choices')[0].get('message').get('content') == outputList[2].get('choices')[0].get(
            'message').get('content')

    def test_minimum_topp_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages='Hi, pls intro yourself',
                                                         stream=True,
                                                         top_p=0.1,
                                                         max_tokens=10):
                outputList.append(output)
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] == responseList[1] or responseList[1] == responseList[2]

    def test_mistake_modelname_return(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model='error', messages='Hi, pls intro yourself',
                                                     temperature=0.01):
            continue
        assert output.get('code') == 404
        assert output.get('message') == 'The model `error` does not exist.'
        assert output.get('object') == 'error'

    def test_mistake_modelname_return_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(model='error',
                                                     messages='Hi, pls intro yourself',
                                                     stream=True,
                                                     max_tokens=5,
                                                     temperature=0.01):
            outputList.append(output)
        assert output.get('code') == 404
        assert output.get('message') == 'The model `error` does not exist.'
        assert output.get('object') == 'error'
        assert len(outputList) == 1

    def test_mutilple_times_response_should_not_same(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(model=model_name, messages='Shanghai is', max_tokens=100):
                outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') != outputList[1].get('choices')[0].get(
            'message').get('content') or outputList[1].get('choices')[0].get('message').get(
                'content') != outputList[2].get('choices')[0].get('message').get('content')

    def test_mutilple_times_response_should_not_same_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages='Shanghai is',
                                                         stream=True,
                                                         max_tokens=100):
                outputList.append(output)
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[1] == responseList[2]

    def test_longtext_input(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself' * 100000,
                                                     temperature=0.01):
            continue
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('choices')[0].get('message').get('content') == ''

    def test_longtext_input_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself' * 100000,
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], model_name, is_last=True)
        assert outputList[0].get('choices')[0].get('finish_reason') == 'length'
        assert outputList[0].get('choices')[0].get('delta').get('content') == ''
        assert len(outputList) == 1

    def test_input_validation(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', top_p=0):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The top_p `0.0` must be in (0, 1].'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', top_p=1.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The top_p `1.01` must be in (0, 1].'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', top_p='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid number' in str(output)

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', n=0):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The n `0` must be a positive int.'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', n='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid integer' in str(output)

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', temperature=-0.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The temperature `-0.01` must be in [0, 2]'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', temperature=2.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The temperature `2.01` must be in [0, 2]'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', temperature='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid number' in str(output)

    def test_input_validation_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, top_p=0):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The top_p `0.0` must be in (0, 1].'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, top_p=1.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The top_p `1.01` must be in (0, 1].'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, top_p='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid number' in str(output)

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, n=0):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The n `0` must be a positive int.'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, n='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid integer' in str(output)

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, temperature=-0.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The temperature `-0.01` must be in [0, 2]'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, temperature=2.01):
            continue
        assert output.get('code') == 400
        assert output.get('message') == 'The temperature `2.01` must be in [0, 2]'
        assert output.get('object') == 'error'

        for output in api_client.chat_completions_v1(model=model_name, messages='Hi', stream=True, temperature='test'):
            continue
        assert output.get('code') is None
        assert 'Input should be a valid number' in str(output)

    def test_ignore_eos(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, what is your name?',
                                                     ignore_eos=True,
                                                     max_tokens=100,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('usage').get('completion_tokens') == 101 or output.get('usage').get(
            'completion_tokens') == 100
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, what is your name?',
                                                     ignore_eos=True,
                                                     stream=True,
                                                     max_tokens=100,
                                                     temperature=0.01):
            outputList.append(output)
        response = ''
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 100 or length == 101

    def __test_max_tokens_or_max_completion_tokens(
        self,
        max_tokens_or_max_completion_tokens: Literal['max_tokens', 'max_completion_tokens'],
    ):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        if max_tokens_or_max_completion_tokens == 'max_tokens':
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages='Hi, pls intro yourself',
                    max_tokens=5,
                    temperature=0.01,
            ):
                continue
        else:
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages='Hi, pls intro yourself',
                    max_completion_tokens=5,
                    temperature=0.01,
            ):
                continue
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    def test_max_tokens(self):
        self.__test_max_tokens_or_max_completion_tokens('max_tokens')

    def test_max_completion_tokens(self):
        self.__test_max_tokens_or_max_completion_tokens('max_completion_tokens')

    def __test_max_tokens_streaming_or_max_completion_tokens_streaming(
        self,
        max_tokens_or_max_completion_tokens: Literal['max_tokens', 'max_completion_tokens'],
    ):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        if max_tokens_or_max_completion_tokens == 'max_tokens':
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages='Hi, pls intro yourself',
                    stream=True,
                    max_tokens=5,
                    temperature=0.01,
            ):
                outputList.append(output)
        else:
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages='Hi, pls intro yourself',
                    stream=True,
                    max_completion_tokens=5,
                    temperature=0.01,
            ):
                outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    def test_max_tokens_streaming(self):
        self.__test_max_tokens_streaming_or_max_completion_tokens_streaming('max_tokens')

    def test_max_completion_tokens_streaming(self):
        self.__test_max_tokens_streaming_or_max_completion_tokens_streaming('max_completion_tokens')

    @pytest.mark.not_pytorch
    def test_logprobs(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     max_tokens=5,
                                                     temperature=0.01,
                                                     logprobs=True,
                                                     top_logprobs=10):
            continue
        assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    @pytest.mark.not_pytorch
    def test_logprobs_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages='Hi, pls intro yourself',
                                                     stream=True,
                                                     max_tokens=5,
                                                     temperature=0.01,
                                                     logprobs=True,
                                                     top_logprobs=10):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True, check_logprobs=True, logprobs_num=10)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name, check_logprobs=True, logprobs_num=10)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6
