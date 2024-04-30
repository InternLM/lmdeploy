import random
from concurrent.futures import ThreadPoolExecutor
from random import randint

import pytest
from tqdm import tqdm
from utils.restful_return_check import (assert_chat_completions_batch_return,
                                        assert_chat_completions_stream_return,
                                        assert_chat_interactive_batch_return,
                                        assert_chat_interactive_stream_return,
                                        get_repeat_times)

from lmdeploy.serve.openai.api_client import APIClient, get_model_list

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2-chat-20b'
MODEL_NAME = 'internlm2'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.pytorch
@pytest.mark.chat
@pytest.mark.completion
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceBase:

    def test_get_model(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        assert model_name == MODEL_NAME, api_client.available_models

        model_list = get_model_list(BASE_URL + '/v1/models')
        assert MODEL_NAME in model_list, model_list

    def test_encode(self):
        api_client = APIClient(BASE_URL)
        input_ids1, length1 = api_client.encode('Hi, pls intro yourself')
        input_ids2, length2 = api_client.encode('Hi, pls intro yourself',
                                                add_bos=False)
        input_ids3, length3 = api_client.encode('Hi, pls intro yourself',
                                                do_preprocess=True)
        input_ids4, length4 = api_client.encode('Hi, pls intro yourself',
                                                do_preprocess=True,
                                                add_bos=False)
        input_ids5, length5 = api_client.encode('Hi, pls intro yourself' * 100,
                                                add_bos=False)

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
@pytest.mark.chat
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceIssue:

    def test_issue1232(self):

        def process_one(question):
            api_client = APIClient(BASE_URL)
            model_name = api_client.available_models[0]

            msg = [dict(role='user', content=question)]

            data = api_client.chat_interactive_v1(msg,
                                                  session_id=randint(1, 100),
                                                  repetition_penalty=1.02,
                                                  request_output_len=224)
            for item in data:
                pass

            data = api_client.chat_completions_v1(model=model_name,
                                                  messages=msg,
                                                  repetition_penalty=1.02,
                                                  stop=['<|im_end|>', '100'],
                                                  max_tokens=10)

            for item in data:
                response = item

            return response

        with ThreadPoolExecutor(max_workers=256) as executor:
            for response in tqdm(executor.map(process_one, ['你是谁'] * 500)):
                continue

    def test_issue1324_illegal_topk(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself', top_k=-1):
            continue
        assert_chat_interactive_batch_return(output)


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatCompletions:

    def test_return_info_with_prompt(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)

    def test_return_info_with_messegae(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': 'Hi, pls intro yourself'
                }],
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)

    def test_return_info_with_prompt_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)

    def test_return_info_with_messegae_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': 'Hi, pls intro yourself'
                }],
                stream=True,
                temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)

    def test_single_stopword(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert ' is' not in output.get('choices')[0].get('message').get(
            'content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_single_stopword_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
            assert ' to' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert ' is' not in output.get('choices')[0].get('message').get(
            'content')
        assert ' 上海' not in output.get('choices')[0].get('message').get(
            'content')
        assert ' to' not in output.get('choices')[0].get('message').get(
            'content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], MODEL_NAME, True,
                                              False)
        assert_chat_completions_stream_return(outputList[-1], MODEL_NAME,
                                              False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
            assert ' is' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
            assert '上海' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
            assert ' to' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
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
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages=message,
                                                     skip_special_tokens=False,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert '<|action_start|><|interpreter|>' in output.get(
            'choices')[0].get('message').get('content')

        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages=message,
                                                     skip_special_tokens=True,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert '<|action_start|><|interpreter|>' not in output.get(
            'choices')[0].get('message').get('content')

    def test_minimum_repetition_penalty(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)
        assert ' is is' * 5 in output.get('choices')[0].get('message').get(
            'content') or ' a a' * 5 in output.get('choices')[0].get(
                'message').get('content')

    def test_minimum_repetition_penalty_streaming(self):
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
            'Hi, pls intro yourself\n' * 5 in response or \
            'pls, pls, ' in response

    def test_repetition_penalty_bigger_than_1(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output, MODEL_NAME)

    def test_repetition_penalty_bigger_than_1_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                repetition_penalty=1.2,
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
            continue

    def test_minimum_topp(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME,
                    messages='Shanghai is',
                    top_p=0.1,
                    max_tokens=10):
                outputList.append(output)
            assert_chat_completions_batch_return(output, MODEL_NAME)
        assert outputList[0].get('choices')[0].get('message').get(
            'content') == outputList[1].get('choices')[0].get('message').get(
                'content')
        assert outputList[1].get('choices')[0].get('message').get(
            'content') == outputList[2].get('choices')[0].get('message').get(
                'content')

    def test_minimum_topp_streaming(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME,
                    messages='Hi, pls intro yourself',
                    stream=True,
                    top_p=0.1,
                    max_tokens=10):
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

    def test_mistake_modelname_return(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model='error', messages='Hi, pls intro yourself',
                temperature=0.01):
            continue
        assert output.get('code') == 404
        assert output.get('message') == 'The model `error` does not exist.'
        assert output.get('object') == 'error'

    def test_mistake_modelname_return_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model='error',
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
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME, messages='Shanghai is', max_tokens=100):
                outputList.append(output)
            assert_chat_completions_batch_return(output, MODEL_NAME)
        assert outputList[0].get('choices')[0].get('message').get(
            'content') != outputList[1].get('choices')[0].get('message').get(
                'content') or outputList[1].get('choices')[0].get(
                    'message').get('content') != outputList[2].get(
                        'choices')[0].get('message').get('content')

    def test_mutilple_times_response_should_not_same_streaming(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME,
                    messages='Shanghai is',
                    stream=True,
                    max_tokens=100):
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
        assert responseList[0] != responseList[1] or responseList[
            1] == responseList[2]

    def test_longtext_input(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself' * 10000,
                temperature=0.01):
            continue
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('choices')[0].get('message').get('content') == ''

    def test_longtext_input_streaming(self):
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
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index],
                                                  MODEL_NAME)
        assert outputList[1].get('choices')[0].get('finish_reason') == 'length'
        assert outputList[1].get('choices')[0].get('delta').get(
            'content') == ''
        assert len(outputList) == 2


@pytest.mark.order(8)
@pytest.mark.turbomind
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatInteractive:

    def test_return_info_with_prompt(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself', temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)

    def test_return_info_with_messegae(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt=[{
                'role':
                'user',
                'content':
                'Hi, pls intro yourself'
        }],
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)

    def test_return_info_with_prompt_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself', stream=True,
                temperature=0.01):
            outputList.append(output)
        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)

    def test_return_info_with_messegae_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(prompt=[{
                'role':
                'user',
                'content':
                'Hi, pls intro yourself'
        }],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)

    def test_single_stopword(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert ' is' not in output.get('text')
        assert output.get('finish_reason') == 'stop'

    def test_single_stopword_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=' is',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
            assert ' to' not in outputList[index].get('text')
        assert output.get('finish_reason') == 'stop'

    def test_array_stopwords(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert ' is' not in output.get('text')
        assert ' 上海' not in output.get('text')
        assert ' to' not in output.get('text')
        assert output.get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
            assert ' is' not in outputList[index].get('text')
            assert '上海' not in outputList[index].get('text')
            assert ' to' not in outputList[index].get('text')
        assert output.get('finish_reason') == 'stop'

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
        for output in api_client.chat_interactive_v1(prompt=message,
                                                     skip_special_tokens=False,
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert '<|action_start|><|interpreter|>' in output.get('text')

        for output in api_client.chat_interactive_v1(prompt=message,
                                                     skip_special_tokens=True,
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert '<|action_start|><|interpreter|>' not in output.get('text')

    def test_minimum_repetition_penalty(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     request_output_len=512):
            continue
        assert_chat_interactive_batch_return(output)
        assert 'a 上海 is a 上海, ' * 5 in output.get('text') or get_repeat_times(
            output.get('text'), 'Shanghai is') > 5

    def test_minimum_repetition_penalty_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     stream=True,
                                                     request_output_len=512):
            outputList.append(output)

        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
            response += outputList[index].get('text')
        assert 'a 上海 is a 上海, ' * 5 in response or get_repeat_times(
            response, 'Shanghai is') > 5

    def test_repetition_penalty_bigger_than_1(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     request_output_len=512):
            continue
        assert_chat_interactive_batch_return(output)

    def test_repetition_penalty_bigger_than_1_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     repetition_penalty=1.2,
                                                     stream=True,
                                                     temperature=0.01,
                                                     request_output_len=512):
            outputList.append(output)
        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 1):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)

    def test_multiple_rounds(self):
        api_client = APIClient(BASE_URL)
        history = 0
        session_id = random.randint(0, 100000)
        for i in range(3):
            for output in api_client.chat_interactive_v1(
                    prompt='Shanghai is',
                    temperature=0.01,
                    interactive_mode=True,
                    session_id=session_id):
                continue
            assert_chat_interactive_batch_return(output)
            assert output.get('history_tokens') == history
            history += output.get('input_tokens') + output.get('tokens')

    def test_multiple_rounds_streaming(self):
        api_client = APIClient(BASE_URL)
        history = 0
        session_id = random.randint(0, 100000)
        for i in range(3):
            outputList = []
            for output in api_client.chat_interactive_v1(
                    prompt='Hi, pls intro yourself',
                    stream=True,
                    temperature=0.01,
                    interactive_mode=True,
                    session_id=session_id):
                outputList.append(output)
            assert_chat_interactive_stream_return(outputList[-1],
                                                  True,
                                                  index=len(outputList) - 2)
            for index in range(0, len(outputList) - 1):
                assert_chat_interactive_stream_return(outputList[index],
                                                      index=index)
            assert outputList[-1].get('history_tokens') == history
            history += outputList[-1].get('input_tokens') + outputList[-1].get(
                'tokens')

    def test_minimum_topp(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_interactive_v1(
                    prompt='Shanghai is', top_p=0.01, request_output_len=10):
                continue
            assert_chat_interactive_batch_return(output)
            outputList.append(output)
        assert outputList[0] == outputList[1]
        assert outputList[1] == outputList[2]

    def test_minimum_topp_streaming(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_interactive_v1(
                    model=MODEL_NAME,
                    prompt='Hi, pls intro yourself',
                    stream=True,
                    top_p=0.01,
                    request_output_len=10):
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

    def test_minimum_topk(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_interactive_v1(
                    prompt='Shanghai is', top_k=1, request_output_len=10):
                continue
            assert_chat_interactive_batch_return(output)
            outputList.append(output)
        assert outputList[0] == outputList[1]
        assert outputList[1] == outputList[2]

    def test_minimum_topk_streaming(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_interactive_v1(
                    model=MODEL_NAME,
                    prompt='Hi, pls intro yourself',
                    stream=True,
                    top_k=1,
                    request_output_len=10):
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

    def test_mutilple_times_response_should_not_same(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_interactive_v1(
                    prompt='Shanghai is', request_output_len=100):
                continue
            assert_chat_interactive_batch_return(output)
            outputList.append(output)
        assert outputList[0] != outputList[1] or outputList[1] != outputList[2]

    def test_mutilple_times_response_should_not_same_streaming(self):
        api_client = APIClient(BASE_URL)
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_interactive_v1(
                    model=MODEL_NAME,
                    prompt='Hi, pls intro yourself',
                    stream=True,
                    request_output_len=100):
                outputList.append(output)
            assert_chat_interactive_stream_return(outputList[-1], True)
            for index in range(0, len(outputList) - 1):
                assert_chat_interactive_stream_return(outputList[index],
                                                      index=index)
                response += outputList[index].get('text')
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[
            1] != responseList[2]

    def test_longtext_input(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself' * 10000, temperature=0.01):
            continue
        assert output.get('finish_reason') == 'length'
        assert output.get('text') == ''

    def test_longtext_input_streaming(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself' * 10000,
                stream=True,
                temperature=0.01):
            outputList.append(output)
        assert outputList[0].get('finish_reason') == 'length', outputList
        assert outputList[0].get('text') == ''
        assert len(outputList) == 1
