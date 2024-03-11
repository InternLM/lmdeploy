import random
from concurrent.futures import ThreadPoolExecutor
from random import randint

import pytest
from tqdm import tqdm

from lmdeploy.serve.openai.api_client import APIClient, get_model_list

BASE_HTTP_URL = 'http://10.140.0.187'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2-chat-20b'
MODEL_NAME = 'internlm2-chat-20b'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(7)
@pytest.mark.restful_interface_turbomind
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceBase:

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


@pytest.mark.order(7)
@pytest.mark.restful_interface_turbomind
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatCompletions:

    def test_chat_completions_check_return_batch1(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)

    def test_chat_completions_check_return_batch2(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': 'Hi, pls intro yourself'
                }],
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)

    def test_chat_completions_check_return_stream1(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                stream=True,
                temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])

    def test_chat_completions_check_return_stream2(self):
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

        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])

    def test_chat_completions_ignore_eos_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, what is your name?',
                ignore_eos=True,
                max_tokens=100,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)
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

        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert len(outputList) == 103

    def test_chat_completions_stopwords_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)
        assert ' is' not in output.get('choices')[0].get('message').get(
            'content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)
        assert ' is' not in output.get('choices')[0].get('message').get(
            'content')
        assert ' 上海' not in output.get('choices')[0].get('message').get(
            'content')
        assert ' to' not in output.get('choices')[0].get('message').get(
            'content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_chat_completions_stopwords_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=' is',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])
            assert ' to' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

        outputList = []
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])
            assert ' is' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
            assert '上海' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
            assert ' to' not in outputList[index].get('choices')[0].get(
                'delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_chat_completions_special_words_batch(self):
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
        assert_chat_completions_batch_return(output)
        assert '<|action_start|><|interpreter|>' in output.get(
            'choices')[0].get('message').get('content')

        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages=message,
                                                     skip_special_tokens=True,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)
        assert '<|action_start|><|interpreter|>' not in output.get(
            'choices')[0].get('message').get('content')

    def test_chat_completions_max_tokens_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself',
                max_tokens=5,
                temperature=0.01):
            continue
        assert_chat_completions_batch_return(output)
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
        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])
        assert outputList[-1].get('choices')[0].get(
            'finish_reason') == 'length'
        assert len(outputList) == 8

    def test_chat_completions_repetition_penalty_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model=MODEL_NAME,
                                                     messages='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output)
        assert ' is is' * 5 in output.get('choices')[0].get('message').get(
            'content')

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
        assert_chat_completions_stream_return(outputList[0], True, False)
        assert_chat_completions_stream_return(outputList[-1], False, True)
        for index in range(1, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index])
            response += outputList[index].get('choices')[0].get('delta').get(
                'content')
        assert 'pls pls ' * 5 in response, response

    def test_chat_completions_topp_min_batch(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(
                    model=MODEL_NAME, messages='Shanghai is', top_p=0.1):
                outputList.append(output)
            assert_chat_completions_batch_return(output)
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
            assert_chat_completions_stream_return(outputList[0], True, False)
            assert_chat_completions_stream_return(outputList[-1], False, True)
            for index in range(1, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index])
                response += outputList[index].get('choices')[0].get(
                    'delta').get('content')
            responseList.append(response)
        assert responseList[0] == responseList[1]
        assert responseList[1] == responseList[2]

    def test_chat_completions_mis_model_name_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model='error', messages='Hi, pls intro yourself',
                temperature=0.01):
            continue
        assert output.get('code') == 404
        assert output.get('message') == 'The model `error` does not exist.'
        assert output.get('object') == 'error'

    def test_chat_completions_mis_model_name_stream(self):
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

    def test_chat_completions_longinput_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself' * 10000,
                temperature=0.01):
            continue
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('choices')[0].get('message').get('content') == ''

    def test_chat_completions_longinput_stream(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(
                model=MODEL_NAME,
                messages='Hi, pls intro yourself' * 10000,
                stream=True,
                temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], True, False)
        assert outputList[1].get('choices')[0].get('finish_reason') == 'length'
        assert outputList[1].get('choices')[0].get('delta').get(
            'content') == ''
        assert len(outputList) == 2


@pytest.mark.order(7)
@pytest.mark.restful_interface_turbomind
@pytest.mark.flaky(reruns=2)
class TestRestfulInterfaceChatInteractive:

    def test_chat_interactive_check_return_batch1(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself', temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)

    def test_chat_interactive_check_return_batch2(self):
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

    def test_chat_interactive_check_return_stream1(self):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself', stream=True,
                temperature=0.01):
            outputList.append(output)
        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        assert_chat_interactive_stream_return(outputList[-2],
                                              False,
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 2):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)

    def test_chat_interactive_check_return_stream2(self):
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
        assert_chat_interactive_stream_return(outputList[-2],
                                              False,
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 2):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)

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

    def test_chat_interactive_stopwords_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert ' is' not in output.get('text')
        assert output.get('finish_reason') == 'stop'

        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_interactive_batch_return(output)
        assert ' is' not in output.get('text')
        assert ' 上海' not in output.get('text')
        assert ' to' not in output.get('text')
        assert output.get('finish_reason') == 'stop'

    def test_chat_interactive_stopwords_stream(self):
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
        assert_chat_interactive_stream_return(outputList[-2],
                                              False,
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 2):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
            assert ' to' not in outputList[index].get('text')
        assert output.get('finish_reason') == 'stop'

        outputList = []
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_interactive_stream_return(outputList[-1],
                                              True,
                                              index=len(outputList) - 2)
        assert_chat_interactive_stream_return(outputList[-2],
                                              False,
                                              True,
                                              index=len(outputList) - 2)
        for index in range(0, len(outputList) - 2):
            assert_chat_interactive_stream_return(outputList[index],
                                                  index=index)
            assert ' is' not in outputList[index].get('text')
            assert '上海' not in outputList[index].get('text')
            assert ' to' not in outputList[index].get('text')
        assert output.get('finish_reason') == 'stop'

    def test_chat_interactive_special_words_batch(self):
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

    def test_chat_interactive_repetition_penalty_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(prompt='Shanghai is',
                                                     repetition_penalty=0.1,
                                                     temperature=0.01,
                                                     request_output_len=512):
            continue
        assert_chat_interactive_batch_return(output)
        assert 'a 上海 is a 上海, ' * 5 in output.get('text')

    def test_chat_interactive_with_history_batch(self):
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

    def test_chat_interactive_with_history_stream(self):
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

    @pytest.mark.tmp
    def test_chat_interactive_longinput_batch(self):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_interactive_v1(
                prompt='Hi, pls intro yourself' * 10000, temperature=0.01):
            continue
        assert output.get('finish_reason') == 'length'
        assert output.get('text') == ''

    @pytest.mark.tmp
    def test_chat_interactive_longinput_stream(self):
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


def assert_chat_completions_batch_return(output):
    assert output.get('usage').get('prompt_tokens') > 0
    assert output.get('usage').get('total_tokens') > 0
    assert output.get('usage').get('completion_tokens') > 0
    assert output.get('usage').get('completion_tokens') + output.get(
        'usage').get('prompt_tokens') == output.get('usage').get(
            'total_tokens')
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion'
    assert output.get('model') == MODEL_NAME
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        assert len(message.get('message').get('content')) > 0
        assert message.get('message').get('role') == 'assistant'


def assert_chat_completions_stream_return(output,
                                          is_first: bool = False,
                                          is_last: bool = False):
    assert output.get('id') is not None
    if is_first is False:
        assert output.get('object') == 'chat.completion.chunk'
    assert output.get('model') == MODEL_NAME
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
                                          is_text_empty: bool = False,
                                          index: int = None):
    assert output.get('input_tokens') > 0
    if index is not None:
        assert output.get('tokens') >= index + 1 and output.get(
            'tokens') <= index + 6
    assert output.get('tokens') > 0
    assert output.get('history_tokens') >= 0
    if is_last:
        assert len(output.get('text')) >= 0
        assert output.get('finish_reason') in ['stop', 'length']
    elif is_text_empty:
        assert len(output.get('text')) == 0
        assert output.get('finish_reason') is None
    else:
        assert len(output.get('text')) >= 0
        assert output.get('finish_reason') is None
