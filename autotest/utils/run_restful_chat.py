import json
import os
import random
import string
import subprocess
from time import sleep, time

import allure
import psutil
from openai import OpenAI
from pytest_assume.plugin import assume
from utils.config_utils import get_cuda_prefix_by_workerid, get_workerid
from utils.get_run_config import get_command_with_extra
from utils.rule_condition_assert import assert_result
from utils.run_client_chat import command_line_test

from lmdeploy.serve.openai.api_client import APIClient
from lmdeploy.utils import is_bf16_supported

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


def start_restful_api(config, param, model, model_path, backend_type, worker_id):
    log_path = config.get('log_path')

    cuda_prefix = param['cuda_prefix']
    tp_num = param['tp_num']
    if 'extra' in param.keys():
        extra = param['extra']
    else:
        extra = None

    if 'modelscope' in param.keys():
        modelscope = param['modelscope']
        if modelscope:
            os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
            model_path = model

    if cuda_prefix is None:
        cuda_prefix = get_cuda_prefix_by_workerid(worker_id, tp_num=tp_num)

    if tp_num > 1 and 'gw' in worker_id:
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)

    worker_num = get_workerid(worker_id)
    if worker_num is None:
        port = DEFAULT_PORT
    else:
        port = DEFAULT_PORT + worker_num

    cmd = get_command_with_extra('lmdeploy serve api_server ' + model_path + ' --session-len 8096 --server-port ' +
                                 str(port),
                                 config,
                                 model,
                                 need_tp=True,
                                 cuda_prefix=cuda_prefix,
                                 extra=extra)

    if backend_type == 'turbomind':
        if ('w4' in model or '4bits' in model or 'awq' in model.lower()):
            cmd += ' --model-format awq'
        elif 'gptq' in model.lower():
            cmd += ' --model-format gptq'
    if backend_type == 'pytorch':
        cmd += ' --backend pytorch'
        if not is_bf16_supported():
            cmd += ' --dtype float16'
    if 'quant_policy' in param.keys() and param['quant_policy'] is not None:
        quant_policy = param['quant_policy']
        cmd += f' --quant-policy {quant_policy}'

    if not is_bf16_supported():
        cmd += ' --cache-max-entry-count 0.5'

    start_log = os.path.join(log_path, 'start_restful_' + model.split('/')[1] + worker_id + '.log')

    print('reproduce command restful: ' + cmd)

    with open(start_log, 'w') as f:
        f.writelines('reproduce command restful: ' + cmd + '\n')

        startRes = subprocess.Popen([cmd], stdout=f, stderr=f, shell=True, text=True, encoding='utf-8')
        pid = startRes.pid

    http_url = BASE_HTTP_URL + ':' + str(port)
    with open(start_log, 'r') as file:
        content = file.read()
        print(content)
    start_time = int(time())

    start_timeout = 300
    if not is_bf16_supported():
        start_timeout = 600

    sleep(5)
    for i in range(start_timeout):
        sleep(1)
        end_time = int(time())
        total_time = end_time - start_time
        result = health_check(http_url)
        if result or total_time >= start_timeout:
            break
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)
    return pid, startRes


def stop_restful_api(pid, startRes, param):
    if pid > 0:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    if 'modelscope' in param.keys():
        modelscope = param['modelscope']
        if modelscope:
            del os.environ['LMDEPLOY_USE_MODELSCOPE']
    if 'MASTER_PORT' in os.environ:
        del os.environ['MASTER_PORT']


def run_all_step(config, cases_info, worker_id: str = '', port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'
    for case in cases_info.keys():
        if ('coder' in model.lower() or 'codellama' in model.lower()) and 'code' not in case:
            continue

        case_info = cases_info.get(case)

        with allure.step(case + ' step1 - command chat regression'):
            chat_result, chat_log, msg = command_line_test(config, case, case_info, model, 'api_client', http_url,
                                                           worker_id)
            allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert chat_result, msg

        with allure.step(case + ' step2 - restful_test - openai chat'):
            restful_result, restful_log, msg = open_chat_test(config, case, case_info, model, http_url, worker_id)
            allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert restful_result, msg

        with allure.step(case + ' step3 - restful_test - interactive chat'):
            active_result, interactive_log, msg = interactive_test(config, case, case_info, model, http_url, worker_id)
            allure.attach.file(interactive_log, attachment_type=allure.attachment_type.TEXT)

        with assume:
            assert active_result, msg


def open_chat_test(config, case, case_info, model, url, worker_id: str = ''):
    log_path = config.get('log_path')

    restful_log = os.path.join(log_path, 'restful_' + model + worker_id + '_' + case + '.log')

    file = open(restful_log, 'w')

    result = True

    api_client = APIClient(url)
    model_name = api_client.available_models[0]

    messages = []
    msg = ''
    for prompt_detail in case_info:
        if not result:
            break
        prompt = list(prompt_detail.keys())[0]
        messages.append({'role': 'user', 'content': prompt})
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1, max_tokens=256):
            output_message = output.get('choices')[0].get('message')
            messages.append(output_message)

            output_content = output_message.get('content')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content, prompt_detail.values(), model_name)
            file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
            if not case_result:
                msg += reason
            result = result & case_result
    file.close()
    return result, restful_log, msg


def interactive_test(config, case, case_info, model, url, worker_id: str = ''):
    log_path = config.get('log_path')

    interactive_log = os.path.join(log_path, 'interactive_' + model + worker_id + '_' + case + '.log')

    file = open(interactive_log, 'w')

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' + ','.join(api_client.available_models) + '\n')

    # Randomly generate 6 characters and concatenate them into a string.
    characters = string.digits
    random_chars = ''.join(random.choice(characters) for i in range(6))

    messages = []
    msg = ''
    for prompt_detail in case_info:
        prompt = list(prompt_detail.keys())[0]
        new_prompt = {'role': 'user', 'content': prompt}
        messages.append(new_prompt)
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_interactive_v1(prompt=prompt,
                                                     interactive_mode=True,
                                                     session_id=random_chars,
                                                     top_k=1,
                                                     request_output_len=256):
            output_content = output.get('text')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content, prompt_detail.values(), model)
            file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
            if not case_result:
                msg += reason
            result = result & case_result
    file.close()
    return result, interactive_log, msg


def health_check(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        messages = []
        messages.append({'role': 'user', 'content': '你好'})
        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            return True
    except Exception:
        return False


def get_model(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        return model_name.split('/')[-1]
    except Exception:
        return None


PIC = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'  # noqa E501
PIC2 = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg'  # noqa E501


def run_vl_testcase(config, port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)
    log_path = config.get('log_path')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    restful_log = os.path.join(log_path, 'restful_vl_' + model_name.split('/')[-1] + str(port) + '.log')
    file = open(restful_log, 'w')

    prompt_messages = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC,
            },
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC2,
            },
        }],
    }]

    response = client.chat.completions.create(model=model_name, messages=prompt_messages, temperature=0.8, top_p=0.8)
    file.writelines(str(response).lower() + '\n')

    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    for item in api_client.chat_completions_v1(model=model_name, messages=prompt_messages):
        continue
    file.writelines(str(item) + '\n')
    file.close()

    allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)

    assert 'tiger' in str(response).lower() or '虎' in str(response).lower() or 'ski' in str(
        response).lower() or '滑雪' in str(response).lower(), response
    assert 'tiger' in str(item).lower() or '虎' in str(item).lower() or 'ski' in str(item).lower() or '滑雪' in str(
        item).lower(), item


def run_reasoning_case(config, port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)
    log_path = config.get('log_path')

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    restful_log = os.path.join(log_path, 'restful_reasoning_' + model + str(port) + '.log')
    file = open(restful_log, 'w')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    with allure.step('step1 - stream'):
        messages = [{'role': 'user', 'content': '9.11 and 9.8, which is greater?'}]
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01, stream=True)
        outputList = []
        final_content = ''
        final_reasoning_content = ''
        for stream_response in response:
            if stream_response.choices[0].delta.content is not None:
                final_content += stream_response.choices[0].delta.content
            if stream_response.choices[0].delta.reasoning_content is not None:
                final_reasoning_content += stream_response.choices[0].delta.reasoning_content
            outputList.append(stream_response)
        file.writelines(str(outputList) + '\n')
        with assume:
            assert '9.11' in final_reasoning_content and '9.11' in final_content and len(outputList) > 1, str(
                outputList)

    with allure.step('step2 - batch'):
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01, stream=False)
        print(response)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        file.writelines(str(outputList) + '\n')
        with assume:
            assert '9.11' in reasoning_content and '9.11' in content and len(outputList) > 1, str(outputList)

    file.close()
    allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)


def test_internlm_multiple_round_prompt(client, model):

    def add(a: int, b: int):
        return a + b

    def mul(a: int, b: int):
        return a * b

    tools = [{
        'type': 'function',
        'function': {
            'name': 'add',
            'description': 'Compute the sum of two numbers',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'int',
                        'description': 'A number',
                    },
                    'b': {
                        'type': 'int',
                        'description': 'A number',
                    },
                },
                'required': ['a', 'b'],
            },
        }
    }, {
        'type': 'function',
        'function': {
            'name': 'mul',
            'description': 'Calculate the product of two numbers',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'int',
                        'description': 'A number',
                    },
                    'b': {
                        'type': 'int',
                        'description': 'A number',
                    },
                },
                'required': ['a', 'b'],
            },
        }
    }]
    messages = [{'role': 'user', 'content': 'Compute (3+5)*2'}]

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.01,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list = [response]
    func1_name = response.choices[0].message.tool_calls[0].function.name
    func1_args = response.choices[0].message.tool_calls[0].function.arguments
    func1_out = eval(f'{func1_name}(**{func1_args})')
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
    with assume:
        assert func1_name == 'add'
    with assume:
        assert func1_args == '{"a": 3, "b": 5}'
    with assume:
        assert func1_out == 8
    with assume:
        assert response.choices[0].message.tool_calls[0].type == 'function'

    messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
    messages.append({'role': 'environment', 'content': f'3+5={func1_out}', 'name': 'plugin'})
    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list.append(response)
    func2_name = response.choices[0].message.tool_calls[0].function.name
    func2_args = response.choices[0].message.tool_calls[0].function.arguments
    func2_out = eval(f'{func2_name}(**{func2_args})')
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
    with assume:
        assert func2_name == 'mul'
    with assume:
        assert func2_args == '{"a": 8, "b": 2}'
    with assume:
        assert func2_out == 16
    with assume:
        assert response.choices[0].message.tool_calls[0].type == 'function'

    return response_list


def test_qwen_multiple_round_prompt(client, model):

    def get_current_temperature(location: str, unit: str = 'celsius'):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            'temperature': 26.1,
            'location': location,
            'unit': unit,
        }

    def get_temperature_date(location: str, date: str, unit: str = 'celsius'):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            'temperature': 25.9,
            'location': location,
            'date': date,
            'unit': unit,
        }

    def get_function_by_name(name):
        if name == 'get_current_temperature':
            return get_current_temperature
        if name == 'get_temperature_date':
            return get_temperature_date

    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_current_temperature',
            'description': 'Get current temperature at a location.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description':
                        'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': ['location']
            }
        }
    }, {
        'type': 'function',
        'function': {
            'name': 'get_temperature_date',
            'description': 'Get temperature at a location and date.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description':
                        'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'date': {
                        'type': 'string',
                        'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': ['location', 'date']
            }
        }
    }]
    messages = [{
        'role': 'user',
        'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?'
    }]

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list = [response]
    with assume:
        assert False, 'test'

    messages.append(response.choices[0].message)

    for tool_call in response.choices[0].message.tool_calls:
        tool_call_args = json.loads(tool_call.function.arguments)
        tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
        messages.append({
            'role': 'tool',
            'name': tool_call.function.name,
            'content': tool_call_result,
            'tool_call_id': tool_call.id
        })

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list.append(response)
    with assume:
        assert False, 'test'

    return response_list


def run_tools_case(config, port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)
    log_path = config.get('log_path')

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    restful_log = os.path.join(log_path, 'restful_reasoning_' + model + str(port) + '.log')
    file = open(restful_log, 'w')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    with allure.step('step1 - one_round_prompt'):
        tools = [{
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'description': 'Get the current weather in a given location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and state, e.g. San Francisco, CA',
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit']
                        },
                    },
                    'required': ['location'],
                },
            }
        }]
        messages = [{'role': 'user', 'content': "What's the weather like in Boston today?"}]
        response = client.chat.completions.create(model=model_name,
                                                  messages=messages,
                                                  temperature=0.01,
                                                  stream=False,
                                                  tools=tools)
        print(response)
        with assume:
            assert response.choices[0].finish_reason == 'tool_calls'
        with assume:
            assert response.choices[0].message.tool_calls[0].function.name == 'get_current_weather'
        with assume:
            assert 'Boston' in response.choices[0].message.tool_calls[0].function.arguments
        with assume:
            assert response.choices[0].message.tool_calls[0].type == 'function'
        file.writelines(str(response) + '\n')

    with allure.step('step2 - search prompt'):
        tools = [{
            'type': 'function',
            'function': {
                'name': 'search',
                'description': 'BING search API',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'list of search query strings'
                        }
                    },
                    'required': ['location']
                }
            }
        }]
        messages = [{'role': 'user', 'content': '搜索最近的人工智能发展趋势'}]
        response = client.chat.completions.create(model=model_name,
                                                  messages=messages,
                                                  temperature=0.01,
                                                  stream=False,
                                                  tools=tools)
        print(response)
        with assume:
            assert response.choices[0].finish_reason == 'tool_calls'
        with assume:
            assert response.choices[0].message.tool_calls[0].function.name == 'search'
        with assume:
            assert '人工智能' in response.choices[0].message.tool_calls[0].function.arguments
        with assume:
            assert response.choices[0].message.tool_calls[0].type == 'function'
        file.writelines(str(response) + '\n')

    with allure.step('step3 - multiple_round_prompt'):
        if 'intern' in model.lower():
            response_list = test_internlm_multiple_round_prompt(client, model_name)
        if 'qwen' in model.lower():
            response_list = test_qwen_multiple_round_prompt(client, model_name)

        file.writelines(str(response_list) + '\n')

    file.close()
    allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)
