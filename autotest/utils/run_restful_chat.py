import json
import os
import subprocess
import time

import allure
import requests
from openai import OpenAI
from pytest_assume.plugin import assume
from utils.config_utils import get_case_str_by_config, get_cli_common_param, get_cuda_prefix_by_workerid, get_workerid
from utils.constant import DEFAULT_PORT
from utils.restful_return_check import assert_chat_completions_batch_return
from utils.rule_condition_assert import assert_result

from lmdeploy.serve.openai.api_client import APIClient

MASTER_ADDR = os.getenv('MASTER_ADDR', 'localhost')
BASE_HTTP_URL = f'http://{MASTER_ADDR}'


def start_openai_service(config, run_config, worker_id):
    case_name = get_case_str_by_config(run_config)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    server_log = os.path.join(config.get('server_log_path'), f'log_{case_name}_{timestamp}.log')

    port = DEFAULT_PORT + get_workerid(worker_id)
    model = run_config.get('model')
    if run_config.get('env', {}).get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        model_path = model
    else:
        model_path = os.path.join(config.get('model_path'), model)

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))

    # Ensure extra_params exists before modifying
    if 'extra_params' not in run_config:
        run_config['extra_params'] = {}
    run_config['extra_params']['server-port'] = str(port)
    run_config['extra_params']['allow-terminate-by-client'] = None
    model_name = case_name if run_config['extra_params'].get(
        'model-name', None) is None else run_config['extra_params'].pop('model-name')
    cmd = ' '.join([
        cuda_prefix, 'lmdeploy serve api_server', model_path,
        get_cli_common_param(run_config), f'--model-name {model_name}'
    ]).strip()

    env = os.environ.copy()
    env['MASTER_PORT'] = str(get_workerid(worker_id) + 29500)
    env.update(run_config.get('env', {}))

    file = open(server_log, 'w')
    print('reproduce command restful: ' + cmd)
    file.write('reproduce command restful: ' + cmd + '\n')
    startRes = subprocess.Popen(cmd,
                                stdout=file,
                                stderr=file,
                                shell=True,
                                text=True,
                                env=env,
                                encoding='utf-8',
                                errors='replace',
                                start_new_session=True)
    pid = startRes.pid

    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    start_time = int(time.time())
    start_timeout = 720

    time.sleep(5)
    for i in range(start_timeout):
        time.sleep(1)
        end_time = int(time.time())
        total_time = end_time - start_time
        result = health_check(http_url, case_name)
        if result or total_time >= start_timeout:
            break
        try:
            # Check if process is still running
            return_code = startRes.wait(timeout=1)  # Small timeout to check status
            if return_code != 0:
                with open(server_log, 'r') as f:
                    content = f.read()
                    print(content)
                return 0, content
        except subprocess.TimeoutExpired:
            continue
    file.close()
    allure.attach.file(server_log, attachment_type=allure.attachment_type.TEXT)
    return pid, ''


def stop_restful_api(pid, startRes):
    if pid > 0:
        startRes.terminate()


def terminate_restful_api(worker_id):
    port = DEFAULT_PORT + get_workerid(worker_id)
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    response = None
    request_error = None
    try:
        response = requests.get(f'{http_url}/terminate')
    except requests.exceptions.RequestException as exc:
        request_error = exc
    if request_error is not None:
        assert False, f'terminate request failed: {request_error}'
    assert response is not None and response.status_code == 200, f'terminate with {response}'


def run_all_step(log_path, case_name, cases_info, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'
    for case in cases_info.keys():
        case_info = cases_info.get(case)

        with allure.step(case + ' restful_test - openai chat'):
            restful_result, restful_log, msg = open_chat_test(log_path, case_name, case, case_info, model, http_url,
                                                              port)
            allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert restful_result, msg


def open_chat_test(log_path, case_name, case, case_info, model, url, port: int = DEFAULT_PORT):
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    restful_log = os.path.join(log_path, f'log_restful_{case_name}_{timestamp}.log')

    file = open(restful_log, 'w')

    result = True

    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{url}/v1')
    model_name = client.models.list().data[0].id

    messages = []
    msg = ''
    for prompt_detail in case_info:
        if not result:
            break
        prompt = list(prompt_detail.keys())[0]
        messages.append({'role': 'user', 'content': prompt})
        file.writelines('prompt:' + prompt + '\n')

        response = client.chat.completions.create(model=model_name,
                                                  messages=messages,
                                                  temperature=0.01,
                                                  top_p=0.8,
                                                  max_completion_tokens=1024)

        output_content = response.choices[0].message.content
        file.writelines('output:' + output_content + '\n')
        messages.append({'role': 'assistant', 'content': output_content})

        case_result, reason = assert_result(output_content, prompt_detail.values(), model_name)
        file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
        if not case_result:
            msg += reason
        result = result and case_result
    file.close()
    return result, restful_log, msg


def health_check(url, model_name):
    try:
        api_client = APIClient(url)
        model_name_current = api_client.available_models[0]
        messages = []
        messages.append({'role': 'user', 'content': '你好'})
        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            # Return True on first successful response
            return model_name == model_name_current
        return False  # No output received
    except Exception:
        return False


def get_model(url):
    print(url)
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        return model_name.split('/')[-1]
    except Exception:
        return None


def _run_logprobs_test(port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    output = None
    for output in api_client.chat_completions_v1(model=model_name,
                                                 messages='Hi, pls intro yourself',
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10):
        continue
    if output is None:
        assert False, 'No output received from logprobs test'
    print(output)
    assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
    assert output.get('choices')[0].get('finish_reason') == 'length'
    assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5


PIC = 'tiger.jpeg'  # noqa E501
PIC2 = 'human-pose.jpg'  # noqa E501


def run_vl_testcase(log_path, resource_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)
    if model is None:
        assert False, 'server not start correctly'

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    simple_model_name = model_name.split('/')[-1]
    restful_log = os.path.join(log_path, f'restful_vl_{simple_model_name}_{str(port)}_{timestamp}.log')  # noqa
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
                'url': f'{resource_path}/{PIC}',
            },
        }, {
            'type': 'image_url',
            'image_url': {
                'url': f'{resource_path}/{PIC2}',
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


def _run_reasoning_case(log_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    restful_log = os.path.join(log_path, f'restful_reasoning_{model}_{str(port)}_{timestamp}.log')
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
    func1_args_dict = json.loads(func1_args)
    func1_out = add(**func1_args_dict) if func1_name == 'add' else mul(**func1_args_dict)
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
    func2_args_dict = json.loads(func2_args)
    func2_out = add(**func2_args_dict) if func2_name == 'add' else mul(**func2_args_dict)
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
            location: The location to get the temperature for, in the format 'City, State, Country'.
            date: The date to get the temperature for, in the format 'Year-Month-Day'.
            unit: The unit to return the temperature in. Defaults to 'celsius'. (choices: ['celsius', 'fahrenheit'])

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
    func1_name = response.choices[0].message.tool_calls[0].function.name
    func1_args = response.choices[0].message.tool_calls[0].function.arguments
    func2_name = response.choices[0].message.tool_calls[1].function.name
    func2_args = response.choices[0].message.tool_calls[1].function.arguments
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
        assert func1_name == 'get_current_temperature'
        assert func1_args == '{"location": "San Francisco, CA, USA"}' \
            or func1_args == '{"location": "San Francisco, California, USA", "unit": "celsius"}'
        assert func2_name == 'get_temperature_date'
        assert func2_args == '{"location": "San Francisco, CA, USA", "date": "2024-11-15"}' \
            or func2_args == '{"location": "San Francisco, California, USA", "date": "2024-11-15", "unit": "celsius"}'
        assert response.choices[0].message.tool_calls[0].type == 'function'

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
        assert response.choices[0].finish_reason == 'stop'
        assert '26.1' in response.choices[0].message.content

    return response_list


def _run_tools_case(log_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    restful_log = os.path.join(log_path, f'restful_toolcall_{model}_{str(port)}_{timestamp}.log')
    file = open(restful_log, 'w')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    with open(restful_log, 'a') as file:
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
            messages = [{'role': 'user', 'content': 'What\'s the weather like in Boston today?'}]
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
            response_list = None
            if 'intern' in model.lower():
                response_list = test_internlm_multiple_round_prompt(client, model_name)
            elif 'qwen' in model.lower():
                response_list = test_qwen_multiple_round_prompt(client, model_name)

            if response_list is not None:
                file.writelines(str(response_list) + '\n')

    allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)


def proxy_health_check(url):
    """Check if proxy server is healthy."""
    try:
        # For proxy server, we check if it responds to the /v1/models endpoint
        import requests
        response = requests.get(f'{url}/v1/models', timeout=5)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False


def start_proxy_server(log_path, port, case_name: str = ''):
    """Start the proxy server for testing with enhanced error handling and
    logging."""
    if log_path is None:
        log_path = '/nvme/qa_test_models/evaluation_report'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    proxy_log = os.path.join(log_path, f'proxy_server_{str(port)}_{timestamp}.log')

    proxy_url = f'http://127.0.0.1:{port}'  # noqa: E231, E261
    try:
        response = requests.get(f'{proxy_url}/nodes/status', timeout=5)
        if response.status_code == 200:
            print(f'Terminating existing nodes on proxy {proxy_url}')
            requests.get(f'{proxy_url}/nodes/terminate_all', timeout=10)
            time.sleep(5)
    except requests.exceptions.RequestException:
        pass

    cmd = (f'lmdeploy serve proxy --server-name 127.0.0.1 --server-port {port} '
           f'--routing-strategy min_expected_latency --serving-strategy Hybrid')

    print(f'Starting proxy server with command: {cmd}')
    print(f'Proxy log will be saved to: {proxy_log}')

    proxy_file = open(proxy_log, 'w')
    proxy_process = subprocess.Popen([cmd],
                                     stdout=proxy_file,
                                     stderr=proxy_file,
                                     shell=True,
                                     text=True,
                                     encoding='utf-8')
    pid = proxy_process.pid

    start_time = int(time.time())
    timeout = 300

    time.sleep(5)
    for i in range(timeout):
        time.sleep(1)
        if proxy_health_check(f'http://127.0.0.1:{port}'):  # noqa: E231, E261
            break

        try:
            # Check if process is still running
            return_code = proxy_process.wait(timeout=1)  # Small timeout to check status
            if return_code != 0:
                with open(proxy_log, 'r') as f:
                    content = f.read()
                    print(content)
                return 0, proxy_process
        except subprocess.TimeoutExpired:
            continue

        end_time = int(time.time())
        total_time = end_time - start_time
        if total_time >= timeout:
            break

    proxy_file.close()
    allure.attach.file(proxy_log, attachment_type=allure.attachment_type.TEXT)

    print(f'Proxy server started successfully with PID: {pid}')
    return pid, proxy_process


def run_llm_test(config, run_config, common_case_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            case_name = get_case_str_by_config(run_config)
            run_all_step(config.get('log_path'),
                         case_name,
                         common_case_config,
                         port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_mllm_test(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            run_vl_testcase(config.get('log_path'),
                            config.get('resource_path'),
                            port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_reasoning_case(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_reasoning_case(config.get('log_path'), port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_tools_case(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_tools_case(config.get('log_path'), port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_logprob_test(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_logprobs_test(port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)
