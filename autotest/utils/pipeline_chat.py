import json
import os
import subprocess
from subprocess import PIPE

import allure
from pytest_assume.plugin import assume
from utils.common_utils import execute_command_with_logging
from utils.config_utils import get_case_str_by_config, get_cuda_prefix_by_workerid, get_workerid
from utils.rule_condition_assert import assert_result


def run_pipeline_llm_test(config, run_config, common_case_config, worker_id: str = '', is_smoke: bool = False):
    model = run_config.pop('model')
    if run_config.get('env', {}).get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        model_path = model
    else:
        model_path = config.get('model_path') + '/' + model

    log_path = config.get('log_path')
    case_name = get_case_str_by_config(run_config)
    os.makedirs(log_path, exist_ok=True)
    pipeline_log = os.path.join(log_path, f'pipeline_llm_{case_name}.log')

    env = os.environ.copy()
    env['MASTER_PORT'] = str(get_workerid(worker_id) + 29500)
    env.update(run_config.pop('env', {}))

    run_config_string = json.dumps(run_config, ensure_ascii=False, indent=None)
    run_config_string = run_config_string.replace(' ', '').replace('"', '\\"').replace(',', '\\,')

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))
    cmd = f'{cuda_prefix} python3 autotest/tools/pipeline/llm_case.py run_pipeline_chat_test {model_path} {run_config_string} autotest/prompt_case.yaml {is_smoke}'  # noqa E501

    result, stderr = execute_command_with_logging(cmd, pipeline_log, timeout=1800, env=env)

    with assume:
        assert result, stderr

    with open(pipeline_log, 'r', encoding='utf-8') as file:
        output_text = file.read()

    file = open(pipeline_log, 'a')
    for case in common_case_config.keys():
        if is_smoke and case != 'memory_test':
            continue

        with allure.step(case):
            case_info = common_case_config.get(case)

            for prompt_detail in case_info:
                prompt = list(prompt_detail.keys())[0]
                case_result, reason = assert_result(get_response_from_output_by_prompt(output_text, case, prompt),
                                                    prompt_detail.values(), model_path)
                if not case_result:
                    print(f'{case} result: {case_result}, reason: {reason} \n')
                file.writelines(f'{case} result: {case_result}, reason: {reason} \n')
            with assume:
                assert case_result, reason
    file.close()
    allure.attach.file(pipeline_log, attachment_type=allure.attachment_type.TEXT)


def run_pipeline_mllm_test(config, run_config, worker_id: str = '', is_smoke: bool = False):
    model = run_config.pop('model')
    if run_config.get('env', {}).get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        model_path = model
    else:
        model_path = config.get('model_path') + '/' + model

    log_path = config.get('log_path')
    os.makedirs(log_path, exist_ok=True)
    case_name = get_case_str_by_config(run_config)
    pipeline_log = os.path.join(log_path, f'pipeline_mllm_{case_name}.log')

    env = os.environ.copy()
    env['MASTER_PORT'] = str(get_workerid(worker_id) + 29500)
    env.update(run_config.pop('env', {}))

    run_config_string = json.dumps(run_config, ensure_ascii=False, indent=None)
    run_config_string = run_config_string.replace(' ', '').replace('"', '\\"').replace(',', '\\,')

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))
    resource_path = config.get('resource_path')
    cmd = f'{cuda_prefix} python3 autotest/tools/pipeline/mllm_case.py run_pipeline_mllm_test {model_path} {run_config_string} {resource_path} {is_smoke}'  # noqa E501

    result, stderr = execute_command_with_logging(cmd, pipeline_log, timeout=1800, env=env)

    with assume:
        assert result, stderr

    with open(pipeline_log, 'r', encoding='utf-8') as file:
        output_text = file.read()

    file = open(pipeline_log, 'a')
    with allure.step('single1 pic'):
        response = get_response_from_output(output_text, 'single1')
        case_result = any(word in response.lower() for word in ['tiger', '虎'])
        file.writelines(f'single1 pic result: {case_result} reason: simple example tiger should in {response} \n')
        with assume:
            assert case_result, f'reason: simple example tiger should in {response}'
    with allure.step('single2 pic'):
        response = get_response_from_output(output_text, 'single2')
        case_result = any(word in response.lower() for word in ['tiger', '虎'])
        file.writelines(f'single2 pic result: {case_result} reason: simple example tiger should in {response} \n')
        with assume:
            assert case_result, f'reason: simple example tiger should in {response}'
    with allure.step('multi-imagese'):
        response = get_response_from_output(output_text, 'multi-imagese')
        case_result = any(word in response.lower() for word in ['tiger', '虎', '滑雪', 'ski'])
        file.writelines(f'multi-imagese pic result: {case_result} reason: tiger or ski should in {response} \n')
        with assume:
            assert case_result, f'reason: Multi-images example: tiger or ski should in {response}'
    with allure.step('batch-example1'):
        response = get_response_from_output(output_text, 'batch-example1')
        case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
        file.writelines(f'batch-example1 pic result: {case_result} reason: ski should in {response} \n')
        with assume:
            assert case_result, f'reason: batch-example1: ski should in {response}'
    with allure.step('batch-example2'):
        response = get_response_from_output(output_text, 'batch-example2')
        case_result = any(word in response.lower() for word in ['tiger', '虎'])
        file.writelines(f'batch-example2 pic result: {case_result} reason: tiger should in {response} \n')
        with assume:
            assert case_result, f'reason: batch-example1: tiger should in {response}'
    with allure.step('multi-turn1'):
        response = get_response_from_output(output_text, 'multi-turn1')
        case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
        file.writelines(f'multi-turn1 pic result: {case_result} reason:  ski should in {response} \n')
        with assume:
            assert case_result, f'reason: batch-example1: ski should in {response}'
    with allure.step('multi-turn2'):
        response = get_response_from_output(output_text, 'multi-turn2')
        case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
        file.writelines(f'multi-turn2 pic result: {case_result} reason: ski should in {response} \n')
        with assume:
            assert case_result, f'reason: batch-example1: ski should in {response}'
    if not is_smoke:
        if 'internvl' in model.lower() and 'internvl2-4b' not in model.lower():
            internvl_vl_testcase(output_text, file)
            internvl_vl_testcase(output_text, file, 'cn')
        if 'minicpm' in model.lower():
            MiniCPM_vl_testcase(output_text, file)
        if 'qwen' in model.lower():
            Qwen_vl_testcase(output_text, file)
    allure.attach.file(pipeline_log, attachment_type=allure.attachment_type.TEXT)


def get_response_from_output(output_text, case):
    return output_text.split('[caseresult ' + case + ' start]')[1].split('[caseresult ' + case + ' end]')[0]


def get_response_from_output_by_prompt(output_text, case, prompt):
    get_response_from_output(output_text, case)
    output_list = output_text.split('[caseresult ' + case + ' start]')[1].split('[caseresult ' + case + ' end]')[0]
    output_dict = json.loads(output_list.rstrip())
    for output in output_dict:
        if output.get('prompt') == prompt:
            return output.get('response')
    return None


def assert_pipeline_single_return(output, logprobs_num: int = 0):
    result = assert_pipeline_single_element(output, is_last=True, logprobs_num=logprobs_num)
    if not result:
        return result, 'single_stream_element is wrong'
    return result & (len(output.token_ids) == output.generate_token_len
                     or len(output.token_ids) == output.generate_token_len - 1), 'token_is len is not correct'


def assert_pipeline_batch_return(output, size: int = 1):
    if len(output) != size:
        return False, 'length is not correct'
    for single_output in output:
        result, msg = assert_pipeline_single_return(single_output)
        if not result:
            return result, msg
    return True, ''


def assert_pipeline_single_stream_return(output, logprobs_num: int = 0):
    for i in range(0, len(output) - 2):
        if not assert_pipeline_single_element(output[i], is_stream=True, logprobs_num=logprobs_num):
            return False, f'single_stream_element is false, index is {i}'
    if assert_pipeline_single_element(output[-1], is_stream=True, is_last=True, logprobs_num=logprobs_num) is False:
        return False, 'last single_stream_element is false'
    return True, ''


def assert_pipeline_batch_stream_return(output, size: int = 1):
    for i in range(size):
        output_list = [item for item in output if item.index == i]
        result, msg = assert_pipeline_single_stream_return(output_list)
        if not result:
            return result, msg
    return True, ''


def assert_pipeline_single_element(output, is_stream: bool = False, is_last: bool = False, logprobs_num: int = 0):
    result = True
    result &= output.generate_token_len > 0
    result &= output.input_token_len > 0
    result &= output.index >= 0
    if is_last:
        result &= len(output.text) >= 0
        result &= output.finish_reason in ['stop', 'length']
        if is_stream:
            result &= output.token_ids is None or output.token_ids == []
        else:
            result &= len(output.token_ids) > 0
    else:
        result &= len(output.text) > 0
        result &= output.finish_reason is None
        result &= len(output.token_ids) > 0
    if logprobs_num == 0 or (is_last and is_stream):
        result &= output.logprobs is None
    else:
        if is_stream:
            result &= len(output.logprobs) >= 1
        else:
            result &= len(output.logprobs) == output.generate_token_len or len(
                output.logprobs) == output.generate_token_len + 1
        if result:
            for content in output.logprobs:
                result &= len(content.keys()) <= logprobs_num
                for key in content.keys():
                    result &= type(content.get(key)) == float
    return result


def internvl_vl_testcase(output_text, file, lang: str = 'en'):
    with allure.step(f'internvl-combined-images-{lang}'):
        response = get_response_from_output(output_text, f'internvl-combined-images-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        file.writelines(f'internvl-combined-images-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step(f'internvl-combined-images2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-combined-images2-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        file.writelines(
            f'internvl-combined-images2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images2: panda should in {response}'
    with allure.step(f'internvl-separate-images-{lang}'):
        response = get_response_from_output(output_text, f'internvl-separate-images-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫', 'same', 'different'])
        file.writelines(f'internvl-separate-images-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: separate images: panda should in {response}'
    with allure.step(f'internvl-separate-images2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-separate-images2-{lang}')
        case_result = any(word in response.lower()
                          for word in ['panda', '熊猫', 'same', 'different', 'difference', 'identical'])
        file.writelines(
            f'internvl-separate-images2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: separate images2: panda should in {response}'
    with allure.step(f'internvl-video-{lang}'):
        response = get_response_from_output(output_text, f'internvl-video-{lang}')
        case_result = any(word in response.lower() for word in ['red panda', 'eat', '熊猫', '竹子', 'food', 'hold'])
        file.writelines(f'internvl-video-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video: panda should in {response}'
    with allure.step(f'internvl-video2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-video2-{lang}')
        case_result = any(word in response.lower() for word in ['red panda', 'eat', '熊猫', '竹子'])
        file.writelines(f'internvl-video2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video2: panda should in {response}'


def MiniCPM_vl_testcase(output_text, file):
    with allure.step('minicpm-combined-images'):
        response = get_response_from_output(output_text, 'minicpm-combined-images')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        file.writelines(f'minicpm-combined-images result: {case_result}, reason:  panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('minicpm-combined-images2'):
        response = get_response_from_output(output_text, 'minicpm-combined-images2')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        file.writelines(f'minicpm-combined-images2 result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images2: panda should in {response}'
    with allure.step('minicpm-fewshot'):
        response = get_response_from_output(output_text, 'minicpm-fewshot')
        case_result = any(word in response.lower() for word in ['2021', '14'])
        file.writelines(f'minicpm-fewshot result: {case_result} reason: 2021 or 14 should in {response} \n')
        with assume:
            assert case_result, f'reason: fewshot: 2021 or 14 should in {response}'
    with allure.step('minicpm-video'):
        response = get_response_from_output(output_text, 'minicpm-video')
        case_result = any(word in response.lower() for word in ['red panda', '熊猫'])
        file.writelines(f'minicpm-video result: {case_result} reason: video: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video: panda should in {response}'


def Qwen_vl_testcase(output_text, file):
    with allure.step('qwen-combined-images'):
        response = get_response_from_output(output_text, 'qwen-combined-images')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        file.writelines(f'qwen-combined-images result: {case_result}, reason: buildings should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('qwen-combined-images2'):
        response = get_response_from_output(output_text, 'qwen-combined-images2')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        file.writelines(f'qwen-combined-images2 result: {case_result}, reason: buildings should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('qwen-performance-images'):
        response = get_response_from_output(output_text, 'qwen-performance-images')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        file.writelines(f'qwen-performance-images result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: performance images: panda should in {response}'
    with allure.step('qwen-performance-images2'):
        response = get_response_from_output(output_text, 'qwen-performance-images2')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        file.writelines(f'qwen-performance-images2 result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: performance images: panda should in {response}'


def save_pipeline_common_log(config, log_name, result, content, msg: str = '', write_type: str = 'w'):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    file = open(config_log, write_type)
    file.writelines(f'result:{result}, reason: {msg}, content: {content}')  # noqa E231
    file.close()


def assert_pipeline_common_log(config, log_name):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)

    msg = 'result is empty, please check again'
    result = False
    with open(config_log, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'result:False, reason:' in line:
                result = False
                msg = line
                break
            if 'result:True, reason:' in line and not result:
                result = True
                msg = ''
    subprocess.run([' '.join(['rm -rf', config_log])],
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8')

    assert result, msg
