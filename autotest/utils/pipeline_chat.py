import json
import os
import subprocess
from subprocess import PIPE

import allure
from pytest_assume.plugin import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result


def run_pipeline_chat_test(config,
                           cases_info,
                           model_case,
                           backend_type,
                           worker_id: str = '',
                           extra: object = None,
                           use_local_model: bool = True,
                           is_smoke: bool = False):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)

    # temp remove testcase because of issue 3434
    if ('InternVL3' in model_case or 'InternVL2_5' in model_case or 'MiniCPM-V-2_6' in model_case
        ) and 'turbomind' in backend_type and extra is not None and 'communicator' in extra and extra.get(
            'communicator') == 'cuda-ipc' and tp > 1:
        return
    model_name = model_name = get_model_name(model_case)
    model_path = config.get('model_path')
    if use_local_model is True:
        hf_path = model_path + '/' + model_case
    else:
        hf_path = model_case

    pipeline_chat_log = os.path.join(
        log_path, '_'.join(['pipeline', 'chat', backend_type, worker_id,
                            model_case.split('/')[1] + '.log']))

    if str(config.get('env_tag')) == '3090' or str(config.get('env_tag')) == '5080':
        if extra is None:
            extra = {}
        extra['cache-max-entry-count'] = 0.6

    if extra is not None:
        extra = json.dumps(extra, ensure_ascii=False, indent=None)
        extra = extra.replace(' ', '').replace('"', '\\"').replace(',', '\\,')
    env = os.environ.copy()
    with open(pipeline_chat_log, 'w') as f:
        cmd = f'python3 autotest/tools/pipeline/llm_case.py run_pipeline_chat_test {hf_path} autotest/prompt_case.yaml {tp} {backend_type} {is_smoke} {extra}'  # noqa E501

        f.writelines('reproduce command: ' + cmd + '\n')
        print('reproduce command: ' + cmd)
        # quantization
        try:
            response = subprocess.run([cmd],
                                      shell=True,
                                      capture_output=True,
                                      text=True,
                                      encoding='utf-8',
                                      env=env,
                                      timeout=900)
        except subprocess.TimeoutExpired as e:
            assert False, f'Test command timed out after 15 minutes: {e.cmd}'

        output_text = response.stdout
        print(output_text)
        f.writelines(output_text)

        if response.returncode != 0:
            assert False, 'system error: ' + response.stderr

        for case in cases_info.keys():
            if ('coder' in model_case or 'CodeLlama' in model_case) and 'code' not in case:
                continue
            if is_smoke and case != 'memory_test':
                continue

            with allure.step(case):
                case_info = cases_info.get(case)

                for prompt_detail in case_info:
                    prompt = list(prompt_detail.keys())[0]
                    case_result, reason = assert_result(get_response_from_output_by_prompt(output_text, case, prompt),
                                                        prompt_detail.values(), model_name)
                    if not case_result:
                        print(f'{case} result: {case_result}, reason: {reason} \n')
                    f.writelines(f'{case} result: {case_result}, reason: {reason} \n')
                with assume:
                    assert case_result, reason
    allure.attach.file(pipeline_chat_log, attachment_type=allure.attachment_type.TEXT)


def run_pipeline_vl_chat_test(config,
                              model_case,
                              backend_type,
                              worker_id: str = '',
                              extra: object = None,
                              is_smoke: bool = False):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_path = config.get('model_path')
    resource_path = config.get('resource_path')
    hf_path = model_path + '/' + model_case

    if ('InternVL3' in model_case or 'InternVL2_5' in model_case or 'MiniCPM-V-2_6' in model_case
        ) and 'turbomind' in backend_type and extra is not None and 'communicator' in extra and extra.get(
            'communicator') == 'cuda-ipc' and tp > 1:
        return

    pipeline_chat_log = os.path.join(
        log_path, '_'.join(['pipeline', 'mllm', backend_type, worker_id,
                            model_case.split('/')[1] + '.log']))

    if str(config.get('env_tag')) == '3090' or str(config.get('env_tag')) == '5080':
        if extra is None:
            extra = {}
        extra['cache-max-entry-count'] = 0.5

    if extra is not None:
        extra = json.dumps(extra, ensure_ascii=False, indent=None)
        extra = extra.replace(' ', '').replace('"', '\\"').replace(',', '\\,')
    env = os.environ.copy()
    with open(pipeline_chat_log, 'w') as f:
        cmd = f'python3 autotest/tools/pipeline/mllm_case.py run_pipeline_mllm_test {hf_path} {resource_path} {tp} {backend_type} {is_smoke} {extra}'  # noqa E501

        f.writelines('reproduce command: ' + cmd + '\n')
        print('reproduce command: ' + cmd)
        # quantization
        try:
            response = subprocess.run([cmd],
                                      shell=True,
                                      capture_output=True,
                                      text=True,
                                      encoding='utf-8',
                                      env=env,
                                      timeout=900)
        except subprocess.TimeoutExpired as e:
            assert False, f'Test command timed out after 15 minutes: {e.cmd}'

        output_text = response.stdout
        print(output_text)
        f.writelines(output_text)

        if response.returncode != 0:
            assert False, 'system error: ' + response.stderr
        with allure.step('single1 pic'):
            response = get_response_from_output(output_text, 'single1')
            case_result = any(word in response.lower() for word in ['tiger', '虎'])
            f.writelines(f'single1 pic result: {case_result} reason: simple example tiger should in {response} \n')
            with assume:
                assert case_result, f'reason: simple example tiger should in {response}'
        with allure.step('single2 pic'):
            response = get_response_from_output(output_text, 'single2')
            case_result = any(word in response.lower() for word in ['tiger', '虎'])
            f.writelines(f'single2 pic result: {case_result} reason: simple example tiger should in {response} \n')
            with assume:
                assert case_result, f'reason: simple example tiger should in {response}'
        with allure.step('multi-imagese'):
            response = get_response_from_output(output_text, 'multi-imagese')
            case_result = any(word in response.lower() for word in ['tiger', '虎', '滑雪', 'ski'])
            f.writelines(f'multi-imagese pic result: {case_result} reason: tiger or ski should in {response} \n')
            with assume:
                assert case_result, f'reason: Multi-images example: tiger or ski should in {response}'
        with allure.step('batch-example1'):
            response = get_response_from_output(output_text, 'batch-example1')
            case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
            f.writelines(f'batch-example1 pic result: {case_result} reason: ski should in {response} \n')
            with assume:
                assert case_result, f'reason: batch-example1: ski should in {response}'
        with allure.step('batch-example2'):
            response = get_response_from_output(output_text, 'batch-example2')
            case_result = any(word in response.lower() for word in ['tiger', '虎'])
            f.writelines(f'batch-example2 pic result: {case_result} reason: tiger should in {response} \n')
            with assume:
                assert case_result, f'reason: batch-example1: tiger should in {response}'
        with allure.step('multi-turn1'):
            response = get_response_from_output(output_text, 'multi-turn1')
            case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
            f.writelines(f'multi-turn1 pic result: {case_result} reason:  ski should in {response} \n')
            with assume:
                assert case_result, f'reason: batch-example1: ski should in {response}'
        with allure.step('multi-turn2'):
            response = get_response_from_output(output_text, 'multi-turn2')
            case_result = any(word in response.lower() for word in ['滑雪', 'ski'])
            f.writelines(f'multi-turn2 pic result: {case_result} reason: ski should in {response} \n')
            with assume:
                assert case_result, f'reason: batch-example1: ski should in {response}'
        if not is_smoke:
            if 'internvl' in model_case.lower() and 'internvl2-4b' not in model_case.lower():
                internvl_vl_testcase(output_text, f)
                internvl_vl_testcase(output_text, f, 'cn')
            if 'minicpm' in model_case.lower():
                MiniCPM_vl_testcase(output_text, f)
            if 'qwen' in model_case.lower():
                Qwen_vl_testcase(output_text, f)
    allure.attach.file(pipeline_chat_log, attachment_type=allure.attachment_type.TEXT)


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
    for i in range(0, len(output) - 1):
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


def internvl_vl_testcase(output_text, f, lang: str = 'en'):
    with allure.step(f'internvl-combined-images-{lang}'):
        response = get_response_from_output(output_text, f'internvl-combined-images-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        f.writelines(f'internvl-combined-images-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step(f'internvl-combined-images2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-combined-images2-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        f.writelines(f'internvl-combined-images2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images2: panda should in {response}'
    with allure.step(f'internvl-separate-images-{lang}'):
        response = get_response_from_output(output_text, f'internvl-separate-images-{lang}')
        case_result = any(word in response.lower() for word in ['panda', '熊猫', 'same', 'different'])
        f.writelines(f'internvl-separate-images-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: separate images: panda should in {response}'
    with allure.step(f'internvl-separate-images2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-separate-images2-{lang}')
        case_result = any(word in response.lower()
                          for word in ['panda', '熊猫', 'same', 'different', 'difference', 'identical'])
        f.writelines(f'internvl-separate-images2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: separate images2: panda should in {response}'
    with allure.step(f'internvl-video-{lang}'):
        response = get_response_from_output(output_text, f'internvl-video-{lang}')
        case_result = any(word in response.lower() for word in ['red panda', 'eat', '熊猫', '竹子', 'food', 'hold'])
        f.writelines(f'internvl-video-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video: panda should in {response}'
    with allure.step(f'internvl-video2-{lang}'):
        response = get_response_from_output(output_text, f'internvl-video2-{lang}')
        case_result = any(word in response.lower() for word in ['red panda', 'eat', '熊猫', '竹子'])
        f.writelines(f'internvl-video2-{lang} result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video2: panda should in {response}'


def MiniCPM_vl_testcase(output_text, f):
    with allure.step('minicpm-combined-images'):
        response = get_response_from_output(output_text, 'minicpm-combined-images')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        f.writelines(f'minicpm-combined-images result: {case_result}, reason:  panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('minicpm-combined-images2'):
        response = get_response_from_output(output_text, 'minicpm-combined-images2')
        case_result = any(word in response.lower() for word in ['panda', '熊猫'])
        f.writelines(f'minicpm-combined-images2 result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images2: panda should in {response}'
    with allure.step('minicpm-fewshot'):
        response = get_response_from_output(output_text, 'minicpm-fewshot')
        case_result = any(word in response.lower() for word in ['2021', '14'])
        f.writelines(f'minicpm-fewshot result: {case_result} reason: 2021 or 14 should in {response} \n')
        with assume:
            assert case_result, f'reason: fewshot: 2021 or 14 should in {response}'
    with allure.step('minicpm-video'):
        response = get_response_from_output(output_text, 'minicpm-video')
        case_result = any(word in response.lower() for word in ['red panda', '熊猫'])
        f.writelines(f'minicpm-video result: {case_result} reason: video: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: video: panda should in {response}'


def Qwen_vl_testcase(output_text, f):
    with allure.step('qwen-combined-images'):
        response = get_response_from_output(output_text, 'qwen-combined-images')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        f.writelines(f'qwen-combined-images result: {case_result}, reason: buildings should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('qwen-combined-images2'):
        response = get_response_from_output(output_text, 'qwen-combined-images2')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        f.writelines(f'qwen-combined-images2 result: {case_result}, reason: buildings should in {response} \n')
        with assume:
            assert case_result, f'reason: combined images: panda should in {response}'
    with allure.step('qwen-performance-images'):
        response = get_response_from_output(output_text, 'qwen-performance-images')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        f.writelines(f'qwen-performance-images result: {case_result}, reason: panda should in {response} \n')
        with assume:
            assert case_result, f'reason: performance images: panda should in {response}'
    with allure.step('qwen-performance-images2'):
        response = get_response_from_output(output_text, 'qwen-performance-images2')
        case_result = any(word in response.lower() for word in ['buildings', '楼', 'skyline', 'city'])
        f.writelines(f'qwen-performance-images2 result: {case_result}, reason: panda should in {response} \n')
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
