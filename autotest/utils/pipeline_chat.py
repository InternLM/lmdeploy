import os
import subprocess
from subprocess import PIPE

import allure
import torch
from pytest import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result

from lmdeploy import pipeline
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


def run_pipeline_chat_test(config,
                           cases_info,
                           model_case,
                           type,
                           worker_id: str = '',
                           extra: object = None,
                           use_local_model: bool = True):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_name = model_name = get_model_name(model_case)
    model_path = config.get('model_path')
    if use_local_model is True:
        hf_path = model_path + '/' + model_case
    else:
        hf_path = model_case

    if 'pytorch' == type:
        backend_config = PytorchEngineConfig(tp=tp)
    elif 'pytorch_lora' == type:
        backend_config = PytorchEngineConfig(tp=tp,
                                             adapters=extra.get('adapters'))
    else:
        backend_config = TurbomindEngineConfig(tp=tp)

    if 'kvint' in type:
        backend_config.quant_policy = extra.get('quant_policy')

    # if llava support kvint or awq, this code should refactor
    if 'llava' in model_case:
        backend_config.model_name = 'vicuna'
    if 'w4' in model_case or ('4bits' in model_case
                              or 'awq' in model_case.lower()):
        backend_config.model_format = 'awq'
    if 'gptq' in model_case.lower():
        backend_config.model_format = 'gptq'

    pipe = pipeline(hf_path, backend_config=backend_config)

    config_log = os.path.join(
        log_path, '_'.join([
            'pipeline', 'config', type, worker_id,
            model_case.split('/')[1] + '.log'
        ]))
    file = open(config_log, 'w')
    log_string = '\n'.join([
        'reproduce config info:',
        'from lmdeploy.messages import PytorchEngineConfig',
        'from lmdeploy.messages import TurbomindEngineConfig',
        'engine_config = ' + str(backend_config),
        'pipe = pipeline("' + hf_path + '",  backend_config=engine_config)',
        'res = pipe("Hi, pls introduce shanghai")'
    ])
    file.writelines(log_string)
    print(log_string)
    file.close

    for case in cases_info.keys():
        if ('coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path, '_'.join([
                'pipeline', 'chat', type, worker_id,
                model_case.split('/')[1], case + '.log'
            ]))

        file = open(pipeline_chat_log, 'w')

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            file.writelines('prompt:' + prompt + '\n')

            response = pipe([prompts])[0].text

            case_result, reason = assert_result(response,
                                                prompt_detail.values(),
                                                model_name)
            prompts.append({'role': 'assistant', 'content': response})
            file.writelines('output:' + response + '\n')
            file.writelines('result:' + str(case_result) + ', reason:' +
                            reason + '\n')
        file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_chat_log(config,
                             cases_info,
                             model_case,
                             type,
                             worker_id: str = ''):
    log_path = config.get('log_path')

    config_log = os.path.join(
        log_path, '_'.join([
            'pipeline', 'config', type, worker_id,
            model_case.split('/')[1] + '.log'
        ]))

    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)

    for case in cases_info.keys():
        if ('coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        msg = 'result is empty, please check again'
        result = False
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, '_'.join([
                    'pipeline', 'chat', type, worker_id,
                    model_case.split('/')[1], case + '.log'
                ]))

            allure.attach.file(pipeline_chat_log,
                               attachment_type=allure.attachment_type.TEXT)

            with open(pipeline_chat_log, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if 'result:False, reason:' in line:
                        result = False
                        msg = line
                        break
                    if 'result:True, reason:' in line and not result:
                        result = True
                        msg = ''

            with assume:
                assert result, msg


def save_pipeline_common_log(config,
                             log_name,
                             result,
                             content,
                             msg: str = '',
                             write_type: str = 'w'):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    file = open(config_log, write_type)
    file.writelines(f'result:{result}, reason: {msg}, content: {content}')
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


def assert_pipeline_single_return(output, logprobs_num: int = 0):
    result = assert_pipeline_single_element(output,
                                            is_last=True,
                                            logprobs_num=logprobs_num)
    if not result:
        return result, 'single_stream_element is wrong'
    return result & (len(output.token_ids) == output.generate_token_len
                     or len(output.token_ids) == output.generate_token_len -
                     1), 'token_is len is not correct'


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
        if not assert_pipeline_single_element(
                output[i], is_stream=True, logprobs_num=logprobs_num):
            return False, f'single_stream_element is false, index is {i}'
    if assert_pipeline_single_element(
            output[-1], is_stream=True, is_last=True,
            logprobs_num=logprobs_num) is False:
        return False, 'last single_stream_element is false'
    return True, ''


def assert_pipeline_batch_stream_return(output, size: int = 1):
    for i in range(size):
        output_list = [item for item in output if item.session_id == i]
        result, msg = assert_pipeline_single_stream_return(output_list)
        if not result:
            return result, msg
    return True, ''


def assert_pipeline_single_element(output,
                                   is_stream: bool = False,
                                   is_last: bool = False,
                                   logprobs_num: int = 0):
    result = True
    result &= output.generate_token_len > 0
    result &= output.input_token_len > 0
    result &= output.session_id >= 0
    if is_last:
        result &= len(output.text) >= 0
        result &= output.finish_reason in ['stop', 'length']
        if is_stream:
            result &= output.token_ids is None
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
            result &= len(output.logprobs) == 1
        else:
            result &= len(output.logprobs) == output.generate_token_len or len(
                output.logprobs) == output.generate_token_len + 1
        if result:
            for content in output.logprobs:
                result &= len(content.keys()) <= logprobs_num
                for key in content.keys():
                    result &= type(content.get(key)) == float
    return result


PIC1 = 'https://raw.githubusercontent.com/' + \
    'open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
PIC2 = 'https://raw.githubusercontent.com/' + \
    'open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg'


def run_pipeline_vl_chat_test(config, model_case):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_path = config.get('model_path')
    hf_path = model_path + '/' + model_case

    if 'llava' in model_case:
        backend_config = TurbomindEngineConfig(tp=tp,
                                               session_len=8192,
                                               model_name='vicuna')
    else:
        backend_config = TurbomindEngineConfig(tp=tp, session_len=8192)
    if '4bit' in model_case.lower() or 'awq' in model_case.lower():
        backend_config.model_format = 'awq'
    pipe = pipeline(hf_path, backend_config=backend_config)

    pipeline_chat_log = os.path.join(
        log_path, 'pipeline_vl_chat_' + model_case.split('/')[1] + '.log')
    file = open(pipeline_chat_log, 'w')

    image = load_image(PIC1)

    if 'deepseek' in model_case:
        prompt = f'describe this image{IMAGE_TOKEN}'
    else:
        prompt = 'describe this image'
    response = pipe((prompt, image))
    result = 'tiger' in response.text.lower() or '虎' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: simple example tiger not in ' + response.text +
                    '\n')

    prompts = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': prompt
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC1
            }
        }]
    }]
    response = pipe(prompts)
    result = 'tiger' in response.text.lower() or '虎' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: OpenAI format example: tiger not in ' +
                    response.text + '\n')

    image_urls = [PIC2, PIC1]
    images = [load_image(img_url) for img_url in image_urls]
    response = pipe((prompt, images))
    result = 'tiger' in response.text.lower() or 'ski' in response.text.lower(
    ) or '虎' in response.text.lower() or '滑雪' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: Multi-images example: tiger or ski not in ' +
                    response.text + '\n')

    image_urls = [PIC2, PIC1]
    prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts)
    result = ('ski' in response[0].text.lower()
              or '滑雪' in response[0].text.lower()) and (
                  'tiger' in response[1].text.lower()
                  or '虎' in response[1].text.lower())
    file.writelines('result:' + str(result) +
                    ', reason: Batch example: ski or tiger not in ' +
                    str(response) + '\n')

    image = load_image(PIC2)
    sess = pipe.chat((prompt, image))
    result = 'ski' in sess.response.text.lower(
    ) or '滑雪' in sess.response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: Multi-turn example: ski not in ' +
                    sess.response.text + '\n')
    sess = pipe.chat('What is the woman doing?', session=sess)
    result = 'ski' in sess.response.text.lower(
    ) or '滑雪' in sess.response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: Multi-turn example: ski not in ' +
                    sess.response.text + '\n')

    file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_vl_chat_log(config, model_case):
    log_path = config.get('log_path')

    pipeline_chat_log = os.path.join(
        log_path, 'pipeline_vl_chat_' + model_case.split('/')[1] + '.log')

    allure.attach.file(pipeline_chat_log,
                       attachment_type=allure.attachment_type.TEXT)

    msg = 'result is empty, please check again'
    result = False
    with open(pipeline_chat_log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'result:False, reason:' in line:
                result = False
                msg = line
                break
            if 'result:True, reason:' in line and not result:
                result = True
                msg = ''

    with assume:
        assert result, msg
