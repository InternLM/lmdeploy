import os

import allure
import torch
from pytest import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result

from lmdeploy import pipeline
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.vl import load_image


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
    elif 'kvint' in type:
        if 'w4' in model_case or ('4bits' in model_case
                                  or 'awq' in model_case.lower()):
            backend_config = TurbomindEngineConfig(
                tp=tp,
                model_format='awq',
                quant_policy=extra.get('quant_policy'))
        else:
            backend_config = TurbomindEngineConfig(
                tp=tp, quant_policy=extra.get('quant_policy'))
    # if llava support kvint or awq, this code should refactor
    elif 'llava' in model_case:
        backend_config = TurbomindEngineConfig(tp=tp, model_name='vicuna')
    else:
        if 'w4' in model_case or ('4bits' in model_case
                                  or 'awq' in model_case.lower()):
            backend_config = TurbomindEngineConfig(tp=tp, model_format='awq')
        else:
            backend_config = TurbomindEngineConfig(tp=tp)
    pipe = pipeline(hf_path, backend_config=backend_config)

    # run testcases
    gen_config = GenerationConfig(top_k=1)

    config_log = os.path.join(
        log_path, '_'.join([
            'pipeline', 'config', type, worker_id,
            model_case.split('/')[1] + '.log'
        ]))
    file = open(config_log, 'w')
    file.writelines(' '.join([
        'reproduce config info:', hf_path,
        str(backend_config),
        str(gen_config)
    ]))
    print(' '.join([
        'reproduce config info:', hf_path,
        str(backend_config),
        str(gen_config)
    ]))
    file.close

    for case in cases_info.keys():
        if ('deepseek-coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path, '_'.join([
                'pipeline', 'chat', type,
                model_case.split('/')[1], case + '.log'
            ]))

        file = open(pipeline_chat_log, 'w')

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            file.writelines('prompt:' + prompt + '\n')

            response = pipe([prompts], gen_config=gen_config)[0].text

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
        if ('deepseek-coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        msg = 'result is empty, please check again'
        result = False
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, '_'.join([
                    'pipeline', 'chat', type,
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
                    if 'result:True, reason:' in line and result is False:
                        result = True
                        msg = ''

            with assume:
                assert result, msg


def save_pipeline_common_log(config,
                             log_name,
                             result,
                             content,
                             write_type: str = 'w'):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    file = open(config_log, write_type)
    file.writelines(f'result:{result}, reason: {content}')
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
            if 'result:True, reason:' in line and result is False:
                result = True
                msg = ''

    assert result, msg


def assert_pipeline_single_return(output):
    result = assert_pipeline_single_stream_element(output, is_last=True)
    return result & (len(output.get('token_ids'))
                     == output.get('generate_token_len'))


def assert_pipeline_batch_return(output, size):
    result = len(output) == size
    for single_output in output:
        result &= assert_pipeline_single_return(single_output)
    return result


def assert_pipeline_single_stream_return(output):
    result = True
    for i in range(0, len(output) - 1):
        result &= assert_pipeline_single_stream_element(output[i])
    result &= assert_pipeline_single_stream_element(output[-1], is_last=True)
    return result


def assert_pipeline_batch_stream_return(output, size):
    result = True
    for i in range(size):
        output_list = [item for item in output if item.get('session_id') == i]
        result &= assert_pipeline_single_stream_return(output_list)
    return result


def assert_pipeline_single_stream_element(output, is_last: bool = False):
    result = True
    result &= len(output.get('text')) > 0
    result &= output.get('generate_token_len') > 0
    result &= output.get('input_token_len') > 0
    result &= output.get('session_id') >= 0
    if is_last:
        result &= output.get('finish_reason') in ['stop', 'length']
    else:
        result &= output.get('finish_reason') is None
    result &= len(output.get('token_ids')) > 0
    result &= output.get('logprobs') is None
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
    pipe = pipeline(hf_path, backend_config=backend_config)

    pipeline_chat_log = os.path.join(
        log_path, 'pipeline_vl_chat_' + model_case.split('/')[1] + '.log')
    file = open(pipeline_chat_log, 'w')

    image = load_image(PIC1)
    response = pipe(('describe this image', image))
    result = 'tiger' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: simple example tiger not in ' + response.text +
                    '\n')

    prompts = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image'
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC1
            }
        }]
    }]
    response = pipe(prompts)
    result = 'tiger' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: OpenAI format example: tiger not in ' +
                    response.text + '\n')

    image_urls = [PIC2, PIC1]
    images = [load_image(img_url) for img_url in image_urls]
    response = pipe(('describe these images', images))
    result = 'tiger' in response.text.lower() or 'ski' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: Multi-images example: tiger or ski not in ' +
                    response.text + '\n')

    image_urls = [PIC2, PIC1]
    prompts = [('describe this image', load_image(img_url))
               for img_url in image_urls]
    response = pipe(prompts)
    result = 'ski' in response[0].text.lower() and (
        'tiger' in response[1].text.lower() or '虎' in response[1].text.lower())
    file.writelines('result:' + str(result) +
                    ', reason: Batch example: ski or tiger not in ' +
                    str(response) + '\n')

    image = load_image(PIC2)
    sess = pipe.chat(('describe this image', image))
    result = 'ski' in sess.response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: Multi-turn example: ski not in ' +
                    sess.response.text + '\n')
    sess = pipe.chat('What is the woman doing?', session=sess)
    result = 'ski' in sess.response.text.lower()
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
            if 'result:True, reason:' in line and result is False:
                result = True
                msg = ''

    with assume:
        assert result, msg
