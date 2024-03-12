import os

import allure
import torch
from pytest import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result

from lmdeploy import pipeline
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)


def run_pipeline_chat_test(config, cases_info, model_case, type):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_name = model_name = get_model_name(model_case)
    model_path = config.get('model_path')
    hf_path = model_path + '/' + model_case

    print(' '.join([
        'reproduce command:', 'python',
        'autotest/tools/pipeline/pipeline_chat_script.py', type, model_case,
        str(tp)
    ]))

    if 'pytorch' == type:
        backend_config = PytorchEngineConfig(tp=tp)
    else:
        if 'kvint8' in model_case and ('w4' in model_case
                                       or '4bits' in model_case):
            backend_config = TurbomindEngineConfig(tp=tp,
                                                   model_format='awq',
                                                   quant_policy=4)
        elif 'kvint8' in model_case:
            backend_config = TurbomindEngineConfig(tp=tp,
                                                   model_format='hf',
                                                   quant_policy=4)
        elif 'w4' in model_case or '4bits' in model_case:
            backend_config = TurbomindEngineConfig(tp=tp, model_format='awq')
        else:
            backend_config = TurbomindEngineConfig(tp=tp)
    pipe = pipeline(hf_path, backend_config=backend_config)

    # run testcases
    gen_config = GenerationConfig(temperature=0.01)
    gen_config = GenerationConfig()
    for case in cases_info.keys():
        if (case == 'memory_test'
                or case == 'emoji_case') and 'chat' not in model_case.lower():
            continue

        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path,
            'pipeline_chat_' + model_case.split('/')[1] + '_' + case + '.log')

        file = open(pipeline_chat_log, 'w')

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            if 'chat' not in model_case.lower():  # base model
                prompts.append(prompt)
            else:  # chat model
                prompts.append({'role': 'user', 'content': prompt})
            file.writelines('prompt:' + prompt + '\n')

            if 'chat' not in model_case.lower():  # base model
                response = pipe(prompts, gen_config=gen_config)[-1].text
            else:  # chat model
                response = pipe([prompts], gen_config=gen_config)[0].text

            case_result, reason = assert_result(response,
                                                prompt_detail.values(),
                                                model_name)
            if 'chat' in model_case.lower():
                prompts.append({'role': 'assistant', 'content': response})
            file.writelines('output:' + response + '\n')
            file.writelines('result:' + str(case_result) + ', reason:' +
                            reason + '\n')
        file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_chat_log(config, cases_info, model_case):
    log_path = config.get('log_path')

    for case in cases_info.keys():
        if (case == 'memory_test'
                or case == 'emoji_case') and 'chat' not in model_case.lower():
            continue

        msg = ''
        result = False
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, 'pipeline_chat_' + model_case.split('/')[1] + '_' +
                case + '.log')

            with open(pipeline_chat_log, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if 'result:False, reason:' in line:
                        result = False
                        msg = line
                        break
                    if 'result:True, reason:' in line and result is False:
                        result = True

            allure.attach.file(pipeline_chat_log,
                               attachment_type=allure.attachment_type.TEXT)
            with assume:
                assert result, msg
