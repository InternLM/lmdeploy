import os
import time
from subprocess import PIPE, Popen

import allure
from utils.config_utils import get_case_str_by_config, get_cli_common_param, get_cuda_prefix_by_workerid
from utils.rule_condition_assert import assert_result

TEMPLATE = 'autotest/template.json'


def run_tests(config, usercase, cli_case_config, run_config, worker_id):
    if 'coder' in run_config['model'].lower() and usercase == 'chat_testcase':
        usercase = 'code_testcase'

    hf_command_line_test(config,
                         usercase,
                         cli_case_config.get(usercase),
                         run_config,
                         cuda_prefix=get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config')))


def hf_command_line_test(config, case, case_info, run_config, cuda_prefix: str = ''):
    model = run_config.get('model')
    if run_config.get('env', {}).get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        model_path = model

    else:
        model_path = config.get('model_path') + '/' + model

    run_config['extra_params']['session_len'] = 4096
    if case == 'base_testcase':
        run_config['extra_params']['chat_template'] = TEMPLATE
        run_config['extra_params']['session_len'] = 512

    print(run_config)

    cmd = ' '.join([cuda_prefix, ' '.join(['lmdeploy chat', model_path, get_cli_common_param(run_config)])]).strip()

    result, chat_log, msg = command_test(config, cmd, run_config, case_info, True)
    if chat_log:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert result, msg


def command_test(config, cmd, run_config, case_info, need_extract_output):
    try:
        log_path = config.get('log_path')
        case_name = get_case_str_by_config(run_config)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        chat_log = os.path.join(log_path, f'chat_{case_name}_{timestamp}.log')

        file = open(chat_log, 'w')

        returncode = -1
        result = True

        print(f'reproduce command chat: {cmd} \n')
        file.writelines(f'reproduce command chat: {cmd} \n')

        spliter = '\n\n'
        # join prompt together
        prompt = ''
        for item in case_info:
            prompt += list(item.keys())[0] + spliter
        prompt += 'exit' + spliter

        msg = ''

        env = os.environ.copy()
        env.update(run_config.get('env', {}))

        with Popen([cmd],
                   stdin=PIPE,
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8',
                   env=env,
                   start_new_session=True) as proc:
            file.writelines('prompt:' + prompt + '\n')

            outputs, errors = proc.communicate(input=prompt)
            returncode = proc.returncode
            if returncode != 0:
                file.writelines('error:' + errors + '\n')
                result = False
                return result, chat_log, errors

            outputDialogs = parse_dialogue(outputs)
            file.writelines('answersize:' + str(len(outputDialogs)) + '\n')

            index = 0
            for prompt_detail in case_info:
                if need_extract_output:
                    output = extract_output(outputDialogs[index], run_config.get('model'))
                else:
                    output = outputDialogs[index]
                case_result, reason = assert_result(output, prompt_detail.values(), run_config.get('model'))
                file.writelines(f'prompt: {list(prompt_detail.keys())[0]}\n')
                file.writelines(f'output: {output}\n')
                file.writelines(f'result: {case_result}, reason: {reason}\n')
                index += 1
                if not case_result:
                    print(f'prompt: {list(prompt_detail.keys())[0]}\n')
                    print(f'output: {output}\n')
                    print(f'result: {case_result}, reason: {reason}\n')
                    msg += reason
                result = result & case_result
            file.writelines('\n\n\n' + 'full log:' + outputs + '\n')

        file.close()
        return result, chat_log, msg
    except Exception as e:
        return False, None, f'Unknown error: {e}'


def parse_dialogue(inputs: str):
    dialogues = inputs.strip()
    sep = 'double enter to end input >>>'
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:-1]


def extract_output(output: str, model: str):
    if 'Qwen' in model or 'internlm2' in model:
        if len(output.split('<|im_start|>assistant')) >= 2:
            return output.split('<|im_start|>assistant')[1]
    if 'Baichuan2' in model:
        if len(output.split('<reserved_107>')) >= 2:
            return output.split('<reserved_107>')[1]
    if 'internlm' in model:
        if len(output.split('<|Bot|>: ')) >= 2:
            return output.split('<|Bot|>: ')[1]
    if 'llama' in model or 'Llama' in model:
        if len(output.split('[/INST]')) >= 2:
            return output.split('[/INST]')[1]

    return output
