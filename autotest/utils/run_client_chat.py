import os
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
        model_path = f"{config.get('model_path')}/{model}"

    # Ensure extra_params exists before modifying
    if 'extra_params' not in run_config:
        run_config['extra_params'] = {}
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

        chat_log = os.path.join(log_path, f'chat_{case_name}.log')

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

        with Popen([cmd], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, text=True, encoding='utf-8',
                   env=env) as proc:
            file.writelines('prompt:' + prompt + '\n')

            outputs, errors = proc.communicate(input=prompt)
            returncode = proc.returncode
            if returncode != 0:
                error_msg = errors if errors else 'Unknown error occurred'
                file.writelines('error:' + error_msg + '\n')
                result = False
                return result, chat_log, error_msg

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
                result = result and case_result
            file.writelines('\n\n\n' + 'full log:' + outputs + '\n')

        file.close()
        return result, chat_log, msg
    except Exception as e:
        return False, None, f'Unknown error: {e}'


def parse_dialogue(inputs: str):
    if not inputs or not inputs.strip():
        return []
    dialogues = inputs.strip()
    sep = 'double enter to end input >>>'
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues if d.strip()]
    # Return all but first and last (which are typically empty or prompt)
    if len(dialogues) <= 2:
        return []
    return dialogues[1:-1]


def extract_output(output: str, model: str):
    if not output:
        return output
    # Check Qwen or internlm2 first (case-sensitive to match original behavior)
    if 'Qwen' in model or 'internlm2' in model:
        parts = output.split('<|im_start|>assistant')
        if len(parts) >= 2:
            return parts[1]
    elif 'Baichuan2' in model:
        parts = output.split('<reserved_107>')
        if len(parts) >= 2:
            return parts[1]
    elif 'internlm' in model.lower():
        parts = output.split('<|Bot|>: ')
        if len(parts) >= 2:
            return parts[1]
    elif 'llama' in model.lower() or 'Llama' in model:
        parts = output.split('[/INST]')
        if len(parts) >= 2:
            return parts[1]

    return output
