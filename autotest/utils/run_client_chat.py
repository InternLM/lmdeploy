import os
from subprocess import PIPE, Popen

from utils.config_utils import _is_bf16_supported_by_device
from utils.get_run_config import get_command_with_extra, get_model_name
from utils.rule_condition_assert import assert_result

TEMPLATE = 'autotest/template.json'


def hf_command_line_test(config,
                         case,
                         case_info,
                         model_case,
                         type,
                         cuda_prefix: str = None,
                         extra: str = '',
                         use_local_model: bool = True):
    if use_local_model:
        model_path = config.get('model_path') + '/' + model_case
    else:
        model_path = model_case

    if str(config.get('env_tag')) == '3090' or str(config.get('env_tag')) == '5080':
        extra += ' --cache-max-entry-count 0.7'

    cmd = get_command_with_extra(' '.join(['lmdeploy chat', model_path, '--backend', type, extra,
                                           '--session-len 4096']),
                                 config,
                                 model_case,
                                 need_tp=True,
                                 cuda_prefix=cuda_prefix)

    if type == 'pytorch':
        if not _is_bf16_supported_by_device():
            cmd += ' --dtype float16'
    if type == 'turbomind':
        if ('w4' in model_case or ('4bits' in model_case or 'awq' in model_case.lower())):
            cmd += ' --model-format awq'
        elif 'gptq' in model_case.lower():
            cmd += ' --model-format gptq'

    if case == 'base_testcase':
        cmd += ' --chat-template ' + TEMPLATE

    # Add device option if specified in environment
    device = os.environ.get('DEVICE', '')
    if device:
        cmd += f' --device {device} '
        if device == 'ascend':
            cmd += '--eager-mode '

    return command_test(config, [cmd], model_case, '_'.join(['hf', type, case]), case_info, True)


def command_test(config, cmd, model, case, case_info, need_extract_output, worker_id: str = ''):
    try:
        log_path = config.get('log_path')
        model_name = get_model_name(model)

        if '/' in model:
            chat_log = os.path.join(log_path, 'chat_' + model.split('/')[1] + worker_id + '_' + case + '.log')
        else:
            chat_log = os.path.join(log_path, 'chat_' + model + worker_id + '_' + case + '.log')

        file = open(chat_log, 'w')

        returncode = -1
        result = True

        print('reproduce command chat: ' + ' '.join(cmd) + '\n')
        file.writelines('reproduce command chat: ' + ' '.join(cmd) + '\n')

        spliter = '\n\n'
        # join prompt together
        prompt = ''
        for item in case_info:
            prompt += list(item.keys())[0] + spliter
        prompt += 'exit' + spliter

        msg = ''

        with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, text=True, encoding='utf-8') as proc:
            file.writelines('prompt:' + prompt + '\n')

            outputs, errors = proc.communicate(input=prompt)
            returncode = proc.returncode
            if returncode != 0:
                file.writelines('error:' + errors + '\n')
                result = False
                return result, chat_log, errors

            outputDialogs = parse_dialogue(outputs, model, spliter)
            file.writelines('answersize:' + str(len(outputDialogs)) + '\n')

            # 结果判断
            index = 0
            for prompt_detail in case_info:
                if need_extract_output:
                    output = extract_output(outputDialogs[index], model)
                else:
                    output = outputDialogs[index]
                case_result, reason = assert_result(output, prompt_detail.values(), model_name)
                file.writelines('prompt:' + list(prompt_detail.keys())[0] + '\n')
                file.writelines('output:' + output + '\n')
                file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
                index += 1
                if not case_result:
                    msg = reason
                result = result & case_result
            file.writelines('\n\n\n' + 'full log:' + outputs + '\n')

        file.close()
        return result, chat_log, msg
    except Exception as e:
        return False, None, f'Unknown error: {e}'


# 从输出中解析模型输出的对话内容
def parse_dialogue(inputs: str, model: str, spliter: str):
    dialogues = inputs.strip()
    sep = 'double enter to end input >>>'
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:-1]  # 去除首尾无用字符


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
