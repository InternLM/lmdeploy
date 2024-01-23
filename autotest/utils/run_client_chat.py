import os
from subprocess import PIPE, Popen

from utils.get_run_config import get_command_with_extra
from utils.rule_condition_assert import assert_result


def command_line_test(config, case, case_info, model, type, extra):
    dst_path = config.get('dst_path')

    if type == 'api_client':
        cmd = get_command_with_extra('lmdeploy serve api_client ' + extra,
                                     config, model)
    elif type == 'triton_client':
        cmd = get_command_with_extra('lmdeploy serve triton_client ' + extra,
                                     config, model)
    else:
        cmd = get_command_with_extra(
            'lmdeploy chat turbomind ' + dst_path + '/workspace_' + model,
            config, model)

    if case == 'session_len_error':
        cmd = cmd + ' --session-len 20'
    return command_test(config, [cmd], model, case_info, type == 'turbomind')


def hf_command_line_test(config, case, case_info, model_case, model_name):
    model_path = config.get('model_path')

    cmd = get_command_with_extra(
        'lmdeploy chat turbomind ' + model_path + '/' + model_case +
        ' --model-name ' + model_name, config, model_case)

    if case == 'session_len_error':
        cmd = cmd + ' --session-len 20'
    return command_test(config, [cmd], model_case, case_info, True)


def pytorch_command_line_test(config, case, case_info, model_case):
    model_path = config.get('model_path')

    cmd = get_command_with_extra(
        'lmdeploy chat torch ' + model_path + '/' + model_case, config,
        model_case)

    if case == 'session_len_error':
        cmd = cmd + ' --session-len 20'
    return command_test(config, [cmd], model_case, case_info, False)


def command_test(config, cmd, model, case_info, need_extract_output):
    try:
        log_path = config.get('log_path')
        model_map = config.get('model_map')
        model_name = model_map.get(model)

        chat_log = os.path.join(log_path, 'chat_' + model + '.log')

        file = open(chat_log, 'w')

        returncode = -1
        result = True

        file.writelines('commondLine: ' + ' '.join(cmd) + '\n')

        spliter = '\n\n'
        if model == 'CodeLlama-7b-Instruct-hf':
            spliter = '\n!!\n'
        # join prompt together
        prompt = ''
        for item in case_info:
            prompt += list(item.keys())[0] + spliter
        prompt += 'exit' + spliter

        msg = ''

        with Popen(cmd,
                   stdin=PIPE,
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8') as proc:
            file.writelines('prompt:' + prompt + '\n')

            outputs, errors = proc.communicate(input=prompt)
            returncode = proc.returncode
            if returncode != 0:
                file.writelines('error:' + errors + '\n')
                result = False
                return result, chat_log, errors

            outputDialogs = parse_dialogue(outputs, model)
            file.writelines('answersize:' + str(len(outputDialogs)) + '\n')

            # 结果判断
            index = 0
            for prompt_detail in case_info:
                if need_extract_output:
                    output = extract_output(outputDialogs[index], model)
                else:
                    output = outputDialogs[index]
                case_result, reason = assert_result(output,
                                                    prompt_detail.values(),
                                                    model_name)
                file.writelines('prompt:' + list(prompt_detail.keys())[0] +
                                '\n')
                file.writelines('output:' + output + '\n')
                file.writelines('result:' + str(case_result) + ',reason:' +
                                reason + '\n')
                index += 1
                if case_result is False:
                    msg = reason
                result = result & case_result

        file.close()
        return result, chat_log, msg
    except Exception as e:
        return False, None, f'Unknown error: {e}'


# 从输出中解析模型输出的对话内容
def parse_dialogue(inputs: str, model: str):
    dialogues = inputs.strip()
    if model == 'CodeLlama-7b-Instruct-hf':
        sep = 'enter !! to end the input >>>'
    else:
        sep = 'double enter to end input >>>'
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    if 'Llama' in model:
        return dialogues
    return dialogues[1:-1]  # 去除首尾无用字符


def extract_output(output: str, model: str):
    if 'internlm' in model:
        if len(output.split('<|Bot|>: ')) >= 2:
            return output.split('<|Bot|>: ')[1]
    if 'Qwen' in model:
        if len(output.split('<|im_start|>assistant')) >= 2:
            return output.split('<|im_start|>assistant')[1]
    if 'Baichuan2' in model:
        if len(output.split('<reserved_107>')) >= 2:
            return output.split('<reserved_107>')[1]
    if 'llama' in model or 'Llama' in model:
        if len(output.split('[/INST]')) >= 2:
            return output.split('[/INST]')[1]

    return output
