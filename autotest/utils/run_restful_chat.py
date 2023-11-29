import os
import random
import string

from utils.rule_condition_assert import assert_result

from lmdeploy.serve.openai.api_client import APIClient


def openAiChatTest(config, case_info, model, url):
    log_path = config.get('log_path')

    restful_log = os.path.join(log_path, 'restful_' + model + '.log')

    file = open(restful_log, 'w')

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' +
                    ','.join(api_client.available_models) + '\n')
    model_name = model

    messages = []
    msg = ''
    for prompt_detail in case_info:
        prompt = list(prompt_detail.keys())[0]
        new_prompt = {'role': 'user', 'content': prompt}
        messages.append(new_prompt)
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=messages):
            output_message = output.get('choices')[0].get('message')
            messages.append(output_message)

            output_content = output_message.get('content')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content,
                                                prompt_detail.values())
            file.writelines('result:' + str(case_result) + ',reason:' +
                            reason + '\n')
            if result is False:
                msg += reason
            result = result & case_result
    file.close()
    return result, restful_log, msg


def interactiveTest(config, case_info, model, url):
    log_path = config.get('log_path')

    interactive_log = os.path.join(log_path, 'interactive_' + model + '.log')

    file = open(interactive_log, 'w')

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' +
                    ','.join(api_client.available_models) + '\n')

    # 定义包含所有可能字符的字符集合
    characters = string.digits

    # 随机生成6个字符并拼接成字符串
    random_chars = ''.join(random.choice(characters) for i in range(6))

    messages = []
    msg = ''
    for prompt_detail in case_info:
        prompt = list(prompt_detail.keys())[0]
        new_prompt = {'role': 'user', 'content': prompt}
        messages.append(new_prompt)
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_interactive_v1(prompt=prompt,
                                                     interactive_mode=True,
                                                     session_id=random_chars):
            output_content = output.get('text')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content,
                                                prompt_detail.values())
            file.writelines('result:' + str(case_result) + ',reason:' +
                            reason + '\n')
            if result is False:
                msg += reason
            result = result & case_result
    file.close()
    return result, interactive_log, msg


if __name__ == '__main__':
    url = 'http://7:60006'
    config = {'log_path': '/home/zhulin'}
    openAiChatTest(config, 'test', url)
