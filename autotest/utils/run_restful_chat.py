import os
import random
import string

from utils.rule_condition_assert import assert_result

from lmdeploy.serve.openai.api_client import APIClient


def open_chat_test(config, case_info, model, url, worker_id: str = 'default'):
    log_path = config.get('log_path')

    restful_log = os.path.join(log_path,
                               'restful_' + model + '_' + worker_id + '.log')

    file = open(restful_log, 'w')

    result = True

    api_client = APIClient(url)
    model_name = api_client.available_models[0]

    messages = []
    msg = ''
    for prompt_detail in case_info:
        if result is False:
            break
        prompt = list(prompt_detail.keys())[0]
        messages.append({'role': 'user', 'content': prompt})
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=messages,
                                                     temperature=0.01):
            output_message = output.get('choices')[0].get('message')
            messages.append(output_message)

            output_content = output_message.get('content')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content,
                                                prompt_detail.values(),
                                                model_name)
            file.writelines('result:' + str(case_result) + ',reason:' +
                            reason + '\n')
            if result is False:
                msg += reason
            result = result & case_result
    file.close()
    return result, restful_log, msg


def interactive_test(config,
                     case_info,
                     model,
                     url,
                     worker_id: str = 'default'):
    log_path = config.get('log_path')

    interactive_log = os.path.join(
        log_path, 'interactive_' + model + '_' + worker_id + '.log')

    file = open(interactive_log, 'w')

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' +
                    ','.join(api_client.available_models) + '\n')

    # Randomly generate 6 characters and concatenate them into a string.
    characters = string.digits
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
                                                     session_id=random_chars,
                                                     temperature=0.01):
            output_content = output.get('text')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content,
                                                prompt_detail.values(), model)
            file.writelines('result:' + str(case_result) + ',reason:' +
                            reason + '\n')
            if result is False:
                msg += reason
            result = result & case_result
    file.close()
    return result, interactive_log, msg


def health_check(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        messages = []
        messages.append({'role': 'user', 'content': '你好'})
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=messages,
                                                     temperature=0.01):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            return True
    except Exception:
        return False


def get_model(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        return model_name
    except Exception:
        return None
