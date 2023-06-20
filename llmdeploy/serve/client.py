# Copyright (c) OpenMMLab. All rights reserved.
import os

import fire

from llmdeploy.serve.fastertransformer.chatbot import Chatbot


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(triton_server_addr: str, model_name: str, session_id: int):
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    chatbot = Chatbot(triton_server_addr,
                      model_name,
                      log_level=log_level,
                      display=True)
    nth_round = 1
    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            chatbot.end(session_id)
        else:
            request_id = f'{session_id}-{nth_round}'
            for status, res, tokens in chatbot.stream_infer(
                    session_id,
                    prompt,
                    request_id=request_id,
                    request_output_len=512):
                continue
            print(f'session {session_id}, {status}, {tokens}, {res}')
        nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
