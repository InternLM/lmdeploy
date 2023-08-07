# Copyright (c) OpenMMLab. All rights reserved.
import os

import fire

from lmdeploy.serve.turbomind.chatbot import Chatbot


def input_prompt():
    """Input a prompt in the console interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(tritonserver_addr: str,
         session_id: int = 1,
         stream_output: bool = True):
    """An example to communicate with inference server through the command line
    interface.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
          triton inference server
        session_id (int): the identical id of a session
        stream_output (bool): indicator for streaming output or not
    """
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'WARNING')
    chatbot = Chatbot(tritonserver_addr,
                      log_level=log_level,
                      display=stream_output)
    nth_round = 1
    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            chatbot.end(session_id)
        else:
            request_id = f'{session_id}-{nth_round}'
            if stream_output:
                for status, res, n_token in chatbot.stream_infer(
                        session_id,
                        prompt,
                        request_id=request_id,
                        request_output_len=512):
                    continue
            else:
                status, res, n_token = chatbot.infer(session_id,
                                                     prompt,
                                                     request_id=request_id,
                                                     request_output_len=512)
                print(res)
        nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
