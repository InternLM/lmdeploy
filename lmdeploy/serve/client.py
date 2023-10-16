# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.serve.turbomind.chatbot import Chatbot


def input_prompt(model_name):
    """Input a prompt in the consolo interface."""
    if model_name == 'codellama':
        print('\nenter !! to end the input >>>\n', end='')
        sentinel = '!!'
    else:
        print('\ndouble enter to end input >>> ', end='')
        sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(tritonserver_addr: str,
         session_id: int = 1,
         cap: str = 'chat',
         stream_output: bool = True,
         **kwargs):
    """An example to communicate with inference server through the command line
    interface.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
          triton inference server
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infill', 'instruct', 'python']
        stream_output (bool): indicator for streaming output or not
        **kwargs (dict): other arguments for initializing model's chat template
    """
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'WARNING')
    kwargs.update(capability=cap)
    chatbot = Chatbot(tritonserver_addr,
                      log_level=log_level,
                      display=stream_output,
                      **kwargs)
    nth_round = 1
    while True:
        prompt = input_prompt(chatbot.model_name)
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
    import fire

    fire.Fire(main)
