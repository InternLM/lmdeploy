# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Iterable, List

import fire
import requests


def get_streaming_response(prompt: str,
                           api_url: str,
                           instance_id: int,
                           request_output_len: int = 512,
                           stream: bool = True,
                           sequence_start: bool = True,
                           sequence_end: bool = True,
                           ignore_eos: bool = False) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    pload = {
        'prompt': prompt,
        'stream': stream,
        'instance_id': instance_id,
        'request_output_len': request_output_len,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'ignore_eos': ignore_eos
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b'\0'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            output = data['text']
            tokens = data['tokens']
            finish_reason = data['finish_reason']
            yield output, tokens, finish_reason


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(server_name: str, server_port: int, session_id: int = 0):
    nth_round = 1
    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        else:
            for output, tokens, finish_reason in get_streaming_response(
                    prompt,
                    f'http://{server_name}:{server_port}/generate',
                    instance_id=session_id,
                    request_output_len=512,
                    sequence_start=(nth_round == 1),
                    sequence_end=False):
                if finish_reason == 'length':
                    print('WARNING: exceed session max length.'
                          ' Please end the session.')
                    continue
                print(output, end='')

            nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
