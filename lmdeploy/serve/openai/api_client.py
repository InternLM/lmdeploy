# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Iterable, List

import fire
import requests


def get_model_list(api_url: str):
    response = requests.get(api_url)
    if hasattr(response, 'text'):
        model_list = json.loads(response.text)
        model_list = model_list.pop('data', [])
        return [item['id'] for item in model_list]
    return None


def get_streaming_response(prompt: str,
                           api_url: str,
                           session_id: int,
                           request_output_len: int = 512,
                           stream: bool = True,
                           sequence_start: bool = True,
                           sequence_end: bool = True,
                           ignore_eos: bool = False,
                           stop: bool = False) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    pload = {
        'prompt': prompt,
        'stream': stream,
        'session_id': session_id,
        'request_output_len': request_output_len,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'ignore_eos': ignore_eos,
        'stop': stop
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b'\n'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            output = data.pop('text', '')
            tokens = data.pop('tokens', 0)
            finish_reason = data.pop('finish_reason', None)
            yield output, tokens, finish_reason


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(restful_api_url: str, session_id: int = 0):
    nth_round = 1
    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            for output, tokens, finish_reason in get_streaming_response(
                    '',
                    f'{restful_api_url}/generate',
                    session_id=session_id,
                    request_output_len=0,
                    sequence_start=(nth_round == 1),
                    sequence_end=True):
                pass
            exit(0)
        else:
            for output, tokens, finish_reason in get_streaming_response(
                    prompt,
                    f'{restful_api_url}/generate',
                    session_id=session_id,
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
