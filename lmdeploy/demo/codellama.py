# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random

import fire

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from lmdeploy.turbomind.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def input_prompt():
    """Input code in the consolo interface."""
    print('\nenter !! to end the input >>>\n', end='')
    sentinel = '!!'  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


def main(model_path,
         session_id: int = 1,
         cap: str = 'completion',
         system_intruction: str = '',
         tp=1,
         stream_output=True):
    """An example to perform turbmind inference of codellama through the
    command line interface.

    Args:
        model_path (str): the path of the deployed model
        session_id (int): the identical id of a session
        cap (str): the capability of codellama among
           ['completion', 'infill', 'instruct', 'python']
        system_intruction (str): the content of 'system' role, which is used by
           instruction model
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
    """
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path, eos_id=tokenizer.eos_token_id, tp=tp)
    generator = tm_model.create_instance()

    seed = random.getrandbits(64)
    model_name = tm_model.model_name
    assert model_name == 'codellama', \
        f'this example is for codellama but got {model_name}'

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model = MODELS.get(model_name)(cap=cap,
                                   default_sys_prompt=system_intruction)

    print(f'session {session_id}')
    while True:
        prompt = input_prompt()

        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            input_ids = tokenizer.encode('')
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=[input_ids],
                                                  request_output_len=0,
                                                  sequence_start=False,
                                                  sequence_end=True,
                                                  stream_output=stream_output):
                pass
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            if cap == 'completion' or cap == 'infill':
                prompt = model.get_prompt(prompt, sequence_start=True)
                sequence_start = True
                sequence_end = True
                step = 0
                # The following parameters comes from https://huggingface.co/spaces/codellama/codellama-playground # noqa: E501
                top_p = 0.9
                temperature = 0.1 if cap == 'completion' else 0.6
                repetition_penalty = 1.05
                request_output_len = 256
            else:
                prompt = model.get_prompt(prompt, nth_round == 1)
                sequence_start = (nth_round == 1)
                sequence_end = False
                temperature = 0.2
                top_p = 0.95
                request_output_len = 1024

            input_ids = tokenizer.encode(prompt)
            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    stream_output=stream_output,
                    request_output_len=request_output_len,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=step,
                    stop=False,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
                res, tokens = outputs[0]
                # decode res
                response = tokenizer.decode(res)[response_size:]
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size += len(response)

            # update step
            step += len(input_ids) + tokens
            print()


if __name__ == '__main__':
    fire.Fire(main)
