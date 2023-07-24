# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random

import fire
import torch

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from lmdeploy.turbomind.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


def main(model_name,
         model_path,
         session_id: int = 1,
         repetition_penalty: float = 1.0,
         tp=torch.cuda.device_count()):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_name (str): the name of the deployed model
        model_path (str): the path of the deployed model
        session_id (int): the identical id of a session
    """
    model = MODELS.get(model_name)()
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path,
                            eos_id=tokenizer.eos_token_id,
                            stop_words=model.stop_words,
                            tp=tp)
    generator = tm_model.create_instance()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)

    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            prompt = model.get_prompt('', nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=[input_ids],
                                                  request_output_len=512,
                                                  sequence_start=False,
                                                  sequence_end=True):
                pass
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            print(f'session {session_id}')
            if step >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue
            prompt = model.get_prompt(prompt, nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    stream_output=True,
                    request_output_len=512,
                    sequence_start=(nth_round == 1),
                    sequence_end=False,
                    step=step,
                    stop=False,
                    top_k=40,
                    top_p=0.8,
                    temperature=0.8,
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

            nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
