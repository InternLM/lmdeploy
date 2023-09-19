# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random

import fire
import torch

from lmdeploy.model import MODELS
from lmdeploy.pytorch_poc import engine as tm
from lmdeploy.pytorch_poc.messages import SamplingParam
from lmdeploy.turbomind.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'

logger = logging.getLogger()


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


def main(
        model_path,
        model_name: str,  # can not get model_name from hf model
        session_id: int = 1,
        top_k=40,
        top_p=0.8,
        temperature=0.00000008,
        repetition_penalty: float = 1.0,
        tp: int = 1,
        trust_remote_code=True,
        stream_output=True):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the huggingface model path
        session_id (int): the identical id of a session
        repetition_penalty (float): parameter to penalize repetition
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
    """
    # tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(model_path)
    tm_model = tm.Engine(model_path,
                         tp=tp,
                         trust_remote_code=trust_remote_code)
    generator = tm_model.create_instance()

    torch.set_default_device(0)

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model = MODELS.get(model_name)()

    for prompt in ['Write a poem about Valencia.', 'exit']:
        # prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            generator.end(session_id)
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
            input_ids = model.update_input_ids(input_ids)
            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            sampling_param = SamplingParam(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=False,
                random_seed=seed,
            )
            logger.debug(f'input_ids = {input_ids}')
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    prompt_token_ids=input_ids,
                    request_output_len=20,
                    step=step,
                    sampling_param=sampling_param):
                status, res, tokens = outputs
                # decode res
                logger.debug(res)
                response = tokenizer.decode(res)[response_size:]
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size += len(response)

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    logging.basicConfig(filename='chat-llama.log',
                        level=logging.DEBUG,
                        filemode='w',
                        format='{%(pathname)s:%(lineno)d}\n  %(message)s')
    torch.set_printoptions(linewidth=122)
    fire.Fire(main)
