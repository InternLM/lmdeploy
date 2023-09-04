# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import os
import os.path as osp
import random

import fire

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS, BaseModel
from lmdeploy.turbomind.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'


@dataclasses.dataclass
class GenParam:
    sequence_start: bool = False
    sequence_end: bool = True
    step: int = 0
    request_output_len: int = 512
    # top_p, top_k, temperature, repetition_penalty are
    # parameters for sampling
    top_p: float = 0.8
    top_k: float = 40
    temperature: float = 0.8
    repetition_penalty: float = 1.0


def input_prompt(model_name):
    """Input a prompt in the consolo interface."""
    if model_name == 'codellama':
        print('\nenter !! to end the input >>>\n', end='')
        sentinel = '!!'
    else:
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


def get_prompt(prompt: str, model: BaseModel, model_name: str, cap: str,
               nth_round: bool):
    if model_name == 'codellama':
        if cap == 'completion' or cap == 'python':
            return prompt
        elif cap == 'infill':
            return model.get_prompt(prompt, sequence_start=True)
        elif cap == 'instruct':
            return model.get_prompt(prompt, nth_round == 1)
        else:
            assert 0, f"{model_name} model hasn't {cap} capability"
    else:
        if cap == 'completion':
            return prompt
        elif cap == 'instruct':
            return model.get_prompt(prompt, nth_round == 1)
        else:
            assert 0, f"{model_name} model hasn't {cap} capability"


def get_gen_param(model_name, cap, nth_round, step):
    if model_name == 'codellama':
        if cap == 'instruct':
            return GenParam(
                sequence_start=(nth_round == 1),
                sequence_end=False,
                step=step,
                # The following parameters comes from https://huggingface.co/spaces/codellama/codellama-13b-chat # noqa: E501
                top_p=0.9,
                top_k=10,
                temperature=0.1,
                request_output_len=1024)
        else:
            return GenParam(
                sequence_start=True,
                sequence_end=True,
                step=0,
                # The following parameters comes from https://huggingface.co/spaces/codellama/codellama-playground # noqa: E501
                top_p=0.9,
                temperature=0.1 if cap == 'completion' else 0.6,
                repetition_penalty=1.05,
                request_output_len=256)
    else:
        if cap == 'instruct':
            return GenParam(sequence_start=(nth_round == 1),
                            sequence_end=False,
                            step=step)
        else:
            return GenParam(sequence_start=True, sequence_end=True, step=0)


def main(model_path,
         session_id: int = 1,
         cap: str = 'instruct',
         sys_instruct: str = 'Provide answers in Python',
         repetition_penalty: float = 1.0,
         tp=1,
         stream_output=True):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infill', 'instruct', 'python']
        sys_instruct (str): the content of 'system' role, which is used by
            conversational model
        repetition_penalty (float): parameter to penalize repetition
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
    """
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path, eos_id=tokenizer.eos_token_id, tp=tp)
    generator = tm_model.create_instance()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model_name = tm_model.model_name
    model = MODELS.get(model_name)(cap=cap, default_sys_prompt=sys_instruct)

    print(f'session {session_id}')
    while True:
        prompt = input_prompt(model_name)
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
            if step >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue
            gen_param = get_gen_param(model_name, step, nth_round, cap)
            prompt = get_prompt(prompt, model, model_name, cap, nth_round)
            input_ids = tokenizer.encode(prompt)
            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    stream_output=stream_output,
                    **dataclasses.asdict(gen_param),
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
                res, tokens = outputs[0]
                # decode res
                response = tokenizer.decode(res.tolist(), offset=response_size)
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size = tokens

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
