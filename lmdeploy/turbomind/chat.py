# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random
from configparser import ConfigParser

import fire
from transformers import AutoTokenizer

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def valid_str(string, coding='utf-8'):
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


def get_session_len(model_path, default_value=2048):
    ini_path = osp.join(model_path, 'triton_models/weights/config.ini')
    try:
        with open(ini_path, 'r') as f:
            parser = ConfigParser()
            parser.read_file(f)
            if 'llama' in parser:
                if 'session_len' in parser.options('llama'):
                    return parser.getint('llama', 'session_len')
    except Exception:
        return default_value
    return default_value


def main(model_name, model_path, tp=1, session_id: int = 1):
    model = MODELS.get(model_name)()
    session_len = get_session_len(model_path)
    tm_model = tm.TurboMind(model_path,
                            stop_words=model.stop_words,
                            session_len=session_len,
                            tensor_parallel_size=tp)
    generator = tm_model.create_instance()
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path,
                                              trust_remote_code=True)

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)

    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            prompt = model.get_prompt('', nth_round == 1)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
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
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
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
                    repetition_penalty=1.05,
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
                res, tokens = outputs[0]
                # decode res
                response = tokenizer.decode(
                    res, skip_special_tokens=True)[response_size:]
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size += len(response)

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
