# Copyright (c) OpenMMLab. All rights reserved.

import os
import random
from typing import List

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.model import MODELS, best_match_model
from lmdeploy.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'


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


def _stop_words(stop_words: List[str], tokenizer: Tokenizer):
    """Return a list of token ids corresponding to stop-words."""
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and \
        all(isinstance(elem, str) for elem in stop_words), \
        f'stop_words must be a list but got {type(stop_words)}'
    stop_words = [
        tokenizer.encode(stop_word, False)[-1] for stop_word in stop_words
    ]
    assert isinstance(stop_words, List) and all(
        isinstance(elem, int) for elem in stop_words), 'invalid stop_words'
    return stop_words


def main(model_path,
         model_name: str = None,
         session_id: int = 1,
         top_k=40,
         top_p=0.8,
         temperature=0.8,
         repetition_penalty: float = 1.0,
         tp: int = 1,
         stream_output=True,
         trust_remote_code=True):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the huggingface model path
        session_id (int): the identical id of a session
        repetition_penalty (float): parameter to penalize repetition
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
    """
    from lmdeploy.pytorch.engine import Engine, EngineConfig
    tm_model = Engine(model_path,
                      engine_config=EngineConfig(tp=tp),
                      trust_remote_code=trust_remote_code)
    tokenizer = tm_model.tokenizer
    generator = tm_model.create_instance()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    if model_name is None:
        model_name = best_match_model(model_path)[0]
        assert model_name is not None, 'Can not find match model template'
        print(f'match template: <{model_name}>')
    model = MODELS.get(model_name)()
    stop_words = _stop_words(model.stop_words, tokenizer)

    while True:
        prompt = input_prompt(model_name)
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            generator.end(session_id)
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            prompt = model.get_prompt(prompt, nth_round == 1)
            input_ids = tokenizer.encode(prompt, nth_round == 1)
            session_len = model.session_len
            if session_len is None:
                session_len = tm_model.session_len
            if step >= session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            gen_config = EngineGenerationConfig(
                max_new_tokens=512,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=False,
                random_seed=seed,
                stop_words=stop_words)
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=input_ids,
                                                  gen_config=gen_config):
                status, res, tokens = outputs
                # decode res
                response = tokenizer.decode(res, offset=response_size)
                # utf-8 char at the end means it's a potential unfinished
                # byte sequence, continue to concate it with the next
                # sequence and decode them together
                if response.endswith('�'):
                    continue
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size = tokens

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    import fire

    fire.Fire(main)
