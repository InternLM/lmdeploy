# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.model import MODELS, ChatTemplateConfig, best_match_model
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.turbomind.utils import (ModelSource,
                                      get_model_name_from_workspace_model,
                                      get_model_source)
from lmdeploy.utils import _stop_words

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


def main(model_path: str,
         model_name: str = None,
         session_id: int = 1,
         top_k: float = 40,
         top_p: float = 0.8,
         temperature: float = 0.8,
         repetition_penalty: float = 1.0,
         cap: str = 'chat',
         tp: int = 1,
         stream_output: bool = True,
         request_output_len: int = 1024,
         **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        model_name (str): the name of deployed model
        session_id (int): the identical id of a session
        top_k (int): sampling top k.
        top_p (int): sampling top p.
        temperature (float): sampling temperature.
        repetition_penalty (float): parameter to penalize repetition
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infilling', 'chat', 'python']
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
        request_output_len (int): output token nums
        **kwarg (dict): other arguments for initializing model's chat template
    """
    # chat template
    if model_name is None:
        model_source = get_model_source(model_path)
        if model_source == ModelSource.WORKSPACE:
            model_name = get_model_name_from_workspace_model(model_path)
        else:
            model_name = best_match_model(model_path)
            assert model_name is not None, 'Can not find match model template'
            print(f'match template: <{model_name}>')
    chat_template_args = {}
    new_kwargs = {}
    for k, v in kwargs.items():
        if hasattr(ChatTemplateConfig, k) and v is not None:
            chat_template_args[k] = v
        else:
            new_kwargs[k] = v
    if 'capability' not in chat_template_args:
        chat_template_args['capability'] = cap
    model = MODELS.get(model_name).from_config(**chat_template_args)

    # engine
    from lmdeploy import turbomind as tm
    kwargs = new_kwargs
    tm_model = tm.TurboMind.from_pretrained(model_path, tp=tp, **kwargs)
    tokenizer = tm_model.tokenizer
    generator = tm_model.create_instance()

    # generateion config
    stop_words = _stop_words(model.stop_words, tokenizer)
    if stop_words is not None:
        stop_words = stop_words[0][0].tolist()
    gen_config = EngineGenerationConfig(max_new_tokens=request_output_len,
                                        top_k=top_k,
                                        top_p=top_p,
                                        temperature=temperature,
                                        repetition_penalty=repetition_penalty,
                                        stop_words=stop_words)

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)

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
            print(f'{prompt} ', end='', flush=True)
            state = DetokenizeState()
            gen_config.random_seed = seed

            if model.capability == 'chat':
                sequence_start = (nth_round == 1)
                sequence_end = False
                step = step
            else:
                sequence_start = True
                sequence_end = True
                step = 0

            if step + len(
                    input_ids) + request_output_len >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    gen_config=gen_config,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=step,
                    stream_output=stream_output):
                _, res, tokens = outputs
                # decode res
                response, state = tokenizer.detokenize_incrementally(
                    res, state=state)
                response = valid_str(response)
                print(f'{response}', end='', flush=True)

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    import fire

    fire.Fire(main)
