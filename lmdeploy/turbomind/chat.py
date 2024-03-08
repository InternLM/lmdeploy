# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.tokenizer import DetokenizeState

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
         cap: str = 'chat',
         tp: int = 1,
         stream_output: bool = True,
         request_output_len: int = 1024,
         chat_template_cfg: ChatTemplateConfig = None,
         **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        model_name (str): the name of deployed model
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infilling', 'chat', 'python']
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
        request_output_len (int): output token nums
        chat_template_cfg (ChatTemplateConfig): Chat template config
        **kwarg (dict): other arguments for initializing model's chat template
    """
    from lmdeploy import turbomind as tm
    if chat_template_cfg is None:
        chat_template_cfg = ChatTemplateConfig(model_name=model_name,
                                               capability=cap)
        new_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(chat_template_cfg, k):
                setattr(chat_template_cfg, k, v)
            else:
                new_kwargs[k] = v
        kwargs = new_kwargs

    engine_cfg = TurbomindEngineConfig(model_name=model_name, tp=tp)
    for k, v in kwargs.items():
        if hasattr(engine_cfg, k):
            setattr(engine_cfg, k, v)

    tm_model = tm.TurboMind.from_pretrained(
        model_path,
        model_name=model_name,
        engine_config=engine_cfg,
        tp=tp,
        capability=cap,
        chat_template_config=chat_template_cfg,
        **kwargs)
    tokenizer = tm_model.tokenizer
    generator = tm_model.create_instance()
    gen_config = EngineGenerationConfig(top_k=40)

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model_name = tm_model.model_name
    model = tm_model.model

    print(f'session {session_id}')
    while True:
        prompt = input_prompt(model_name)
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            prompt = model.get_prompt('', nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    request_output_len=request_output_len,
                    sequence_start=False,
                    sequence_end=True,
                    stream_output=stream_output):
                pass
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            prompt = model.get_prompt(prompt, nth_round == 1)
            input_ids = tokenizer.encode(prompt, nth_round == 1)
            if step + len(
                    input_ids) + request_output_len >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            sequence_start = (nth_round == 1)
            sequence_end = False
            if cap != 'chat':  # not interactive for other capability
                sequence_start, sequence_end = True, True
                step = 0

            print(f'{prompt} ', end='', flush=True)
            state = DetokenizeState()
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=step,
                    stream_output=stream_output,
                    gen_config=gen_config,
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
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
