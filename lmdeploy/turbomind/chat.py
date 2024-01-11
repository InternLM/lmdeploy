# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from configparser import ConfigParser

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.model import ChatTemplateConfig, best_match_model

from .deploy.target_model.base import TurbomindModelConfig
from .engine_config import EngineConfig
from .utils import ModelSource, get_model_source

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


def update_engine_config(input_engine_config_is_none: bool,
                         engine_config: EngineConfig, model_path: str):

    model_source = get_model_source(model_path)
    if model_source == ModelSource.HF_MODEL:
        if engine_config.model_name is None:
            engine_config.model_name = best_match_model(model_path)
    else:
        ini_path = os.path.join(model_path, 'triton_models', 'weights',
                                'config.ini')
        # load cfg
        with open(ini_path, 'r') as f:
            parser = ConfigParser()
            parser.read_file(f)
        section_name = 'llama'
        _cfg = parser._sections[section_name]
        cfg = TurbomindModelConfig.from_dict(_cfg)
        if engine_config.model_name is None:
            engine_config.model_name = cfg.model_name

        # if read from workspace and doesn't pass engine_config
        if input_engine_config_is_none:
            for k, v in cfg.__dict__.items():
                if hasattr(engine_config, k):
                    setattr(engine_config, k, v)
    return engine_config


def main(model_path,
         model_name: str = None,
         session_id: int = 1,
         cap: str = 'chat',
         tp: int = 1,
         stream_output: bool = True,
         request_output_len: int = 512,
         engine_config: EngineConfig = None,
         **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infilling', 'chat', 'python']
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
        **kwarg (dict): other arguments for initializing model's chat template
    """
    from lmdeploy import turbomind as tm

    def _create_cfg(cls, *args, **kwargs):
        used_args = {}
        for k, v in kwargs.items():
            if v and hasattr(cls, k):
                used_args[k] = v
        return cls(*args, **used_args)

    # engine config
    input_engine_config_is_none = engine_config is None
    if engine_config is None:
        engine_config = _create_cfg(EngineConfig,
                                    **kwargs,
                                    model_name=model_name,
                                    tp=tp)
    engine_config = update_engine_config(input_engine_config_is_none,
                                         engine_config, model_path)

    # chat template
    chat_template_config = _create_cfg(ChatTemplateConfig,
                                       engine_config.model_name,
                                       **kwargs,
                                       capability=cap)

    tm_model = tm.TurboMind.from_pretrained(
        model_path,
        engine_config=engine_config,
        chat_template_config=chat_template_config)
    tokenizer = tm_model.tokenizer
    generator = tm_model.create_instance()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model_name = tm_model.model_name
    model = tm_model.model

    # gen_config
    gen_config = _create_cfg(EngineGenerationConfig,
                             **model.sampling_param.__dict__)

    print(f'session {session_id}')
    while True:
        prompt = input_prompt(model_name)
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            prompt = model.get_prompt('', nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            gen_config.max_new_tokens = 0
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=[input_ids],
                                                  gen_config=gen_config,
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

            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            gen_config.max_new_tokens = request_output_len
            gen_config.random_seed = seed
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    stream_output=stream_output,
                    step=step,
                    sequence_start=(nth_round == 1),
                    gen_config=gen_config):
                _, res, tokens = outputs
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
