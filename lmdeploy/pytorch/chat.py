# Copyright (c) OpenMMLab. All rights reserved.

import os
import random
from typing import List, Optional

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import get_names_from_model
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import _get_and_verify_max_len

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def input_prompt(chat_template_name):
    """Input a prompt in the consolo interface."""
    if chat_template_name == 'codellama':
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


def run_chat(model_path: str,
             engine_config: PytorchEngineConfig,
             gen_config: GenerationConfig = None,
             session_id: int = 1,
             trust_remote_code: bool = True,
             chat_template_config: Optional[ChatTemplateConfig] = None):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the huggingface model path.
        engine_config (PytorchEngineConfig): Config of engine.
        gen_config (GenerationConfig): Config of generation.
        session_id (int): the identical id of a session.
        trust_remote_code (bool): trust remote code.
    """
    from lmdeploy.pytorch.engine import Engine
    tm_model = Engine.from_pretrained(model_path,
                                      engine_config=engine_config,
                                      trust_remote_code=trust_remote_code)
    tokenizer = tm_model.tokenizer
    generator = tm_model.create_instance()
    adapter_name = None
    if engine_config.adapters is not None:
        adapter_name = next(iter(engine_config.adapters.keys()))

    if gen_config is None:
        gen_config = GenerationConfig()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)

    _, chat_template_name = get_names_from_model(model_path)
    if chat_template_config is None:
        chat_template_config = ChatTemplateConfig(chat_template_name)
    model = chat_template_config.chat_template

    stop_words = _stop_words(model.stop_words, tokenizer)

    _, model_config = get_model_arch(model_path)
    session_len = _get_and_verify_max_len(model_config, None)

    while True:
        prompt = input_prompt(chat_template_name)
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
            if step >= session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            print(f'{prompt}', end='', flush=True)
            state = DetokenizeState(len(input_ids))
            gen_config.random_seed = seed
            gen_config.stop_token_ids = stop_words
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=input_ids,
                                                  gen_config=gen_config,
                                                  adapter_name=adapter_name):
                res, tokens = input_ids + outputs.token_ids, outputs.num_token
                # decode res
                response, state = tokenizer.detokenize_incrementally(
                    res, state)
                response = valid_str(response)
                print(f'{response}', end='', flush=True)

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


def main(model_path: str,
         session_id: int = 1,
         top_k: float = 40,
         top_p: float = 0.8,
         temperature: float = 0.8,
         repetition_penalty: float = 1.0,
         tp: int = 1,
         adapter: str = None,
         trust_remote_code: bool = True,
         chat_template: str = None):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the huggingface model path
        session_id (int): the identical id of a session
        top_k (int): sampling top k.
        top_p (int): sampling top p.
        temperature (float): sampling temperature.
        repetition_penalty (float): parameter to penalize repetition
        tp (int): GPU number used in tensor parallelism
        adapter (str): path to lora adapter.
        trust_remote_code (bool): Trust remote code.
        chat_template (str): A JSON file or string that specifies the
            chat template configuration.
    """
    adapters = None
    if adapter is not None:
        adapters = dict(default=adapter)
    engine_config = PytorchEngineConfig(tp=tp, adapters=adapters)
    gen_config = GenerationConfig(max_new_tokens=512,
                                  top_k=top_k,
                                  top_p=top_p,
                                  temperature=temperature,
                                  repetition_penalty=repetition_penalty,
                                  ignore_eos=False)
    chat_template_config = None
    if chat_template is not None and os.path.exists(chat_template):
        chat_template_config = ChatTemplateConfig.from_json(chat_template)
    return run_chat(model_path,
                    engine_config,
                    gen_config,
                    session_id=session_id,
                    trust_remote_code=trust_remote_code,
                    chat_template_config=chat_template_config)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
