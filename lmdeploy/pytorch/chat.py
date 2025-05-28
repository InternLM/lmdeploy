# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import os
import random
from typing import Optional

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import get_names_from_model

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
    from lmdeploy import pipeline

    if gen_config is None:
        gen_config = GenerationConfig(do_sample=True)

    adapter_name = None
    if engine_config.adapters is not None:
        adapter_name = next(iter(engine_config.adapters.keys()))

    chat_count = 0

    def __reset_chat_state():
        """Reset chat state."""
        nonlocal chat_count
        seed = random.getrandbits(64)
        gen_config.random_seed = seed

    async def __generate(prompt: str):
        """Chat generate."""
        nonlocal chat_count
        print()
        async for out in pipe.generate(
                prompt,
                session_id,
                gen_config=gen_config,
                sequence_start=chat_count == 0,
                sequence_end=False,
                adapter_name=adapter_name,
        ):
            print(f'{out.response}', end='', flush=True)
        print()
        chat_count += 1

    async def __chat_step(prompt: str):
        """Chat step."""
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            await pipe.stop_session(session_id)
            __reset_chat_state()
        else:
            await __generate(prompt)

    async def __chat_loop(model_path: str):
        """Chat loop."""
        __reset_chat_state()
        _, chat_template_name = get_names_from_model(model_path)
        while True:
            prompt = input_prompt(chat_template_name)
            await __chat_step(prompt)

    with pipeline(
            model_path,
            backend_config=engine_config,
            chat_template_config=chat_template_config,
    ) as pipe:
        try:
            asyncio.run(__chat_loop(model_path))
        except KeyboardInterrupt:
            exit(0)


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
