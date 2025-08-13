# Copyright (c) OpenMMLab. All rights reserved.
import fire

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.archs import autoget_backend


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def build_pipe(model_path, backend, **kwargs):
    # # set enable_prefix_cache
    # disable_prefix_cache = kwargs.pop('disable_prefix_cache', False)
    # kwargs.update(enable_prefix_caching=not disable_prefix_cache)
    # set engine config
    engine_config = None
    if backend == 'turbomind':
        engine_config = TurbomindEngineConfig()
        for key, value in kwargs.items():
            if hasattr(TurbomindEngineConfig, key):
                setattr(engine_config, key, value)
    else:
        engine_config = PytorchEngineConfig()
        for key, value in kwargs.items():
            if hasattr(PytorchEngineConfig, key):
                setattr(engine_config, key, value)
        if kwargs.get('adapters', None):
            from .utils import get_lora_adapters
            adapters = get_lora_adapters(kwargs['adapters'])
            engine_config.adapters = adapters
    # set chat template config
    chat_template = kwargs.get('chat_template', None)
    chat_template_config = None
    if chat_template:
        from .utils import get_chat_template
        chat_template_config = get_chat_template(chat_template)

    pipe = pipeline(model_path,
                    backend_config=engine_config,
                    chat_template_config=chat_template_config,
                    log_level='ERROR',
                    **kwargs)
    return pipe


def build_gen_config(**kwargs):
    gen_config = GenerationConfig(max_new_tokens=1024, top_k=40, top_p=0.8, temperature=0.8, repetition_penalty=1.0)
    for key, value in kwargs.items():
        if hasattr(GenerationConfig, key):
            setattr(gen_config, key, value)
    return gen_config


def main(model_path, backend, **kwargs):
    if backend != 'pytorch':
        # set auto backend mode
        backend = autoget_backend(model_path)

    pipe = build_pipe(model_path, backend, **kwargs)
    gen_config = build_gen_config(**kwargs)

    quit = False
    while True:
        with pipe.session(gen_config) as sess:
            while True:
                try:
                    prompt = input_prompt()
                except KeyboardInterrupt:
                    quit = True
                    break
                if prompt == 'end':
                    sess.close()
                    break
                if prompt == 'exit':
                    quit = True
                    break
                resps = sess(prompt)
                try:
                    for resp in resps:
                        print(resp.text, end='', flush=True)
                except KeyboardInterrupt:
                    sess.stop()
        if quit:
            print('exiting...')
            break


if __name__ == '__main__':
    fire.Fire(main)
