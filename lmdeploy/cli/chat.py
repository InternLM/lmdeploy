# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import closing

import fire

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.archs import autoget_backend


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def build_pipe(model_path, backend, **kwargs):
    engine_config = None
    if kwargs.get('enable_prefix_caching', False):
        print('interactive chat cannot be used when prefix caching is enabled')
        exit(-1)
    if backend == 'turbomind':
        engine_config = TurbomindEngineConfig()
        for key, value in kwargs.items():
            if hasattr(TurbomindEngineConfig, key):
                setattr(engine_config, key, value)
    else:
        engine_config = PytorchEngineConfig()
        for key, value in kwargs.items():
            key = 'device_type' if key == 'device' else key
            if hasattr(PytorchEngineConfig, key):
                setattr(engine_config, key, value)
        if kwargs.get('adapters', None):
            from .utils import get_lora_adapters
            adapters = get_lora_adapters(kwargs['adapters'])
            engine_config.adapters = adapters
    # disable metrics to avoid installing prometheus_client, which is not needed
    # in interactive chat
    engine_config.enable_metrics = False

    # set chat template config
    chat_template = kwargs.get('chat_template', None)
    chat_template_config = None
    if chat_template:
        from .utils import get_chat_template
        chat_template_config = get_chat_template(chat_template, model_path)
    pipe = pipeline(model_path,
                    backend_config=engine_config,
                    chat_template_config=chat_template_config,
                    log_level='ERROR',
                    **kwargs)
    return pipe


def build_gen_config(**kwargs):
    gen_config = GenerationConfig(do_sample=True, max_new_tokens=4096)
    for key, value in kwargs.items():
        if hasattr(GenerationConfig, key):
            setattr(gen_config, key, value)
    return gen_config


def get_adapter_name(adapters=None, **kwargs):
    if adapters is None:
        return None
    from .utils import get_lora_adapters
    adapters = get_lora_adapters(adapters)
    return list(adapters.keys())[0]


def main(model_path, backend, **kwargs):
    if backend != 'pytorch':
        # set auto backend mode
        backend = autoget_backend(model_path)
    quit = False
    with build_pipe(model_path, backend, **kwargs) as pipe:
        gen_config = build_gen_config(**kwargs)
        adapter_name = get_adapter_name(**kwargs)
        while not quit:
            with closing(pipe.session()) as sess:
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
                    if prompt.strip() == '':
                        continue
                    resps = pipe.chat(prompt,
                                      session=sess,
                                      gen_config=gen_config,
                                      adapter_name=adapter_name,
                                      stream_response=True)
                    try:
                        for resp in resps:
                            print(resp.text, end='', flush=True)
                    except KeyboardInterrupt:
                        sess.abort()
        else:
            print('exiting...')


if __name__ == '__main__':
    fire.Fire(main)
