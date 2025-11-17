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

    pipe = build_pipe(model_path, backend, **kwargs)
    gen_config = build_gen_config(**kwargs)
    adapter_name = get_adapter_name(**kwargs)

    quit = False
    while not quit:
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
                if prompt.strip() == '':
                    continue
                resps = sess(prompt, adapter_name=adapter_name)
                try:
                    for resp in resps:
                        print(resp.text, end='', flush=True)
                except KeyboardInterrupt:
                    sess.stop()
    else:
        print('exiting...')


if __name__ == '__main__':
    fire.Fire(main)
