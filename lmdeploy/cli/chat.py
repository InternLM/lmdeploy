# Copyright (c) OpenMMLab. All rights reserved.
import os
from urllib.parse import urlparse

import fire

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.archs import autoget_backend

IMAGE_COMMAND = '/image'


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def normalize_image_source(image_url: str):
    """Resolve local relative image paths against the CLI working directory."""
    if urlparse(image_url).scheme or os.path.isabs(image_url):
        return image_url
    return os.path.abspath(image_url)


def parse_image_command(stripped_line: str):
    """Parse one stripped /image command line into an OpenAI content block."""
    if stripped_line == IMAGE_COMMAND:
        raise ValueError('/image requires an image path or URL')
    if not stripped_line.startswith(f'{IMAGE_COMMAND} '):
        return None

    image_url = stripped_line[len(IMAGE_COMMAND):].strip()
    if not image_url:
        raise ValueError('/image requires an image path or URL')
    image_url = normalize_image_source(image_url)

    return {
        'kind': 'content',
        'block': {
            'type': 'image_url',
            'image_url': {
                'url': image_url
            }
        },
    }


def parse_prompt_line(line: str):
    """Parse one raw prompt line into a text or content segment."""
    stripped = line.strip()
    for parse_command in (parse_image_command, ):
        segment = parse_command(stripped)
        if segment is not None:
            return segment
    return {'kind': 'text', 'text': line}


def merge_text_segments(segments: list[dict]) -> list[dict]:
    """Merge adjacent text segments into OpenAI text content blocks."""
    content = []
    pending_text = []

    def append_pending_text():
        if any(line.strip() for line in pending_text):
            content.append({'type': 'text', 'text': '\n'.join(pending_text)})
        pending_text.clear()

    for segment in segments:
        if segment['kind'] == 'text':
            pending_text.append(segment['text'])
            continue

        append_pending_text()
        content.append(segment['block'])

    append_pending_text()
    return content


def parse_interactive_prompt(prompt: str):
    """Parse a completed interactive chat prompt block.

    Text-only prompts return the original string to preserve existing CLI behavior. Prompts with image commands return
    ordered OpenAI multimodal content blocks.
    """
    segments = [parse_prompt_line(line) for line in prompt.splitlines()]
    content = merge_text_segments(segments)
    has_image = any(block['type'] == 'image_url' for block in content)
    return content if has_image else prompt


def build_pipe(model_path, backend, trust_remote_code=False, **kwargs):
    engine_config = None
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
                    trust_remote_code=trust_remote_code,
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


def main(model_path, backend, trust_remote_code=False, **kwargs):
    if backend != 'pytorch':
        # set auto backend mode
        backend = autoget_backend(model_path, trust_remote_code=trust_remote_code)
    quit = False
    messages = []
    with build_pipe(model_path, backend, trust_remote_code=trust_remote_code, **kwargs) as pipe:
        gen_config = build_gen_config(**kwargs)
        adapter_name = get_adapter_name(**kwargs)
        while not quit:
            try:
                prompt = input_prompt()
            except KeyboardInterrupt:
                quit = True
                continue
            if prompt == 'end':
                messages.clear()
                continue
            if prompt == 'exit':
                quit = True
                continue
            if prompt.strip() == '':
                continue

            try:
                content = parse_interactive_prompt(prompt)
            except ValueError as exc:
                print(f'Error: {exc}')
                continue

            messages.append({'role': 'user', 'content': content})
            request_messages = [message.copy() for message in messages]
            # This session is only a per-request cancellation handle; the
            # conversation state lives in the Python transcript above.
            request_session = pipe.session()
            response_text = ''
            resps = pipe.stream_infer(request_messages,
                                      sessions=request_session,
                                      gen_config=gen_config,
                                      adapter_name=adapter_name,
                                      stream_response=True,
                                      sequence_start=True,
                                      sequence_end=True)
            try:
                for resp in resps:
                    print(resp.text, end='', flush=True)
                    response_text += resp.text
            except KeyboardInterrupt:
                request_session.abort()
                messages.pop()
                print()
                continue

            messages.append({'role': 'assistant', 'content': response_text})
        else:
            print('exiting...')


if __name__ == '__main__':
    fire.Fire(main)
