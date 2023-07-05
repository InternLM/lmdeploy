# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.model'])


@MODELS.register_module(name='vicuna')
class Vicuna:

    def __init__(self):
        self.system = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """  # noqa: E501
        self.user = 'USER'
        self.assistant = 'ASSISTANT'

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'{self.system} {self.user}: {prompt} {self.assistant}:'
        else:
            return f'</s>{self.user}: {prompt} {self.assistant}:'

    @property
    def stop_words(self):
        return None


@MODELS.register_module(name='internlm')
class Puyu:

    def __init__(self):
        self.system = ''
        self.user = '<|User|>'
        self.eou = '<TOKENS_UNUSED_0>'
        self.assistant = '<|Bot|>'

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'{self.system}\n' \
                   f'{self.user}:{prompt}{self.eou}\n' \
                   f'{self.assistant}:'
        else:
            return f'\n{self.user}:{prompt}{self.eou}\n{self.assistant}:'

    @property
    def stop_words(self):
        return [103027, 103028]


@MODELS.register_module(name='llama')
class Llama:

    def __init__(self):
        pass

    def get_prompt(self, prompt, sequence_start=True):
        return prompt

    @property
    def stop_words(self):
        return None


def main(model_name: str = 'test'):
    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'
    model = MODELS.get('vicuna--1')()
    prompt = model.get_prompt(prompt='hi')
    print(prompt)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
