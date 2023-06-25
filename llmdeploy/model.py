# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

MODELS = Registry('model', locations=['llmdeploy.model'])


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


@MODELS.register_module(name='puyu')
class Puyu:

    def __init__(self):
        self.system = """meta instruction
You are an AI assistant whose name is InternLM (书生·浦语).
- 书生·浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 书生·浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation"""  # noqa: E501
        self.user = '<|Human|>'
        self.eou = 'െ'
        self.assistant = '<|Assistant|>'

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'{self.system}\n' \
                   f'{self.user}:{prompt}{self.eou}\n' \
                   f'{self.assistant}:'
        else:
            return f'\n{self.user}:{prompt}{self.eou}\n{self.assistant}:'

    @property
    def stop_words(self):
        return [45623]


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
