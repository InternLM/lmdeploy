# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.model'])


@MODELS.register_module(name='llama')
class BaseModel:
    """Base model."""

    def __init__(self):
        self.session_len = 2048
        self.top_p = 0.8
        self.top_k = None
        self.temperature = 0.8
        self.repetition_penalty = 1.0

    @staticmethod
    def get_prompt(prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        return prompt

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return None


@MODELS.register_module(name='vicuna')
class Vicuna(BaseModel):
    """Chat template of vicuna model."""

    def __init__(self):
        super().__init__()
        self.system = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """  # noqa: E501
        self.user = 'USER'
        self.assistant = 'ASSISTANT'

    def get_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        if sequence_start:
            return f'{self.system} {self.user}: {prompt} {self.assistant}:'
        else:
            return f'</s>{self.user}: {prompt} {self.assistant}:'


@MODELS.register_module(name='internlm')
class InternLM(BaseModel):

    def __init__(self):
        super().__init__()


@MODELS.register_module(name='internlm-chat-7b')
class InternLMChat7B(BaseModel):
    """Chat template of InternLM model."""

    def __init__(self):
        super().__init__()
        self.system = ''
        self.user = '<|User|>'
        self.eoh = '<eoh>'
        self.eoa = '<eoa>'
        self.assistant = '<|Bot|>'

    def get_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        if sequence_start:
            return f'<BOS>{self.user}:{prompt}{self.eoh}\n' \
                   f'{self.assistant}:'
        else:
            return f'\n{self.user}:{prompt}{self.eoh}\n' \
                   f'{self.assistant}:'

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return [103027, 103028]


@MODELS.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):

    def __init__(self):
        super(InternLMChat7B8K, self).__init__()
        self.session_len = 8192


@MODELS.register_module(name='puyu')
class Puyu(BaseModel):
    """Chat template of puyu model.This is only for internal usage in Shanghai
    AI Laboratory."""

    def __init__(self):
        super().__init__()
        self.system = """meta instruction
You are an AI assistant whose name is InternLM (书生·浦语).
- 书生·浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 书生·浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation"""  # noqa: E501
        self.user = '<|Human|>'
        self.eoh = 'െ'
        self.assistant = '<|Assistant|>'

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'<BOS>{self.system}\n' \
                   f'{self.user}:{prompt}{self.eoh}\n' \
                   f'{self.assistant}:'
        else:
            return f'\n{self.user}:{prompt}{self.eoh}\n{self.assistant}:'

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return [45623]


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseModel):

    def __init__(self):
        super().__init__()
        self.repetition_penalty = 1.1


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
