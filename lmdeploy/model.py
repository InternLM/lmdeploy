# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.model'])


@MODELS.register_module(name='llama')
class BaseModel:
    """Base model."""

    def __init__(self,
                 session_len=2048,
                 top_p=0.8,
                 top_k=None,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """  # noqa: E501
        self.user = kwargs.get('user', 'USER')
        self.assistant = kwargs.get('assistant', 'ASSISTANT')
        self.system = kwargs.get('system', self.system)

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@MODELS.register_module(name='internlm-chat-7b')
class InternLMChat7B(BaseModel):
    """Chat template of InternLM model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system = kwargs.get('system', '')
        self.user = kwargs.get('user', '<|User|>')
        self.eoh = kwargs.get('eoh', '<eoh>')
        self.eoa = kwargs.get('eoa', '<eoa>')
        self.assistant = kwargs.get('assistant', '<|Bot|>')

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

    def __init__(self, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = kwargs.get('session_len', 8192)


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repetition_penalty = kwargs[
            'repetition_penalty'] if 'repetition_penalty' in kwargs else 1.1


@MODELS.register_module(name='puyu')
class Puyu(BaseModel):
    """Chat template of puyu model.This is only for internal usage in Shanghai
    AI Laboratory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta_instruction = kwargs.get('meta_instruction', '')
        self.user = kwargs.get('user', '<|Human|>: ')
        self.eoh = kwargs.get('eoh', '')
        self.eosys = kwargs.get('eosys', '')
        self.assistant = kwargs.get('assistant', '<|Assistant|>: ')
        self.system = kwargs.get('system', '<|System|>: ')

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'<BOS>{self.system}{self.meta_instruction}{self.eosys}\n' \
                   f'{self.user}{prompt}{self.eoh}\n' \
                   f'{self.assistant}'
        else:
            return f'\n{self.user}{prompt}{self.eoh}\n{self.assistant}'

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return [45623]


@MODELS.register_module(name='llama2')
class Llama2(BaseModel):
    """Chat template of LLaMA2 model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        B_INST, E_INST = '[INST]', '[/INST]'
        B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

        DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501

        self.b_inst = kwargs.get('b_inst', B_INST)
        self.e_inst = kwargs.get('e_inst', E_INST)
        self.b_sys = kwargs.get('b_sys', B_SYS)
        self.e_sys = kwargs.get('e_sys', E_SYS)
        self.default_sys_prompt = kwargs.get('default_sys_prompt',
                                             DEFAULT_SYSTEM_PROMPT)
        self.session_len = kwargs.get('session_len', 4096)

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
            return f'<BOS>{self.b_inst} ' \
                   f'{self.b_sys} {self.default_sys_prompt} {self.e_sys}' \
                   f'{prompt} {self.e_inst} '

        return f'{self.b_inst} {prompt} {self.e_inst} '


@MODELS.register_module(name='qwen-7b')
class Qwen7BChat(BaseModel):
    """Chat template for Qwen-7B-Chat."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_len = kwargs.get('session_len', 8192)
        self.top_p = kwargs.get('top_p', 0.5)
        self.top_k = kwargs.get('top_k', 40)
        self.temperature = kwargs.get('temperature', 1.0)

        self.im_start = kwargs.get('im_start', '<|im_start|>')
        self.im_end = kwargs.get('im_end', '<|im_end|>')
        self.system = kwargs.get('system', 'You are a helpful assistant.')

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'{self.im_start}system\n{self.system}{self.im_end}' \
                   f'\n{self.im_start}user\n{prompt}{self.im_end}' \
                   f'\n{self.im_start}assistant\n'

        return f'\n{self.im_start}user\n{prompt}{self.im_end}' \
               f'\n{self.im_start}assistant\n'

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return [151645]  # <|im_end|>


def main(model_name: str = 'test'):
    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'
    model = MODELS.get(model_name)()
    prompt = model.get_prompt(prompt='hi')
    print(prompt)
    print(f'session_len: {model.session_len}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
