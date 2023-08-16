# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

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

    @staticmethod
    def _translate_messages(messages: List):
        """Translate messages into system, user speaking list, assistant
        speaking list.

        Args:
            messages (List): chat history
        Returns:
            Turple: consists of system (str), users (List[str]),
                assistants (List[str])
        """
        system = None
        users = []
        assistants = []
        assert isinstance(messages, List)
        for message in messages:
            msg_role = message['role']
            if msg_role == 'system':
                system = message['content']
            elif msg_role == 'user':
                users.append(message['content'])
            elif msg_role == 'assistant':
                assistants.append(message['content'])
            else:
                raise ValueError(f'Unknown role: {msg_role}')
        assistants.append(None)
        return system, users, assistants

    @abstractmethod
    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template. When messages arg is a string, return
        self.get_prompt(messages). When messages arg is a chat history, return
        translated prompt from chat history.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages)
        # chat history processing in derived classes

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
            return f'{self.system} {self.user}: {prompt} {self.assistant}: '
        else:
            return f'</s>{self.user}: {prompt} {self.assistant}: '

    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        system, users, assistants = self._translate_messages(messages)
        system = self.system if not system else system
        ret = system + ' '
        for user, assistant in zip(users, assistants):
            if assistant:
                ret += f'{self.user}: {user} {self.assistant}: {assistant}</s>'
            else:
                ret += f'{self.user}: {user} {self.assistant}: '
        return ret


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

    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        system, users, assistants = self._translate_messages(messages)
        ret = '<BOS>'
        for user, assistant in zip(users, assistants):
            if assistant:
                ret += f'{self.user}:{user}{self.eoh}\n{self.assistant}:' \
                       f'{assistant}{self.eoa}'
            else:
                ret += f'{self.user}:{user}{self.eoh}\n{self.assistant}:'
        return ret

    @property
    def stop_words(self):
        """Return the stop-words' token ids."""
        return [103027, 103028]


@MODELS.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):

    def __init__(self):
        super(InternLMChat7B8K, self).__init__()
        self.session_len = 8192


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseModel):

    def __init__(self):
        super().__init__()
        self.repetition_penalty = 1.1


@MODELS.register_module(name='llama2')
class Llama2(BaseModel):
    """Chat template of LLaMA2 model."""

    def __init__(self):
        super().__init__()
        B_INST, E_INST = '[INST]', '[/INST]'
        B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

        DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501

        self.b_inst = B_INST
        self.e_inst = E_INST
        self.b_sys = B_SYS
        self.e_sys = E_SYS
        self.default_sys_prompt = DEFAULT_SYSTEM_PROMPT
        self.session_len = 4096

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
