# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

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
        self.stop_words = None

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


@MODELS.register_module(name='vicuna')
class Vicuna(BaseModel):
    """Chat template of vicuna model."""

    def __init__(
            self,
            system="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """,  # noqa: E501
            user='USER',
            assistant='ASSISTANT',
            **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.user = user
        self.assistant = assistant

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@MODELS.register_module(name='internlm-chat-7b')
class InternLMChat7B(BaseModel):
    """Chat template of InternLM model."""

    def __init__(self,
                 system='',
                 user='<|User|>',
                 eoh='<eoh>',
                 eoa='<eoa>',
                 assistant='<|Bot|>',
                 stop_words=[103027, 103028],
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.assistant = assistant
        self.stop_words = stop_words

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


@MODELS.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):

    def __init__(self, session_len=8192, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = session_len


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseModel):

    def __init__(self, repetition_penalty=1.1, **kwargs):
        super().__init__(**kwargs)
        self.repetition_penalty = repetition_penalty


@MODELS.register_module(name='puyu')
class Puyu(BaseModel):
    """Chat template of puyu model.This is only for internal usage in Shanghai
    AI Laboratory."""

    def __init__(self,
                 meta_instruction='',
                 user='<|Human|>: ',
                 eoh='',
                 eosys='',
                 assistant='<|Assistant|>: ',
                 system='<|System|>: ',
                 stop_words=[45623],
                 **kwargs):
        super().__init__(**kwargs)
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eosys = eosys
        self.assistant = assistant
        self.system = system
        self.stop_words = stop_words

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'<BOS>{self.system}{self.meta_instruction}{self.eosys}\n' \
                   f'{self.user}{prompt}{self.eoh}\n' \
                   f'{self.assistant}'
        else:
            return f'\n{self.user}{prompt}{self.eoh}\n{self.assistant}'


@MODELS.register_module(name='llama2')
class Llama2(BaseModel):
    """Chat template of LLaMA2 model."""

    def __init__(
            self,
            b_inst='[INST]',
            e_inst='[/INST]',
            b_sys='<<SYS>>\n',
            e_sys='\n<</SYS>>\n\n',
            default_sys_prompt="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",  # noqa: E501
            session_len=4096,
            **kwargs):
        super().__init__(**kwargs)
        self.b_inst = b_inst
        self.e_inst = e_inst
        self.b_sys = b_sys
        self.e_sys = e_sys
        self.default_sys_prompt = default_sys_prompt
        self.session_len = session_len

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
        system = self.default_sys_prompt if not system else system
        ret = f'<BOS>{self.b_inst} {self.b_sys} {system} {self.e_sys}'
        for i, (user, assistant) in enumerate(zip(users, assistants)):
            if i != 0:
                ret += f'{self.b_inst} '
            if assistant:
                ret += f'{user} {self.e_inst} {assistant}'
            else:
                ret += f'{user} {self.e_inst} '
        return ret


@MODELS.register_module(name='qwen-7b')
class Qwen7BChat(BaseModel):
    """Chat template for Qwen-7B-Chat."""

    def __init__(self,
                 session_len=8192,
                 top_p=0.5,
                 top_k=40,
                 temperature=1.0,
                 im_start='<|im_start|>',
                 im_end='<|im_end|>',
                 system='You are a helpful assistant.',
                 stop_words=[151645],
                 **kwargs):
        super().__init__(**kwargs)
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

        self.im_start = im_start
        self.im_end = im_end
        self.system = system
        self.stop_words = stop_words

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return f'{self.im_start}system\n{self.system}{self.im_end}' \
                   f'\n{self.im_start}user\n{prompt}{self.im_end}' \
                   f'\n{self.im_start}assistant\n'

        return f'\n{self.im_start}user\n{prompt}{self.im_end}' \
               f'\n{self.im_start}assistant\n'


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
