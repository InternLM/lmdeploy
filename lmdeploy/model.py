# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import os
from abc import abstractmethod
from typing import List, Literal, Optional

from fuzzywuzzy import fuzz, process
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.model'])


@dataclasses.dataclass
class SamplingParam:
    top_p: float = 0.8
    top_k: float = None
    temperature: float = 0.8
    repetition_penalty: float = 1.0


@dataclasses.dataclass
class ChatTemplateConfig:
    """Parameters for chat template.

    Args:
        model_name (str): the name of the deployed model. Determine which chat template will be applied
        system (str | None): begin of the system prompt
        meta_instruction (str | None): system prompt
        eosys (str | None): end of the system prompt
        user (str | None): begin of the user prompt
        eoh (str | None): end of the user prompt
        assistant (str | None): begin of the assistant prompt
        eoa (str | None): end of the assistant prompt
        capability: ('completion' | 'infilling' | 'chat' | 'python') = None
    """  # noqa: E501

    model_name: str
    system: Optional[str] = None
    meta_instruction: Optional[str] = None
    eosys: Optional[str] = None
    user: Optional[str] = None
    eoh: Optional[str] = None
    assistant: Optional[str] = None
    eoa: Optional[str] = None
    capability: Optional[Literal['completion', 'infilling', 'chat',
                                 'python']] = None

    @property
    def chat_template(self):
        attrs = {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
        model: BaseModel = MODELS.get(self.model_name).from_config(**attrs)
        return model


@MODELS.register_module(name='internlm')
@MODELS.register_module(name='llama')
@MODELS.register_module(name='base')
class BaseModel:
    """Base model."""

    def __init__(self,
                 session_len=2048,
                 top_p=0.8,
                 top_k=None,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 capability='chat',
                 stop_words=None,
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.stop_words = stop_words
        self.capability = capability

    @classmethod
    def from_config(cls, **kwargs):
        """Build model from ModelConfig parameters."""
        return cls(**kwargs)

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
        if self.capability == 'completion':
            return prompt
        else:
            return self.decorate_prompt(prompt, sequence_start)

    @abstractmethod
    def decorate_prompt(self, prompt, sequence_start):
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
    def sampling_param(self):
        return SamplingParam(top_p=self.top_p,
                             top_k=self.top_k,
                             temperature=self.temperature,
                             repetition_penalty=self.repetition_penalty)


@MODELS.register_module(name='wizardlm')
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

    @classmethod
    def from_config(cls, **kwargs):
        """Build model from ModelConfig parameters."""
        mapping = {'meta_instruction': 'system'}
        new_kwargs = {
            mapping.get(key, key): value
            for key, value in kwargs.items()
        }
        return cls(**new_kwargs)

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
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


@MODELS.register_module(name='internlm-chat')
@MODELS.register_module(name='internlm-chat-7b')
class InternLMChat7B(BaseModel):
    """Chat template of InternLM model."""

    def __init__(
            self,
            system='<|System|>:',
            meta_instruction="""You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
""",  # noqa: E501
            user='<|User|>:',
            eoh='\n',
            eoa='<eoa>\n',
            eosys='\n',
            assistant='<|Bot|>:',
            stop_words=['<eoa>'],
            **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.eosys = eosys
        self.assistant = assistant
        self.stop_words = stop_words

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            ret = ''
            if self.meta_instruction:
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
            ret += f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
            return ret

        else:
            return f'\n{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'

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
        eox_map = dict(user=self.eoh, assistant=self.eoa, system=self.eosys)
        ret = ''
        if self.meta_instruction:
            ret += f'{self.system}{self.meta_instruction}{self.eosys}'

        for message in messages:
            role = message['role']
            content = message['content']
            ret += f'{eval(f"self.{role}")}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


@MODELS.register_module(name='internlm-chat-20b')
@MODELS.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):
    """Chat template and generation parameters of InternLM-Chat-7B-8K and
    InternLM-Chat-20B models."""

    def __init__(self, session_len=8192, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = session_len


@MODELS.register_module(name='internlm-20b')
class InternLMBaseModel20B(BaseModel):
    """Generation parameters of InternLM-20B-Base model."""

    def __init__(self, session_len=4096, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)


@MODELS.register_module(name=['internlm2-7b', 'internlm2-20b'])
class InternLM2BaseModel7B(BaseModel):
    """Generation parameters of InternLM2-7B-Base model."""

    def __init__(self, session_len=32768, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)


@MODELS.register_module(name=['internlm2-chat-7b', 'internlm2-chat-20b'])
class InternLM2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM2-Chat-7B."""

    def __init__(self,
                 session_len=32768,
                 system='<|im_start|>system\n',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
        super(InternLM2Chat7B, self).__init__(session_len=session_len,
                                              system=system,
                                              user=user,
                                              assistant=assistant,
                                              eosys=eosys,
                                              eoh=eoh,
                                              eoa=eoa,
                                              stop_words=stop_words,
                                              **kwargs)


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseModel):
    """Generation parameters of Baichuan-7B base model."""

    def __init__(self, repetition_penalty=1.1, **kwargs):
        super().__init__(**kwargs)
        self.repetition_penalty = repetition_penalty


@MODELS.register_module(name='baichuan2-7b')
class Baichuan2_7B(BaseModel):
    """Chat template and generation parameters of Baichuan2-7B-Base and
    Baichuan2-7B-Chat models."""

    def __init__(self,
                 temperature=0.3,
                 top_k=5,
                 top_p=0.85,
                 repetition_penalty=1.05,
                 **kwargs):
        super().__init__(temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         repetition_penalty=repetition_penalty,
                         **kwargs)
        self.user_token = '<reserved_106>'  # id = 195
        self.assistant_token = '<reserved_107>'  # id = 196

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        return f'{self.user_token}{prompt}{self.assistant_token}'

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
        ret = ''
        for user, assistant in zip(users, assistants):
            ret += f'{self.user_token}{user}{self.assistant_token}'
            if assistant:
                ret += f'{assistant}'
        return ret


@MODELS.register_module(name='puyu')
class Puyu(BaseModel):
    """Chat template of puyu model.This is only for internal usage in Shanghai
    AI Laboratory."""

    def __init__(self,
                 meta_instruction='',
                 system='',
                 eosys='',
                 user='',
                 eoh='',
                 assistant='',
                 eoa='',
                 stop_words=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.meta_instruction = meta_instruction
        self.system = system
        self.user = user
        self.assistant = assistant
        self.stop_words = stop_words
        self.eosys = eosys
        self.eoh = eoh
        self.eoa = eoa

    def decorate_prompt(self, prompt, sequence_start=True):
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.system}{self.meta_instruction}{self.eosys}' \
                   f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
        else:
            return f'{self.eoa}{self.user}{prompt}{self.eoh}{self.assistant}'

    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
            sequence_start (bool): flag to start the sequence
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        eox_map = dict(user=self.eoh, assistant=self.eoa, system=self.eosys)
        ret = ''
        if self.meta_instruction:
            ret += f'{self.system}{self.meta_instruction}{self.eosys}'

        for message in messages:
            role = message['role']
            content = message['content']
            ret += f'{eval(f"self.{role}")}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


@MODELS.register_module(name=['llama2', 'llama-2', 'llama-2-chat'])
class Llama2(BaseModel):
    """Chat template of LLaMA2 model."""

    def __init__(
            self,
            b_inst='[INST]',
            e_inst='[/INST]',
            b_sys='<<SYS>>\n',
            e_sys='\n<</SYS>>\n\n',
            system="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",  # noqa: E501
            session_len=4096,
            **kwargs):
        super().__init__(**kwargs)
        self.b_inst = b_inst
        self.e_inst = e_inst
        self.b_sys = b_sys
        self.e_sys = e_sys
        self.default_sys_prompt = system
        self.session_len = session_len

    @classmethod
    def from_config(cls, **kwargs):
        """Build model from ModelConfig parameters."""
        mapping = {
            'system': 'b_sys',
            'eosys': 'e_sys',
            'meta_instruction': 'system'
        }
        new_kwargs = {
            mapping.get(key, key): value
            for key, value in kwargs.items()
        }
        return cls(**new_kwargs)

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.b_inst} ' \
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
        ret = f'{self.b_inst} {self.b_sys} {system} {self.e_sys}'
        for i, (user, assistant) in enumerate(zip(users, assistants)):
            if i != 0:
                ret += f'{self.b_inst} '
            if assistant:
                ret += f'{user} {self.e_inst} {assistant}'
            else:
                ret += f'{user} {self.e_inst} '
        return ret


@MODELS.register_module(name='qwen-14b')
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
                 stop_words=['<|im_end|>'],
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

    @classmethod
    def from_config(cls, **kwargs):
        """Build model from ModelConfig parameters."""
        # TODO system, eosys, user, eoh, assistant, eoa.
        mapping = {'meta_instruction': 'system'}
        new_kwargs = {
            mapping.get(key, key): value
            for key, value in kwargs.items()
        }
        return cls(**new_kwargs)

    def decorate_prompt(self, prompt, sequence_start=True):
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.im_start}system\n{self.system}{self.im_end}' \
                   f'\n{self.im_start}user\n{prompt}{self.im_end}' \
                   f'\n{self.im_start}assistant\n'

        return f'\n{self.im_start}user\n{prompt}{self.im_end}' \
               f'\n{self.im_start}assistant\n'

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
        ret = f'{self.im_start}system\n{system}{self.im_end}'
        for user, assistant in zip(users, assistants):
            if assistant:
                ret += f'\n{self.im_start}user\n{user}{self.im_end}' \
                       f'\n{self.im_start}assistant\n{assistant}'
            else:
                ret += f'\n{self.im_start}user\n{user}{self.im_end}' \
                       f'\n{self.im_start}assistant\n'
        return ret


@MODELS.register_module(name='codellama')
class CodeLlama(Llama2):

    def __init__(self,
                 system='',
                 session_len=4096,
                 suffix_first=False,
                 stop_words=None,
                 **kwargs):
        super().__init__(**kwargs)
        caps = ['completion', 'infilling', 'chat', 'python']
        assert self.capability in caps, \
            f'{self.capability} is not supported. ' \
            f'The supported capabilities are: {caps}'
        self.default_sys_prompt = system
        self.session_len = session_len
        self.suffix_first = suffix_first
        self.stop_words = stop_words

        # The following sampling parameters refers to https://github.com/facebookresearch/codellama # noqa: E501
        if self.capability == 'completion' or self.capability == 'python':
            self.top_p = kwargs.get('top_p', 0.9)
            self.temperature = kwargs.get('temperature', 0.2)
        if self.capability == 'chat':
            self.top_p = kwargs.get('top_p', 0.95)
            self.temperature = kwargs.get('temperature', 0.2)
        elif self.capability == 'infilling':
            self.top_p = kwargs.get('top_p', 0.9)
            self.temperature = kwargs.get('temperature', 0.0)
            if self.stop_words is None:
                self.stop_words = ['<EOT>']

    def decorate_prompt(self, prompt, sequence_start=True):
        if self.capability == 'infilling':
            return self._infill_prompt(prompt)
        elif self.capability == 'chat':
            return self._get_prompt(prompt, sequence_start)
        else:  # python speicalist
            return prompt

    def _infill_prompt(self, prompt):
        prefix, suffix = prompt.split('<FILL>')
        if self.suffix_first:
            # format as "<PRE> <SUF>{suf} <MID> {pre}"
            prompt = f'<PRE> <SUF>{suffix} <MID> {prefix}'
        else:
            # format as "<PRE> {pre} <SUF>{suf} <MID>"
            prompt = f'<PRE> {prefix} <SUF>{suffix} <MID>'
        return prompt

    def _get_prompt(self, prompt, sequence_start):
        prompt = prompt.strip()
        if sequence_start:
            return f'{self.b_inst} ' \
                   f'{self.b_sys}{self.default_sys_prompt}{self.e_sys}' \
                   f'{prompt} {self.e_inst}'

        return f'{self.b_inst} {prompt} {self.e_inst}'

    def messages2prompt(self, messages, sequence_start=True):
        assert self.capability == 'chat', \
            f'codellama message2prompt only supports chat mode ' \
            f'but got {self.cap} mode'
        return super().messages2prompt(messages, sequence_start)


@MODELS.register_module(name='falcon')
class Falcon(BaseModel):

    def __init__(self):
        super().__init__()

    def get_prompt(self, prompt, sequence_start=True):
        prompt = super().get_prompt(prompt, sequence_start)
        if len(prompt) == 0:
            return '<|endoftext|>'
        return prompt


@MODELS.register_module(name='chatglm2-6b')
class ChatGLM2(BaseModel):

    def __init__(self):
        super().__init__()
        self.count = 0

    def get_prompt(self, prompt, sequence_start=True):
        # need more check
        # https://github.com/THUDM/ChatGLM2-6B/issues/48
        # [64790, 64792] to be prepended
        self.count += 1
        return f'[Round {self.count}]\n\n问：{prompt}\n\n答：'


@MODELS.register_module(name=['solar', 'solar-70b'])
class SOLAR(BaseModel):
    """Chat template of SOLAR model.

    `https://huggingface.co/upstage/SOLAR-0-70b-16bit`
    """

    def __init__(self,
                 system='### System:\n',
                 eosys='\n\n',
                 user='### User:\n',
                 eoh='\n\n',
                 assistant='### Assistant:\n',
                 eoa='\n\n',
                 meta_instruction='',
                 session_len=2048,
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.eosys = eosys
        self.user = user
        self.eoh = eoh
        self.assistant = assistant
        self.eoa = eoa
        self.meta_instruction = meta_instruction
        self.session_len = session_len

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.system}{self.meta_instruction}{self.eosys}' \
                   f'{self.user}{prompt}{self.eoh}{self.assistant}'

        return f'{self.user}{prompt}{self.eoh}{self.assistant}'

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
        system = self.meta_instruction if not system else system
        ret = f'{self.system}{system}{self.eosys}'
        for i, (user, assistant) in enumerate(zip(users, assistants)):
            ret += f'{self.user}{user}{self.eoh}{self.assistant}'
            if assistant:
                ret += f'{assistant}{self.eoa}'
        return ret


@MODELS.register_module(name='ultracm')
@MODELS.register_module(name='ultralm')
class UltraChat(BaseModel):
    """Template of UltraCM and UltraLM models.

    `https://huggingface.co/openbmb/UltraCM-13b`
    `https://huggingface.co/openbmb/UltraLM-13b`
    """

    def __init__(
            self,
            meta_instruction="""User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>""",  # noqa: E501
            eos='</s>',
            user='User: ',
            assistant='Assistant: ',
            session_len=2048,
            **kwargs):
        super().__init__(**kwargs)
        self.meta_instruction = meta_instruction
        self.eos = eos
        self.session_len = session_len
        self.user = user
        self.assistant = assistant

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): the input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.meta_instruction}\n{self.user}{prompt}{self.eos}' \
                   f'\n{self.assistant}'

        return f'\n{self.user}{prompt}{self.eos}' \
               f'\n{self.assistant}'

    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template. Only evaluate the last instruction completion pair.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        system, users, assistants = self._translate_messages(messages)
        system = self.meta_instruction if not system else system
        ret = f'{system}'
        for user, assistant in zip(users, assistants):
            if assistant:
                ret += f'\n{self.user}{user}{self.eos}' \
                       f'\n{self.assistant}{assistant}{self.eos}'
            else:
                ret += f'\n{self.user}{user}{self.eos}' \
                       f'\n{self.assistant}'
        return ret


@MODELS.register_module(name=['yi', 'yi-chat', 'yi-200k', 'yi-34b'])
class Yi(BaseModel):
    """Chat template of Yi model."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction=None,
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>\n',
                 eosys='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 stop_words=['<|im_end|>', '<|endoftext|>'],
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.eosys = eosys
        self.assistant = assistant
        self.stop_words = stop_words

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            if self.meta_instruction is None:
                return f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
            return f'{self.system}{self.meta_instruction}{self.eosys}' \
                   f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
        else:
            return f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'

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
        eox_map = dict(user=self.eoh, assistant=self.eoa, system=self.eosys)
        ret = ''
        if self.meta_instruction:
            ret += f'{self.system}:{self.meta_instruction}{self.eosys}'

        for message in messages:
            role = message['role']
            content = message['content']
            ret += f'{eval(f"self.{role}")}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


def best_match_model(query: str, similarity_cutoff: float = 0.5):
    """Get the model that matches the query.

    Args:
        query (str): the input query. Could be a model path.
        similarity_cutoff (float): similarities below the limit are ignored.

    Return:
        List[str] | None: the possible model names or none.
    """
    model_names = list(MODELS.module_dict.keys())
    if query.endswith('/'):
        query = query[:-1]
    base_name = os.path.basename(query).lower()
    max_ratio, matched_name = float('-inf'), None
    for model_name in model_names:
        if model_name in base_name:
            ratio = fuzz.ratio(model_name.lower(), base_name)
            if ratio > max_ratio and model_name != 'base':  # skip base model
                max_ratio = ratio
                matched_name = model_name
    if matched_name:
        return matched_name

    # Using fuzzy matching
    matches = process.extract(base_name, model_names, scorer=fuzz.ratio)

    # Ignore matches with score below similarity_cutoff
    matches = [
        match for match, score in matches if score / 100 >= similarity_cutoff
    ]

    return matches[0] if matches else None
