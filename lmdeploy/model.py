# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from abc import abstractmethod
from typing import Literal, Optional

from mmengine import Registry

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
MODELS = Registry('model', locations=['lmdeploy.model'])


@dataclasses.dataclass
class ChatTemplateConfig:
    """Parameters for chat template.

    Args:
        model_name (str): the name of the deployed model. Determine which chat template will be applied.
            All the chat template names: `lmdeploy list`
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
    separator: Optional[str] = None
    capability: Optional[Literal['completion', 'infilling', 'chat',
                                 'python']] = None

    @property
    def chat_template(self):
        attrs = {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
        if self.model_name in MODELS.module_dict.keys():
            model: BaseModel = MODELS.get(self.model_name)(**attrs)
        else:
            logger.warning(
                f'Could not find {self.model_name} in registered models. '
                f'Register {self.model_name} using the BaseChatTemplate.')
            model = BaseChatTemplate(**attrs)
        return model


@MODELS.register_module(name='internlm')
@MODELS.register_module(name='llama')
@MODELS.register_module(name='base')
class BaseModel:
    """Base model."""

    def __init__(self,
                 session_len=2048,
                 capability='chat',
                 stop_words=None,
                 **kwargs):
        self.session_len = session_len
        self.stop_words = stop_words
        self.capability = capability

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
        return prompt

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

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        return None


class BaseChatTemplate(BaseModel):
    """Base Chat template."""

    def __init__(self,
                 system='',
                 meta_instruction='',
                 eosys='',
                 user='',
                 eoh='',
                 assistant='',
                 eoa='',
                 separator='',
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.separator = separator
        self.eosys = eosys
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
        if self.capability == 'completion':
            return prompt
        if sequence_start:
            return f'{self.system}{self.meta_instruction}{self.eosys}' \
                   f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
        else:
            return f'{self.separator}{self.user}{prompt}{self.eoh}' \
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
        box_map = dict(user=self.user,
                       assistant=self.assistant,
                       system=self.system)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.separator,
                       system=self.eosys)
        ret = ''
        for message in messages:
            role = message['role']
            content = message['content']
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


@MODELS.register_module(name='wizardlm')
@MODELS.register_module(name='vicuna')
class Vicuna(BaseChatTemplate):
    """Chat template of vicuna model."""

    def __init__(
            self,
            meta_instruction="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n""",  # noqa: E501
            user='USER: ',
            eoh='\n',
            assistant='ASSISTANT: ',
            eoa='</s>',
            separator='\n',
            stop_words=['</s>'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'vicuna' in model_path.lower():
            return 'vicuna'
        if 'wizardlm' in model_path.lower():
            return 'wizardlm'


@MODELS.register_module(name='internlm-chat')
@MODELS.register_module(name='internlm-chat-7b')
class InternLMChat7B(BaseChatTemplate):
    """Chat template of InternLM model."""

    def __init__(
            self,
            system='<|System|>:',
            meta_instruction="""You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
""",  # noqa: E501
            eosys='\n',
            user='<|User|>:',
            eoh='\n',
            assistant='<|Bot|>:',
            eoa='<eoa>',
            separator='\n',
            stop_words=['<eoa>'],
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if all([c not in path for c in ['internlm2', '8k']]) and \
                all([c in path for c in ['internlm', 'chat']]):
            return 'internlm-chat'


@MODELS.register_module(name='internlm-chat-20b')
@MODELS.register_module(name='internlm-chat-7b-8k')
@MODELS.register_module(name='internlm-chat-8k')
class InternLMChat7B8K(InternLMChat7B):
    """Chat template and generation parameters of InternLM-Chat-7B-8K and
    InternLM-Chat-20B models."""

    def __init__(self, session_len=8192, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = session_len

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'intenlm' in path and 'internlm2' not in path and 'chat' in path:
            if '20b' in path or '8k' in path:
                return 'internlm-chat-8k'


@MODELS.register_module(name='internlm-20b')
class InternLMBaseModel20B(BaseChatTemplate):
    """Generation parameters of InternLM-20B-Base model."""

    def __init__(self, session_len=4096, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'intenlm' in path and 'internlm2' not in path:
            if '20b' in path and 'chat' not in path:
                return 'internlm-20b'


@MODELS.register_module(
    name=['internlm2', 'internlm2-1_8b', 'internlm2-7b', 'internlm2-20b'])
class InternLM2BaseModel7B(BaseChatTemplate):
    """Generation parameters of InternLM2-7B-Base model."""

    def __init__(self, session_len=32768, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'intenlm2' in path and 'chat' not in path:
            return 'internlm2'


@MODELS.register_module(name=[
    'internlm2-chat', 'internlm2-chat-1_8b', 'internlm2-chat-7b',
    'internlm2-chat-20b'
])
class InternLM2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM2-Chat-7B."""

    def __init__(self,
                 session_len=32768,
                 system='<|im_start|>system\n',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
        super(InternLM2Chat7B, self).__init__(session_len=session_len,
                                              system=system,
                                              user=user,
                                              assistant=assistant,
                                              eosys=eosys,
                                              eoh=eoh,
                                              eoa=eoa,
                                              separator=separator,
                                              stop_words=stop_words,
                                              **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'intenlm2' in path and 'chat' in path:
            return 'internlm2-chat'


@MODELS.register_module(name='baichuan-7b')
@MODELS.register_module(name='baichuan-base')
class Baichuan7B(BaseChatTemplate):
    """Generation parameters of Baichuan-7B base model."""

    def __init__(self, repetition_penalty=1.1, **kwargs):
        super().__init__(**kwargs)
        self.repetition_penalty = repetition_penalty

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'baichuan' in path and 'chat' not in path \
                and 'baichuan2' not in path:
            return 'baichuan-base'


@MODELS.register_module(name='baichuan2-7b')
@MODELS.register_module(name='baichuan2-chat')
class Baichuan2_7B(BaseChatTemplate):
    """Chat template and generation parameters of Baichuan2-7B-Base and
    Baichuan2-7B-Chat models."""

    def __init__(self,
                 user='<reserved_106>',
                 assistant='<reserved_107>',
                 **kwargs):
        super().__init__(user=user, assistant=assistant, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'baichuan2' in path and 'chat' in path:
            return 'baichuan2-chat'


@MODELS.register_module(name='puyu')
class Puyu(BaseChatTemplate):
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
        super().__init__(meta_instruction=meta_instruction,
                         system=system,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_words=stop_words,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'puyu' in model_path.lower():
            return 'puyu'


@MODELS.register_module(name=['llama2', 'llama-2', 'llama-2-chat'])
class Llama2(BaseChatTemplate):
    """Chat template of LLaMA2 model."""

    def __init__(
            self,
            system='[INST] <<SYS>>\n',
            meta_instruction="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",  # noqa: E501
            eosys='\n<</SYS>>\n\n',
            assistant=' [/INST] ',
            eoa='</s>',
            separator='<s>[INST] ',
            session_len=4096,
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         session_len=session_len,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'llama-2' in model_path.lower() or 'llama2' in model_path.lower():
            return 'llama-2'


@MODELS.register_module(name='qwen-14b')
@MODELS.register_module(name='qwen-7b')
@MODELS.register_module(name='qwen-chat')
class Qwen7BChat(BaseChatTemplate):
    """Chat template for Qwen-7B-Chat."""

    def __init__(self,
                 session_len=8192,
                 system='<|im_start|>system\n',
                 meta_instruction='You are a helpful assistant.',
                 eosys='<|im_end|>\n',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>'],
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         session_len=session_len,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'qwen' in model_path.lower():
            return 'qwen-chat'


@MODELS.register_module(name='codellama')
class CodeLlama(Llama2):

    def __init__(self,
                 meta_instruction='',
                 session_len=4096,
                 suffix_first=False,
                 stop_words=None,
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         session_len=session_len,
                         stop_words=stop_words,
                         **kwargs)
        caps = ['completion', 'infilling', 'chat', 'python']
        assert self.capability in caps, \
            f'{self.capability} is not supported. ' \
            f'The supported capabilities are: {caps}'
        self.meta_instruction = meta_instruction
        self.session_len = session_len
        self.suffix_first = suffix_first
        self.stop_words = stop_words
        if self.capability == 'infilling':
            if self.stop_words is None:
                self.stop_words = ['<EOT>']

    def get_prompt(self, prompt, sequence_start=True):
        if self.capability == 'infilling':
            return self._infill_prompt(prompt)
        elif self.capability == 'chat':
            return super().get_prompt(prompt, sequence_start)
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

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'codellama' in model_path.lower():
            return 'codellama'


@MODELS.register_module(name='falcon')
class Falcon(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'falcon' in model_path.lower():
            return 'falcon'


@MODELS.register_module(name='chatglm2-6b')
@MODELS.register_module(name='chatglm2')
class ChatGLM2(BaseModel):

    def __init__(self,
                 user='问：',
                 eoh='\n\n',
                 assistant='答：',
                 eoa='\n\n',
                 **kwargs):
        super().__init__(**kwargs)
        self._user = user
        self._assistant = assistant
        self._eoh = eoh
        self._eoa = eoa
        self.count = 0

    def get_prompt(self, prompt, sequence_start=True):
        """get prompt."""
        # need more check
        # https://github.com/THUDM/ChatGLM2-6B/issues/48
        # [64790, 64792] to be prepended
        self.count += 1
        ret = f'[Round {self.count}]\n\n'
        ret += f'{self._user}{prompt}{self._eoh}'
        ret += f'{self._assistant}'
        return ret

    def messages2prompt(self, messages, sequence_start=True):
        """message to prompt."""
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        ret = ''
        count = 0
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                count += 1
                ret += f'[Round {count}]\n\n'
                ret += f'{self._user}{content}{self._eoh}'
                ret += f'{self._assistant}'
            if role == 'assistant':
                ret += f'{content}'
        return ret

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'chatglm2' in model_path.lower():
            return 'chatglm2'


@MODELS.register_module(name=['solar', 'solar-70b'])
class SOLAR(BaseChatTemplate):
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

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'solar' in model_path.lower():
            return 'solar'


@MODELS.register_module(name='ultracm')
@MODELS.register_module(name='ultralm')
class UltraChat(BaseChatTemplate):
    """Template of UltraCM and UltraLM models.

    `https://huggingface.co/openbmb/UltraCM-13b`
    `https://huggingface.co/openbmb/UltraLM-13b`
    """

    def __init__(
            self,
            system='User: ',
            meta_instruction="""A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.""",  # noqa: E501
            eosys='</s>\n',
            user='User: ',
            eoh='</s>\n',
            assistant='Assistant: ',
            eoa='</s>',
            separator='\n',
            stop_words=['</s>'],
            session_len=2048,
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         session_len=session_len,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'ultracm' in model_path.lower():
            return 'ultracm'
        if 'ultralm' in model_path.lower():
            return 'ultralm'


@MODELS.register_module(name=['yi', 'yi-chat', 'yi-200k', 'yi-34b'])
class Yi(BaseChatTemplate):
    """Chat template of Yi model."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='',
                 eosys='<|im_end|>\n',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|endoftext|>'],
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'yi' in model_path.lower():
            return 'yi'


@MODELS.register_module(name=['mistral-instruct', 'mixtral-instruct'])
class MistralChat(BaseModel):
    """Template of Mistral and Mixtral Instruct models.

    `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1`
    `https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1`
    """

    def __init__(self,
                 user='[INST] ',
                 eoh=' [/INST]',
                 eoa='</s>',
                 session_len=2048,
                 **kwargs):
        super().__init__(user=user,
                         eoh=eoh,
                         eoa=eoa,
                         session_len=session_len,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'instruct' in model_path.lower():
            if 'mistral' in model_path.lower():
                return 'mistral-instruct'
            if 'mixtral' in model_path.lower():
                return 'mixtral-instruct'


@MODELS.register_module(name=['gemma'])
class Gemma(BaseModel):
    """Template of Gemma models.

    `https://huggingface.co/google/gemma-7b-it`
    """

    def __init__(self,
                 user='<start_of_turn>user\n',
                 eoh='<end_of_turn>\n',
                 assistant='<start_of_turn>model\n',
                 eoa='<end_of_turn>\n',
                 **kwargs):
        super().__init__(user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'gemma' in model_path.lower():
            return 'gemma'


@MODELS.register_module(name=['deepseek-chat'])
class Deepseek(BaseModel):

    def __init__(self,
                 user='User: ',
                 eoh='\n\n',
                 assistant='Assistant: ',
                 eoa='<｜end▁of▁sentence｜>',
                 **kwargs):
        super().__init__(user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'deepseek' in model_path.lower():
            return 'deepseek-chat'


def best_match_model(query: str) -> Optional[str]:
    """Get the model that matches the query.

    Args:
        query (str): the input query. Could be a model path.

    Return:
        str | None: the possible model name or none.
    """
    for name, model in MODELS.module_dict.items():
        if model.match(query):
            return model.match(query)
