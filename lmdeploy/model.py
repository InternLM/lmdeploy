# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import json
import uuid
from abc import abstractmethod
from typing import List, Literal, Optional

from mmengine import Registry

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
MODELS = Registry('model', locations=['lmdeploy.model'])


def random_uuid() -> str:
    """Return a random uuid."""
    return str(uuid.uuid4().hex)


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
    stop_words: Optional[List[str]] = None

    @property
    def chat_template(self):
        attrs = {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
        attrs.pop('model_name', None)
        if self.model_name in MODELS.module_dict.keys():
            model: BaseModel = MODELS.get(self.model_name)(**attrs)
        else:
            logger.warning(
                f'Could not find {self.model_name} in registered models. '
                f'Register {self.model_name} using the BaseChatTemplate.')
            model = BaseChatTemplate(**attrs)
        return model

    def to_json(self, file_path=None):
        """Convert the dataclass instance to a JSON formatted string and
        optionally save to a file."""
        json_str = json.dumps(dataclasses.asdict(self),
                              ensure_ascii=False,
                              indent=4)
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, file_or_string):
        """Construct a dataclass instance from a JSON file or JSON string."""
        try:
            # Try to open the input_data as a file path
            with open(file_or_string, 'r', encoding='utf-8') as file:
                json_data = file.read()
        except FileNotFoundError:
            # If it's not a file path, assume it's a JSON string
            json_data = file_or_string
        except IOError:
            # If it's not a file path and not a valid JSON string, raise error
            raise ValueError(
                'Invalid input. Must be a file path or a valid JSON string.')
        json_data = json.loads(json_data)
        if json_data.get('model_name', None) is None:
            json_data['model_name'] = random_uuid()
        if json_data['model_name'] not in MODELS.module_dict.keys():
            MODELS.register_module(json_data['model_name'],
                                   module=BaseChatTemplate)
        return cls(**json_data)


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
            # None is different from ''
            if self.meta_instruction is not None:
                return f'{self.system}{self.meta_instruction}{self.eosys}' \
                    f'{self.user}{prompt}{self.eoh}' \
                    f'{self.assistant}'
            else:
                return f'{self.user}{prompt}{self.eoh}' \
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
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
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
            meta_instruction="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",  # noqa: E501
            eosys=' ',
            user='USER: ',
            eoh=' ',
            assistant='ASSISTANT: ',
            eoa='</s>',
            stop_words=['</s>'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
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
        path = model_path.lower()
        if 'llava' in path and 'v1.5' in path:
            return 'vicuna'
        if 'vicuna' in path:
            return 'vicuna'
        if 'wizardlm' in path:
            return 'wizardlm'


@MODELS.register_module(name='mini-gemini-vicuna')
class MiniGemini(Vicuna):
    """Chat template of vicuna model."""

    def __init__(self, session_len=4096, **kwargs):
        super().__init__(session_len=session_len, **kwargs)

    def get_prompt(self, prompt, sequence_start=True):
        return super().get_prompt(prompt, sequence_start)[:-1]

    def messages2prompt(self, messages, sequence_start=True):
        return super().messages2prompt(messages, sequence_start)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'mgm-7b' in path or 'mgm-13b' in path or 'mgm-34b' in path:
            return 'mini-gemini-vicuna'
        if 'mini-gemini-7b' in path or 'mini-gemini-13b' in path:
            return 'mini-gemini-vicuna'


@MODELS.register_module(name='internlm-chat')
@MODELS.register_module(name='internlm-chat-7b')
@MODELS.register_module(name='internlm')
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
            return 'internlm'


@MODELS.register_module(name='internlm-chat-20b')
@MODELS.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):
    """Chat template and generation parameters of InternLM-Chat-7B-8K and
    InternLM-Chat-20B models."""

    def __init__(self, session_len=8192, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = session_len


@MODELS.register_module(name='internlm-20b')
class InternLMBaseModel20B(BaseChatTemplate):
    """Generation parameters of InternLM-20B-Base model."""

    def __init__(self, session_len=4096, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)


@MODELS.register_module(
    name=['internlm2-1_8b', 'internlm2-7b', 'internlm2-20b'])
class InternLM2BaseModel7B(BaseChatTemplate):
    """Generation parameters of InternLM2-7B-Base model."""

    def __init__(self, session_len=32768, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)


@MODELS.register_module(name=[
    'internlm2-chat', 'internlm2-chat-1_8b', 'internlm2-chat-7b',
    'internlm2-chat-20b'
])
@MODELS.register_module(name='internlm2')
class InternLM2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM2-Chat-7B."""

    def __init__(self,
                 session_len=32768,
                 system='<|im_start|>system\n',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 environment='<|im_start|>environment\n',
                 plugin='<|plugin|>',
                 interpreter='<|interpreter|>',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 eoenv='<|im_end|>\n',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
        self.plugin = plugin
        self.interpreter = interpreter
        self.environment = environment
        self.eoenv = eoenv
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
        if 'internlm2' in path and ('chat' in path or 'math' in path):
            return 'internlm2'

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
                       system=self.system,
                       environment=self.environment)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.separator,
                       system=self.eosys,
                       environment=self.eoenv)
        name_map = dict(plugin=self.plugin, interpreter=self.interpreter)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role']
            content = message['content']
            begin = box_map[role].strip(
            ) + f" name={name_map[message['name']]}\n" if 'name' in message else box_map[
                role]
            ret += f'{begin}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


@MODELS.register_module(name='internvl-internlm2')
class InternVLInternLM2Chat(InternLM2Chat7B):

    def __init__(
            self,
            meta_instruction='You are an AI assistant whose name is InternLM (书生·浦语).',
            **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'internvl' in path and 'v1-5' in path:
            return 'internvl-internlm2'


@MODELS.register_module(name='internlm-xcomposer2')
class InternLMXComposer2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM-XComposer2-7b."""

    def __init__(
            self,
            session_len=4096,
            system='[UNUSED_TOKEN_146]system\n',
            meta_instruction="""You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.""",
            user='[UNUSED_TOKEN_146]user\n',
            assistant='[UNUSED_TOKEN_146]assistant\n',
            eosys='[UNUSED_TOKEN_145]\n',
            eoh='[UNUSED_TOKEN_145]\n',
            eoa='[UNUSED_TOKEN_145]\n',
            separator='\n',
            stop_words=['[UNUSED_TOKEN_145]'],
            **kwargs):
        super().__init__(session_len=session_len,
                         system=system,
                         meta_instruction=meta_instruction,
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
        if 'internlm' in path and 'xcomposer2' in path and '4khd' not in path:
            return 'internlm-xcomposer2'


@MODELS.register_module(name='internlm-xcomposer2-4khd')
class InternLMXComposer24khdChat7B(InternLMXComposer2Chat7B):
    """Chat template and generation parameters of InternLM-
    XComposer2-4khd-7b."""

    def __init__(self, session_len=16384, **kwargs):
        super().__init__(session_len=session_len, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'internlm' in path and 'xcomposer2' in path and '4khd' in path:
            return 'internlm-xcomposer2-4khd'


@MODELS.register_module(name='baichuan-7b')
@MODELS.register_module(name='baichuan-base')
class Baichuan7B(BaseChatTemplate):
    """Generation parameters of Baichuan-7B base model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@MODELS.register_module(name='baichuan2-7b')
@MODELS.register_module(name='baichuan2')
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
            return 'baichuan2'


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
            return 'llama2'


@MODELS.register_module(name='llama3')
class Llama3(BaseChatTemplate):
    """Chat template of LLaMA3 model."""

    def __init__(self,
                 system='<|start_header_id|>system<|end_header_id|>\n\n',
                 meta_instruction=None,
                 eosys='<|eot_id|>',
                 assistant='<|start_header_id|>assistant<|end_header_id|>\n\n',
                 eoa='<|eot_id|>',
                 user='<|start_header_id|>user<|end_header_id|>\n\n',
                 eoh='<|eot_id|>',
                 stop_words=['<|eot_id|>', '<|end_of_text|>'],
                 session_len=8192,
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         assistant=assistant,
                         eoa=eoa,
                         user=user,
                         eoh=eoh,
                         stop_words=stop_words,
                         session_len=session_len,
                         **kwargs)

    def get_prompt(self, prompt, sequence_start=True):
        if sequence_start:
            return '<|begin_of_text|>' + super().get_prompt(
                prompt, sequence_start)
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True):
        if sequence_start and not isinstance(messages, str):
            return '<|begin_of_text|>' + super().messages2prompt(
                messages, sequence_start)[:-1]
        return super().messages2prompt(messages, sequence_start)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'llama-3-' in model_path.lower() or 'llama3-' in model_path.lower():
            return 'llama3'


@MODELS.register_module(name='qwen-14b')
@MODELS.register_module(name='qwen-7b')
@MODELS.register_module(name='qwen')
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
            return 'qwen'


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
@MODELS.register_module(name='chatglm')
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
        if 'chatglm' in model_path.lower():
            return 'chatglm'


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
                 meta_instruction='',
                 session_len=2048,
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.eosys = eosys
        self.user = user
        self.eoh = eoh
        self.assistant = assistant
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
                 meta_instruction=None,
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
        path = model_path.lower()
        if 'yi' in path and 'vl' not in path:
            return 'yi'


@MODELS.register_module(name=['mistral', 'mixtral'])
@MODELS.register_module(name=['Mistral-7B-Instruct', 'Mixtral-8x7B-Instruct'])
class MistralChat(BaseChatTemplate):
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
                return 'mistral'
            if 'mixtral' in model_path.lower():
                return 'mixtral'


@MODELS.register_module(name=['gemma'])
class Gemma(BaseChatTemplate):
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
@MODELS.register_module(name=['deepseek'])
class Deepseek(BaseChatTemplate):

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

    def get_prompt(self, prompt, sequence_start=True):
        return super().get_prompt(prompt, sequence_start)[:-1]

    def messages2prompt(self, messages, sequence_start=True):
        return super().messages2prompt(messages, sequence_start)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'deepseek' in path and 'chat' in path and 'vl' not in path:
            return 'deepseek'


@MODELS.register_module(name=['internvl-zh'])
class InternVLZH(BaseChatTemplate):

    def __init__(self,
                 user='<human>: ',
                 eoh=' ',
                 assistant='<bot>: ',
                 eoa='</s>',
                 session_len=4096,
                 **kwargs):
        super().__init__(user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         session_len=session_len,
                         **kwargs)

    def get_prompt(self, prompt, sequence_start=True):
        return super().get_prompt(prompt, sequence_start)[:-1]

    def messages2prompt(self, messages, sequence_start=True):
        return super().messages2prompt(messages, sequence_start)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'internvl-chat' in path and 'v1-1' in path:
            return 'internvl-zh'


@MODELS.register_module(name=['deepseek-vl'])
class DeepseekVL(BaseChatTemplate):

    def __init__(
            self,
            meta_instruction="""You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.""",  # noqa: E501
            eosys='\n\n',
            user='User: ',
            eoh='\n\n',
            assistant='Assistant: ',
            eoa='<｜end▁of▁sentence｜>',
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
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
        path = model_path.lower()
        if 'deepseek-vl' in path and 'chat' in path:
            return 'deepseek-vl'


@MODELS.register_module(name='deepseek-coder')
class DeepSeek(BaseChatTemplate):
    """Chat template of deepseek model."""

    def __init__(
            self,
            session_len=4096,
            system='',
            meta_instruction="""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n""",  # noqa: E501
            eosys='',
            user='### Instruction:\n',
            eoh='\n',
            assistant='### Response:\n',
            eoa='\n<|EOT|>',
            separator='\n',
            stop_words=['<|EOT|>'],
            **kwargs):
        super().__init__(session_len=session_len,
                         system=system,
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
        if 'deepseek-coder' in path:
            return 'deepseek-coder'


@MODELS.register_module(name=['yi-vl'])
class YiVL(BaseChatTemplate):

    def __init__(
            self,
            meta_instruction="""This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n""",  # noqa: E501
            user='### Human: ',
            eoh='\n',
            assistant='### Assistant:',
            eoa='\n',
            stop_words=['###'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
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
        path = model_path.lower()
        if 'yi-vl' in path:
            return 'yi-vl'


# flake8: noqa: E501
def dbrx_system_prompt():
    # This is inspired by the Claude3 prompt.
    # source: https://twitter.com/AmandaAskell/status/1765207842993434880
    # Identity and knowledge
    prompt = 'You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\n'
    prompt += 'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\n'
    # Capabilities (and reminder to use ``` for JSON blocks and tables, which it can forget). Also a reminder that it can't browse the internet or run code.
    prompt += 'You assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n'
    prompt += '(You do not have real-time data access or code execution capabilities. '
    # Ethical guidelines
    prompt += 'You avoid stereotyping and provide balanced perspectives on controversial topics. '
    # Data: the model doesn't know what it was trained on; it thinks that everything that it is aware of was in its training data. This is a reminder that it wasn't.
    # We also encourage it not to try to generate lyrics or poems
    prompt += 'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\n'
    # The model really wants to talk about its system prompt, to the point where it is annoying, so encourage it not to
    prompt += 'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\n'
    prompt += 'You do not mention any of this information about yourself unless the information is directly pertinent to the user\\\'s query.'.upper(
    )
    return prompt


@MODELS.register_module(name=['dbrx'])
class DbrxInstruct(BaseChatTemplate):

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction=dbrx_system_prompt(),
                 eosys='<|im_end|>\n',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 **kwargs):
        super().__init__(system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'dbrx' in path:
            return 'dbrx'


@MODELS.register_module(name=['internvl-zh-hermes2'])
@MODELS.register_module(name=['llava-chatml'])
class ChatmlDirect(BaseChatTemplate):

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='Answer the questions.',
                 eosys='<|im_end|>\n',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 session_len=4096,
                 **kwargs):
        super().__init__(system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
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
        path = model_path.lower()
        if 'llava' in path and 'v1.6-34b' in path:
            return 'llava-chatml'
        if 'internvl-chat' in path and 'v1-2' in path:
            return 'internvl-zh-hermes2'


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
    try:
        from transformers import AutoTokenizer
        tokenizer_config = AutoTokenizer.from_pretrained(
            query, trust_remote_code=True)
        if tokenizer_config.chat_template is None:
            return 'base'
    except Exception as e:
        assert type(e) == OSError
