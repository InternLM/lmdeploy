# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import os
from abc import abstractmethod
from typing import Literal, Optional

from fuzzywuzzy import fuzz, process
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.model'])


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
    stop_word_suffix: Optional[str] = None
    capability: Optional[Literal['completion', 'infilling', 'chat',
                                 'python']] = None

    @property
    def chat_template(self):
        attrs = {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
        model: BaseModel = MODELS.get(self.model_name)(**attrs)
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
                 stop_word_suffix='',
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.stop_word_suffix = stop_word_suffix
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
            return f'{self.stop_word_suffix}{self.user}{prompt}{self.eoh}' \
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
                       assistant=self.eoa + self.stop_word_suffix,
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
            stop_word_suffix='\n',
            stop_words=['</s>'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_word_suffix=stop_word_suffix,
                         stop_words=stop_words,
                         **kwargs)


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
            stop_word_suffix='\n',
            stop_words=['<eoa>'],
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_word_suffix=stop_word_suffix,
                         stop_words=stop_words,
                         **kwargs)


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
    name=['internlm2', 'internlm2-1_8b', 'internlm2-7b', 'internlm2-20b'])
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
                 stop_word_suffix='\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
        super(InternLM2Chat7B,
              self).__init__(session_len=session_len,
                             system=system,
                             user=user,
                             assistant=assistant,
                             eosys=eosys,
                             eoh=eoh,
                             eoa=eoa,
                             stop_word_suffix=stop_word_suffix,
                             stop_words=stop_words,
                             **kwargs)


@MODELS.register_module(name='baichuan-7b')
class Baichuan7B(BaseChatTemplate):
    """Generation parameters of Baichuan-7B base model."""

    def __init__(self, repetition_penalty=1.1, **kwargs):
        super().__init__(**kwargs)
        self.repetition_penalty = repetition_penalty


@MODELS.register_module(name='baichuan2-7b')
class Baichuan2_7B(BaseChatTemplate):
    """Chat template and generation parameters of Baichuan2-7B-Base and
    Baichuan2-7B-Chat models."""

    def __init__(self,
                 user='<reserved_106>',
                 assistant='<reserved_107>',
                 **kwargs):
        super().__init__(user=user, assistant=assistant, **kwargs)


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
            stop_word_suffix='<s>[INST] ',
            session_len=4096,
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         assistant=assistant,
                         eoa=eoa,
                         stop_word_suffix=stop_word_suffix,
                         session_len=session_len,
                         **kwargs)


@MODELS.register_module(name='qwen-14b')
@MODELS.register_module(name='qwen-7b')
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
                 stop_word_suffix='\n',
                 stop_words=['<|im_end|>'],
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_word_suffix=stop_word_suffix,
                         stop_words=stop_words,
                         session_len=session_len,
                         **kwargs)


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
            stop_word_suffix='\n',
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
                         stop_word_suffix=stop_word_suffix,
                         stop_words=stop_words,
                         session_len=session_len,
                         **kwargs)


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
                 stop_word_suffix='\n',
                 stop_words=['<|im_end|>', '<|endoftext|>'],
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_word_suffix=stop_word_suffix,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name=['Mistral-7B-Instruct', 'Mixtral-8x7B-Instruct'])
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


def best_match_model(query: str, similarity_cutoff: float = 0.5):
    """Get the model that matches the query.

    Args:
        query (str): the input query. Could be a model path.
        similarity_cutoff (float): similarities below the limit are ignored.

    Return:
        List[str] | None: the possible model names or none.
    """
    model_names = list(MODELS.module_dict.keys())
    if ('models--' in query) and ('snapshots' in query):
        paths = query.split(os.sep)
        paths = [x for x in paths if 'models--' in x]
        query = paths[0].split('--')[-1]
    elif query.endswith('/'):
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
