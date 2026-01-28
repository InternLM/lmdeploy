# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import json
import uuid
from typing import List, Literal, Optional, Union

from mmengine import Registry

from lmdeploy.archs import get_model_arch
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
MODELS = Registry('model', locations=['lmdeploy.model'])


def random_uuid() -> str:
    """Return a random uuid."""
    return str(uuid.uuid4().hex)


def get_text(content: Union[str, List[dict]]):
    """Within the OpenAI API, the content field may be specified as either a
    string or a list of ChatCompletionContentPartTextParam (defined in openai).

    When a list is provided, lmdeploy selects the first element to incorporate into the chat template, as the manner in
    which OpenAI processes lists is not explicitly defined.
    """

    if isinstance(content, str):
        return content
    return content[0]['text']


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
        tool (str | None): begin of the tool prompt
        eotool (str | None): end of the tool prompt
        capability: ('completion' | 'infilling' | 'chat' | 'python') = None
    """  # noqa: E501

    model_name: str
    model_path: Optional[str] = None
    system: Optional[str] = None
    meta_instruction: Optional[str] = None
    eosys: Optional[str] = None
    user: Optional[str] = None
    eoh: Optional[str] = None
    assistant: Optional[str] = None
    eoa: Optional[str] = None
    tool: Optional[str] = None
    eotool: Optional[str] = None
    separator: Optional[str] = None
    capability: Optional[Literal['completion', 'infilling', 'chat', 'python']] = None
    stop_words: Optional[List[str]] = None

    @property
    def chat_template(self):
        attrs = {key: value for key, value in dataclasses.asdict(self).items() if value is not None}
        attrs.pop('model_name', None)
        if self.model_name in MODELS.module_dict.keys():
            model = MODELS.get(self.model_name)(**attrs)
        else:
            logger.warning(f'Could not find {self.model_name} in registered models. '
                           f'Register {self.model_name} using the BaseChatTemplate.')
            model = BaseChatTemplate(**attrs)
        return model

    def to_json(self, file_path=None):
        """Convert the dataclass instance to a JSON formatted string and
        optionally save to a file."""
        json_str = json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=4)
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
            raise ValueError('Invalid input. Must be a file path or a valid JSON string.')
        json_data = json.loads(json_data)
        if json_data.get('model_name', None) is None:
            json_data['model_name'] = random_uuid()
        if json_data['model_name'] not in MODELS.module_dict.keys():
            MODELS.register_module(json_data['model_name'], module=BaseChatTemplate)
        return cls(**json_data)


@MODELS.register_module(name='base')
class BaseChatTemplate:
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
                 tool='',
                 eotool='',
                 capability='chat',
                 stop_words=None,
                 **kwargs):
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.separator = separator
        self.eosys = eosys
        self.assistant = assistant
        self.tool = tool
        self.eotool = eotool
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

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(user=self.user, assistant=self.assistant, system=self.system, tool=self.tool)
        eox_map = dict(user=self.eoh, assistant=self.eoa + self.separator, system=self.eosys, tool=self.eotool)
        ret = ''
        if self.meta_instruction is not None and sequence_start:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role']
            content = get_text(message['content'])
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        if len(messages) and messages[-1]['role'] == 'assistant' and len(eox_map['assistant']) > 0:
            return ret[:-len(eox_map['assistant'])]  # prefix of response
        ret += f'{self.assistant}'
        return ret

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        return None


@MODELS.register_module(name='cogvlm')
class CogVLM(BaseChatTemplate):
    """Chat template of CogVLM model."""

    def __init__(self,
                 meta_instruction='',
                 eosys='',
                 user='Question: ',
                 separator='\n',
                 eoh=' ',
                 assistant='Answer:',
                 eoa='</s>',
                 stop_words=['</s>'],
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         separator=separator,
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
        if 'cogvlm' in path and 'cogvlm2' not in path:
            return 'cogvlm'


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

    def get_prompt(self, prompt, sequence_start=True):
        if self.capability == 'chat':
            return super().get_prompt(prompt, sequence_start)[:-1]
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'vicuna' in path and 'llava' not in path:
            return 'vicuna'
        if 'wizardlm' in path:
            return 'wizardlm'


@MODELS.register_module(name='llava-v1')
class Llavav1(Vicuna):
    """Chat template of llava-v1 model."""

    def __init__(
            self,
            meta_instruction="""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.""",  # noqa: E501
            **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'llava' in path and 'v1' in path and 'v1.6-34b' not in path \
                and 'mistral' not in path:
            return 'llava-v1'
        elif 'llava-1.5' in path:
            return 'llava-v1'


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
        if all([c not in path for c in ['internlm3', 'internlm2', '8k']]) and \
                all([c in path for c in ['internlm', 'chat']]):
            return 'internlm'


@MODELS.register_module(name='baichuan2')
class Baichuan2(BaseChatTemplate):
    """Chat template and generation parameters of Baichuan2-7B-Base and
    Baichuan2-7B-Chat models."""

    def __init__(self, user='<reserved_106>', assistant='<reserved_107>', **kwargs):
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


@MODELS.register_module(name='llama2')
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


@MODELS.register_module(name='codellama')
class CodeLlama(Llama2):

    def __init__(self, meta_instruction='', suffix_first=False, stop_words=None, **kwargs):
        super().__init__(meta_instruction=meta_instruction, stop_words=stop_words, **kwargs)
        caps = ['completion', 'infilling', 'chat', 'python']
        assert self.capability in caps, \
            f'{self.capability} is not supported. ' \
            f'The supported capabilities are: {caps}'
        self.meta_instruction = meta_instruction
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


@MODELS.register_module(name='chatglm')
class ChatGLM2(BaseChatTemplate):

    def __init__(self, user='问：', eoh='\n\n', assistant='答：', eoa='\n\n', **kwargs):
        super().__init__(**kwargs)
        self._user = user
        self._assistant = assistant
        self._eoh = eoh
        self._eoa = eoa
        self.count = 0

    def get_prompt(self, prompt, sequence_start=True):
        """Get prompt."""
        # need more check
        # https://github.com/THUDM/ChatGLM2-6B/issues/48
        # [64790, 64792] to be prepended
        self.count += 1
        ret = f'[Round {self.count}]\n\n'
        ret += f'{self._user}{prompt}{self._eoh}'
        ret += f'{self._assistant}'
        return ret

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        """Message to prompt."""
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        ret = ''
        count = 0
        for message in messages:
            role = message['role']
            content = get_text(message['content'])
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
        path = model_path.lower()
        if 'chatglm2' in path:
            return 'chatglm'


@MODELS.register_module(name=['mistral', 'mixtral'])
class MistralChat(BaseChatTemplate):
    """Template of Mistral and Mixtral Instruct models.

    `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1`
    `https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1`
    """

    def __init__(self, user='[INST] ', eoh=' [/INST]', eoa='</s>', **kwargs):
        super().__init__(user=user, eoh=eoh, eoa=eoa, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        model_path = model_path.lower()
        if 'instruct' in model_path or 'llava' in model_path:
            if 'mistral' in model_path:
                return 'mistral'
            if 'mixtral' in model_path:
                return 'mixtral'


@MODELS.register_module(name=['internvl-zh'])
class InternVLZH(BaseChatTemplate):

    def __init__(self, user='<human>: ', eoh=' ', assistant='<bot>: ', eoa='</s>', **kwargs):
        super().__init__(user=user, eoh=eoh, assistant=assistant, eoa=eoa, **kwargs)

    def get_prompt(self, prompt, sequence_start=True):
        if self.capability == 'chat':
            return super().get_prompt(prompt, sequence_start)[:-1]
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]

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

    def get_prompt(self, prompt, sequence_start=True):
        if self.capability == 'chat':
            return super().get_prompt(prompt, sequence_start)[:-1]
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'deepseek-vl' in path and 'chat' in path:
            return 'deepseek-vl'


@MODELS.register_module(name=['deepseek-vl2'])
class DeepseekVL2(BaseChatTemplate):

    def __init__(self,
                 meta_instruction='',
                 eosys='',
                 user='<|User|>: ',
                 eoh='\n\n',
                 assistant='<|Assistant|>: ',
                 eoa='<｜end▁of▁sentence｜>',
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         **kwargs)

    def get_prompt(self, prompt, sequence_start=True):
        return super().get_prompt(prompt, sequence_start)[:-1]

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'deepseek-vl2' in path:
            return 'deepseek-vl2'


@MODELS.register_module(name=['llava-chatml'])
class ChatmlDirect(BaseChatTemplate):

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='Answer the questions.',
                 eosys='<|im_end|>',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='',
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
        if 'llava' in path and 'v1.6-34b' in path:
            return 'llava-chatml'


@MODELS.register_module(name=['hf'])
class HFChatTemplate(BaseChatTemplate):
    """Chat template for HuggingFace models with `apply_chat_template` method.

    It MUST be at the end of @MODLES registry
    """

    def __init__(self, model_path: str = '', **kwargs):

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Verify if the model can perform apply_chat_template with user roles.
            self._role_instruction('user')
            self.stop_words = []
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.stop_words.append(self.tokenizer.eos_token)
            if hasattr(self.tokenizer, 'eot_token') and self.tokenizer.eot_token is not None:
                self.stop_words.append(self.tokenizer.eot_token)
            arch, _ = get_model_arch(model_path)
            self.is_gpt_oss = arch == 'GptOssForCausalLM'
            if self.is_gpt_oss:
                self.stop_words.append('<|call|>')
        except Exception as e:
            raise ValueError(f'Try apply_chat_template failed: {e}')
        try:
            _, _, self.sentinel_system_messages, self.sentinel_system_prompt = self._role_instruction('system')
            self.assistant_start, self.assistant_end, _, _ = self._role_instruction('assistant')
        except Exception:
            # Some models, such as google/gemma-2-2b-it, do not support a system role in the message structure.
            # They require conversation roles to strictly alternate between 'user' and 'assistant'
            # (e.g., user/assistant/user/assistant...). Consequently, we cannot directly obtain the special
            # tokens for the system and assistant roles.
            arch, _ = get_model_arch(model_path)
            if arch in ['Gemma2ForCausalLM', 'Gemma3ForConditionalGeneration']:
                self.sentinel_system_messages = []
                self.sentinel_system_prompt = '<bos>'
                self.assistant_start = '<start_of_turn>model\n'
                self.assistant_end = '<end_of_turn>\n'
            else:
                raise RuntimeError(f'apply chat template failed with model {arch} ')

    def get_prompt(self, prompt, sequence_start=True, **kwargs):
        messages = [{'role': 'user', 'content': prompt}]
        return self.messages2prompt(messages, sequence_start, **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        assert all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages), \
            'Each message should be a dict with "role" and "content" keys.'

        if 'enable_thinking' in kwargs and kwargs['enable_thinking'] is None:
            # Workaround for internlm/Intern-S1: when enable_thinking=None passed apply_chat_template,
            # the <think> tag is not generated.
            kwargs.pop('enable_thinking')
        if 'reasoning_effort' in kwargs and kwargs['reasoning_effort'] is None:
            kwargs.pop('reasoning_effort')
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        if sequence_start:
            prompt = self.tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=add_generation_prompt,
                                                        **kwargs)
        else:
            # Use a sentinel position to avoid the influence of default system role in the tokenizer's chat template
            # in interactive chat mode
            messages = self.sentinel_system_messages + messages if self.sentinel_system_messages else messages
            prompt = self.tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=add_generation_prompt,
                                                        **kwargs)
            # Remove the sentinel part.
            prompt = prompt[len(self.sentinel_system_prompt):] if len(self.sentinel_system_prompt) > 0 else prompt
        if messages[-1]['role'] == 'assistant' and len(self.assistant_end) > 0:
            prompt = prompt[:-len(self.assistant_end)]  # prefix of response to let the model complete the response
        if self.is_gpt_oss and not kwargs.get('tools'):
            # for gpt-oss model, remove this seems more conducive to instruction following.
            prompt = prompt.replace('commentary, ', '', 1)
        return prompt

    def _role_instruction(self, role):
        messages = [{'role': role, 'content': 'sentinel'}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        role_pos = prompt.find('sentinel')
        role_start = prompt[:role_pos]
        role_end = prompt[role_pos + len('sentinel'):]
        return role_start, role_end, messages, prompt

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        try:
            cls(model_path)
        except Exception:
            return False
        return True


def get_chat_template(model_path: str, config: Optional[ChatTemplateConfig] = None) -> BaseChatTemplate:
    """Get the chat template for the model.

    Args:
        model_path (str): the model path.
        config (Optional[ChatTemplateConfig]): the chat template config.
    Returns:
        BaseChatTemplate: the chat template.
    """
    if config is not None:
        return config.chat_template
    chat_template_name = 'base'
    for name, model in MODELS.module_dict.items():
        if model.match(model_path):
            chat_template_name = name
            break
    config = ChatTemplateConfig(chat_template_name, model_path=model_path)
    return config.chat_template
