# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

from lmdeploy.utils import get_logger

# this file will be copied to triton server, make sure all
# importing are starting from the package root lmdeploy


@dataclass
class DetokenizeState:
    """A state collection of incrementally detekenization.

    Args:
        ids_offset (int): offset to all input ids. In LMDeploy, the output
            ids length is not one by one. It could be random by random.
        prev_tokens (List[str] | None): for incrementally decoding.
            Default to None, which means the first round.
        prefix_offset (int): the start index of tokens to be converted to
            string (prev + new tokens). Default to 0 for the first round.
        read_offset (int): the end index of tokens to be converted to
            string (prev token). Default to 0 for the first round.
    """
    ids_offset: int = 0
    prev_tokens: Optional[List[str]] = None
    prefix_offset: int = 0
    read_offset: int = 0

    def as_tuple(self) -> Tuple:
        """Return a tuple of states."""
        return (self.ids_offset, self.prev_tokens, self.prefix_offset, self.read_offset)


class HuggingFaceTokenizer:
    """A wrapper of transformers' AutoTokenizer.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        self._check_transformers_version(model_dir)
        from transformers import AutoTokenizer
        self.logger = get_logger('lmdeploy')
        self.model = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self._prefix_space_tokens = None

        if self.model.eos_token_id is None:
            generation_config_file = osp.join(model_dir, 'generation_config.json')
            if osp.exists(generation_config_file):
                with open(generation_config_file, 'r') as f:
                    cfg = json.load(f)
                    self.model.eos_token_id = cfg['eos_token_id']
            elif hasattr(self.model, 'eod_id'):  # Qwen remote
                self.model.eos_token_id = self.model.eod_id

        # for stop words
        self._vocab_size_with_added: int = None
        self._maybe_decode_bytes: bool = None
        # TODO maybe lack a constant.py
        self._indexes_tokens_deque = deque(maxlen=10)
        self.max_indexes_num = 5
        self.token2id = {}

    def _check_transformers_version(self, model_dir: str):
        import transformers
        from packaging import version

        from lmdeploy.archs import get_model_arch

        logger = get_logger('lmdeploy')

        current_transformers_version = version.parse(transformers.__version__)
        cfg = get_model_arch(model_dir)[1]
        cfg_ver = getattr(cfg, 'transformers_version', None)
        if cfg_ver is None:
            llm_config = getattr(cfg, 'llm_config', None)
            if llm_config:
                cfg_ver = getattr(llm_config, 'transformers_version', None)
        if cfg_ver is None:
            return
        required_transformers_version = version.parse(cfg_ver)
        if current_transformers_version < required_transformers_version:
            logger.warning(
                f'The current version of `transformers` is transformers=={current_transformers_version}, '  # noqa: E501
                f'which is lower than the required version transformers=={required_transformers_version}. '  # noqa: E501
                'Please upgrade to the required version.')

    def get_vocab(self):
        """Get vocab."""
        return self.model.get_vocab()

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.model.vocab_size

    @property
    def vocab_size_with_added(self):
        """Vocabulary size with added vocab."""
        if self._vocab_size_with_added is not None:
            return self._vocab_size_with_added
        self._vocab_size_with_added = len(self.model.get_vocab())
        return self._vocab_size_with_added

    @property
    def bos_token_id(self):
        """Begin of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """End of the sentence token id."""
        return self.model.eos_token_id

    @property
    def prefix_space_tokens(self):
        """Tokens without prefix space."""
        if self._prefix_space_tokens is None:
            vocab = self.model.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if tok.startswith('▁' if isinstance(tok, str) else b' ')
            }
        return self._prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens: List[int], decoded: str):
        """Maybe add prefix space for incremental decoding."""
        if len(tokens) and not decoded.startswith(' ') and\
                tokens[0] in self.prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    @property
    def maybe_decode_bytes(self):
        """Check if self.model.convert_ids_to_tokens return not a str value."""
        if self._maybe_decode_bytes is None:
            self._maybe_decode_bytes = False
            vocab = self.model.convert_ids_to_tokens(list(range(self.vocab_size)))
            for tok in vocab:
                if not isinstance(tok, str):
                    self._maybe_decode_bytes = True
                    break
        return self._maybe_decode_bytes

    def indexes_containing_token(self, token: str):
        """Return all the possible indexes, whose decoding output may contain
        the input token."""
        # traversing vocab is time consuming, can not be accelerated with
        # multi threads (computation) or multi process (can't pickle tokenizer)
        # so, we maintain latest 10 stop words and return directly if matched
        for _token, _indexes in self._indexes_tokens_deque:
            if token == _token:
                return _indexes

        if self.token2id == {}:
            # decode is slower than convert_ids_to_tokens
            if self.maybe_decode_bytes:
                for i in range(self.vocab_size):
                    try:
                        self.token2id[self.model.decode(i)] = i
                    except:  # noqa: E722
                        # some tokens just can't be decoded by `decode`
                        pass
            else:
                self.token2id = {self.model.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        if token == ' ':  # ' ' is special
            token = '▁'
        indexes = [i for _token, i in self.token2id.items() if token in _token]
        if len(indexes) > self.max_indexes_num:
            # multiple id decode to same token
            indexes = [i for i in indexes if self.decode([i]) == token]
            indexes = indexes[:self.max_indexes_num]
            self.logger.warning(f'There are too many(>{self.max_indexes_num}) possible '
                                f'indexes may decoding {token}, we will use {indexes} only')
        # there might be token id that exceeds self.vocab_size
        if len(indexes) == 0:
            indexes = self.encode(token, False)
            if len(indexes) != 1:
                self.logger.warning(f'The token {token}, its length of indexes {indexes} is '
                                    'not 1. Currently, it can not be used as stop words')
                indexes = []
        self._indexes_tokens_deque.append((token, indexes))
        return indexes

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
            add_bos (bool): Whether to add `bos` token id when encoding
                the prompt
            add_special_tokens (bool): Whether or not to add special tokens
                when encoding the prompt
        Returns:
            list[int]: token ids
        """
        encoded = self.model.encode(s, add_special_tokens=add_special_tokens, **kwargs)
        if not add_bos:
            # in the middle of a session
            if len(encoded) and encoded[0] == self.bos_token_id:
                encoded = encoded[1:]
        return encoded

    def decode(self, t: Sequence[int], offset: Optional[int] = None, skip_special_tokens: bool = True):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
            skip_special_tokens (bool): Whether or not to remove special
                tokens in the decoding.
        Returns:
            str: text of decoding tokens
        """
        t = t[offset:]
        out_string = self.model.decode(t, skip_special_tokens=skip_special_tokens)
        if offset:
            logger = get_logger('lmdeploy')
            logger.warning('For incrementally detokenization, please try '
                           'detokenize_incrementally function instead.')
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    @staticmethod
    def _convert_tokens_to_string_with_added_encoders(
        tokenizer,
        output_tokens: List[str],
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ) -> str:
        if tokenizer.is_fast or not tokenizer.get_added_vocab():
            return tokenizer.convert_tokens_to_string(output_tokens)
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/v0.2.7/vllm/transformers_utils/tokenizer.py#L68-L99
        sub_texts = []
        current_sub_text = []
        all_special_tokens = set(tokenizer.all_special_tokens)
        for token in output_tokens:
            if skip_special_tokens and token in all_special_tokens:
                continue
            if token in tokenizer.get_added_vocab():
                if current_sub_text:
                    sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                    sub_texts.append(sub_text)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
            sub_texts.append(sub_text)
        if spaces_between_special_tokens:
            return ' '.join(sub_texts)
        else:
            return ''.join(sub_texts)

    # Based on
    # https://github.com/vllm-project/vllm/blob/v0.2.7/vllm/transformers_utils/tokenizer.py#L105-L165
    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input indexes.

        Args:
            all_input_ids (List[int]): a list of token ids. Expected to be
                different sections of a long sequence.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                between special tokens. Default to be True.
        Returns:
            str: decoding output string of the current round.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
        """
        tokenizer = self.model
        ids_offset, prev_tokens, prefix_offset, read_offset = state.as_tuple()
        # This is the first iteration for this sequence
        new_tokens = tokenizer.convert_ids_to_tokens(all_input_ids[ids_offset:],
                                                     skip_special_tokens=skip_special_tokens)
        # `convert_ids_to_tokens` returns None for out-of-range token_id
        new_tokens = new_tokens or []
        new_tokens = [x for x in new_tokens if x is not None] if None in new_tokens else new_tokens
        if prev_tokens is None:
            # Please notice that in VLLM, indexes are detokenized one by one
            # while in LMDeploy, every turn, the detokenized indexes length
            # can be different.
            prev_tokens = tokenizer.convert_ids_to_tokens(all_input_ids[:ids_offset],
                                                          skip_special_tokens=skip_special_tokens)
            # `convert_ids_to_tokens` returns None for out-of-range token_id
            prev_tokens = prev_tokens or []
            prev_tokens = [x for x in prev_tokens if x is not None] if None in prev_tokens else prev_tokens
            read_offset = len(prev_tokens)
            if skip_special_tokens and new_tokens and new_tokens[0] in tokenizer.all_special_ids:
                read_offset = read_offset + 1  # skip special token

        output_tokens = prev_tokens + new_tokens
        prev_tokens += new_tokens
        prefix_text = self._convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = self._convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        # update state and get final decoded output
        if len(new_text) > len(prefix_text) and not new_text.endswith('�'):
            # utf-8 char at the end means it's a potential unfinished byte
            # sequence from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            prefix_offset = read_offset
            read_offset = len(output_tokens)
            new_text = new_text[len(prefix_text):]
        else:
            new_text = ''

        return new_text, DetokenizeState(len(all_input_ids), prev_tokens, prefix_offset, read_offset)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)


class ChatGLM4Tokenizer(HuggingFaceTokenizer):
    """Tokenizer of GLM4."""

    def __init__(self, model_path):
        super(ChatGLM4Tokenizer, self).__init__(model_path)
        original_pad = self.model._pad

        def __pad(*args, **kwargs):
            if 'padding_side' in kwargs:
                kwargs.pop('padding_side')
            return original_pad(*args, **kwargs)

        # fix for transformers>4.45.0
        self.model._pad = __pad

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt."""
        # ChtGLM4Tokenizer hardcode `add_speical_tokens=False` when tokenizing
        # a prompt. Refer to https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py#L227 # noqa E501
        return super(ChatGLM4Tokenizer, self).encode(s, add_bos, add_special_tokens=False, **kwargs)


class ChatGLMTokenizer(HuggingFaceTokenizer):
    """Tokenizer of GLM2."""

    def __init__(self, model_path):
        super(ChatGLMTokenizer, self).__init__(model_path)
        original_pad = self.model._pad

        def __pad(*args, **kwargs):
            if 'padding_side' in kwargs:
                kwargs.pop('padding_side')
            return original_pad(*args, **kwargs)

        # fix for transformers>4.45.0
        self.model._pad = __pad


class GptOssTokenizer(HuggingFaceTokenizer):
    """Tokenizer of GPT-OSS."""

    def __init__(self, model_dir: str):
        super(GptOssTokenizer, self).__init__(model_dir)
        from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.role = Role.ASSISTANT
        self.parser = partial(StreamableParser, encoding, role=Role.ASSISTANT)

    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        if not hasattr(state, 'stream'):
            state.stream = self.parser()

        response = ''
        stream = state.stream
        for token_id in all_input_ids[state.ids_offset:]:
            stream.process(token_id)
            if stream.current_channel in ['final', 'analysis'] and stream.current_role == self.role:
                response += stream.last_content_delta or ''

        state.ids_offset = len(all_input_ids)
        return response, state


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_path (str): the path of the tokenizer model
    """

    def __init__(self, model_path: str):
        from transformers import AutoConfig, PretrainedConfig
        try:
            model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:  # noqa
            model_cfg = PretrainedConfig.from_pretrained(model_path, trust_remote_code=True)
        is_gpt_oss = getattr(model_cfg, 'model_type', '') == 'gpt_oss'
        from transformers.models.auto.tokenization_auto import get_tokenizer_config
        tokenizer_config = get_tokenizer_config(model_path, trust_remote_code=True)
        config_tokenizer_class = tokenizer_config.get('tokenizer_class')
        if config_tokenizer_class == 'ChatGLM4Tokenizer':
            self.model = ChatGLM4Tokenizer(model_path)
        elif config_tokenizer_class == 'ChatGLMTokenizer':
            self.model = ChatGLMTokenizer(model_path)
        elif is_gpt_oss:
            self.model = GptOssTokenizer(model_path)
        else:
            self.model = HuggingFaceTokenizer(model_path)
        self.logger = get_logger('lmdeploy')

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """Begin of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """End of the sentence token id."""
        return self.model.eos_token_id

    def get_vocab(self):
        """Get vocab."""
        return self.model.get_vocab()

    def encode(self, s: str, add_bos: bool = True, add_special_tokens: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
            add_bos (bool): Whether to add `bos` token id when encoding
                the prompt
            add_special_tokens (bool): Whether or not to add special tokens
                when encoding the prompt
        Returns:
            list[int]: token ids
        """
        encoded = self.model.encode(s, add_bos, add_special_tokens, **kwargs)
        if encoded[:2] == [self.bos_token_id] * 2:
            self.logger.warning(f'Detected duplicate bos token {self.bos_token_id} in prompt, '
                                'this will likely reduce response quality, one of them will be'
                                'removed')
            encoded = encoded[1:]
        return encoded

    def decode(
        self,
        t: Sequence[int],
        offset: Optional[int] = None,
        skip_special_tokens: bool = True,
    ):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
            skip_special_tokens (bool): Whether or not to remove special
                tokens in the decoding.
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t, offset, skip_special_tokens)

    def detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        """Incrementally detokenize the input indexes.

        Args:
            all_input_ids (List[int]): a list of token ids. Expected to be
                different sections of a long sequence.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                between special tokens. Default to be True.
        Returns:
            str: decoding output string of the current round.
            state (DetokenizeState): an instance of DetokenizeState. Consists
                of incrementally decoding states.
        """
        return self.model.detokenize_incrementally(all_input_ids,
                                                   state=state,
                                                   skip_special_tokens=skip_special_tokens,
                                                   spaces_between_special_tokens=spaces_between_special_tokens)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        return self.model(s)

    def indexes_containing_token(self, token):
        """Return all the possible indexes, whose decoding output may contain
        the input token."""
        encoded = self.encode(token, add_bos=False)
        if len(encoded) > 1:
            self.logger.warning(f'The token {token}, its length of indexes {encoded} is over '
                                'than 1. Currently, it can not be used as stop words')
            return []
        return self.model.indexes_containing_token(token)
