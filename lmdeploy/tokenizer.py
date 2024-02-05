# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from lmdeploy.model import MODELS, BaseModel, best_match_model
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
        return (self.ids_offset, self.prev_tokens, self.prefix_offset,
                self.read_offset)


class SentencePieceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)
        self._prefix_space_tokens = None
        # for stop words
        self._maybe_decode_bytes: bool = None
        # TODO maybe lack a constant.py
        self._indexes_tokens_deque = deque(maxlen=10)
        self.max_indexes_num = 5
        self.logger = get_logger('lmdeploy')

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size()

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_id()

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_id()

    @property
    def prefix_space_tokens(self):
        """tokens without prefix space."""
        if self._prefix_space_tokens is None:
            vocab = self.model.IdToPiece(list(range(self.vocab_size)))
            self._prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if tok.startswith('▁')
            }
        return self._prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens, decoded):
        """maybe add prefix space for incremental decoding."""
        if len(tokens) and not decoded.startswith(' ') and\
                tokens[0] in self.prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def indexes_containing_token(self, token: str):
        """Return all the possible indexes, whose decoding output may contain
        the input token."""
        # traversing vocab is time consuming, can not be accelerated with
        # multi threads (computation) or multi process (can't pickle tokenizer)
        # so, we maintain latest 10 stop words and return directly if matched
        for _token, _indexes in self._indexes_tokens_deque:
            if token == _token:
                return _indexes
        if token == ' ':  # ' ' is special
            token = '▁'
        vocab = self.model.IdToPiece(list(range(self.vocab_size)))
        indexes = [i for i, voc in enumerate(vocab) if token in voc]
        if len(indexes) > self.max_indexes_num:
            indexes = self.encode(token, add_bos=False)[-1:]
            self.logger.warning(
                f'There are too many(>{self.max_indexes_num}) possible '
                f'indexes may decoding {token}, we will use {indexes} only')
        self._indexes_tokens_deque.append((token, indexes))
        return indexes

    def encode(self, s: str, add_bos: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        return self.model.Encode(s, add_bos=add_bos, **kwargs)

    def decode(self,
               t: Sequence[int],
               offset: Optional[int] = None,
               skip_special_tokens: bool = True,
               **kwargs):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
            skip_special_tokens (boo): not used in SentencePieceTokenizer.
        Returns:
            str: text of decoding tokens
        """
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        t = t[offset:]
        out_string = self.model.Decode(t)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

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
        out_string = self.model.Decode(all_input_ids)
        if state.prev_tokens is not None:
            out_string = self._maybe_add_prefix_space(all_input_ids,
                                                      out_string)
        state.prev_tokens = []  # not None for the above condition
        return out_string, state

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        import addict
        add_bos = False
        add_eos = False

        input_ids = self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)
        return addict.Addict(input_ids=input_ids)


class HuggingFaceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer
        self.logger = get_logger('lmdeploy')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)

        # get chat template from hf
        self.chat_template = self.model.chat_template
        deduced_name = best_match_model(model_dir)
        self.lmdeploy_chat_template = None  # for interactive chat
        if deduced_name is not None:
            # will apply if hf chat template is None
            self.lmdeploy_chat_template = MODELS.get(deduced_name)()
            self.chat_template = self.lmdeploy_chat_template

        self._prefix_space_tokens = None

        if self.model.eos_token_id is None:
            generation_config_file = osp.join(model_dir,
                                              'generation_config.json')
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

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def vocab_size_with_added(self):
        """vocabulary size with added vocab."""
        if self._vocab_size_with_added is not None:
            return self._vocab_size_with_added
        self._vocab_size_with_added = len(self.model.get_vocab())
        return self._vocab_size_with_added

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    @property
    def prefix_space_tokens(self):
        """tokens without prefix space."""
        if self._prefix_space_tokens is None:
            vocab = self.model.convert_ids_to_tokens(
                list(range(self.vocab_size)))
            self._prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab)
                if tok.startswith('▁' if isinstance(tok, str) else b' ')
            }
        return self._prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens: List[int], decoded: str):
        """maybe add prefix space for incremental decoding."""
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
            vocab = self.model.convert_ids_to_tokens(
                list(range(self.vocab_size)))
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
                self.token2id = {
                    self.model.decode(i): i
                    for i in range(self.vocab_size)
                }
            else:
                self.token2id = {
                    self.model.convert_ids_to_tokens(i): i
                    for i in range(self.vocab_size)
                }
        if token == ' ':  # ' ' is special
            token = '▁'
        indexes = [i for _token, i in self.token2id.items() if token in _token]
        if len(indexes) > self.max_indexes_num:
            indexes = self.encode(token, add_bos=False)[-1:]
            self.logger.warning(
                f'There are too many(>{self.max_indexes_num}) possible '
                f'indexes may decoding {token}, we will use {indexes} only')
        # there might be token id that exceeds self.vocab_size
        if len(indexes) == 0:
            indexes = self.encode(token, False)
            if len(indexes) != 1:
                self.logger.warning(
                    f'The token {token}, its length of indexes {indexes} is '
                    'not 1. Currently, it can not be used as stop words')
                indexes = []
        self._indexes_tokens_deque.append((token, indexes))
        return indexes

    def encode(self, s: str, add_bos: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        encoded = self.model.encode(s, **kwargs)
        if not add_bos:
            # in the middle of a session
            if len(encoded) and encoded[0] == self.bos_token_id:
                encoded = encoded[1:]
        return encoded

    def decode(self,
               t: Sequence[int],
               offset: Optional[int] = None,
               skip_special_tokens: bool = True):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        t = t[offset:]
        out_string = self.model.decode(t,
                                       skip_special_tokens=skip_special_tokens)
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
                    sub_text = tokenizer.convert_tokens_to_string(
                        current_sub_text)
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
        new_tokens = tokenizer.convert_ids_to_tokens(
            all_input_ids[ids_offset:],
            skip_special_tokens=skip_special_tokens)
        if prev_tokens is None:
            # Please notice that in VLLM, indexes are detokenized one by one
            # while in LMDeploy, every turn, the detokenized indexes length
            # can be different.
            if skip_special_tokens and new_tokens and new_tokens[
                    0] in tokenizer.all_special_ids:
                read_offset = 1  # skip special token
            output_tokens = new_tokens
            prev_tokens = new_tokens
        else:
            # Put new_token_id in a list so skip_special_tokens is respected
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

        return new_text, DetokenizeState(len(all_input_ids), prev_tokens,
                                         prefix_offset, read_offset)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[Union[str, BaseModel]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **tokenizer_kwargs,
    ) -> Union[str, List[int]]:
        """This is a function compatible with huggingface
        AutoTokenizer.apply_chat_template.

        Args:
            conversation (List | str): type string for interactive chat.
                List refers to OpenAI format messages.
        """
        if chat_template is None:
            chat_template = self.chat_template
        if hasattr(chat_template, 'messages2prompt'):
            prompt = chat_template.messages2prompt(conversation)
            if tokenize:
                return self.encode(prompt)
            else:
                return prompt
        elif isinstance(chat_template, str) or chat_template is None:
            # apply hf chat template
            return self.model.apply_chat_template(
                conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **tokenizer_kwargs)
        else:
            raise TypeError(f'Unsupported chat_template type: {chat_template}'
                            f' for {conversation}')


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            model_folder = osp.split(model_file)[0]
        else:
            model_folder = model_file
            model_file = osp.join(model_folder, 'tokenizer.model')
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')

        model_file_exists = osp.exists(model_file)
        config_exists = osp.exists(tokenizer_config_file)
        use_hf_model = config_exists or not model_file_exists
        self.logger = get_logger('lmdeploy')
        if not use_hf_model:
            self.model = SentencePieceTokenizer(model_file)
        else:
            self.model = HuggingFaceTokenizer(model_folder)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str, add_bos: bool = True, **kwargs):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        return self.model.encode(s, add_bos, **kwargs)

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
        return self.model.detokenize_incrementally(
            all_input_ids,
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
            self.logger.warning(
                f'The token {token}, its length of indexes {encoded} is over '
                'than 1. Currently, it can not be used as stop words')
            return []
        return self.model.indexes_containing_token(token)

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], str],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **tokenizer_kwargs,
    ) -> Union[str, List[int]]:
        """This is a function compatible with huggingface
        AutoTokenizer.apply_chat_template."""
        return self.model.apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **tokenizer_kwargs)
