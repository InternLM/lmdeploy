# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from collections import deque
from typing import List, Optional, Sequence, Union

import torch

from lmdeploy.utils import get_logger

# this file will be copied to triton server, make sure all
# importing are starting from the package root lmdeploy


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

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
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

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        skip_special_tokens = True
        t = t[offset:]
        out_string = self.model.decode(t,
                                       skip_special_tokens=skip_special_tokens)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)


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

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t, offset)

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
