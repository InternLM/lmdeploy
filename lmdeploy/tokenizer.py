# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from typing import Optional, Sequence, Union

import torch


class SentencePieceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)
        self._prefix_space_tokens = None

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
        model_file = osp.join(model_dir, 'tokenizer.model')
        backend_tokenizer_file = osp.join(model_dir, 'tokenizer.json')
        model_file_exists = osp.exists(model_file)
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            print('WARNING: Can not find tokenizer.json. '
                  'It may take long time to initialize the tokenizer.')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)
        self._prefix_space_tokens = None
        # save tokenizer.json to reuse
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            if hasattr(self.model, 'backend_tokenizer'):
                if os.access(model_dir, os.W_OK):
                    self.model.backend_tokenizer.save(backend_tokenizer_file)

        if self.model.eos_token_id is None:
            generation_config_file = osp.join(model_dir,
                                              'generation_config.json')
            if osp.exists(generation_config_file):
                with open(generation_config_file, 'r') as f:
                    cfg = json.load(f)
                    self.model.eos_token_id = cfg['eos_token_id']
            elif hasattr(self.model, 'eod_id'):  # Qwen remote
                self.model.eos_token_id = self.model.eod_id

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

    def _maybe_add_prefix_space(self, tokens, decoded):
        """maybe add prefix space for incremental decoding."""
        if len(tokens) and not decoded.startswith(' ') and\
                tokens[0] in self.prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

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
