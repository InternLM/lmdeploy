# Copyright (c) OpenMMLab. All rights reserved.
# Borrowed from xgrammar's TokenizerInfo
"""This module provides the tokenizer info class to handle the tokenizer
information."""

import json
import logging
from enum import Enum
from typing import List, Optional, Union

import _xgrammar as _xgr  # noqa: E402

try:
    import sentencepiece
except ImportError:
    sentencepiece = None
try:
    import tiktoken
except ImportError:
    tiktoken = None

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class VocabType(Enum):
    """The type of the vocabulary.

    Used in TokenizerInfo. XGrammar supports three types of
    vocabularies: RAW, BYTE_FALLBACK, BYTE_LEVEL.
    """

    RAW = 0
    """The vocabulary is in the raw format.

    The tokens in the vocabulary are kept in their original form without any processing. This kind of tokenizer includes
    the tiktoken tokenizer, e.g. microsoft/Phi-3-small-8k-instruct, Qwen/Qwen-7B-Chat, etc.
    """

    BYTE_FALLBACK = 1
    r"""The vocabulary used in the byte fallback BPE tokenizer.

    The tokens are encoded through the byte-fallback conversion. E.g. "\u001b" -> "<0x1B>", " apple" -> "▁apple". This
    kind of tokenizer includes meta-llama/Llama-2-7b-chat, microsoft/Phi-3.5-mini-instruct, etc.
    """

    BYTE_LEVEL = 2
    """The vocabulary used in the byte level BPE tokenizer.

    The tokens are encoded through the byte-to-unicode conversion, as in
    https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59

    This kind of tokenizer includes meta-llama/Meta-Llama-3-8B-Instruct,
    meta-llama/Meta-Llama-3.1-8B-Instruct, etc.
    """


class TokenizerInfo(_xgr.TokenizerInfo):
    """The tokenizer info contains the vocabulary, the type of the vocabulary,
    and necessary information for the grammar-guided generation.

    Note that although some tokenizers will encode the tokens in a special format, e.g. "<0x1B>" for "\u001b" in the
    ByteFallback tokenizer, and "Ġ" for " " in the Byte-Level BPE tokenizer, TokenizerInfo always decodes the vocabulary
    to the original format (e.g. "\u001b" and " ").

    Also note that some models (e.g. Phi-3 and Deepseek-V2) may pad the vocabulary to a multiple of 32. In this case,
    the model's vocab_size is larger than the tokenizer's vocabulary size. Please pass the model's vocab_size to the
    vocab_size parameter in the constructor, because this information is used to determine the size of the token mask.
    """

    def __init__(
        self,
        encoded_vocab: Union[List[bytes], List[str]],
        vocab_type: VocabType = VocabType.RAW,
        *,
        vocab_size: Optional[int] = None,
        stop_token_ids: Optional[Union[List[int], int]] = None,
        add_prefix_space: bool = False,
    ) -> None:
        """Construct the tokenizer info.

        Parameters
        ----------
        encoded_vocab : Union[List[bytes], List[str]]
            The encoded vocabulary of the tokenizer.

        vocab_type : VocabType, default: VocabType.RAW
            The type of the vocabulary. See also VocabType.

        vocab_size : Optional[int], default: None
            The size of the vocabulary. If not provided, the vocabulary size will be len(encoded_vocab).

        stop_token_ids : Optional[List[int]], default: None
            The stop token ids. If not provided, the stop token ids will be auto detected (but may not
            be correct).

        add_prefix_space : bool, default: False
            Whether the tokenizer will prepend a space before the text in the tokenization process.
        """
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        super().__init__(encoded_vocab, vocab_type.value, vocab_size, stop_token_ids, add_prefix_space)

    @staticmethod
    def _is_tiktoken_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
        if tiktoken is None:
            return False

        # helper to check if tokenizer is a tiktoken tokenizer
        has_tiktoken_encoding = hasattr(tokenizer, 'tokenizer') and isinstance(tokenizer.tokenizer, tiktoken.Encoding)

        filename_pattern = (hasattr(tokenizer, 'vocab_files_names') and 'vocab_file' in tokenizer.vocab_files_names
                            and 'tiktoken' in tokenizer.vocab_files_names['vocab_file'])

        return has_tiktoken_encoding or filename_pattern

    @staticmethod
    def _is_sentencepiece_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
        if sentencepiece is None:
            return False

        # helper to check if tokenizer is a sentence piece tokenizer
        has_sp_model_attr = hasattr(tokenizer, 'sp_model') and isinstance(tokenizer.sp_model,
                                                                          sentencepiece.SentencePieceProcessor)

        has_nested_sp_model_attr = (hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'sp_model')
                                    and isinstance(tokenizer.tokenizer.sp_model, sentencepiece.SentencePieceProcessor))

        return has_sp_model_attr or has_nested_sp_model_attr

    @staticmethod
    def from_huggingface(
        tokenizer: PreTrainedTokenizerBase,
        *,
        vocab_size: Optional[int] = None,
        stop_token_ids: Optional[Union[List[int], int]] = None,
    ) -> 'TokenizerInfo':
        """Construct the tokenizer info from the huggingface tokenizer. This
        constructor supports various tokenizer backends, including the
        huggingface fast tokenizer and tiktoken tokenizer. Necessary
        information is automatically detected from the tokenizer.

        The vocab_size parameter is introduced to handle the misalignment between the model's
        vocab_size and the tokenizer's vocabulary size. User should pass the model's vocab_size
        (could be defined in the model config) here. See docs of vocab_size for more details.

        The stop token ids is by default the eos_token_id of the tokenizer. If there are other
        stop tokens, you can specify them manually.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The huggingface tokenizer.

        vocab_size : Optional[int], default: None
            The vocabulary size **defined by the model** (**not the tokenizer**). This equals to the
            vocab dimension of the model's lm_head. This is the size of the token mask.

            It can be:

            1. the same as the tokenizer's vocabulary size. This is the most common case.
            2. larger than the tokenizer's vocabulary size. This happens when the model has padding
               to lm_head, possibly due to aligning lm_head to the power of 2.
               E.g. Phi-3 and Deepseek-V2.
            3. smaller than the tokenizer's vocabulary size. This happens when the tokenizer has
               some added tokens that will not supported by the model. E.g.
               Llama-3.2 Vision and Molmo-72B-0924 has padded `<|image|>` tokens, but they will not
               be considered in lm_head or generated by the model.

            model_vocab_size need to be provided for case 2 and 3. If not provided, it will be
            set to the tokenizer's vocabulary size.

        stop_token_ids : Optional[List[int]], default: None
            The stop token ids. If not provided, the eos_token_id of the tokenizer will be used.

        Returns
        -------
        tokenizer_info : TokenizerInfo
            The tokenizer info.
        """
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        if isinstance(stop_token_ids, list) and len(stop_token_ids) == 0:
            raise ValueError('stop_token_ids cannot be empty')

        try:
            vocab_dict = tokenizer.get_vocab()
        except AttributeError as e:
            msg = (f'Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer '
                   'should have a get_vocab method.')
            raise ValueError(msg) from e

        # Some tokenizer don't have token id 0 or 1 or 2. So the max_id could be larger than the
        # number of tokens.
        max_id = max(vocab_dict.values())
        tokenizer_vocab_size = max(len(vocab_dict), max_id + 1)

        vocab_size = vocab_size or tokenizer_vocab_size

        # maintain tokenizer's indexing
        encoded_vocab = [''] * vocab_size
        for token, idx in vocab_dict.items():
            if idx < vocab_size:
                encoded_vocab[idx] = token

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            # huggingface fast tokenizer
            # - the vocabulary is directly obtained from tokenizer.get_vocab()
            #   (tokenizer.backend_tokenizer.to_str() may not contain the full vocab, special
            #   tokens may be omitted)
            # - the vocab size is obtained from len(tokenizer.get_vocab()) or provided by user
            # - the vocab type and add_prefix_space are obtained from
            #   tokenizer.backend_tokenizer.to_str()
            # - stop token id is provided by user, or auto detected.
            backend_str = tokenizer.backend_tokenizer.to_str()
            if stop_token_ids is None:
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
                else:
                    logger.warning('When constructing TokenizerInfo from a huggingface tokenizer, '
                                   'stop_token_ids is neither provided by user nor found from the tokenizer. '
                                   'It will be automatically detected.')
            metadata = json.loads(TokenizerInfo._detect_metadata_from_hf(backend_str))
            return TokenizerInfo(
                encoded_vocab,
                vocab_type=VocabType(metadata['vocab_type']),
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=metadata['add_prefix_space'],
            )

        elif TokenizerInfo._is_tiktoken_tokenizer(tokenizer):
            # tiktoken tokenizer
            # e.g. Phi-3-small-8k-instruct, Qwen-7B-Chat, stablelm-2-12b-chat (previously)
            if stop_token_ids is None:
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
                else:
                    logger.warning('When constructing TokenizerInfo from a huggingface tokenizer, '
                                   'stop_token_ids is neither provided by user nor found from the tokenizer. '
                                   'It will be automatically detected.')
            return TokenizerInfo(
                encoded_vocab,
                VocabType.RAW,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=False,
            )

        elif TokenizerInfo._is_sentencepiece_tokenizer(tokenizer):
            # sentencepiece tokenizer
            # e.g. Chatglm3-6b
            if hasattr(tokenizer, 'sp_model'):
                sp_model = tokenizer.sp_model
            elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'sp_model'):
                sp_model = tokenizer.tokenizer.sp_model

            if stop_token_ids is None:
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
                else:
                    eos_id = sp_model.eos_id()
                    if eos_id != -1:
                        stop_token_ids = [eos_id]
                    else:
                        logger.warning('When constructing TokenizerInfo from a huggingface tokenizer, '
                                       'stop_token_ids is neither provided by user nor found from the tokenizer. '
                                       'It will be automatically detected.')
            # detect vocab_type of tokenizer
            if '<0x0A>' in vocab_dict:
                vocab_type = VocabType.BYTE_FALLBACK
            else:
                vocab_type = VocabType.RAW

            return TokenizerInfo(
                encoded_vocab,
                vocab_type=vocab_type,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )

        else:
            # TODO(yixin): unsupported tokenizer
            raise ValueError(f'Unsupported tokenizer type: {type(tokenizer)}')

    @property
    def vocab_type(self) -> VocabType:
        """The type of the vocabulary."""
        return VocabType(self._handle.vocab_type)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""
        return self._handle.vocab_size

    @property
    def add_prefix_space(self) -> bool:
        """Whether the tokenizer will prepend a space before the text in the
        tokenization process."""
        return self._handle.add_prefix_space

    @property
    def prepend_space_in_tokenization(self) -> bool:
        """Whether the tokenizer will prepend a space before the text in the
        tokenization process.

        This property is deprecated. Use add_prefix_space instead.
        """
        logger.warning('prepend_space_in_tokenization is deprecated. Use add_prefix_space instead.')
        return self.add_prefix_space

    @property
    def decoded_vocab(self) -> List[bytes]:
        """The decoded vocabulary of the tokenizer.

        This converts the tokens in the LLM's vocabulary back to the original format of the input text. E.g. for type
        ByteFallback, the token <0x1B> is converted back to "\u001b".
        """
        return self._handle.decoded_vocab

    @property
    def stop_token_ids(self) -> List[int]:
        """The stop token ids."""
        return self._handle.stop_token_ids

    @property
    def special_token_ids(self) -> List[int]:
        """The special token ids.

        Special tokens include control tokens, reserved tokens, padded tokens, etc. Now it is automatically detected
        from the vocabulary.
        """
        return self._handle.special_token_ids

    def dump_metadata(self) -> str:
        """Dump the metadata of the tokenizer to a json string.

        It can be used to construct the tokenizer info from the vocabulary and the metadata string.
        """
        return self._handle.dump_metadata()
