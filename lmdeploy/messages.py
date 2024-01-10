# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from typing import List

from lmdeploy.utils import get_logger

from .tokenizer import Tokenizer

logger = get_logger('lmdeploy')


@dataclass
class GenerationConfig:
    """generation parameters used by inference engines."""

    n: int = 1  # How many chat completion choices to generate for each input message. # noqa E501
    max_new_tokens: int = 512  # The maximum number of tokens that can be generated in the chat completion. # noqa E501
    top_p: float = 1.0  # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass # noqa E501
    top_k: int = 1  # An alternative to sampling with temperature, where the model considers the top_k tokens with the highest probability # noqa E501
    temperature: float = 0.8  # sampling temperature
    repetition_penalty: float = 1.0  # penalty to prevent the model from generating repeated words or phrases. A value larger than 1 discourages repetition. # noqa E501
    ignore_eos: bool = False  # indicator to ignore the eos_token_id or not
    random_seed: int = None  # seed used when sampling a token
    stop_words: List[str] = None  # words that stop generating further tokens
    bad_words: List[str] = None  # words that the engine will never generate


@dataclass
class EngineGenerationConfig(GenerationConfig):
    """generation parameter used by the inference engines."""
    stop_words: List[int] = None
    bad_words: List[int] = None

    @staticmethod
    def From(gen_config: GenerationConfig, tokenizer: Tokenizer):
        """convert `GenerationConfig` to `EngineGenerationConfig`
        Args:
            gen_config (GenerationConfig): an instance of class `GenerationConfig`
            tokenizer (Tokenizer): a tokenizer to encode the `stop_words` and `bad_words` in `gen_config`

        Returns:
            EngineGenerationConfig: the generation config used by inference engines

        Examples:
            >>> from lmdeploy import Tokenizer, GenerationConfig, EngineGenerationConfig
            >>> tokenizer = Tokenizer('internlm/internlm-chat-7b')
            >>> gen_config = GenerationConfig(stop_words=['<eoa>'])
            >>> gen_config = EngineGenerationConfig.From(gen_config, tokenizer)
        """ # noqa E501

        def special_word_token_ids(words):
            if words is not None:
                assert isinstance(words, List) and \
                    all(isinstance(elem, str) for elem in words), \
                    f'stop_words must be a list of str but got {type(words)}'
                indexes = []
                for word in words:
                    indexes += tokenizer.indexes_containing_token(word)
                return indexes
            return None

        return EngineGenerationConfig(
            n=gen_config.n,
            max_new_tokens=gen_config.max_new_tokens,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            ignore_eos=gen_config.ignore_eos,
            random_seed=gen_config.random_seed,
            stop_words=special_word_token_ids(gen_config.stop_words),
            bad_words=special_word_token_ids(gen_config.bad_words))


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    SESSION_REPEAT = enum.auto()
    SESSION_NOT_EXIST = enum.auto()
    HANDLER_NOT_EXIST = enum.auto()
