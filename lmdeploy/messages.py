# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass
from typing import List

from .tokenizer import Tokenizer


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
class EngineGenerationConfig:
    """generation parameter used by the inference engines."""
    n: int = 1
    max_new_tokens: int = 512
    top_p: float = 1.0
    top_k: int = 1
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
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
        stop_words = gen_config.stop_words
        if stop_words is not None:
            assert isinstance(stop_words, List) and \
                all(isinstance(elem, str) for elem in stop_words), \
                f'stop_words must be a list of str but got {type(stop_words)}'
            stop_words = [tokenizer(word).input_ids for word in stop_words]

        bad_words = gen_config.bad_words
        if bad_words is not None:
            assert isinstance(bad_words, List) and \
                all(isinstance(elem, str) for elem in bad_words), \
                f'bad_words must be a list of str but got {type(bad_words)}'
            bad_words = [tokenizer(word).input_ids for word in bad_words]

        return EngineGenerationConfig(
            n=gen_config.n,
            max_new_tokens=gen_config.max_new_tokens,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            ignore_eos=gen_config.ignore_eos,
            random_seed=gen_config.random_seed,
            stop_words=stop_words,
            bad_words=bad_words)
