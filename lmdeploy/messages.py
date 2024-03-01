# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from pydantic.dataclasses import dataclass as pydantic_dataclass

from .tokenizer import Tokenizer


@dataclass
class GenerationConfig:
    """generation parameters used by inference engines.

    Args:
        n (int): Define how many chat completion choices to generate for each
            input message
        max_new_tokens (int): The maximum number of tokens that can be
            generated in the chat completion
        top_p (float): An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass
        top_k (int): An alternative to sampling with temperature, where
            the model considers the top_k tokens with the highest probability
        temperature (float): Sampling temperature
        repetition_penalty (float): Penalty to prevent the model from
            generating repeated words or phrases. A value larger than
            1 discourages repetition
        ignore_eos (bool): Indicator to ignore the eos_token_id or not
        random_seed (int): Seed used when sampling a token
        stop_words (List[str]): Words that stop generating further tokens
        bad_words (List[str]): Words that the engine will never generate
        min_new_tokens (int): The minimum numbers of tokens to generate,
            ignoring the number of tokens in the prompt.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
    """

    n: int = 1
    max_new_tokens: int = 512
    top_p: float = 1.0
    top_k: int = 1
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: List[str] = None
    bad_words: List[str] = None
    min_new_tokens: int = None
    skip_special_tokens: bool = True


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
        """  # noqa E501

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
            min_new_tokens=gen_config.min_new_tokens,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            ignore_eos=gen_config.ignore_eos,
            random_seed=gen_config.random_seed,
            skip_special_tokens=gen_config.skip_special_tokens,
            stop_words=special_word_token_ids(gen_config.stop_words),
            bad_words=special_word_token_ids(gen_config.bad_words))


@pydantic_dataclass
class TurbomindEngineConfig:
    """TurboMind Engine config.

    Args:
        model_name (str): the name of the deployed model, deprecated and has no effect when version > 0.2.1
        model_format (str): the layout of the deployed model. It can be one of the following values [hf, llama, awq], `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by AWQ.
        tp (int): the number of GPU cards used in tensor parallelism, default to 1
        session_len (int): the max session length of a sequence, default to None
        max_batch_size (int): the max batch size during inference, default to 128
        cache_max_entry_count (float): the percentage of gpu memory occupied by the k/v cache.
            For versions of lmdeploy between `v0.2.0` and `v0.2.1`, it defaults to 0.5, depicting the percentage of TOTAL GPU memory to be allocated to the k/v cache.
            For lmdeploy versions greater than `v0.2.1`, it defaults to 0.8, signifying the percentage of FREE GPU memory to be reserved for the k/v cache
        quant_policy (int): , default to 0. When k/v is quantized into 8 bit, set it to 4
        rope_scaling_factor (int): scaling factor used for dynamic ntk, default to 0. TurboMind follows the implementation of transformer LlamaAttention
        use_logn_attn (bool): whether or not to use log attn: default to False
        download_dir (str): Directory to download and load the weights, default to the default cache directory of huggingface.
        revision (str): The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.
        max_prefill_token_num(int): the number of tokens each iteration during prefill, default to 8192
    """  # noqa: E501

    model_name: Optional[str] = None
    model_format: Optional[str] = None
    tp: int = 1
    session_len: Optional[int] = None
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
    download_dir: Optional[str] = None
    revision: Optional[str] = None
    max_prefill_token_num: int = 8192


@dataclass
class PytorchEngineConfig:
    """PyTorch Engine Config.

    Args:
        model_name (str): name of the given model.
        tp (int): Tensor Parallelism. default 1.
        session_len (int): Max session length. Default None.
        max_batch_size (int): Max batch size. Default 128.
        cache_max_entry_count (float): the percentage of gpu memory occupied
            by the k/v cache. For lmdeploy versions greater than `v0.2.1`,
            it defaults to 0.8, signifying the percentage of FREE GPU memory
            to be reserved for the k/v cache
        eviction_type (str): What action to perform when kv cache
            is full, ['recompute', 'copy'], Default 'recompute'.
        prefill_interval (int): Interval to perform prefill,
            Default 16.
        block_size (int): paging cache block size, default 64.
        num_cpu_blocks (int): Num cpu blocks. If num is 0, cache
            would be allocate according to current environment.
        num_gpu_blocks (int): Num gpu blocks. If num is 0, cache
            would be allocate according to current environment.
        adapters (dict): The path configs to lora adapters.
        max_prefill_token_num (int): tokens per iteration.
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
        revision (str): The specific model version to use.
            It can be a branch name, a tag name, or a commit id.
            If unspecified, will use the default version.
    """
    model_name: str = ''
    tp: int = 1
    session_len: int = None
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.8
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0
    adapters: Dict[str, str] = None
    max_prefill_token_num: int = 8192
    download_dir: str = None
    revision: str = None


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    SESSION_REPEAT = enum.auto()
    SESSION_NOT_EXIST = enum.auto()
    HANDLER_NOT_EXIST = enum.auto()


@dataclass
class Response:
    """Pack all response information together.

    Args:
        text (str): the response text from the server. If the output text is
            an empty str and the finish_reason is length, it means the session
            length is reached.
        generate_token_len (int): the response token length.
        input_token_len (int): the input prompt token length. Note that it may
            contains chat template part.
        session_id (int): the id for running the session. Basically, it refers
            to the position index of the input request batch.
        finish_reason ('stop' | 'length' | None): the reason the model stopped
            generating tokens. This will be 'stop' if the model hit a natural
            stop point or a provided stop sequence, 'length' if the maximum
            number of tokens specified in the request was reached.
    """
    text: str
    generate_token_len: int
    input_token_len: int
    session_id: int
    finish_reason: Optional[Literal['stop', 'length']] = None
