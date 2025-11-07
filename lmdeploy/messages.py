# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from pydantic.dataclasses import dataclass as pydantic_dataclass

from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest

from .tokenizer import Tokenizer
from .utils import get_logger

logger = get_logger('lmdeploy')

LogitsProcessor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""LogitsProcessor is a function that takes a tensor of input_ids, the logits
tensor for the next token, and returns a modified tensor of logits to sample
from."""


@dataclass
class GenerationConfig:
    """Generation parameters used by inference engines.

    Args:
        n (int): Define how many chat completion choices to generate for each
            input message. **Only 1** is supported now.
        max_new_tokens (int): The maximum number of tokens that can be
            generated in the chat completion
        do_sample (bool):  Whether or not to use sampling, use greedy
            decoding otherwise. Default to be False.
        top_p (float): An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass
        top_k (int): An alternative to sampling with temperature, where
            the model considers the top_k tokens with the highest probability
        min_p (float): Minimum token probability, which will be scaled by the
            probability of the most likely token. It must be a value between
            0 and 1. Typical values are in the 0.01-0.2 range, comparably
            selective as setting `top_p` in the 0.99-0.8 range (use the
            opposite of normal `top_p` values)
        temperature (float): Sampling temperature
        repetition_penalty (float): Penalty to prevent the model from
            generating repeated words or phrases. A value larger than
            1 discourages repetition
        ignore_eos (bool): Indicator to ignore the eos_token_id or not
        random_seed (int): Seed used when sampling a token
        stop_words (List[str]): Words that stop generating further tokens
        bad_words (List[str]): Words that the engine will never generate
        stop_token_ids (List[int]): List of tokens that stop the generation
            when they are generated. The returned output will not contain
            the stop tokens.
        bad_token_ids (List[str]): List of tokens that the engine will never
            generate.
        min_new_tokens (int): The minimum numbers of tokens to generate,
            ignoring the number of tokens in the prompt.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        spaces_between_special_tokens (bool): Whether or not to add spaces
            around special tokens. The behavior of Fast tokenizers is to have
            this to False. This is setup to True in slow tokenizers.
        logprobs (int): Number of log probabilities to return per output token.
        response_format (Dict): Generate responses according to given formatting.
        response. Examples:
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                    "properties": {
                        "name": {
                        "type": "string"
                        }
                    },
                    "required": ["name"],
                    "type": "object"
                    }
                }
            }
        or,
            {
                "type": "regex_schema",
                "regex_schema": "call me [A-Za-z]{1,10}"
            }
        logits_processors (List[Callable]): Custom logit processors.
    """

    n: int = 1
    max_new_tokens: int = 512
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.0
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: List[str] = None
    bad_words: List[str] = None
    stop_token_ids: List[int] = None
    bad_token_ids: List[int] = None
    min_new_tokens: int = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logprobs: int = None
    response_format: Optional[Dict] = None
    logits_processors: Optional[List[LogitsProcessor]] = None
    output_logits: Literal['all', 'generation'] = None
    output_last_hidden_state: Literal['all', 'generation'] = None
    include_stop_str_in_output: bool = False

    # for disaggregation
    with_cache: bool = False
    preserve_cache: bool = False
    migration_request: Optional[MigrationRequest] = None

    def convert_stop_bad_words_to_ids(self, tokenizer: Tokenizer):
        """Convert stop_words/bad_sords to ids and append the ids to
        stop_token_ids/bad_token_ids."""

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

        stop_token_ids = special_word_token_ids(self.stop_words) or []
        bad_token_ids = special_word_token_ids(self.bad_words) or []
        stop_token_ids.extend(self.stop_token_ids or [])
        bad_token_ids.extend(self.bad_token_ids or [])
        self.stop_token_ids = list(set(stop_token_ids)) or None
        self.bad_token_ids = list(set(bad_token_ids)) or None

    def update_from_hf_gen_cfg(self, generation_config, tokenizer_eos_token_id):
        """Update the stop_token_ids."""
        stop_token_ids = set(self.stop_token_ids or [])

        # add tokenizer's eos_token_id
        if tokenizer_eos_token_id is not None:
            stop_token_ids.add(tokenizer_eos_token_id)

        # add eos_token_id from model's generation_config.json file if there
        # is any.
        eos_token_id = generation_config.get('eos_token_id')
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                stop_token_ids.add(eos_token_id)
            else:
                stop_token_ids.update(eos_token_id)

        self.stop_token_ids = list(stop_token_ids)

    def __post_init__(self):
        """Check input validation."""
        assert type(self.n) == int and self.n > 0, 'n is not a positive integer'
        assert self.top_p >= 0 and self.top_p <= 1  # [0, 1]
        assert self.top_k >= 0, 'top_k can not be a negative integer'
        assert self.temperature >= 0 and self.temperature <= 2  # [0,2]
        assert 0 <= self.min_p <= 1, \
            f'min_p should be in range [0, 1], but found {self.min_p}'


@pydantic_dataclass
class TurbomindEngineConfig:
    """TurboMind Engine config.

    Args:
        dtype (str): data type for model weights and activations. It can be
            one of the following values, ['auto', 'float16', 'bfloat16']
            The `auto` option will use FP16 precision for FP32 and FP16
            models, and BF16 precision for BF16 models.
        model_format (str): the layout of the deployed model. It can be one
            of the following values [hf, awq, gptq],`hf` meaning
            huggingface model(.bin, .safetensors), `awq` and `gptq` meaning
            the quantized model by AWQ and GPTQ, respectively. If it is not
            specified, i.e. None, it will be extracted from the input model
        tp (int): the number of GPU cards used in tensor parallelism,
            default to 1
        session_len (int): the max session length of a sequence, default to
            None
        max_batch_size (int): the max batch size during inference. If it is
            not specified, the engine will automatically set it according to
            the device
        cache_max_entry_count (float): the percentage of gpu memory occupied
            by the k/v cache.
            For versions of lmdeploy between `v0.2.0` and `v0.2.1`, it
            defaults to 0.5, depicting the percentage of TOTAL GPU memory to
            be allocated to the k/v cache.
            For lmdeploy versions greater than `v0.2.1`, it defaults to 0.8,
            signifying the percentage of FREE GPU memory to be reserved for
            the k/v cache.
            When it's an integer > 0, it represents the total number of k/v
            blocks.
        cache_chunk_size (int): The policy to apply for KV block from
            the block manager, default to -1.
        cache_block_seq_len (int): the length of the token sequence in
            a k/v block, default to 64
        enable_prefix_caching (bool): enable cache prompts for block reuse,
            default to False
        quant_policy (int): default to 0. When k/v is quantized into 4 or 8
            bit, set it to 4 or 8, respectively
        rope_scaling_factor (float): scaling factor used for dynamic ntk,
            default to 0. TurboMind follows the implementation of transformer
            LlamaAttention
        use_logn_attn (bool): whether or not to use log attn: default to False
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
        revision (str): The specific model version to use. It can be a branch
            name, a tag name, or a commit id. If unspecified, will use the
            default version.
        max_prefill_token_num(int): the number of tokens each iteration during
            prefill, default to 8192
        num_tokens_per_iter(int): the number of tokens processed in each
            forward pass. Working with `max_prefill_iters` enables the
            "Dynamic SplitFuse"-like scheduling
        max_prefill_iters(int): the max number of forward pass during prefill
            stage
        devices(List[int]): the used devices
        empty_init (bool): Whether to load the model weights, you should set
            it to True if you want to update weights after create the pipeline
        hf_overrides (Dict[str, Any]): Huggingface overrides for the model.
            It can be used to override the default config of the model
        enable_metrics (bool): enable metrics system
    """

    dtype: str = 'auto'
    model_format: Optional[str] = None
    tp: int = 1
    dp: int = 1
    device_num: int = None
    attn_tp_size: int = None
    attn_dp_size: int = None
    mlp_tp_size: int = None
    mlp_dp_size: int = None
    outer_dp_size: int = None
    session_len: Optional[int] = None
    max_batch_size: int = None
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    cache_block_seq_len: int = 64
    enable_prefix_caching: bool = False
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
    download_dir: Optional[str] = None
    revision: Optional[str] = None
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    devices: Optional[List[int]] = None
    empty_init: bool = False
    communicator: str = 'nccl'
    hf_overrides: Optional[Dict[str, Any]] = None
    enable_metrics: bool = True

    def __post_init__(self):
        """Check input validation."""
        assert self.dtype in ['auto', 'float16', 'bfloat16']
        assert self.tp >= 1, 'tp must be a positive integer'
        assert self.cache_max_entry_count > 0, 'invalid cache_max_entry_count'
        assert self.quant_policy in (0, 4, 8), 'invalid quant_policy'
        assert self.rope_scaling_factor >= 0, 'invalid rope_scaling_factor'
        assert self.max_prefill_token_num >= 0, \
            'invalid max_prefill_token_num'
        assert self.num_tokens_per_iter >= 0, 'invalid num_tokens_per_iter'


@dataclass
class PytorchEngineConfig:
    """PyTorch Engine Config.

    Args:
        dtype (str): data type for model weights and activations. It can be
            one of the following values, ['auto', 'float16', 'bfloat16']
            The `auto` option will use FP16 precision for FP32 and FP16
            models, and BF16 precision for BF16 models.
        tp (int): Tensor Parallelism. default 1.
        dp (int): Data Parallelism. default 1.
        dp_rank (int): rank of dp.
        ep (int): Expert Parallelism. default 1.
        session_len (int): Max session length. Default None.
        max_batch_size (int): Max batch size. If it is not specified,
            the engine will automatically set it according to the device
        cache_max_entry_count (float): the percentage of gpu memory occupied
            by the k/v cache. For lmdeploy versions greater than `v0.2.1`,
            it defaults to 0.8, signifying the percentage of FREE GPU memory
            to be reserved for the k/v cache
        prefill_interval (int): Interval to perform prefill,
            Default 16.
        block_size (int): paging cache block size, default 64.
        num_cpu_blocks (int): Num cpu blocks. If num is 0, cache
            would be allocate according to current environment.
        num_gpu_blocks (int): Num gpu blocks. If num is 0, cache
            would be allocate according to current environment.
        adapters (dict): The path configs to lora adapters.
        max_prefill_token_num (int): tokens per iteration.
        thread_safe (bool): thread safe engine instance.
        enable_prefix_caching (bool): Enable token match and sharing caches.
        device_type (str): The inference device type, options ['cuda']
        eager_mode (bool): Enable "eager" mode or not
        custom_module_map (Dict): nn module map customized by users. Once
            provided, the original nn modules of the model will be
            substituted by the mapping ones
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
        revision (str): The specific model version to use.
            It can be a branch name, a tag name, or a commit id.
            If unspecified, will use the default version.
        quant_policy (int): default to 0. When k/v is quantized into 4 or 8
            bit, set it to 4 or 8, respectively
        distributed_executor_backend (str): backend of distributed backend,
            options: ['uni', 'mp', 'ray']
        empty_init (bool): Whether to load the model weights, you should set
            it to True if you want to update weights after create the pipeline
        enable_microbatch (bool): enable microbatch for specified model
        enable_eplb (bool): enable eplb for specified model
        enable_metrics (bool): enable metrics system
        role (EngineRole): role of engin, options: ['Hybrid', 'Prefill',
            'Decode']. Default to `EngineRole.Hybrid`.
        migration_backend: migration backend. options: ['DLSlime'].
            Default to `MigrationBackend.DLSlime`.
        enable_mp_engine (bool): run engine in multi-process mode.
        mp_engine_backend (str): backend of mp engine, options:
            ['mp', 'ray']. Default to `mp`.
        model_format (str): weight quantization policy, options: ['fp8'].
        hf_overrides (Dict[str, Any]): Huggingface overrides for the model.
            It can be used to override the default config of the model,
        disable_vision_encoder (bool): Whether to disable loading vision
            encoder. Default to False.
        logprobs_mode (str): The mode of logprob, options: ['raw_logits', 'raw_logprobs']
        dllm_block_length (int): Block size of block diffusion model.
        dllm_unmasking_strategy (str): Dllm unmasking strategy, options:
            ['low_confidence_dynamic', 'low_confidence_static', 'sequential'].
        dllm_denoising_steps (int): Dllm denoising steps.
        dllm_confidence_threshold (float): dllm unmasking threshold for
            dynamic unmasking.
    """
    dtype: str = 'auto'
    tp: int = 1
    dp: int = 1
    dp_rank: int = 0
    ep: int = 1
    session_len: int = None
    max_batch_size: int = None
    cache_max_entry_count: float = 0.8
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0
    adapters: Dict[str, str] = None
    max_prefill_token_num: int = 4096
    thread_safe: bool = False
    enable_prefix_caching: bool = False
    device_type: str = 'cuda'
    eager_mode: bool = False
    custom_module_map: Dict[str, str] = None
    download_dir: str = None
    revision: str = None
    quant_policy: Literal[0, 4, 8] = 0
    distributed_executor_backend: str = None
    empty_init: bool = False
    enable_microbatch: bool = False
    enable_eplb: bool = False
    enable_mp_engine: bool = False
    mp_engine_backend: str = 'mp'
    model_format: str = None
    enable_metrics: bool = True
    hf_overrides: Optional[Dict[str, Any]] = None
    disable_vision_encoder: bool = False
    logprobs_mode: str = None

    # dllm
    dllm_block_length: int = None
    dllm_unmasking_strategy: str = 'low_confidence_dynamic'
    dllm_denoising_steps: int = None
    dllm_confidence_threshold: float = 0.85

    role: EngineRole = EngineRole.Hybrid
    migration_backend: MigrationBackend = MigrationBackend.DLSlime

    def __post_init__(self):
        """Check input validation."""
        assert self.dtype in ['auto', 'float16', 'bfloat16']
        assert self.tp >= 1, 'invalid tp'
        assert self.dp >= 1, 'invalid dp'
        assert self.ep >= 1, 'invalid ep'
        assert 0 < self.cache_max_entry_count < 1, \
            'invalid cache_max_entry_count'
        assert self.num_cpu_blocks >= 0, 'invalid num_cpu_blocks'
        assert self.max_prefill_token_num >= 0, \
            'invalid max_prefill_token_num'
        assert self.num_gpu_blocks >= 0, 'invalid num_gpu_blocks'
        assert self.quant_policy in (0, 4, 8), 'invalid quant_policy'
        assert self.device_type in ['cuda', 'ascend', 'maca', 'camb'], (f'invalid device_type: {self.device_type}')
        assert self.block_size >= 16 and (self.block_size & (self.block_size - 1)) == 0, \
            f'block_size must be >= 16 and a power of 2, but got {self.block_size}'
        if self.quant_policy > 0 and self.device_type not in ['cuda', 'ascend']:
            assert False, \
                   'kv cache quantization only works for CUDA and ASCEND.'
        if self.device_type == 'camb' and self.block_size != 16:
            self.block_size = 16
            logger.warning('Currently, camb device requires block size to be 16, \
                    setting block size to 16')


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    SESSION_REPEAT = enum.auto()
    SESSION_NOT_EXIST = enum.auto()
    HANDLER_NOT_EXIST = enum.auto()
    INPUT_LENGTH_ERROR = enum.auto()
    INTERNAL_ENGINE_ERROR = enum.auto()
    CANCEL = enum.auto()
    PREFIX_CACHE_CONFLICT_INTERACTIVE_MODE = enum.auto()


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
        session_id (int): the id for running the session.
        finish_reason ('stop' | 'length' | None): the reason the model stopped
            generating tokens. This will be 'stop' if the model hit a natural
            stop point or a provided stop sequence, 'length' if the maximum
            number of tokens specified in the request was reached.
        token_ids: (List[int]): the output token ids.
        logprobs: (List[Dict[int, float]]): the top logprobs for each output
            position.
        index (int): it refers to the position index of the input request
            batch
    """
    text: str
    generate_token_len: int
    input_token_len: int
    finish_reason: Optional[Literal['stop', 'length']] = None
    token_ids: List[int] = field(default_factory=list)
    logprobs: List[Dict[int, float]] = None
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    index: int = 0

    def __repr__(self):
        logits = 'logits=None' if self.logits is None else f'logits.shape={self.logits.shape}\nlogits={self.logits}'
        hidden_state = (
            'last_hidden_state=None' if self.last_hidden_state is None else
            f'last_hidden_state.shape={self.last_hidden_state.shape}\nlast_hidden_state={self.last_hidden_state}')
        s = (f'text={self.text}\ngenerate_token_len={self.generate_token_len}\nfinish_reason="{self.finish_reason}"\n'
             f'token_ids={self.token_ids}\nlog_probs={self.logprobs}\n{logits}\n{hidden_state}')
        return s


# modified from https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py
class EventType(enum.IntEnum):
    """The type of request event.

    QUEUED - when the request was enqued by the engine
    SCHEDULED - when the request was first scheduled for execution
    PREEMPTED - the request has been put back in the waiting queue in order to make room for other requests to complete.
                It will be re-scheduled in future and re-start its prefill phase
    """
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3  # FIXME, currently ignored for simplicity


# modified from https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py
@dataclass
class EngineEvent:
    """A timestamped engine event associated with a request.

    Attributes:
        type: the type of an event associated with a request during its life cycle
        timestamp: the WALL-CLOCK time when the event happens.
    """
    type: EventType
    timestamp: float

    @classmethod
    def new_event(cls, event_type: EventType, timestamp: Optional[float] = None) -> 'EngineEvent':
        # Timestamps MUST use wall-clock time (time.time()) to maintain consistency
        # between csrc(std::chrono::system_clock) and python
        timestamp = time.time() if timestamp is None else timestamp
        return cls(event_type, timestamp)


@dataclass
class ScheduleMetrics:
    active_seqs: int = 0
    waiting_seqs: int = 0
    total_blocks: int = 0
    active_blocks: int = 0
    cached_blocks: int = 0
    free_blocks: int = 0


@dataclass
class RequestMetrics:
    """Basic metrics for a request.

    Attributes:
        token_timestamp: A wall-clock time when a token is generated.
        engine_events: List of engine events during inference.
    """
    token_timestamp: float = 0.0
    engine_events: List[EngineEvent] = field(default_factory=list)


@dataclass
class EngineOutput:
    """Engine output from turbomind/pytorch engine.

    Args:
        status (ResponseType): the response type.
        token_ids (List[int]): the newly generated token ids in each iteration.
        logprobs (List[Dict[int, float]]): the top logprobs for each output
            position.
        cache_block_ids (List[int]): send cache blocks back for migration in
            Disaggregated LLM Serving when Prefill Engine is Done.
        req_metrics (RequestMetrics): request metrics information
    """
    status: ResponseType
    token_ids: List[int]
    logprobs: List[Dict[int, float]] = None
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    cache_block_ids: Optional[List[int]] = None
    req_metrics: Optional[RequestMetrics] = None


@dataclass
class VisionConfig:
    """Vison model configs.

    Args:
        max_batch_size (int): the max image size passed to the model, since
            some models will use image patch, the actual running batch could
            be larger than this value.
        thread_safe (bool): Specifies whether the engine instance is
            thread-safe. Please set it to True when using the pipeline
            in a multi-threaded environment.
    """
    max_batch_size: int = 1
    thread_safe: bool = False
