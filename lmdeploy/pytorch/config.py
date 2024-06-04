# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import torch


def _update_torch_dtype(config: 'ModelConfig', default: str = 'float16'):
    """Update the torch dtype from the model config.

    Args:
        config (ModelConfig): The input model config.
        default (str): default device type.
    """
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    def __hack_qwen():
        """hack qwen."""
        is_bf16_supported = torch.cuda.is_bf16_supported()
        torch_dtype = 'bfloat16' if is_bf16_supported else 'float16'
        if config.hf_config.bf16 and is_bf16_supported:
            torch_dtype = 'bfloat16'
        elif config.hf_config.fp16:
            torch_dtype = 'float16'
        return torch_dtype

    def __hack_cogvlm():
        """hack cogvlm."""
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

    torch_dtype = getattr(config.hf_config, 'torch_dtype', None)
    if torch_dtype is None:
        if config.hf_config.model_type == 'qwen':
            torch_dtype = __hack_qwen()
        elif config.model_arch == 'CogVLMForCausalLM':
            torch_dtype = __hack_cogvlm()
        else:
            logger.warning('Model config does not have `torch_dtype`,'
                           f' use default: {default}')
            torch_dtype = default
        # update hf_config as well
        setattr(config.hf_config, 'torch_dtype', torch_dtype)

    config.dtype = eval(f'torch.{torch_dtype}')
    return config


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int = 512
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    max_active_adapters: int = 64


@dataclass
class CacheConfig:
    """Config of key value cache."""

    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int
    window_size: int = -1
    cache_max_entry_count: float = 0.8
    max_prefill_token_num: int = 4096
    enable_prefix_caching: bool = False

    def __post_init__(self):
        """post init."""
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        if self.window_size > 1 and self.enable_prefix_caching:
            logger.warning(
                'Prefix caching is not available for window attention.')
            self.enable_prefix_caching = False


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    bos_token_id: int
    eos_token_id: List[int]
    head_dim: int
    sliding_window: int = -1
    dtype: torch.dtype = torch.float16
    multi_query_attention: bool = False
    vocab_size: int = 40000
    hf_config: Any = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_arch: str = None
    unused_modules: List[str] = None
    task_type: Literal['llm', 'vlm'] = 'llm'

    def get_head_size(self):
        """get head size."""
        return self.head_dim

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        trust_remote_code: bool = True):
        """build ModelConfig from model path or name."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return cls.from_hf_config(hf_config, pretrained_model_name_or_path)

    @classmethod
    def from_hf_config(cls, hf_config: Any, model_path: str = None):
        """from huggingface config."""
        from lmdeploy.archs import check_vl_llm

        if model_path is None:
            model_path = ''

        def __build_falcon():
            """build falcon."""
            num_attention_heads = hf_config.num_attention_heads
            if hf_config.new_decoder_architecture:
                # 40b-instruct, GQA
                kv_head = hf_config.num_kv_heads
            if hf_config.multi_query:
                # 7b-instruct, MQA
                kv_head = 1
            else:
                # rw-1b, MHA
                kv_head = num_attention_heads
            head_dim = hf_config.hidden_size // num_attention_heads
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=kv_head,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=head_dim,
                multi_query_attention=hf_config.multi_query,
                vocab_size=hf_config.vocab_size,
            )

        def __build_chatglm():
            """build chatglm."""
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            bos_token_id = hf_config.bos_token_id
            if bos_token_id is None:
                bos_token_id = hf_config.pad_token_id
            init_kwargs = dict(empty_init=False)
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.multi_query_group_num,
                bos_token_id=bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=head_dim,
                vocab_size=hf_config.padded_vocab_size,
                init_kwargs=init_kwargs)

        def __build_gemma():
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                head_dim=hf_config.head_dim,
                vocab_size=hf_config.vocab_size)

        def __build_dbrx():
            hidden_size = hf_config.d_model
            num_heads = hf_config.n_heads
            head_dim = hidden_size // num_heads
            eos_token_id = getattr(hf_config, 'eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = 100257
            bos_token_id = getattr(hf_config, 'bos_token_id', None)
            if bos_token_id is None:
                bos_token_id = eos_token_id
            return ModelConfig(
                hidden_size=hidden_size,
                num_layers=hf_config.n_layers,
                num_attention_heads=num_heads,
                num_key_value_heads=hf_config.attn_config.kv_n_heads,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                head_dim=head_dim,
                vocab_size=hf_config.vocab_size)

        def __build_default():
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            num_attention_heads = hf_config.num_attention_heads
            num_key_value_heads = getattr(hf_config, 'num_key_value_heads',
                                          num_attention_heads)
            use_sliding_window = getattr(hf_config, 'use_sliding_window', True)
            sliding_window = -1
            if use_sliding_window:
                sliding_window = getattr(hf_config, 'sliding_window',
                                         sliding_window) or -1
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                sliding_window=sliding_window,
                head_dim=head_dim,
                vocab_size=hf_config.vocab_size)

        def __build_qwen():
            cfg = __build_default()
            if cfg.bos_token_id is None:
                cfg.bos_token_id = 151644
            if cfg.eos_token_id is None:
                cfg.eos_token_id = 151645
            return cfg

        def __build_cogvlm():
            cfg = __build_default()
            if getattr(hf_config, 'num_multi_query_heads', None):
                cfg.num_key_value_heads = hf_config.num_multi_query_heads
            cfg.unused_modules = ['model.vision']
            return cfg

        model_arch = getattr(hf_config, 'architectures', [None])[0]

        if hf_config.model_type == 'falcon':
            model_config = __build_falcon()
        elif hf_config.model_type == 'chatglm':
            model_config = __build_chatglm()
        elif hf_config.model_type == 'gemma':
            model_config = __build_gemma()
        elif hf_config.model_type == 'dbrx':
            model_config = __build_dbrx()
        elif hf_config.model_type == 'qwen':
            model_config = __build_qwen()
        elif model_arch == 'CogVLMForCausalLM':
            model_config = __build_cogvlm()
        else:
            model_config = __build_default()

        model_config.hf_config = hf_config
        model_config.model_arch = model_arch

        # should after setting `hf_config` and `model_arch` attributes
        model_config = _update_torch_dtype(model_config)

        if check_vl_llm(hf_config.to_dict()):
            model_config.task_type = 'vlm'

        # update eos_token_id to list
        if isinstance(model_config.eos_token_id, int):
            model_config.eos_token_id = [model_config.eos_token_id]

        return model_config
