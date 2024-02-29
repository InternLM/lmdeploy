# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, Optional, Union

from transformers import AutoConfig

from .messages import PytorchEngineConfig, TurbomindEngineConfig
from .utils import get_logger

logger = get_logger('lmdeploy')

_SUPPORTED_ARCHS = dict(
    # baichuan-7b
    BaiChuanForCausalLM=dict(pytorch=False, turbomind=True),
    # baichuan2-7b, baichuan-13b, baichuan2-13b
    BaichuanForCausalLM=dict(pytorch=True, turbomind=True),
    # chatglm2-6b, chatglm3-6b
    ChatGLMModel=dict(pytorch=True, turbomind=False),
    # deepseek-moe
    DeepseekForCausalLM=dict(pytorch=True, turbomind=False),
    # falcon-7b
    FalconForCausalLM=dict(pytorch=True, turbomind=False),
    # gemma-7b
    GemmaForCausalLM=dict(pytorch=True, turbomind=False),
    # internlm
    InternLMForCausalLM=dict(pytorch=True, turbomind=True),
    # internlm2
    InternLM2ForCausalLM=dict(pytorch=True, turbomind=True),
    # internlm-xcomposer
    InternLM2XComposerForCausalLM=dict(pytorch=False, turbomind=True),
    # llama, llama2, alpaca, vicuna, codellama, ultracm, yi,
    # deepseek-coder, deepseek-llm
    LlamaForCausalLM=dict(pytorch=True, turbomind=True),
    # Mistral-7B
    MistralForCausalLM=dict(pytorch=True, turbomind=False),
    # Mixtral-8x7B
    MixtralForCausalLM=dict(pytorch=True, turbomind=False),
    # Qwen 7B-72B, Qwen-VL-7B
    QWenLMHeadModel=dict(pytorch=False, turbomind=True),
    # Qwen1.5 7B-72B
    Qwen2ForCausalLM=dict(pytorch=True, turbomind=False),
)


def _is_support_by(model_path: str):
    """Check whether supported by pytorch or turbomind engine.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
    Returns:
        support_by_torch (bool): Whether input model is supported by pytorch engine
        support_by_turbomind (bool): Whether input model is supported by turbomind engine
    """  # noqa: E501
    import os

    support_by_torch, support_by_turbomind = False, False

    triton_model_path = os.path.join(model_path, 'triton_models')
    if os.path.exists(triton_model_path):
        support_by_turbomind = True
        return support_by_torch, support_by_turbomind

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(cfg, 'architectures'):
        arch = cfg.architectures[0]
    elif hasattr(cfg, 'auto_map') and 'AutoModelForCausalLM' in cfg.auto_map:
        arch = cfg.auto_map['AutoModelForCausalLM'].split('.')[-1]
    else:
        raise RuntimeError(
            f'Could not find model architecture from config: {cfg}')

    if arch in _SUPPORTED_ARCHS:
        info = _SUPPORTED_ARCHS[arch]
        support_by_torch = info['pytorch']
        support_by_turbomind = info['turbomind']
        # special cases
        if arch == 'BaichuanForCausalLM':
            num_attn_head = cfg.num_attention_heads
            if num_attn_head == 40:
                # baichuan-13B, baichuan2-13B not supported by turbomind
                support_by_turbomind = False
                if cfg.vocab_size == 64000:
                    # baichuan-13B not supported by pytorch
                    support_by_torch = False
    return support_by_torch, support_by_turbomind


def autoget_backend(model_path: str) -> Union[Literal['turbomind', 'pytorch']]:
    """Get backend type in auto backend mode.

    Args:
         model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.

    Returns:
        str: the backend type.
    """
    pytorch_has, turbomind_has = _is_support_by(model_path)
    if not (pytorch_has or turbomind_has):
        logger.warning(f'{model_path} is not explicitly supported by lmdeploy.'
                       f' Try to run with lmdeploy pytorch engine.')
    backend = 'turbomind' if turbomind_has else 'pytorch'
    return backend


def autoget_backend_config(
    model_path: str,
    backend_config: Optional[Union[PytorchEngineConfig,
                                   TurbomindEngineConfig]] = None
) -> Union[PytorchEngineConfig, TurbomindEngineConfig]:
    """Get backend config automatically.

    Args:
        model_path (str): The input model path.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): The
            input backend config. Default to None.

    Returns:
        (PytorchEngineConfig | TurbomindEngineConfig): The auto-determined
            backend engine config.
    """
    from dataclasses import asdict

    backend = autoget_backend(model_path)
    if backend == 'pytorch':
        config = PytorchEngineConfig()
    else:
        config = TurbomindEngineConfig()
    if backend_config is not None:
        data = asdict(backend_config)
        for k, v in data.items():
            if v and hasattr(config, k):
                setattr(config, k, v)
    return config
