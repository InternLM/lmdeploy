# Copyright (c) OpenMMLab. All rights reserved.

from transformers import AutoConfig

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_SUPPORTED_ARCHS = dict(
    # baichuan-7b
    BaiChuanForCausalLM=False,
    # baichuan2-7b, baichuan-13b, baichuan2-13b
    BaichuanForCausalLM=True,
    # chatglm2-6b, chatglm3-6b
    ChatGLMModel=True,
    # deepseek-moe
    DeepseekForCausalLM=True,
    # falcon-7b
    FalconForCausalLM=True,
    # gemma-7b
    GemmaForCausalLM=True,
    # internlm
    InternLMForCausalLM=True,
    # internlm2
    InternLM2ForCausalLM=True,
    # internlm-xcomposer
    InternLMXComposerForCausalLM=False,
    # internlm2-xcomposer
    InternLM2XComposerForCausalLM=False,
    # llama, llama2, alpaca, vicuna, codellama, ultracm, yi,
    # deepseek-coder, deepseek-llm
    LlamaForCausalLM=True,
    # Mistral-7B
    MistralForCausalLM=True,
    # Mixtral-8x7B
    MixtralForCausalLM=True,
    # Qwen 7B-72B, Qwen-VL-7B
    QWenLMHeadModel=True,
    # Qwen1.5 7B-72B
    Qwen2ForCausalLM=True,
)


def is_supported(model_path: str):
    """Check whether supported by pytorch engine.

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
    """  # noqa: E501
    import os

    support_by_torch = False

    triton_model_path = os.path.join(model_path, 'triton_models')
    if os.path.exists(triton_model_path):
        logger.warning(f'{model_path} seems to be a turbomind workspace, '
                       'which can only be ran with turbomind engine.')
    else:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(cfg, 'architectures'):
            arch = cfg.architectures[0]
        elif hasattr(cfg,
                     'auto_map') and 'AutoModelForCausalLM' in cfg.auto_map:
            arch = cfg.auto_map['AutoModelForCausalLM'].split('.')[-1]
        else:
            raise RuntimeError(
                f'Could not find model architecture from config: {cfg}')

        if arch in _SUPPORTED_ARCHS:
            support_by_torch = _SUPPORTED_ARCHS[arch]
            # special cases
            if arch == 'BaichuanForCausalLM':
                # baichuan-13B not supported by pytorch
                if cfg.num_attention_heads == 40 and cfg.vocab_size == 64000:
                    support_by_torch = False
            elif arch == 'QWenLMHeadModel':
                # qwen-vl not supported by pytorch
                if getattr(cfg, 'visual', None):
                    support_by_torch = False

    return support_by_torch
