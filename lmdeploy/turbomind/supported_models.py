# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_SUPPORTED_ARCHS = dict(
    # baichuan-7b
    BaiChuanForCausalLM=True,
    # baichuan2-7b, baichuan-13b, baichuan2-13b
    BaichuanForCausalLM=True,
    # chatglm2-6b, chatglm3-6b
    ChatGLMModel=False,
    # deepseek-moe
    DeepseekForCausalLM=False,
    # falcon-7b
    FalconForCausalLM=False,
    # gemma-7b
    GemmaForCausalLM=False,
    # internlm
    InternLMForCausalLM=True,
    # internlm2
    InternLM2ForCausalLM=True,
    # internlm-xcomposer
    InternLMXComposerForCausalLM=True,
    # internlm2-xcomposer
    InternLM2XComposerForCausalLM=False,
    # llama, llama2, alpaca, vicuna, codellama, ultracm, yi,
    # deepseek-coder, deepseek-llm
    LlamaForCausalLM=True,
    # Mistral-7B
    MistralForCausalLM=False,
    # Mixtral-8x7B
    MixtralForCausalLM=False,
    # Qwen 7B-72B, Qwen-VL-7B
    QWenLMHeadModel=True,
    # Qwen1.5 7B-72B
    Qwen2ForCausalLM=False,
    # llava
    LlavaLlamaForCausalLM=True,
    # deepseek-vl
    MultiModalityCausalLM=True)


def is_supported(model_path: str):
    """Check whether supported by turbomind engine.

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
        support_by_turbomind (bool): Whether input model is supported by turbomind engine
    """  # noqa: E501
    import os

    support_by_turbomind = False
    triton_model_path = os.path.join(model_path, 'triton_models')
    if os.path.exists(triton_model_path):
        support_by_turbomind = True
    else:
        try:
            cfg = AutoConfig.from_pretrained(model_path,
                                             trust_remote_code=True).to_dict()
        except Exception as e:  # noqa
            logger.warning('AutoConfig.from_pretrained failed for '
                           f'{model_path}, try PretrainedConfig.'
                           'get_config_dict instead.')
            from transformers import PretrainedConfig
            cfg = PretrainedConfig.get_config_dict(model_path)[0]

        if cfg.get('architectures', None):
            arch = cfg['architectures'][0]
        elif cfg.get('auto_map',
                     None) and 'AutoModelForCausalLM' in cfg['auto_map']:
            arch = cfg['auto_map']['AutoModelForCausalLM'].split('.')[-1]
        else:
            raise RuntimeError(
                f'Could not find model architecture from config: {cfg}')

        if arch in _SUPPORTED_ARCHS:
            support_by_turbomind = _SUPPORTED_ARCHS[arch]
            # special cases
            if arch == 'BaichuanForCausalLM':
                num_attn_head = cfg['num_attention_heads']
                if num_attn_head == 40:
                    # baichuan-13B, baichuan2-13B not supported by turbomind
                    support_by_turbomind = False
    return support_by_turbomind
