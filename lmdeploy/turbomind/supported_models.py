# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

SUPPORTED_ARCHS = dict(
    # Qwen2 / Qwen2-MoE
    Qwen2ForCausalLM='qwen2',
    Qwen2MoeForCausalLM='qwen2-moe',
    # Qwen3
    Qwen3ForCausalLM='qwen3',
    Qwen3MoeForCausalLM='qwen3-moe',
    # Qwen 3.5
    Qwen3_5ForConditionalGeneration='qwen3_5',
    Qwen3_5MoeForConditionalGeneration='qwen3_5-moe',
    # InternVL family
    InternVLChatModel='internvl',
    InternVLForConditionalGeneration='internvl',
    InternS1ForConditionalGeneration='internvl',
    # Llama (2, 3, 3.1, 3.2) + InternLM3
    LlamaForCausalLM='llama',
    InternLM2ForCausalLM='internlm2',
    InternLM3ForCausalLM='llama',
    # glm4-moe-lite (e.g. GLM-4.7-Flash)
    Glm4MoeLiteForCausalLM='glm4-moe-lite',
    # gpt-oss
    GptOssForCausalLM='gpt-oss',
)


def is_supported(model_path: str, trust_remote_code: bool = False):
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
        arch, cfg = get_model_arch(model_path, trust_remote_code=trust_remote_code)
        quant_method = search_nested_config(cfg.to_dict(), 'quant_method')
        if quant_method and quant_method in ['smooth_quant']:
            return False

        if arch in SUPPORTED_ARCHS.keys():
            support_by_turbomind = True
            if arch == 'Glm4MoeLiteForCausalLM':
                if getattr(cfg, 'vision_config', None) is not None:
                    support_by_turbomind = False

    return support_by_turbomind
