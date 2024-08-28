# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import get_model_arch
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

SUPPORTED_ARCHS = dict(
    # baichuan-7b
    BaiChuanForCausalLM='baichuan',
    # baichuan2-7b, baichuan-13b, baichuan2-13b
    BaichuanForCausalLM='baichuan2',
    # internlm
    InternLMForCausalLM='llama',
    # internlm2
    InternLM2ForCausalLM='internlm2',
    # llama, llama2, alpaca, vicuna, codellama, ultracm, yi,
    # deepseek-coder, deepseek-llm
    LlamaForCausalLM='llama',
    # Qwen 7B-72B, Qwen-VL-7B
    QWenLMHeadModel='qwen',
    # Qwen2
    Qwen2ForCausalLM='qwen2',
    # mistral
    MistralForCausalLM='llama',
    # llava
    LlavaLlamaForCausalLM='llama',
    LlavaMistralForCausalLM='llama',
    # xcomposer2
    InternLMXComposer2ForCausalLM='xcomposer2',
    # internvl
    InternVLChatModel='internvl',
    # deepseek-vl
    MultiModalityCausalLM='deepseekvl',
    # MiniCPMV
    MiniCPMV='minicpmv',
    # mini gemini
    MGMLlamaForCausalLM='llama',
    MiniGeminiLlamaForCausalLM='llama',
    # chatglm2/3, glm4
    ChatGLMModel='glm4',
    ChatGLMForConditionalGeneration='glm4')


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

    def _is_head_dim_128(cfg):
        num_attn_head = cfg.num_attention_heads
        hidden_size = cfg.hidden_size
        # turbomind support head_dim=128
        return (hidden_size // num_attn_head) == 128

    support_by_turbomind = False
    triton_model_path = os.path.join(model_path, 'triton_models')
    if os.path.exists(triton_model_path):
        support_by_turbomind = True
    else:
        arch, cfg = get_model_arch(model_path)

        if arch in SUPPORTED_ARCHS.keys():
            support_by_turbomind = True
            # special cases
            if arch == 'BaichuanForCausalLM':
                num_attn_head = cfg.num_attention_heads
                if num_attn_head == 40:
                    # baichuan-13B, baichuan2-13B not supported by turbomind
                    support_by_turbomind = False
            elif arch == 'Qwen2ForCausalLM':
                # qwen2 0.5b size_per_head is 64, which hasn't been supported
                # by turbomind yet
                support_by_turbomind = _is_head_dim_128(cfg)
            elif arch in ('ChatGLMModel', 'ChatGLMForConditionalGeneration'):
                # chatglm1/2/3 is not working yet
                support_by_turbomind = cfg.num_layers == 40
                if getattr(cfg, 'vision_config', None) is not None:
                    # glm-4v-9b not supported
                    support_by_turbomind = False
            elif arch == 'InternVLChatModel':
                # internvl2-4b,internlm2-1b are not working yet
                support_by_turbomind = _is_head_dim_128(cfg.llm_config)

    return support_by_turbomind
