# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

SUPPORTED_ARCHS = dict(
    # baichuan-7b
    BaiChuanForCausalLM='baichuan',
    # baichuan2-7b, baichuan-13b, baichuan2-13b
    BaichuanForCausalLM='baichuan2',
    # gpt-oss
    GptOssForCausalLM='gpt-oss',
    # internlm
    InternLMForCausalLM='llama',
    # internlm2
    InternLM2ForCausalLM='internlm2',
    # internlm3
    InternLM3ForCausalLM='llama',
    # llama, llama2, alpaca, vicuna, codellama, ultracm, yi,
    # deepseek-coder, deepseek-llm
    LlamaForCausalLM='llama',
    # Qwen 7B-72B, Qwen-VL-7B
    QWenLMHeadModel='qwen',
    # Qwen2
    Qwen2ForCausalLM='qwen2',
    Qwen2MoeForCausalLM='qwen2-moe',
    # Qwen2-VL
    Qwen2VLForConditionalGeneration='qwen2',
    # Qwen2.5-VL
    Qwen2_5_VLForConditionalGeneration='qwen2',
    # Qwen3
    Qwen3ForCausalLM='qwen3',
    Qwen3MoeForCausalLM='qwen3-moe',
    # mistral
    MistralForCausalLM='llama',
    # llava
    LlavaLlamaForCausalLM='llama',
    LlavaMistralForCausalLM='llama',
    LlavaForConditionalGeneration='llava',
    # xcomposer2
    InternLMXComposer2ForCausalLM='xcomposer2',
    # internvl
    InternVLChatModel='internvl',
    # internvl3
    InternVLForConditionalGeneration='internvl',
    InternS1ForConditionalGeneration='internvl',
    # deepseek-vl
    MultiModalityCausalLM='deepseekvl',
    DeepseekV2ForCausalLM='deepseek2',
    # MiniCPMV
    MiniCPMV='minicpmv',
    # chatglm2/3, glm4
    ChatGLMModel='glm4',
    ChatGLMForConditionalGeneration='glm4',
    # mixtral
    MixtralForCausalLM='mixtral',
    MolmoForCausalLM='molmo',
)


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

    def _is_head_dim_supported(cfg):
        head_dim = cfg.head_dim if hasattr(cfg, 'head_dim') else cfg.hidden_size // cfg.num_attention_heads
        return head_dim in [128, 64]

    support_by_turbomind = False
    triton_model_path = os.path.join(model_path, 'triton_models')
    if os.path.exists(triton_model_path):
        support_by_turbomind = True
    else:

        arch, cfg = get_model_arch(model_path)
        quant_method = search_nested_config(cfg.to_dict(), 'quant_method')
        if quant_method and quant_method in ['smooth_quant']:
            # tm hasn't support quantized models by applying smoothquant
            return False

        if arch in SUPPORTED_ARCHS.keys():
            support_by_turbomind = True
            # special cases
            if arch == 'BaichuanForCausalLM':
                num_attn_head = cfg.num_attention_heads
                if num_attn_head == 40:
                    # baichuan-13B, baichuan2-13B not supported by turbomind
                    support_by_turbomind = False
            elif arch in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
                support_by_turbomind = _is_head_dim_supported(cfg)
            elif arch in ('ChatGLMModel', 'ChatGLMForConditionalGeneration'):
                # chatglm1/2/3 is not working yet
                support_by_turbomind = cfg.num_layers == 40
                if getattr(cfg, 'vision_config', None) is not None:
                    # glm-4v-9b not supported
                    support_by_turbomind = False
            elif arch == 'InternVLChatModel':
                llm_arch = cfg.llm_config.architectures[0]
                support_by_turbomind = (llm_arch in SUPPORTED_ARCHS and _is_head_dim_supported(cfg.llm_config))
            elif arch in ['LlavaForConditionalGeneration', 'InternVLForConditionalGeneration']:
                llm_arch = cfg.text_config.architectures[0]
                if llm_arch in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
                    support_by_turbomind = _is_head_dim_supported(cfg.text_config)
            elif arch == 'MolmoForCausalLM':
                kv_heads = cfg.num_key_value_heads
                # TM hasn't supported allenai/Molmo-7B-O-0924 yet
                support_by_turbomind = kv_heads is not None
            elif arch == 'DeepseekV2ForCausalLM':
                if getattr(cfg, 'vision_config', None) is not None:
                    support_by_turbomind = False

    return support_by_turbomind
