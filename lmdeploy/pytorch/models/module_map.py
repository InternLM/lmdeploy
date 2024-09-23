# Copyright (c) OpenMMLab. All rights reserved.

LMDEPLOY_PYTORCH_MODEL_PATH = 'lmdeploy.pytorch.models'

# ascend module
MODULE_MAP = dict()
ASCEND_MODULE_MAP = dict()

DEVICE_SPECIAL_MODULE_MAP = dict(ascend=ASCEND_MODULE_MAP)

# llama
MODULE_MAP.update({
    'LlamaForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
})

# Falcon Models in transformer / on hub
MODULE_MAP.update({
    'FalconForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.FalconForCausalLM',
})

# baichuan
MODULE_MAP.update({
    'BaichuanForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.BaichuanForCausalLM',
})

# chatglm
MODULE_MAP.update({
    'ChatGLMForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.ChatGLMForConditionalGeneration',  # noqa: E501
})

# internlm
MODULE_MAP.update({
    'InternLMForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm.InternLMForCausalLM',
})

# internlm2
MODULE_MAP.update({
    'InternLM2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2.InternLM2ForCausalLM',
})

# mistral
MODULE_MAP.update({
    'MistralForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.MistralForCausalLM',
})

# mixtral
MODULE_MAP.update({
    'MixtralForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.MixtralForCausalLM',
})

# gemma
MODULE_MAP.update({
    'GemmaForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.GemmaForCausalLM',
})

# gemma2
MODULE_MAP.update({
    'Gemma2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.GemmaForCausalLM',
})

# deepseek
MODULE_MAP.update({
    'DeepseekForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.DeepseekForCausalLM',
})

# deepseek-v2
MODULE_MAP.update({
    'DeepseekV2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek_v2.DeepseekV2ForCausalLM'
})

# llava
MODULE_MAP.update(
    {
        'LlavaLlamaForCausalLM':
        f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlavaLlamaForCausalLM',
        'LlavaMistralForCausalLM':
        f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.LlavaMistralForCausalLM',
        'LlavaForConditionalGeneration':
        f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llava.LlavaForConditionalGeneration',  # noqa: E501
        'LlavaNextForConditionalGeneration':  # noqa: E501
        f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llava.LlavaForConditionalGeneration'
    })

# qwen
MODULE_MAP.update({
    'QWenLMHeadModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.QWenLMHeadModel',
})

# qwen1.5
MODULE_MAP.update({
    'Qwen2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.Qwen2ForCausalLM',
})

# qwen2 moe
MODULE_MAP.update({
    'Qwen2MoeForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_moe.Qwen2MoeForCausalLM',
})

# qwen2_vl
MODULE_MAP.update({
    'Qwen2VLForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_vl.Qwen2VLForConditionalGeneration',
})

# dbrx
MODULE_MAP.update({
    'DbrxForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.DbrxForCausalLM',
})

# starcoder2
MODULE_MAP.update({
    'Starcoder2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.Starcoder2ForCausalLM',
})

# phi-3
MODULE_MAP.update({
    'Phi3ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.Phi3ForCausalLM',
})

# cogvlm
MODULE_MAP.update({
    'CogVLMForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.cogvlm.CogVLMForCausalLM',
})

# internvl
MODULE_MAP.update({
    'InternVLChatModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internvl.InternVLChatModel'
})

# phi3 vision
MODULE_MAP.update({
    'Phi3VForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.Phi3VForCausalLM',
})

# phi-3.5-moe
MODULE_MAP.update({
    'PhiMoEForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3_moe.PhiMoEForCausalLM',
})

# minicpm3
MODULE_MAP.update({
    'MiniCPM3ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.minicpm3.MiniCPM3ForCausalLM',
})

CUSTOM_MODULE_MAP = dict()
