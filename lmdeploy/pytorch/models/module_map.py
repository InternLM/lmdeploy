# Copyright (c) OpenMMLab. All rights reserved.

LMDEPLOY_PYTORCH_MODEL_PATH = 'lmdeploy.pytorch.models'

# ascend module
MODULE_MAP = dict()
ASCEND_MODULE_MAP = dict()
MACA_MODULE_MAP = dict()
CAMB_MODULE_MAP = dict()

DEVICE_SPECIAL_MODULE_MAP = dict(ascend=ASCEND_MODULE_MAP, maca=MACA_MODULE_MAP, camb=CAMB_MODULE_MAP)

# llama
MODULE_MAP.update({
    'LlamaForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
})

# llama4
MODULE_MAP.update({
    'Llama4ForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama4.Llama4ForConditionalGeneration',
})

# baichuan
MODULE_MAP.update({
    'BaichuanForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.BaichuanForCausalLM',
})

# chatglm
MODULE_MAP.update({
    'ChatGLMForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.ChatGLMForConditionalGeneration',  # noqa: E501
})

# glm4-0414
MODULE_MAP.update({
    'Glm4ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.glm4.Glm4ForCausalLM',
})

# glm4.1-v
MODULE_MAP.update({
    'Glm4vForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.glm4_1v.Glm4vForConditionalGeneration',
})

# glm4.5
MODULE_MAP.update({
    'Glm4MoeForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.glm4_moe.Glm4MoeForCausalLM',
})

# internlm
MODULE_MAP.update({
    'InternLMForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm.InternLMForCausalLM',
})

# internlm2
MODULE_MAP.update({
    'InternLM2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2.InternLM2ForCausalLM',
})

# mistral
MODULE_MAP.update({
    'MistralForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.MistralForCausalLM',
})

# mixtral
MODULE_MAP.update({
    'MixtralForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.MixtralForCausalLM',
})

# gemma
MODULE_MAP.update({
    'GemmaForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.GemmaForCausalLM',
})

# gemma2
MODULE_MAP.update({
    'Gemma2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.GemmaForCausalLM',
})

# gemma3 text
MODULE_MAP.update({
    'Gemma3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.GemmaForCausalLM',
})

# gemma3 VL
MODULE_MAP.update({
    'Gemma3ForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma3_vl.Gemma3ForConditionalGeneration',
})

# deepseek
MODULE_MAP.update({
    'DeepseekForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.DeepseekForCausalLM',
})

# deepseek-v2
MODULE_MAP.update({'DeepseekV2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek_v2.DeepseekV2ForCausalLM'})

# deepseek-v3
MODULE_MAP.update({'DeepseekV3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek_v2.DeepseekV2ForCausalLM'})

# deepseek-vl2
MODULE_MAP.update({'DeepseekVLV2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek_vl2.DeepseekVLV2ForCausalLM'})

# llava
MODULE_MAP.update({
    'LlavaForConditionalGeneration': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llava.LlavaForConditionalGeneration',  # noqa: E501
    'LlavaNextForConditionalGeneration':  # noqa: E501
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llava.LlavaNextForConditionalGeneration'  # noqa: E501
})

# qwen
MODULE_MAP.update({
    'QWenLMHeadModel': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.QWenLMHeadModel',
})

# qwen1.5
MODULE_MAP.update({
    'Qwen2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.Qwen2ForCausalLM',
})

# qwen2 moe
MODULE_MAP.update({
    'Qwen2MoeForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_moe.Qwen2MoeForCausalLM',
})

# qwen3
MODULE_MAP.update({
    'Qwen3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3.Qwen3ForCausalLM',
})

# qwen3 moe
MODULE_MAP.update({
    'Qwen3MoeForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3_moe.Qwen3MoeForCausalLM',
})

# qwen2_vl
MODULE_MAP.update({
    'Qwen2VLForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_vl.Qwen2VLForConditionalGeneration',
})

# qwen2_5_vl
MODULE_MAP.update({
    'Qwen2_5_VLForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration',
})

# qwen3_vl
MODULE_MAP.update({
    'Qwen3VLForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3_vl.Qwen3VLForConditionalGeneration',
})

# qwen3_vl_moe
MODULE_MAP.update({
    'Qwen3VLMoeForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration',
})

# starcoder2
MODULE_MAP.update({
    'Starcoder2ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.Starcoder2ForCausalLM',
})

# phi-3
MODULE_MAP.update({
    'Phi3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.Phi3ForCausalLM',
})

# cogvlm
MODULE_MAP.update({
    'CogVLMForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.cogvlm.CogVLMForCausalLM',
})

# internvl
MODULE_MAP.update({'InternVLChatModel': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internvl.InternVLChatModel'})

# internvl3-hf
MODULE_MAP.update({
    'InternVLForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internvl3_hf.InternVLForConditionalGeneration'
})

# interns1-hf
MODULE_MAP.update({
    'InternS1ForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internvl3_hf.InternVLForConditionalGeneration'
})

# mono-internvl
MODULE_MAP.update({
    'InternLM2VEForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2_ve.InternLM2VEForCausalLM',
})

# phi3 vision
MODULE_MAP.update({
    'Phi3VForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3_v.Phi3VForCausalLM',
})

# phi-3.5-moe
MODULE_MAP.update({
    'PhiMoEForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3_moe.PhiMoEForCausalLM',
})

# minicpm3
MODULE_MAP.update({
    'MiniCPM3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.minicpm3.MiniCPM3ForCausalLM',
})

# minicpmv2_6
MODULE_MAP.update({
    'MiniCPMV': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.minicpmv26.MiniCPMVForCausalLM',
})

# mllama
MODULE_MAP.update({
    'MllamaForConditionalGeneration':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mllama.MllamaForConditionalGeneration',
})

# internlm3
MODULE_MAP.update({
    'InternLM3ForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm3.InternLM3ForCausalLM',
})

# internlm2 reward model
MODULE_MAP.update(
    {'InternLM2ForRewardModel': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2_reward.InternLM2ForRewardModel'})

# qwen2 reward model
MODULE_MAP.update({'Qwen2ForRewardModel': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_reward.Qwen2ForRewardModel'})

# gpt-oss
MODULE_MAP.update({
    'GptOssForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gpt_oss.GptOssForCausalLM',
})

# qwen3 next model
MODULE_MAP.update({
    'Qwen3NextForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3_next.Qwen3NextForCausalLM',
})

# SDAR
MODULE_MAP.update({
    'SDARForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.sdar.SDARForCausalLM',
    'SDARMoeForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.sdar_moe.SDARMoeForCausalLM',
})

CUSTOM_MODULE_MAP = dict()
