# Copyright (c) OpenMMLab. All rights reserved.

LMDEPLOY_PYTORCH_MODEL_PATH = 'lmdeploy.pytorch.models'

# llama
MODULE_MAP = {
    'LlamaForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
}

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
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
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
    'modeling_deepseek.DeepseekAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.PatchedDeepseekAttention',
    'modeling_deepseek.DeepseekFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.PatchedDeepseekAttention',
    'modeling_deepseek.DeepseekSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.PatchedDeepseekAttention',
    'modeling_deepseek.DeepseekModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'modeling_deepseek.DeepseekMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'modeling_deepseek.DeepseekRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
    'modeling_deepseek.DeepseekMoE':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek.PatchedDeepseekMoE',
})

# qwen
MODULE_MAP.update({
    'modeling_qwen.QWenAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.PatchedQWenAttention',
    'modeling_qwen.FlashSelfAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.PatchedQWenAttention',
    'modeling_qwen.QWenModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.PatchedQWenModel',
    'modeling_qwen.QWenMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.PatchedQWenMLP',
    'modeling_qwen.RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen.PatchedRMSNorm',
})

# qwen1.5
MODULE_MAP.update({
    'transformers.models.qwen2.modeling_qwen2.Qwen2Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2.modeling_qwen2.Qwen2Model':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'transformers.models.qwen2.modeling_qwen2.Qwen2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# qwen2 moe
MODULE_MAP.update({
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2.PatchedQwen2Attention',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_moe.PatchedQwen2MoeModel',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen2_moe.PatchedQwen2MoeSparseMoeBlock',
    'transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# mixtral
MODULE_MAP.update({
    'MixtralForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.MixtralForCausalLM',
})

# dbrx
MODULE_MAP.update({
    'modeling_dbrx.DbrxAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxAttention',
    'modeling_dbrx.DbrxFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxAttention',
    'modeling_dbrx.DbrxSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxAttention',
    'modeling_dbrx.DbrxModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxModel',
    'modeling_dbrx.DbrxExpertGLU':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxExpertGLU',
    'modeling_dbrx.DbrxExperts':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.dbrx.PatchedDbrxExperts',
})

# starcoder2
MODULE_MAP.update({
    'transformers.models.starcoder2.modeling_starcoder2.Starcoder2Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.PatchedStarcoder2Attention',
    'transformers.models.starcoder2.modeling_starcoder2.Starcoder2FlashAttention2':    # noqa: E501
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.PatchedStarcoder2Attention',
    'transformers.models.starcoder2.modeling_starcoder2.Starcoder2SdpaAttention':    # noqa: E501
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.PatchedStarcoder2Attention',
    'transformers.models.starcoder2.modeling_starcoder2.Starcoder2Model':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'transformers.models.starcoder2.modeling_starcoder2.Starcoder2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.starcoder2.PatchedStarcoder2MLP',
})

# phi-3
MODULE_MAP.update({
    'modeling_phi3.Phi3Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3.Phi3FlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3.Phi3SdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3.Phi3Model':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Model',
    'modeling_phi3.Phi3MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3MLP',
    'modeling_phi3.Phi3RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# deepseek-v2
MODULE_MAP.update({
    'DeepseekV2ForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.deepseek_v2.DeepseekV2ForCausalLM'
})

# cogvlm
MODULE_MAP.update({
    'modeling_cogvlm.RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
    'modeling_cogvlm.MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'modeling_cogvlm.VisionExpertMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.cogvlm.PatchedVisionExpertMLP',
    'modeling_cogvlm.VisionExpertAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.cogvlm.PatchedVisionExpertAttention',
    'modeling_cogvlm.CogVLMModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.cogvlm.PatchedCogVLMModel',
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

# internvl
MODULE_MAP.update({
    'modeling_internvl_chat.InternVLChatModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internvl.PatchedInternVLChatModel'
})

# phi3 vision
MODULE_MAP.update({
    'modeling_phi3_v.Phi3Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3_v.Phi3FlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3_v.Phi3SdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Attention',
    'modeling_phi3_v.Phi3VModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3Model',
    'modeling_phi3_v.Phi3MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3.PatchedPhi3MLP',
    'modeling_phi3_v.Phi3RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# ascend module
ASCEND_MODULE_MAP = dict()

DEVICE_SPECIAL_MODULE_MAP = dict(ascend=ASCEND_MODULE_MAP)

# phi-3.5-moe
MODULE_MAP.update({
    'modeling_phimoe.PhiMoEAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'modeling_phimoe.PhiMoEFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'modeling_phimoe.PhiMoESdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'modeling_phimoe.PhiMoEModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralModel',
    'modeling_phimoe.PhiMoEBlockSparseTop2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralBLockSparseTop2MLP',
    'modeling_phimoe.PhiMoEBLockSparseTop2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralBLockSparseTop2MLP',
    'modeling_phimoe.PhiMoESparseMoeBlock':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.phi3_moe.PatchedPhiMoESparseMoeBlock',
})

CUSTOM_MODULE_MAP = dict()
