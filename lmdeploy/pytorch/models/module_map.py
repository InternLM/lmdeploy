# Copyright (c) OpenMMLab. All rights reserved.

LMDEPLOY_PYTORCH_MODEL_PATH = 'lmdeploy.pytorch.models'

# llama
MODULE_MAP = {
    'transformers.models.llama.modeling_llama.LlamaFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'transformers.models.llama.modeling_llama.LlamaMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'transformers.models.llama.modeling_llama.LlamaRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
    # support modeling rewritten in lmdeploy
    'modeling_llama.LlamaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaAttention',
    'modeling_llama.LlamaModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'modeling_llama.LlamaMLP': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
}

# Falcon Models in transformer / on hub
MODULE_MAP.update({
    'modeling_falcon.FalconAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconAttention',
    'modeling_falcon.FalconModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconModel',
    'modeling_falcon.FalconMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconMLP',
    'modeling_falcon.FalconForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconForCausalLM',
    # for old implementations on hub
    'modelling_RW.Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconAttention',
    'modelling_RW.MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconMLP',
    'modelling_RW.RWModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconModel',
    'modelling_RW.RotaryEmbedding':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconRotaryEmbedding',
})

# baichuan
MODULE_MAP.update({
    'modeling_baichuan.Model':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',  # noqa
    'modeling_baichuan.BaichuanModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.BaichuanModel',  # noqa
    'modeling_baichuan.Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.Attention',  # noqa
    'modeling_baichuan.BaichuanAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.BaichuanAttention',  # noqa
    'modeling_baichuan.MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',  # noqa
    'modeling_baichuan.RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.baichuan.PatchedRMSNorm',
})

# chatglm2
MODULE_MAP.update({
    'modeling_chatglm.SelfAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.PatchedSelfAttention',
    'modeling_chatglm.ChatGLMModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.PatchedChatGLMModel',
    'modeling_chatglm.MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.MLP',
    'modeling_chatglm.RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.chatglm2.PatchedRMSNorm',
})

# internlm
MODULE_MAP.update({
    'modeling_internlm.InternLMAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm.PatchedInternLMAttention',
    'modeling_internlm.InternLMModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'modeling_internlm.InternLMMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'modeling_internlm.InternLMRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# internlm2
MODULE_MAP.update({
    'modeling_internlm2.InternLM2Attention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2.PatchedInternLM2Attention',
    'modeling_internlm2.InternLM2Model':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2.PatchedInternLM2Model',
    'modeling_internlm2.InternLM2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.internlm2.PatchedInternLM2MLP',
    'modeling_internlm2.InternLM2RMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# mistral
MODULE_MAP.update({
    'transformers.models.mistral.modeling_mistral.MistralAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.MistralFlashAttention2',
    'transformers.models.mistral.modeling_mistral.MistralFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.MistralFlashAttention2',
    'transformers.models.mistral.modeling_mistral.MistralSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mistral.MistralFlashAttention2',
    'transformers.models.mistral.modeling_mistral.MistralModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaModel',
    'transformers.models.mistral.modeling_mistral.MistralMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'transformers.models.mistral.modeling_mistral.MistralRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})

# gemma
MODULE_MAP.update({
    'transformers.models.gemma.modeling_gemma.GemmaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.PatchedGemmaAttention',
    'transformers.models.gemma.modeling_gemma.GemmaFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.PatchedGemmaAttention',
    'transformers.models.gemma.modeling_gemma.GemmaSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.PatchedGemmaAttention',
    'transformers.models.gemma.modeling_gemma.GemmaModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.PatchedGemmaModel',
    'transformers.models.gemma.modeling_gemma.modeling_mistral.GemmaMLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaMLP',
    'transformers.models.gemma.modeling_gemma.GemmaRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.gemma.PatchedGemmaRMSNorm',
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

# peft
MODULE_MAP.update({
    'peft.tuners.lora.layer.Linear':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.peft.LoRALinear'
})

# mixtral
MODULE_MAP.update({
    'transformers.models.mixtral.modeling_mixtral.MixtralAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'transformers.models.mixtral.modeling_mixtral.MixtralSdpaAttention':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralAttention',
    'transformers.models.mixtral.modeling_mixtral.MixtralModel':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralModel',
    'transformers.models.mixtral.modeling_mixtral.MixtralBLockSparseTop2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralBLockSparseTop2MLP',
    'transformers.models.mixtral.modeling_mixtral.MixtralBlockSparseTop2MLP':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.mixtral.PatchedMixtralBLockSparseTop2MLP',
    'transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaRMSNorm',
})
