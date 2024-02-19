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
    'modeling_falcon.FalconRotaryEmbedding':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.falcon.PatchedFalconRotaryEmbedding',
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

# peft
MODULE_MAP.update({
    'peft.tuners.lora.layer.Linear':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.peft.LoRALinear'
})
