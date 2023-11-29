# Copyright (c) OpenMMLab. All rights reserved.

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.model import LoadWoInit

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',  # Baichuan 7B
    'BaichuanForCausalLM': 'DecoderLayer',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}


def load_hf_from_pretrained(pretrained_model_name_or_path, **kwargs):

    kwargs.pop('config', None)

    hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                           torch_dtype=torch.float16,
                                           trust_remote_code=True)

    # hard code for qwen, other configs do not have the `fp16` attribute.
    hf_config.fp16 = True

    with init_empty_weights():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=hf_config, **kwargs)
        model.config.use_cache = False
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    decoder_layers = collect_target_modules(model, layer_type)
    # Infer device map
    device_map = infer_auto_device_map(model,
                                       no_split_module_classes=[layer_type])
    for name in device_map.keys():
        if name in decoder_layers or 'lm_head' in name:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    if 'device_map' in kwargs:
        kwargs.pop('device_map')
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            config=hf_config,
            **kwargs)
        model.config.use_cache = False

    return model
