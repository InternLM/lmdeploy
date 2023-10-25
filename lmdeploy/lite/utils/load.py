# Copyright (c) OpenMMLab. All rights reserved.

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM

from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.model import LoadWoInit

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}


def load_hf_from_pretrained(pretrained_model_name_or_path, **kwargs):
    with init_empty_weights():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs)
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
            pretrained_model_name_or_path, device_map=device_map, **kwargs)
        model.config.use_cache = False

    return model
