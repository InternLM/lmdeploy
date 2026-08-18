# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig


def config_from_pretrained(pretrained_model_name_or_path: str, **kwargs):
    return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
