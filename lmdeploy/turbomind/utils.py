# Copyright (c) OpenMMLab. All rights reserved.
import os

from transformers.utils import ExplicitEnum

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ModelSource(ExplicitEnum):
    """Turbomind model source."""
    WORKSPACE = 'workspace'
    HF_MODEL = 'hf_model'


def get_model_source(pretrained_model_name_or_path: str,
                     **kwargs) -> ModelSource:
    """Get model source."""
    triton_model_path = os.path.join(pretrained_model_name_or_path,
                                     'triton_models')
    if os.path.exists(triton_model_path):
        return ModelSource.WORKSPACE
    return ModelSource.HF_MODEL
