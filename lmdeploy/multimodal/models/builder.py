# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union

import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.multimodal.models.base import BASE_MODELS
from lmdeploy.utils import get_logger, get_model

from .internvl3_hf import InternVL3VisionModel  # noqa F401

logger = get_logger('lmdeploy')


def load_mm_model(model_path: str,
                  backend: str = '',
                  with_llm: bool = False,
                  backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None):
    """Load multimodal model.

    Args:
        model_path(str): the path or repo_id from model hub of the model
        backend(str): the name of inference backend
        with_llm(bool): load LLM model or not. Set it to False for VLM
            inference scenarios and True for VLM quantization
        backend_config: the config of the inference engine
    """
    if not os.path.exists(model_path):
        revision = getattr(backend_config, 'revision', None)
        download_dir = getattr(backend_config, 'download_dir', None)
        model_path = get_model(model_path, revision=revision, download_dir=download_dir)

    max_memory = None
    if not with_llm:
        tp = getattr(backend_config, 'tp', 1)
        max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(tp)}

    _, hf_config = get_model_arch(model_path)
    kwargs = dict(model_path=model_path, with_llm=with_llm, max_memory=max_memory, hf_config=hf_config, backend=backend)

    for name, module in BASE_MODELS.module_dict.items():
        try:
            if module.match(hf_config):
                logger.info(f'matching multimodal model: {name}')
                model = module(**kwargs)
                model.build_preprocessor()
                model.build_model()
                return model
        except Exception as e:
            logger.error(f'build multimodal model {name} failed, {e}')
            raise

    raise ValueError(f'unsupported multimodal model with config {hf_config}')
