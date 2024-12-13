# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union

import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.utils import get_logger, get_model
from lmdeploy.vl.model.base import VISION_MODELS

from .cogvlm import CogVLMVisionModel  # noqa F401
from .deepseek import DeepSeekVisionModel  # noqa F401
from .glm_4v import GLM4VisionModel  # noqa F401
from .internvl import InternVLVisionModel  # noqa F401
from .internvl_llava import InternVLLlavaVisionModel  # noqa F401
from .llava import LlavaVisionModel  # noqa F401
from .llava_hf import LlavaHfVisionModel  # noqa F401
from .llava_next import LlavaNextVisionModel  # noqa F401
from .mini_gemeni import MiniGeminiVisionModel  # noqa F401
from .minicpmv import MiniCPMVModel  # noqa F401
from .mllama import MllamaVLModel  # noqa F401
from .molmo import MolmoVisionModel  # noqa F401
from .phi3_vision import Phi3VisionModel  # noqa F401
from .qwen import QwenVisionModel  # noqa F401
from .qwen2 import Qwen2VLModel  # noqa F401
from .xcomposer2 import Xcomposer2VisionModel  # noqa F401
from .yi import YiVisionModel  # noqa F401

logger = get_logger('lmdeploy')


def load_vl_model(model_path: str,
                  backend: str,
                  backend_config: Optional[Union[TurbomindEngineConfig,
                                                 PytorchEngineConfig]] = None):
    """load visual model.

    Args:
        model_path(str): the path or repo_id from model hub of the model
        backend(str): the name of inference backend
        backend_config: the config of the inference engine
    """
    if not os.path.exists(model_path):
        revision = getattr(backend_config, 'revision', None)
        download_dir = getattr(backend_config, 'download_dir', None)
        model_path = get_model(model_path,
                               revision=revision,
                               download_dir=download_dir)

    max_memory = None
    tp = getattr(backend_config, 'tp', 1)
    max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(tp)}

    _, hf_config = get_model_arch(model_path)
    kwargs = dict(model_path=model_path,
                  max_memory=max_memory,
                  hf_config=hf_config,
                  backend=backend)
    for name, module in VISION_MODELS.module_dict.items():
        try:
            if module.match(hf_config):
                logger.info(f'matching vision model: {name}')
                model = module(**kwargs)
                model.build_preprocessor()
                if backend == 'turbomind':
                    model.build_model()
                return model
        except Exception:
            logger.error(f'matching vision model: {name} failed')
            raise

    raise ValueError(f'unsupported vl model with config {hf_config}')
