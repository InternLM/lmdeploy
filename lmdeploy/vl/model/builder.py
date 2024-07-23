# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union

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
from .phi3_vision import Phi3VisionModel  # noqa F401
from .qwen import QwenVisionModel  # noqa F401
from .xcomposer2 import Xcomposer2VisionModel  # noqa F401
from .yi import YiVisionModel  # noqa F401

logger = get_logger('lmdeploy')


def load_vl_model(model_path: str,
                  with_llm: bool = False,
                  backend_config: Optional[Union[TurbomindEngineConfig,
                                                 PytorchEngineConfig]] = None):
    """load visual model."""
    if not os.path.exists(model_path):
        revision = getattr(backend_config, 'revision', None)
        download_dir = getattr(backend_config, 'download_dir', None)
        model_path = get_model(model_path,
                               revision=revision,
                               download_dir=download_dir)

    max_memory = None
    if not with_llm:
        import torch
        tp = getattr(backend_config, 'tp', 1)
        max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(tp)}

    _, hf_config = get_model_arch(model_path)
    kwargs = dict(model_path=model_path,
                  with_llm=with_llm,
                  max_memory=max_memory,
                  hf_config=hf_config)
    for name, module in VISION_MODELS.module_dict.items():
        try:
            if module.match(hf_config):
                logger.info(f'matching vision model: {name}')
                return module(**kwargs)
        except Exception:
            logger.error(f'matching vision model: {name} failed')
            raise

    raise ValueError(f'unsupported vl model with config {hf_config}')


def vl_model_with_tokenizer(model_path: str, with_llm: bool = True):
    """load visual model."""
    vl_model = load_vl_model(model_path, with_llm).vl_model
    llm = vl_model
    if hasattr(vl_model, 'language_model'):  # deepseek vl
        llm = vl_model.language_model
    if hasattr(vl_model, 'llm'):  # MiniCPMV
        llm = vl_model.llm
    llm.config.use_cache = False
    llm.half().eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    return vl_model, llm, tokenizer
