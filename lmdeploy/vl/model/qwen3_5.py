# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS

from .qwen3 import Qwen3VLModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model."""

    _arch = ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]
        self.mm_processor_kwargs = None
