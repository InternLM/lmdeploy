# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS

from .qwen3 import Qwen3VLModel, check_qwen3_vl_deps_install


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model (TurboMind vision path is inherited from
    `Qwen3VLModel`)."""

    _arch = ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']

    def build_preprocessor(self):
        check_qwen3_vl_deps_install()

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token
        self.mm_processor_kwargs = None
